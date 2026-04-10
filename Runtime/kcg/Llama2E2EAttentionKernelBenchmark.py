#!/usr/bin/env python3
"""
Benchmark a Llama-2-style decoder stack end-to-end with:
1. pure PyTorch attention formula
2. our attention kernel for the selected op, PyTorch for the rest

Supported ops:
  attn
  h2o
  gemma2

Preferred config jsons are loaded from Runtime/kcg/*.json. If the preferred
aggregated json is absent locally, the script falls back to:
  Runtime/kcg/testattn_<op>_combined_<seqlen>.json
  bench_attention/A100/testattn_<op>_combined_<seqlen>.json

Examples:
  python Runtime/kcg/Llama2E2EAttentionKernelBenchmark.py --op attn --seqlen 2048 --dtype float32
  python Runtime/kcg/Llama2E2EAttentionKernelBenchmark.py --op h2o --seqlen 4096 --layers 32
  python Runtime/kcg/Llama2E2EAttentionKernelBenchmark.py --op gemma2 --seqlen 2048
"""

import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from kcg.Kernel import CompiledKernel, KernelConfigs
from kcg.Operators.attention_gemma2 import _gemma2_split_k1, _gemma2_split_k2
from kcg.Operators.attention_h2o import _h2o_split_k1, _h2o_split_k2, _h2o_split_k3
from kcg.Operators.attention_split import _attn_split_k1, _attn_split_k2
from kcg.TorchNamespace import dev_name, torch_ns
from kcg.Utils import DeviceInfo, EnumBackendType, PathManager, is_hip


GEMMA2_TANH_SCALE = 50.0

OP_SPECS = {
    "attn": {
        "required_names": ("k1_name", "k2_name"),
        "candidate_jsons": ("testattn_name_combined_seqlen.json",),
        "runtime_pattern": "testattn_split_combined_{seqlen}.json",
        "fallback_pattern": "testattn_split_combined_{seqlen}.json",
    },
    "h2o": {
        "required_names": ("k1_name", "k2_name", "k3_name"),
        "candidate_jsons": (
            "testh2o_name_combined_seqlen.json",
            "testattn_h2o_name_combined_seqlen.json",
            "testattn_h2o_combined.json",
        ),
        "runtime_pattern": "testattn_h2o_combined_{seqlen}.json",
        "fallback_pattern": "testattn_h2o_combined_{seqlen}.json",
    },
    "gemma2": {
        "required_names": ("k1_name", "k2_name"),
        "candidate_jsons": (
            "testgemma2_name_combined_seqlen.json",
            "testattn_gemma2_name_combined_seqlen.json",
            "testattn_gemma2_combined.json",
        ),
        "runtime_pattern": "testattn_gemma2_combined_{seqlen}.json",
        "fallback_pattern": "testattn_gemma2_combined_{seqlen}.json",
    },
}

OP_CHOICES = tuple(OP_SPECS.keys()) + ("all",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Llama-2-style decoder stack using PyTorch attention vs our selected attention kernel."
    )
    parser.add_argument("--op", choices=OP_CHOICES, default="attn")
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length to benchmark.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--layers", type=int, default=32, help="Number of decoder layers. Defaults to Llama-2-7B style.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-output-head", action="store_true", help="Include the final LM head projection.")
    parser.add_argument("--config-json", default="", help="Optional explicit config json path.")
    parser.add_argument(
        "--fallback-root",
        default=str(Path(__file__).resolve().parents[2] / "bench_attention"),
        help="Fallback root containing per-seqlen benchmark jsons.",
    )
    parser.add_argument("--hidden-size", type=int, default=None, help="Optional override. Must match heads * head_dim from the selected kernel.")
    parser.add_argument("--num-heads", type=int, default=None, help="Optional override. Must match the selected kernel.")
    parser.add_argument("--head-dim", type=int, default=None, help="Optional override. Must match the selected kernel.")
    return parser.parse_args()


def to_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]

def summarize_times(times_ms) -> Dict[str, float]:
    return {
        "median_ms": float(statistics.median(times_ms)),
        "mean_ms": float(statistics.mean(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def parse_kernel_name(name: str) -> Tuple[Tuple[int, int, int, int], Dict[str, int]]:
    shape_m = re.match(r"kcg_[A-Za-z0-9]+_(\d+)_(\d+)_(\d+)_(\d+)_", name)
    if shape_m is None:
        raise ValueError(f"Cannot parse shape from kernel name: {name}")
    shape = tuple(int(shape_m.group(i)) for i in range(1, 5))

    param_pattern = {
        "Br": r"Br(\d+)",
        "Bc": r"Bc(\d+)",
        "Hd": r"Hd(\d+)",
        "Slice1": r"_Sa(\d+)",
        "Slice2": r"Sb(\d+)",
        "PTr": r"PTr(\d+)",
        "PTc": r"PTc(\d+)",
        "OTr": r"OTr(\d+)",
        "OTc": r"OTc(\d+)",
        "GLOB_LOAD_WIDTH_Q": r"GLWQ(\d+)",
        "GLOB_LOAD_WIDTH_K": r"GLWK(\d+)",
        "GLOB_LOAD_WIDTH_V": r"GLWV(\d+)",
        "BLOCK_LAYOUT_P_Y": r"BLPY(\d+)",
        "BLOCK_LAYOUT_P_X": r"BLPX(\d+)",
        "WARP_LAYOUT_P_Y": r"WLPY(\d+)",
        "WARP_LAYOUT_P_X": r"WLPX(\d+)",
        "BLOCK_SCATTER_WIDTH_Q": r"BSWQ(\d+)",
        "BLOCK_SCATTER_WIDTH_K": r"BSWK(\d+)",
        "WARP_SCATTER_WIDTH_Q": r"WSWQ(\d+)",
        "WARP_SCATTER_WIDTH_K": r"WSWK(\d+)",
        "BLOCK_LAYOUT_O_Y": r"BLOY(\d+)",
        "BLOCK_LAYOUT_O_X": r"BLOX(\d+)",
        "WARP_LAYOUT_O_Y": r"WLOY(\d+)",
        "WARP_LAYOUT_O_X": r"WLOX(\d+)",
        "BLOCK_SCATTER_WIDTH_P": r"BSWP(\d+)",
        "BLOCK_SCATTER_WIDTH_V": r"BSWV(\d+)",
        "WARP_SCATTER_WIDTH_P": r"WSWP(\d+)",
        "WARP_SCATTER_WIDTH_V": r"WSWV(\d+)",
        "UNROLL_NUM": r"Un(\d+)",
        "WARP_SIZE": r"W(\d+)",
        "LOAD_CONTINUOUS_P": r"LCP(\d+)",
        "LOAD_CONTINUOUS_O": r"LCO(\d+)",
        "SHARED_PREFETCH_P": r"SPP(\d+)",
        "REG_PREFETCH_P": r"RPP(\d+)",
        "REG_PREFETCH_O": r"RPO(\d+)",
        "SHUFFLE_P": r"SHP(\d+)",
        "SPLITK_PV": r"SKP(\d+)",
    }
    cfg: Dict[str, int] = {}
    for key, pat in param_pattern.items():
        m = re.search(pat, name)
        if m is not None:
            cfg[key] = int(m.group(1))
    return shape, cfg


def fill_defaults(cfg: Dict[str, int]) -> Dict[str, int]:
    defaults = {
        "Slice2": 4,
        "OTr": cfg.get("PTr", 4),
        "OTc": 8,
        "GLOB_LOAD_WIDTH_V": 4,
        "BLOCK_LAYOUT_O_Y": cfg.get("BLOCK_LAYOUT_P_Y", 2),
        "BLOCK_LAYOUT_O_X": cfg.get("BLOCK_LAYOUT_P_X", 1),
        "WARP_LAYOUT_O_Y": cfg.get("WARP_LAYOUT_P_Y", 4),
        "WARP_LAYOUT_O_X": cfg.get("WARP_LAYOUT_P_X", 8),
        "BLOCK_SCATTER_WIDTH_P": cfg.get("BLOCK_SCATTER_WIDTH_Q", 4),
        "BLOCK_SCATTER_WIDTH_V": 4,
        "WARP_SCATTER_WIDTH_P": cfg.get("WARP_SCATTER_WIDTH_Q", 4),
        "WARP_SCATTER_WIDTH_V": 4,
        "LOAD_CONTINUOUS_O": 1,
        "REG_PREFETCH_O": 0,
        "SHUFFLE_P": 0,
        "SPLITK_PV": 0,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def compute_launch_params(shape: Tuple[int, int, int, int], cfg: Dict[str, int], dtype: torch.dtype):
    _, heads, seqlen, head_dim = shape
    br = cfg["Br"]
    bc = cfg["Bc"]
    hd = cfg.get("Hd", head_dim)
    ptr = cfg["PTr"]
    ptc = cfg["PTc"]
    slice1 = cfg["Slice1"]
    slice2 = cfg.get("Slice2", 4)
    th_num = (br // ptr) * (bc // ptc)
    shm = br * slice1 + bc * slice1 + br * bc + hd * slice2 + 3 * br
    if slice1 != hd:
        shm += hd * br
    if cfg.get("SHARED_PREFETCH_P", 0) == 1:
        shm += bc * slice1
    type_width = 4 if dtype == torch.float32 else 2
    grid = [seqlen // br, heads, shape[0]]
    block = [th_num, 1, 1]
    return grid, block, shm * type_width


def explicit_candidate_paths(op_name: str, config_json: str):
    if config_json:
        return [config_json]
    base = Path(__file__).resolve().parent
    paths = [str(base / name) for name in OP_SPECS[op_name]["candidate_jsons"]]
    runtime_pattern = OP_SPECS[op_name].get("runtime_pattern")
    if runtime_pattern:
        paths.append(str(base / runtime_pattern.format(seqlen="{seqlen}")))
    return paths


def find_runtime_local_json(op_name: str, seqlen: int) -> Optional[str]:
    runtime_pattern = OP_SPECS[op_name].get("runtime_pattern")
    if not runtime_pattern:
        return None
    path = Path(__file__).resolve().parent / runtime_pattern.format(seqlen=seqlen)
    if path.exists():
        return str(path)
    return None


def find_fallback_json(op_name: str, seqlen: int, fallback_root: str) -> Optional[str]:
    root = Path(fallback_root)
    pattern = OP_SPECS[op_name]["fallback_pattern"].format(seqlen=seqlen)
    preferred = root / "A100" / pattern
    if preferred.exists():
        return str(preferred)

    matches = sorted(root.glob(f"*/{pattern}"))
    if matches:
        return str(matches[0])
    return None


def _extract_kernel_name_field(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "kernel_name", "kernelName"):
            inner = value.get(key)
            if isinstance(inner, str):
                return inner
    return None


def _infer_seqlen_from_kernel_name(name: str) -> Optional[int]:
    try:
        shape, _ = parse_kernel_name(name)
        return int(shape[2])
    except Exception:
        return None


def _path_mentions_seqlen(path_tokens, seqlen: int) -> bool:
    target = str(seqlen)
    for token in path_tokens:
        text = str(token)
        if text == target:
            return True
        if re.search(rf"(^|[^0-9]){re.escape(target)}([^0-9]|$)", text):
            return True
    return False


def _coerce_kernel_entry(candidate: Any, seqlen: int, required_names, path_tokens=()) -> Optional[Dict[str, Any]]:
    if isinstance(candidate, (list, tuple)):
        if len(candidate) < len(required_names):
            return None
        normalized = {}
        for key, value in zip(required_names, candidate):
            kernel_name = _extract_kernel_name_field(value)
            if kernel_name is None:
                return None
            normalized[key] = kernel_name
        candidate = normalized

    if not isinstance(candidate, dict):
        return None

    alias_map = {
        "k1_name": "k1",
        "k2_name": "k2",
        "k3_name": "k3",
    }
    entry = dict(candidate)
    for key in required_names:
        if key in entry:
            kernel_name = _extract_kernel_name_field(entry[key])
            if kernel_name is None:
                return None
            entry[key] = kernel_name
            continue
        alias = alias_map.get(key)
        if alias is None or alias not in entry:
            return None
        kernel_name = _extract_kernel_name_field(entry[alias])
        if kernel_name is None:
            return None
        entry[key] = kernel_name

    inferred_seqlen = None
    if "seqlen" in entry:
        try:
            inferred_seqlen = int(entry["seqlen"])
        except Exception:
            inferred_seqlen = None
    if inferred_seqlen is None:
        for key in required_names:
            inferred_seqlen = _infer_seqlen_from_kernel_name(entry[key])
            if inferred_seqlen is not None:
                break
    if inferred_seqlen is None and _path_mentions_seqlen(path_tokens, seqlen):
        inferred_seqlen = seqlen

    if inferred_seqlen is not None and inferred_seqlen != seqlen:
        return None

    entry["seqlen"] = seqlen
    return entry


def _search_kernel_entry(obj: Any, seqlen: int, required_names, path_tokens=()) -> Optional[Dict[str, Any]]:
    direct = _coerce_kernel_entry(obj, seqlen, required_names, path_tokens=path_tokens)
    if direct is not None:
        return direct

    if isinstance(obj, dict):
        for key, value in obj.items():
            nested = _search_kernel_entry(value, seqlen, required_names, path_tokens + (key,))
            if nested is not None:
                return nested
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            nested = _search_kernel_entry(item, seqlen, required_names, path_tokens + (str(index),))
            if nested is not None:
                return nested
    return None


def load_kernel_entry(op_name: str, seqlen: int, config_json: str, fallback_root: str) -> Tuple[Dict[str, Any], str]:
    required_names = OP_SPECS[op_name]["required_names"]
    candidate_paths = []
    if config_json:
        candidate_paths.append(config_json)
    else:
        runtime_local_json = find_runtime_local_json(op_name, seqlen)
        if runtime_local_json is not None:
            candidate_paths.append(runtime_local_json)
        candidate_paths.extend(
            str(Path(__file__).resolve().parent / name)
            for name in OP_SPECS[op_name]["candidate_jsons"]
        )
    fallback_json = find_fallback_json(op_name, seqlen, fallback_root)
    if fallback_json is not None and fallback_json not in candidate_paths:
        candidate_paths.append(fallback_json)

    missing_paths = []
    unparsable_paths = []
    for path in candidate_paths:
        if not path:
            continue
        if not os.path.exists(path):
            missing_paths.append(path)
            continue
        with open(path) as f:
            payload = json.load(f)

        direct = _coerce_kernel_entry(payload, seqlen, required_names)
        if direct is not None:
            return direct, path

        if isinstance(payload, dict):
            keyed = payload.get(str(seqlen))
            direct = _coerce_kernel_entry(keyed, seqlen, required_names, path_tokens=(str(seqlen),))
            if direct is not None:
                return direct, path

            for list_key in ("results", "items", "entries", "configs", "data"):
                items = payload.get(list_key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    direct = _coerce_kernel_entry(item, seqlen, required_names, path_tokens=(list_key,))
                    if direct is not None:
                        return direct, path

        recursive = _search_kernel_entry(payload, seqlen, required_names)
        if recursive is not None:
            return recursive, path
        unparsable_paths.append(path)

    searched = ", ".join(candidate_paths)
    details = []
    if missing_paths:
        details.append("missing=" + ", ".join(missing_paths))
    if unparsable_paths:
        details.append("no_matching_entry=" + ", ".join(unparsable_paths))
    detail_text = f" ({'; '.join(details)})" if details else ""
    raise FileNotFoundError(
        f"Cannot find a kernel entry for op={op_name}, seqlen={seqlen}. Searched: {searched}{detail_text}"
    )


def load_deepgen_module():
    lib_path = PathManager.kcg_lib_deepgen_path()
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"libdeepgen not found: {lib_path}")
    spec = importlib.util.spec_from_file_location("deepgen", lib_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def backend_and_arch(device_id: Optional[int] = None) -> Tuple[int, EnumBackendType, str]:
    if is_hip():
        return 2, EnumBackendType.HIP, "906"
    device_count = int(torch.cuda.device_count())
    candidate_ids = []
    if device_id is not None:
        candidate_ids.append(device_id)
    try:
        candidate_ids.append(torch.cuda.current_device())
    except Exception:
        pass
    candidate_ids.extend(range(device_count))

    visited = set()
    for cand in candidate_ids:
        if cand in visited or cand is None:
            continue
        visited.add(cand)
        if not (0 <= int(cand) < device_count):
            continue
        try:
            major, minor = torch.cuda.get_device_capability(int(cand))
            return 1, EnumBackendType.CUDA, f"{major}{minor}"
        except Exception:
            continue

    raise RuntimeError(
        f"Unable to query CUDA device capability. device_count={device_count}, candidates={candidate_ids}"
    )


def select_runtime_device(requested_device_id: int) -> int:
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        raise RuntimeError("No CUDA devices are visible in the current process.")

    if 0 <= requested_device_id < device_count:
        runtime_id = requested_device_id
    elif device_count == 1:
        runtime_id = 0
        print(
            f"[device] requested GPU {requested_device_id} is not a valid logical id in this process; "
            f"falling back to visible logical device 0",
            flush=True,
        )
    else:
        raise RuntimeError(
            f"Requested device id {requested_device_id} is invalid for the current process. "
            f"Visible logical device count: {device_count}. "
            f"If you selected a physical GPU via CUDA_VISIBLE_DEVICES, pass its logical id instead."
        )

    DeviceInfo.set_current_device(runtime_id)
    return DeviceInfo.get_current_device()


def _sig_attn_k1(dtype: torch.dtype):
    return lambda: _attn_split_k1(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
    )


def _sig_attn_k2(dtype: torch.dtype):
    return lambda: _attn_split_k2(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.empty(1, 3, 100, 100, dtype=dtype),
    )


def _sig_h2o_k1(dtype: torch.dtype):
    return lambda: _h2o_split_k1(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
    )


def _sig_h2o_k2(dtype: torch.dtype):
    return lambda: _h2o_split_k2(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, dtype=dtype),
    )


def _sig_h2o_k3(dtype: torch.dtype):
    return lambda: _h2o_split_k3(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.empty(1, 3, 100, 100, dtype=dtype),
    )


def _sig_gemma2_k1(dtype: torch.dtype):
    return lambda: _gemma2_split_k1(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
    )


def _sig_gemma2_k2(dtype: torch.dtype):
    return lambda: _gemma2_split_k2(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.randn(1, 3, 100, 1, dtype=dtype),
        torch.empty(1, 3, 100, 100, dtype=dtype),
    )


def make_kernel(
    binary_path: str,
    kernel_name: str,
    shape: Tuple[int, int, int, int],
    cfg: Dict[str, int],
    n_dtypes: int,
    sig_func,
    backend: EnumBackendType,
    dtype: torch.dtype,
    device_id: int,
) -> CompiledKernel:
    grid, block, shm = compute_launch_params(shape, cfg, dtype)
    kernel_cfg = KernelConfigs(binary_path, kernel_name, [dtype] * n_dtypes, backend)
    kernel_cfg.m_gridDims = grid
    kernel_cfg.m_blockDims = block
    kernel_cfg.shmBytes = shm
    signature = sig_func()
    return CompiledKernel(
        kernel_cfg.backend,
        kernel_cfg.binaryPath,
        kernel_cfg.kernelFuncName,
        kernel_cfg.sharedMem(),
        signature,
        kernel_cfg.gridDims(),
        kernel_cfg.blockDims(),
        device_id,
    )


class AttentionKernelRunner:
    def __init__(self, op_name: str, kernel_entry: Dict[str, Any], dtype_name: str, device_id: int):
        self.op_name = op_name
        self.kernel_entry = dict(kernel_entry)
        self.dtype_name = dtype_name
        self.dtype = to_torch_dtype(dtype_name)
        self.requested_device_id = device_id
        self.device_id = device_id
        self.device = None
        self.workspace: Dict[Tuple[int, int, int, int, torch.dtype, str], Tuple[torch.Tensor, ...]] = {}

        PathManager.init()
        if not torch_ns.is_available():
            raise RuntimeError("No GPU runtime is available for this benchmark.")

        self.device_id = select_runtime_device(device_id)
        self.device = torch.device(dev_name(self.device_id))

        required_names = OP_SPECS[op_name]["required_names"]
        self.kernel_names = [self.kernel_entry[key] for key in required_names]
        self.shapes = []
        self.cfgs = []
        for kernel_name in self.kernel_names:
            shape, cfg = parse_kernel_name(kernel_name)
            fill_defaults(cfg)
            self.shapes.append(shape)
            self.cfgs.append(cfg)

        if len(set(self.shapes)) != 1:
            raise ValueError(f"Kernel shape mismatch: {self.shapes}")
        self.shape = self.shapes[0]

        compile_begin = time.perf_counter()
        mod = load_deepgen_module()
        backend_id, backend, arch = backend_and_arch(self.device_id)
        mod.set_platform(backend_id, arch)
        self.backend = backend

        if op_name == "attn":
            compile_names = ("compile_attn_split_k1", "compile_attn_split_k2")
            n_dtypes = (4, 6)
            sig_builders = (_sig_attn_k1, _sig_attn_k2)
        elif op_name == "h2o":
            compile_names = ("compile_h2o_split_k1", "compile_h2o_split_k2", "compile_h2o_split_k3")
            n_dtypes = (4, 5, 6)
            sig_builders = (_sig_h2o_k1, _sig_h2o_k2, _sig_h2o_k3)
        elif op_name == "gemma2":
            compile_names = ("compile_gemma2_split_k1", "compile_gemma2_split_k2")
            n_dtypes = (4, 6)
            sig_builders = (_sig_gemma2_k1, _sig_gemma2_k2)
        else:
            raise ValueError(f"Unsupported op: {op_name}")

        self.kernels = []
        for kernel_name, cfg, compile_name, dtype_count, sig_builder in zip(
            self.kernel_names, self.cfgs, compile_names, n_dtypes, sig_builders
        ):
            compile_func = getattr(mod, compile_name, None)
            if compile_func is None:
                raise RuntimeError(f"{compile_name} not found in libdeepgen")
            if hasattr(mod, "set_kernel_name"):
                mod.set_kernel_name(kernel_name)
            hsaco = compile_func(list(self.shape), {kernel_name: cfg}, dtype_name)
            if not hsaco:
                raise RuntimeError(f"Compilation failed for {kernel_name}")
            self.kernels.append(
                make_kernel(hsaco, kernel_name, self.shape, cfg, dtype_count, sig_builder(self.dtype), backend, self.dtype, self.device_id)
            )

        self.compile_ms = (time.perf_counter() - compile_begin) * 1000.0

    def _get_workspace(self, batch_size: int, heads: int, seqlen: int, head_dim: int):
        key = (batch_size, heads, seqlen, head_dim, self.dtype, str(self.device))
        if key not in self.workspace:
            em = torch.empty((batch_size, heads, seqlen, 1), dtype=self.dtype, device=self.device)
            denom = torch.empty((batch_size, heads, seqlen, 1), dtype=self.dtype, device=self.device)
            out = torch.empty((batch_size, heads, seqlen, head_dim), dtype=self.dtype, device=self.device)
            if self.op_name == "h2o":
                row_sum = torch.empty((batch_size, heads, seqlen), dtype=self.dtype, device=self.device)
                self.workspace[key] = (em, denom, row_sum, out)
            else:
                self.workspace[key] = (em, denom, out)
        return self.workspace[key]

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, heads, seqlen, head_dim = q.shape
        expected = self.shape
        current = (batch_size, heads, seqlen, head_dim)
        if current != expected:
            raise ValueError(f"Input shape {current} does not match compiled kernel shape {expected}")
        if q.dtype != self.dtype or k.dtype != self.dtype or v.dtype != self.dtype:
            raise TypeError(f"Input dtype must be {self.dtype}, got {q.dtype}, {k.dtype}, {v.dtype}")

        qq = q.transpose(-1, -2).contiguous()
        kk = k.transpose(-1, -2).contiguous()

        if self.op_name == "h2o":
            em, denom, row_sum, out = self._get_workspace(batch_size, heads, seqlen, head_dim)
            self.kernels[0].run(qq, kk, em, denom)
            self.kernels[1].run(kk, qq, em, denom, row_sum)
            self.kernels[2].run(qq, kk, v, em, denom, out)
            return out

        em, denom, out = self._get_workspace(batch_size, heads, seqlen, head_dim)
        self.kernels[0].run(qq, kk, em, denom)
        self.kernels[1].run(qq, kk, v, em, denom, out)
        return out


@dataclass
class LlamaLikeConfig:
    batch_size: int
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    max_seq_len: int
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    with_output_head: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaLikeAttention(nn.Module):
    def __init__(self, config: LlamaLikeConfig, head_dim: int, op_name: str, kernel_runner: AttentionKernelRunner):
        super().__init__()
        self.op_name = op_name
        self.n_heads = config.n_heads
        self.head_dim = head_dim
        self.impl_mode = "torch"
        self.kernel_runner = kernel_runner

        proj_dim = self.n_heads * self.head_dim
        self.wq = nn.Linear(config.dim, proj_dim, bias=False)
        self.wk = nn.Linear(config.dim, proj_dim, bias=False)
        self.wv = nn.Linear(config.dim, proj_dim, bias=False)
        self.wo = nn.Linear(proj_dim, config.dim, bias=False)

    def set_impl_mode(self, mode: str) -> None:
        if mode not in ("torch", "kernel"):
            raise ValueError(f"Unsupported attention impl mode: {mode}")
        self.impl_mode = mode

    def _torch_attention(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.op_name == "gemma2":
            scores = torch.tanh(scores / GEMMA2_TANH_SCALE) * GEMMA2_TANH_SCALE
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores.float(), dim=-1).type_as(query)
        if self.op_name == "h2o":
            _ = probs.sum(dim=2, keepdim=False)
        return torch.matmul(probs, values)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seqlen, _ = x.shape

        xq = self.wq(x).view(batch_size, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        query = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        if self.impl_mode == "kernel":
            output = self.kernel_runner(query, keys, values)
        else:
            output = self._torch_attention(query, keys, values, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * (4 * dim) / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * ffn_dim_multiplier)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaLikeConfig, head_dim: int, op_name: str, kernel_runner: AttentionKernelRunner):
        super().__init__()
        self.attention = LlamaLikeAttention(config, head_dim, op_name, kernel_runner)
        self.feed_forward = FeedForward(config.dim, config.multiple_of, config.ffn_dim_multiplier)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def set_attention_impl(self, mode: str) -> None:
        self.attention.set_impl_mode(mode)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class LlamaLikeModel(nn.Module):
    def __init__(self, config: LlamaLikeConfig, head_dim: int, op_name: str, kernel_runner: AttentionKernelRunner):
        super().__init__()
        self.config = config
        self.op_name = op_name
        self.head_dim = head_dim
        self.attention_impl = "torch"

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, head_dim, op_name, kernel_runner) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False) if config.with_output_head else None

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len * 2),
            persistent=False,
        )

    def set_attention_impl(self, mode: str) -> None:
        self.attention_impl = mode
        for layer in self.layers:
            layer.set_attention_impl(mode)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)

        mask = None
        if self.attention_impl == "torch" and seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        if self.output is not None:
            h = self.output(h)
        return h


def benchmark_model(model: LlamaLikeModel, tokens: torch.Tensor, mode: str, warmup: int, iters: int):
    model.set_attention_impl(mode)
    times_ms = []
    output = None

    with torch.inference_mode():
        for _ in range(warmup):
            output = model(tokens)
        torch_ns.synchronize()

        for _ in range(iters):
            start = torch_ns.Event(enable_timing=True)
            end = torch_ns.Event(enable_timing=True)
            start.record()
            output = model(tokens)
            end.record()
            torch_ns.synchronize()
            times_ms.append(float(start.elapsed_time(end)))

    return output, summarize_times(times_ms)


def build_model_config(args: argparse.Namespace, kernel_shape: Tuple[int, int, int, int]) -> Tuple[LlamaLikeConfig, int]:
    batch_size, kernel_heads, _, kernel_head_dim = kernel_shape

    num_heads = args.num_heads if args.num_heads is not None else kernel_heads
    head_dim = args.head_dim if args.head_dim is not None else kernel_head_dim
    hidden_size = args.hidden_size if args.hidden_size is not None else num_heads * head_dim

    if num_heads != kernel_heads:
        raise ValueError(f"--num-heads={num_heads} does not match kernel heads={kernel_heads}")
    if head_dim != kernel_head_dim:
        raise ValueError(f"--head-dim={head_dim} does not match kernel head_dim={kernel_head_dim}")
    if hidden_size != kernel_heads * kernel_head_dim:
        raise ValueError(
            f"--hidden-size={hidden_size} does not match kernel hidden size={kernel_heads * kernel_head_dim}"
        )

    cfg = LlamaLikeConfig(
        batch_size=batch_size,
        dim=hidden_size,
        n_layers=args.layers,
        n_heads=num_heads,
        vocab_size=args.vocab_size,
        max_seq_len=args.seqlen,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        norm_eps=args.norm_eps,
        with_output_head=args.with_output_head,
    )
    return cfg, head_dim


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    kernel_entry, kernel_source = load_kernel_entry(op_name, args.seqlen, args.config_json, args.fallback_root)
    runner = AttentionKernelRunner(op_name, kernel_entry, args.dtype, args.device_id)
    model_cfg, head_dim = build_model_config(args, runner.shape)

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(dev_name(runner.device_id))

    model = LlamaLikeModel(model_cfg, head_dim, op_name, runner).to(device=device, dtype=dtype)
    model.eval()

    tokens = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(model_cfg.batch_size, model_cfg.max_seq_len),
        dtype=torch.long,
        device=device,
    )

    _, torch_stats = benchmark_model(model, tokens, "torch", args.warmup, args.iters)
    _, kernel_stats = benchmark_model(model, tokens, "kernel", args.warmup, args.iters)

    ffn_hidden = model.layers[0].feed_forward.hidden_dim if model.layers else 0
    param_count = int(sum(p.numel() for p in model.parameters()))
    speedup = torch_stats["median_ms"] / kernel_stats["median_ms"] if kernel_stats["median_ms"] > 0 else 0.0

    result = {
        "op": op_name,
        "kernel_config_source": kernel_source,
        "kernel_entry": kernel_entry,
        "compile_ms": runner.compile_ms,
        "model": {
            "family": "llama2_style",
            "requested_device_id": args.device_id,
            "runtime_device_id": runner.device_id,
            "batch_size": model_cfg.batch_size,
            "seqlen": model_cfg.max_seq_len,
            "hidden_size": model_cfg.dim,
            "num_layers": model_cfg.n_layers,
            "num_heads": model_cfg.n_heads,
            "head_dim": head_dim,
            "ffn_hidden_size": ffn_hidden,
            "multiple_of": model_cfg.multiple_of,
            "with_output_head": model_cfg.with_output_head,
            "parameter_count": param_count,
            "dtype": args.dtype,
            "device": str(device),
        },
        "torch_e2e": torch_stats,
        "kernel_e2e": kernel_stats,
        "speedup": float(speedup),
    }

    print(f"[op] {op_name}")
    print(f"[config] source={kernel_source}")
    print(f"[device] requested={args.device_id} runtime={runner.device_id}")
    for idx, kernel_name in enumerate(runner.kernel_names, start=1):
        print(f"[kernel{idx}] {kernel_name}")
    print(
        f"[model] batch={model_cfg.batch_size} seqlen={model_cfg.max_seq_len} dim={model_cfg.dim} "
        f"layers={model_cfg.n_layers} heads={model_cfg.n_heads} head_dim={head_dim} ffn={ffn_hidden} "
        f"output_head={model_cfg.with_output_head}"
    )
    print(f"[compile] {runner.compile_ms:.3f} ms")
    print(f"[torch] median={torch_stats['median_ms']:.3f} ms mean={torch_stats['mean_ms']:.3f} ms")
    print(f"[kernel] median={kernel_stats['median_ms']:.3f} ms mean={kernel_stats['mean_ms']:.3f} ms")
    print(f"[speedup] {speedup:.4f}x")

    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.op == "all":
        results = []
        for index, op_name in enumerate(OP_SPECS.keys(), start=1):
            if index > 1:
                print("")
            print(f"========== [{index}/{len(OP_SPECS)}] {op_name} ==========")
            results.append(run_single_op(op_name, args))

        summary = {
            "mode": "all",
            "ops": {item["op"]: item for item in results},
            "speedup_summary": {
                item["op"]: item["speedup"] for item in results
            },
        }

        print("")
        print("[summary]")
        for item in results:
            print(
                f"{item['op']}: torch={item['torch_e2e']['median_ms']:.3f} ms "
                f"kernel={item['kernel_e2e']['median_ms']:.3f} ms "
                f"speedup={item['speedup']:.4f}x"
            )
    else:
        summary = run_single_op(args.op, args)

if __name__ == "__main__":
    main()
