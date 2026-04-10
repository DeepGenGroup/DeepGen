#!/usr/bin/env python3
"""
Three-way ablation benchmark for attention operators.

It compares:
1. PyTorch baseline for the original sequence
2. Origin fused kernel path
3. Combined graph-optimized kernel path

Supported ops:
- attn
- gemma2
- h2o

Examples:
  python Runtime/kcg/AttentionKernelAblationBenchmark.py --op attn --seqlen 2048
  python Runtime/kcg/AttentionKernelAblationBenchmark.py --op gemma2 --seqlen 4096 --dtype float16
  python Runtime/kcg/AttentionKernelAblationBenchmark.py --op all --seqlen 4096 --iters 20
  python Runtime/kcg/AttentionKernelAblationBenchmark.py --op gemma2 --seqlen 4096 \
      --origin-json Runtime/kcg/testattn_gemma2_origin_4096.json \
      --combined-json Runtime/kcg/testattn_gemma2_combined_4096.json
"""

import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from kcg.Kernel import CompiledKernel, KernelConfigs
from kcg.Operators.attention import _attention
from kcg.Operators.attention_gemma2 import _gemma2_split_k1, _gemma2_split_k2
from kcg.Operators.attention_h2o import _h2o_split_k1, _h2o_split_k2, _h2o_split_k3
from kcg.Operators.attention_h2o_origin import _h2o_origin
from kcg.Operators.attention_split import _attn_split_k1, _attn_split_k2
from kcg.TorchNamespace import dev_name, torch_ns
from kcg.Utils import DeviceInfo, EnumBackendType, PathManager, is_hip


GEMMA2_TANH_SCALE = 50.0
TIMING_METHOD = "torch_ns.Event device time"

SEQUENCE_DESCRIPTIONS = {
    "attn": {
        "baseline": (
            "scores=q@k^T -> scores=scores/sqrt(d) -> mask -> row_max -> "
            "p_scaled=exp(scores-row_max) -> denom=sum(p_scaled) -> probs=p_scaled/denom -> out=probs@v"
        ),
        "origin": (
            "single fused kernel for the original sequence: q_scale@k^T -> mask -> "
            "max/sub/exp/sum -> pv -> out"
        ),
        "combined": (
            "K1: q_scale@k^T -> mask -> row_max/em=exp(row_max), denom=sum(exp(scores))/em; "
            "K2: q_scale@k^T -> mask -> p_scaled=exp(scores-row_max) -> pv=p_scaled@v -> out=pv/denom"
        ),
    },
    "gemma2": {
        "baseline": (
            "scores=q@k^T -> scores=scores/sqrt(d) -> y=tanh(scores/50)*50 -> mask -> row_max -> "
            "p_scaled=exp(y-row_max) -> denom=sum(p_scaled) -> probs=p_scaled/denom -> out=probs@v"
        ),
        "origin": (
            "single fused kernel for the original sequence: q_scale@k^T -> softcap -> mask -> "
            "max/sub/exp/sum -> pv -> out"
        ),
        "combined": (
            "K1: q_scale@k^T -> softcap -> mask -> row_max/em=exp(row_max), denom=sum(exp(y-row_max)); "
            "K2: q_scale@k^T -> softcap -> mask -> p_scaled=exp(y-row_max) -> pv=p_scaled@v -> out=pv/denom"
        ),
    },
    "h2o": {
        "baseline": (
            "scores=q@k^T -> scores=scores/sqrt(d) -> mask -> row_max -> "
            "p_scaled=exp(scores-row_max) -> denom=sum(p_scaled) -> probs=p_scaled/denom -> out=probs@v; "
            "row_sum=sum(probs, dim=query)"
        ),
        "origin": (
            "single fused kernel for the original sequence: q_scale@k^T -> mask -> "
            "max/sub/exp/sum -> pv -> out, plus in-kernel row_sum accumulation"
        ),
        "combined": (
            "K1: q_scale@k^T -> mask -> row_max/em=exp(row_max), denom=sum(exp(scores))/em; "
            "K2: transposed score path -> p_scaled_t/denom_t -> row_sum; "
            "K3: q_scale@k^T -> mask -> p_scaled=exp(scores-row_max) -> pv=p_scaled@v -> out=pv/denom"
        ),
    },
}

OP_SPECS = {
    "attn": {
        "origin": {
            "required_names": ("kernel_name",),
            "candidate_jsons": (
                "testattn_origin_{seqlen}.json",
                "testattn_origin.json",
                "testattn_split_origin_{seqlen}.json",
                "testattn_split_origin.json",
            ),
            "fallback_pattern": "testattn_origin_{seqlen}.json",
        },
        "combined": {
            "required_names": ("k1_name", "k2_name"),
            "candidate_jsons": (
                "testattn_split_combined_{seqlen}.json",
                "testattn_split_combined.json",
                "testattn_name_combined_seqlen.json",
            ),
            "fallback_pattern": "testattn_split_combined_{seqlen}.json",
        },
    },
    "gemma2": {
        "origin": {
            "required_names": ("kernel_name",),
            "candidate_jsons": (
                "testattn_gemma2_origin_{seqlen}.json",
                "testattn_gemma2_origin.json",
            ),
            "fallback_pattern": "testattn_gemma2_origin_{seqlen}.json",
        },
        "combined": {
            "required_names": ("k1_name", "k2_name"),
            "candidate_jsons": (
                "testattn_gemma2_combined_{seqlen}.json",
                "testattn_gemma2_combined.json",
                "testgemma2_name_combined_seqlen.json",
            ),
            "fallback_pattern": "testattn_gemma2_combined_{seqlen}.json",
        },
    },
    "h2o": {
        "origin": {
            "required_names": ("kernel_name",),
            "candidate_jsons": (
                "testattn_h2o_origin_{seqlen}.json",
                "testattn_h2o_origin.json",
            ),
            "fallback_pattern": "testattn_h2o_origin_{seqlen}.json",
        },
        "combined": {
            "required_names": ("k1_name", "k2_name", "k3_name"),
            "candidate_jsons": (
                "testattn_h2o_combined_{seqlen}.json",
                "testattn_h2o_combined.json",
                "testh2o_name_combined_seqlen.json",
            ),
            "fallback_pattern": "testattn_h2o_combined_{seqlen}.json",
        },
    },
}

OP_CHOICES = tuple(OP_SPECS.keys()) + ("all",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PyTorch baseline vs origin fused kernel vs combined graph-optimized kernels "
            "for attn/gemma2/h2o."
        )
    )
    parser.add_argument("--op", choices=OP_CHOICES, default="attn")
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length to benchmark.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-pattern", choices=("ones", "randn"), default="ones")
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--skip-check", action="store_true", help="Skip correctness comparison.")
    parser.add_argument("--origin-json", default="", help="Optional explicit origin kernel result json for a single op.")
    parser.add_argument(
        "--combined-json",
        default="",
        help="Optional explicit combined kernel json for a single op.",
    )
    parser.add_argument("--config-json", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fallback-root",
        default=str(Path(__file__).resolve().parents[2] / "bench_attention"),
        help="Fallback root containing per-seqlen benchmark jsons.",
    )
    parser.add_argument("--output-json", default="", help="Optional output path for benchmark results.")
    return parser.parse_args()


def to_torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def default_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 2e-2, 2e-2


def summarize_times(times_ms: Sequence[float]) -> Dict[str, float]:
    return {
        "median_ms": float(statistics.median(times_ms)),
        "mean_ms": float(statistics.mean(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def tensor_error_summary(actual: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    diff = (actual - ref).abs()
    rel = diff / (ref.abs() + 1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
    }


def compare_outputs(ref: torch.Tensor, test: torch.Tensor, rtol: float, atol: float) -> Dict[str, Any]:
    return {
        "ok": bool(torch.allclose(ref, test, rtol=rtol, atol=atol)),
        **tensor_error_summary(test, ref),
        "rtol": float(rtol),
        "atol": float(atol),
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
        match = re.search(pat, name)
        if match is not None:
            cfg[key] = int(match.group(1))
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
    batch_size, heads, seqlen, head_dim = shape
    br = cfg["Br"]
    bc = cfg["Bc"]
    hd = cfg.get("Hd", head_dim)
    ptr = cfg["PTr"]
    ptc = cfg["PTc"]
    slice1 = cfg["Slice1"]
    slice2 = cfg.get("Slice2", 4)
    threads = (br // ptr) * (bc // ptc)
    shm = br * slice1 + bc * slice1 + br * bc + hd * slice2 + 3 * br
    if slice1 != hd:
        shm += hd * br
    if cfg.get("SHARED_PREFETCH_P", 0) == 1:
        shm += bc * slice1
    type_width = 4 if dtype == torch.float32 else 2
    grid = [seqlen // br, heads, batch_size]
    block = [threads, 1, 1]
    return grid, block, shm * type_width


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


def _path_mentions_seqlen(path_tokens: Sequence[str], seqlen: int) -> bool:
    target = str(seqlen)
    for token in path_tokens:
        text = str(token)
        if text == target:
            return True
        if re.search(rf"(^|[^0-9]){re.escape(target)}([^0-9]|$)", text):
            return True
    return False


def _alias_candidates(key: str) -> Tuple[str, ...]:
    aliases = {
        "kernel_name": ("name", "kernel", "origin_name"),
        "k1_name": ("k1", "kernel1", "k1Name"),
        "k2_name": ("k2", "kernel2", "k2Name"),
        "k3_name": ("k3", "kernel3", "k3Name"),
    }
    return aliases.get(key, ())


def _coerce_kernel_entry(candidate: Any, seqlen: int, required_names, path_tokens=()) -> Optional[Dict[str, Any]]:
    if isinstance(candidate, str) and len(required_names) == 1:
        candidate = {required_names[0]: candidate}

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

    entry = dict(candidate)
    for key in required_names:
        if key in entry:
            kernel_name = _extract_kernel_name_field(entry[key])
            if kernel_name is None:
                return None
            entry[key] = kernel_name
            continue

        found = None
        for alias in _alias_candidates(key):
            if alias in entry:
                found = _extract_kernel_name_field(entry[alias])
                if found is not None:
                    break
        if found is None:
            return None
        entry[key] = found

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


def explicit_candidate_paths(op_name: str, variant: str, seqlen: int, config_json: str) -> List[str]:
    if config_json:
        return [config_json]
    base = Path(__file__).resolve().parent
    return [str(base / pattern.format(seqlen=seqlen)) for pattern in OP_SPECS[op_name][variant]["candidate_jsons"]]


def find_fallback_json(op_name: str, variant: str, seqlen: int, fallback_root: str) -> Optional[str]:
    pattern = OP_SPECS[op_name][variant].get("fallback_pattern")
    if not pattern:
        return None

    root = Path(fallback_root)
    target = pattern.format(seqlen=seqlen)
    preferred = root / "A100" / target
    if preferred.exists():
        return str(preferred)
    matches = sorted(root.glob(f"*/{target}"))
    if matches:
        return str(matches[0])
    return None


def load_kernel_entry(
    op_name: str,
    variant: str,
    seqlen: int,
    config_json: str,
    fallback_root: str,
) -> Tuple[Dict[str, Any], str]:
    required_names = OP_SPECS[op_name][variant]["required_names"]
    candidate_paths = explicit_candidate_paths(op_name, variant, seqlen, config_json)
    fallback_json = find_fallback_json(op_name, variant, seqlen, fallback_root)
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

        with open(path) as handle:
            payload = json.load(handle)

        direct = _coerce_kernel_entry(payload, seqlen, required_names)
        if direct is not None:
            return direct, path

        if isinstance(payload, dict):
            keyed = payload.get(str(seqlen))
            direct = _coerce_kernel_entry(keyed, seqlen, required_names, path_tokens=(str(seqlen),))
            if direct is not None:
                return direct, path

            for list_key in ("results", "items", "entries", "configs", "data", "testResult"):
                items = payload.get(list_key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    direct = _coerce_kernel_entry(item, seqlen, required_names, path_tokens=(list_key,))
                    if direct is not None:
                        return direct, path

        recursive = _search_kernel_entry(payload, seqlen, required_names)
        if recursive is not None:
            return recursive, path
        unparsable_paths.append(path)

    searched = ", ".join(candidate_paths)
    detail_parts = []
    if missing_paths:
        detail_parts.append("missing=" + ", ".join(missing_paths))
    if unparsable_paths:
        detail_parts.append("no_matching_entry=" + ", ".join(unparsable_paths))
    detail_text = f" ({'; '.join(detail_parts)})" if detail_parts else ""
    raise FileNotFoundError(
        f"Cannot find a kernel entry for op={op_name}, variant={variant}, seqlen={seqlen}. "
        f"Searched: {searched}{detail_text}"
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
            f"Visible logical device count: {device_count}."
        )

    DeviceInfo.set_current_device(runtime_id)
    return DeviceInfo.get_current_device()


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


def _sig_attention_origin(dtype: torch.dtype):
    return lambda: _attention(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.empty(1, 3, 100, 100, dtype=dtype),
    )


def _sig_h2o_origin(dtype: torch.dtype):
    return lambda: _h2o_origin(
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.randn(1, 3, 100, 100, dtype=dtype),
        torch.empty(1, 3, 100, 100, dtype=dtype),
        torch.empty(1, 3, 100, dtype=dtype),
    )


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


class OriginKernelRunner:
    def __init__(self, op_name: str, kernel_entry: Dict[str, Any], dtype_name: str, device_id: int):
        self.op_name = op_name
        self.kernel_entry = dict(kernel_entry)
        self.dtype_name = dtype_name
        self.dtype = to_torch_dtype(dtype_name)
        self.requested_device_id = device_id
        self.device_id = device_id
        self.device = None
        self.workspace = {}

        PathManager.init()
        DeviceInfo.get_current_device()
        if not torch_ns.is_available():
            raise RuntimeError("No GPU runtime is available for this benchmark.")

        self.device_id = select_runtime_device(device_id)
        self.device = torch.device(dev_name(self.device_id))

        self.kernel_name = self.kernel_entry["kernel_name"]
        self.shape, self.cfg = parse_kernel_name(self.kernel_name)
        fill_defaults(self.cfg)

        mod = load_deepgen_module()
        backend_id, backend, arch = backend_and_arch(self.device_id)
        mod.set_platform(backend_id, arch)
        self.backend = backend

        if op_name == "attn":
            compile_name = "compile_attention_origin"
            n_dtypes = 4
            sig_builder = _sig_attention_origin
        elif op_name == "gemma2":
            compile_name = "compile_gemma2_origin"
            n_dtypes = 4
            sig_builder = _sig_attention_origin
        elif op_name == "h2o":
            compile_name = "compile_h2o_origin"
            n_dtypes = 5
            sig_builder = _sig_h2o_origin
        else:
            raise ValueError(f"Unsupported op: {op_name}")

        compile_func = getattr(mod, compile_name, None)
        if compile_func is None:
            raise RuntimeError(f"{compile_name} not found in libdeepgen")

        if hasattr(mod, "set_kernel_name"):
            mod.set_kernel_name(self.kernel_name)
        hsaco = compile_func(list(self.shape), {self.kernel_name: self.cfg}, self.dtype_name)
        if not hsaco:
            raise RuntimeError(f"Compilation failed for {self.kernel_name}")

        self.kernel = make_kernel(
            hsaco,
            self.kernel_name,
            self.shape,
            self.cfg,
            n_dtypes,
            sig_builder(self.dtype),
            backend,
            self.dtype,
            self.device_id,
        )

    def _get_workspace(self, batch_size: int, heads: int, seqlen: int, head_dim: int):
        key = (batch_size, heads, seqlen, head_dim, self.dtype, str(self.device))
        if key not in self.workspace:
            out = torch.empty((batch_size, heads, seqlen, head_dim), dtype=self.dtype, device=self.device)
            if self.op_name == "h2o":
                row_sum = torch.empty((batch_size, heads, seqlen), dtype=self.dtype, device=self.device)
                self.workspace[key] = (out, row_sum)
            else:
                self.workspace[key] = out
        return self.workspace[key]

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, heads, seqlen, head_dim = q.shape
        current = (batch_size, heads, seqlen, head_dim)
        if current != self.shape:
            raise ValueError(f"Input shape {current} does not match compiled kernel shape {self.shape}")
        if q.dtype != self.dtype or k.dtype != self.dtype or v.dtype != self.dtype:
            raise TypeError(f"Input dtype must be {self.dtype}")

        qq = q.transpose(-1, -2).contiguous()
        kk = k.transpose(-1, -2).contiguous()

        if self.op_name == "h2o":
            out, row_sum = self._get_workspace(batch_size, heads, seqlen, head_dim)
            row_sum.zero_()
            self.kernel.run(qq, kk, v, out, row_sum)
            return {"out": out, "row_sum": row_sum}

        out = self._get_workspace(batch_size, heads, seqlen, head_dim)
        self.kernel.run(qq, kk, v, out)
        return {"out": out}


class CombinedKernelRunner:
    def __init__(self, op_name: str, kernel_entry: Dict[str, Any], dtype_name: str, device_id: int):
        self.op_name = op_name
        self.kernel_entry = dict(kernel_entry)
        self.dtype_name = dtype_name
        self.dtype = to_torch_dtype(dtype_name)
        self.requested_device_id = device_id
        self.device_id = device_id
        self.device = None
        self.workspace = {}

        PathManager.init()
        DeviceInfo.get_current_device()
        if not torch_ns.is_available():
            raise RuntimeError("No GPU runtime is available for this benchmark.")

        self.device_id = select_runtime_device(device_id)
        self.device = torch.device(dev_name(self.device_id))

        required_names = OP_SPECS[op_name]["combined"]["required_names"]
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

        mod = load_deepgen_module()
        backend_id, backend, arch = backend_and_arch(self.device_id)
        mod.set_platform(backend_id, arch)
        self.backend = backend

        if op_name == "attn":
            compile_names = ("compile_attn_split_k1", "compile_attn_split_k2")
            n_dtypes = (4, 6)
            sig_builders = (_sig_attn_k1, _sig_attn_k2)
        elif op_name == "gemma2":
            compile_names = ("compile_gemma2_split_k1", "compile_gemma2_split_k2")
            n_dtypes = (4, 6)
            sig_builders = (_sig_gemma2_k1, _sig_gemma2_k2)
        elif op_name == "h2o":
            compile_names = ("compile_h2o_split_k1", "compile_h2o_split_k2", "compile_h2o_split_k3")
            n_dtypes = (4, 5, 6)
            sig_builders = (_sig_h2o_k1, _sig_h2o_k2, _sig_h2o_k3)
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
            hsaco = compile_func(list(self.shape), {kernel_name: cfg}, self.dtype_name)
            if not hsaco:
                raise RuntimeError(f"Compilation failed for {kernel_name}")
            self.kernels.append(
                make_kernel(
                    hsaco,
                    kernel_name,
                    self.shape,
                    cfg,
                    dtype_count,
                    sig_builder(self.dtype),
                    backend,
                    self.dtype,
                    self.device_id,
                )
            )

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

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, heads, seqlen, head_dim = q.shape
        current = (batch_size, heads, seqlen, head_dim)
        if current != self.shape:
            raise ValueError(f"Input shape {current} does not match compiled kernel shape {self.shape}")
        if q.dtype != self.dtype or k.dtype != self.dtype or v.dtype != self.dtype:
            raise TypeError(f"Input dtype must be {self.dtype}")

        qq = q.transpose(-1, -2).contiguous()
        kk = k.transpose(-1, -2).contiguous()

        if self.op_name == "h2o":
            em, denom, row_sum, out = self._get_workspace(batch_size, heads, seqlen, head_dim)
            self.kernels[0].run(qq, kk, em, denom)
            self.kernels[1].run(kk, qq, em, denom, row_sum)
            self.kernels[2].run(qq, kk, v, em, denom, out)
            return {"out": out, "row_sum": row_sum, "em": em, "denom": denom}

        em, denom, out = self._get_workspace(batch_size, heads, seqlen, head_dim)
        self.kernels[0].run(qq, kk, em, denom)
        self.kernels[1].run(qq, kk, v, em, denom, out)
        return {"out": out, "em": em, "denom": denom}


def maybe_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    manual_seed_all = getattr(torch_ns, "manual_seed_all", None)
    if manual_seed_all is not None:
        manual_seed_all(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_inputs(
    shape: Tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    pattern: str,
    seed: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maybe_seed_all(seed)

    batch_size, heads, seqlen, head_dim = shape
    if pattern == "ones":
        q = torch.ones((batch_size, heads, seqlen, head_dim), dtype=dtype, device=device)
        k = torch.ones((batch_size, heads, seqlen, head_dim), dtype=dtype, device=device)
        v = torch.ones((batch_size, heads, seqlen, head_dim), dtype=dtype, device=device)
    else:
        q = torch.randn((batch_size, heads, seqlen, head_dim), dtype=dtype, device=device) * scale
        k = torch.randn((batch_size, heads, seqlen, head_dim), dtype=dtype, device=device) * scale
        v = torch.randn((batch_size, heads, seqlen, head_dim), dtype=dtype, device=device) * scale
    return q, k, v


def causal_mask(seqlen: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)


def scaled_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(2, 3))
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    return torch.mul(scores, scale)


def stable_p_scaled_and_denom(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores_fp32 = scores.float()
    row_max = torch.max(scores_fp32, dim=-1, keepdim=True).values
    p_scaled = torch.exp(scores_fp32 - row_max)
    denom = torch.sum(p_scaled, dim=-1, keepdim=True)
    return p_scaled, denom


def baseline_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
    scores = scaled_scores(q, k)
    scores = scores + causal_mask(q.shape[-2], q.device, q.dtype)
    p_scaled, denom = stable_p_scaled_and_denom(scores)
    probs = (p_scaled / denom).to(dtype=v.dtype)
    return {"out": torch.matmul(probs, v)}


def baseline_gemma2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
    scores = scaled_scores(q, k)
    scores = torch.tanh(scores / GEMMA2_TANH_SCALE) * GEMMA2_TANH_SCALE
    scores = scores + causal_mask(q.shape[-2], q.device, q.dtype)
    p_scaled, denom = stable_p_scaled_and_denom(scores)
    probs = (p_scaled / denom).to(dtype=v.dtype)
    return {"out": torch.matmul(probs, v)}


def baseline_h2o(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
    scores = scaled_scores(q, k)
    scores = scores + causal_mask(q.shape[-2], q.device, q.dtype)
    p_scaled, denom = stable_p_scaled_and_denom(scores)
    probs = (p_scaled / denom).to(dtype=v.dtype)
    return {
        "out": torch.matmul(probs, v),
        "row_sum": probs.sum(dim=2),
    }


def measure_gpu_benchmark(fn, warmup: int, iters: int):
    times_ms = []
    result = None
    with torch.inference_mode():
        for _ in range(warmup):
            result = fn()
        torch_ns.synchronize()

        for _ in range(iters):
            start = torch_ns.Event(enable_timing=True)
            end = torch_ns.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch_ns.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
    return result, summarize_times(times_ms)


def build_correctness(
    baseline_ret: Dict[str, torch.Tensor],
    test_ret: Dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    rtol, atol = default_tolerances(dtype)
    result = {
        "full_output": compare_outputs(baseline_ret["out"], test_ret["out"], rtol=rtol, atol=atol),
        "last_token_output": compare_outputs(
            baseline_ret["out"][..., -1, :],
            test_ret["out"][..., -1, :],
            rtol=rtol,
            atol=atol,
        ),
    }
    if "row_sum" in baseline_ret and "row_sum" in test_ret:
        result["row_sum"] = compare_outputs(
            baseline_ret["row_sum"],
            test_ret["row_sum"],
            rtol=rtol,
            atol=atol,
        )
    return result


def baseline_fn_for_op(op_name: str):
    if op_name == "attn":
        return baseline_attention
    if op_name == "gemma2":
        return baseline_gemma2
    if op_name == "h2o":
        return baseline_h2o
    raise ValueError(f"Unsupported op: {op_name}")


def _print_check(prefix: str, check: Dict[str, Any]) -> None:
    full = check["full_output"]
    last = check["last_token_output"]
    print(
        f"[{prefix}-check-full] ok={full['ok']} max_abs={full['max_abs']:.6g} "
        f"mean_abs={full['mean_abs']:.6g} max_rel={full['max_rel']:.6g}"
    )
    print(
        f"[{prefix}-check-last] ok={last['ok']} max_abs={last['max_abs']:.6g} "
        f"mean_abs={last['mean_abs']:.6g} max_rel={last['max_rel']:.6g}"
    )
    if "row_sum" in check:
        row = check["row_sum"]
        print(
            f"[{prefix}-check-row-sum] ok={row['ok']} max_abs={row['max_abs']:.6g} "
            f"mean_abs={row['mean_abs']:.6g} max_rel={row['max_rel']:.6g}"
        )


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    origin_entry, origin_source = load_kernel_entry(
        op_name, "origin", args.seqlen, args.origin_json, args.fallback_root
    )
    combined_entry, combined_source = load_kernel_entry(
        op_name, "combined", args.seqlen, args.combined_json, args.fallback_root
    )

    origin_runner = OriginKernelRunner(op_name, origin_entry, args.dtype, args.device_id)
    combined_runner = CombinedKernelRunner(op_name, combined_entry, args.dtype, args.device_id)

    if origin_runner.shape != combined_runner.shape:
        raise ValueError(
            f"Origin and combined shapes do not match for op={op_name}: "
            f"{origin_runner.shape} vs {combined_runner.shape}"
        )
    if origin_runner.device_id != combined_runner.device_id:
        raise ValueError(
            f"Origin and combined resolved to different runtime devices: "
            f"{origin_runner.device_id} vs {combined_runner.device_id}"
        )

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(dev_name(origin_runner.device_id))
    q, k, v = make_inputs(origin_runner.shape, dtype, device, args.input_pattern, args.seed, args.input_scale)
    baseline_impl = baseline_fn_for_op(op_name)

    correctness = None
    if not args.skip_check:
        with torch.inference_mode():
            baseline_ret = baseline_impl(q, k, v)
            origin_ret = origin_runner(q, k, v)
            combined_ret = combined_runner(q, k, v)
            correctness = {
                "origin_vs_baseline": build_correctness(baseline_ret, origin_ret, dtype),
                "combined_vs_baseline": build_correctness(baseline_ret, combined_ret, dtype),
            }

    _, baseline_stats = measure_gpu_benchmark(lambda: baseline_impl(q, k, v), args.warmup, args.iters)
    _, origin_stats = measure_gpu_benchmark(lambda: origin_runner(q, k, v), args.warmup, args.iters)
    _, combined_stats = measure_gpu_benchmark(lambda: combined_runner(q, k, v), args.warmup, args.iters)

    baseline_to_origin = baseline_stats["median_ms"] / origin_stats["median_ms"] if origin_stats["median_ms"] > 0 else 0.0
    baseline_to_combined = (
        baseline_stats["median_ms"] / combined_stats["median_ms"] if combined_stats["median_ms"] > 0 else 0.0
    )
    origin_to_combined = origin_stats["median_ms"] / combined_stats["median_ms"] if combined_stats["median_ms"] > 0 else 0.0

    result = {
        "op": op_name,
        "timing_method": TIMING_METHOD,
        "baseline_sequence": SEQUENCE_DESCRIPTIONS[op_name]["baseline"],
        "origin_sequence": SEQUENCE_DESCRIPTIONS[op_name]["origin"],
        "combined_sequence": SEQUENCE_DESCRIPTIONS[op_name]["combined"],
        "kernel_config_source": {
            "origin": origin_source,
            "combined": combined_source,
        },
        "kernel_entry": {
            "origin": origin_entry,
            "combined": combined_entry,
        },
        "shape": {
            "batch_size": origin_runner.shape[0],
            "num_heads": origin_runner.shape[1],
            "seqlen": origin_runner.shape[2],
            "head_dim": origin_runner.shape[3],
        },
        "dtype": args.dtype,
        "device": {
            "requested_device_id": args.device_id,
            "runtime_device_id": origin_runner.device_id,
            "name": str(device),
        },
        "input_pattern": args.input_pattern,
        "input_scale": float(args.input_scale),
        "correctness": correctness,
        "baseline": baseline_stats,
        "origin": origin_stats,
        "combined": combined_stats,
        "speedup": {
            "baseline_to_origin": float(baseline_to_origin),
            "baseline_to_combined": float(baseline_to_combined),
            "origin_to_combined": float(origin_to_combined),
        },
    }

    print(f"[op] {op_name}")
    print(f"[origin-config] source={origin_source}")
    print(f"[combined-config] source={combined_source}")
    print(f"[device] requested={args.device_id} runtime={origin_runner.device_id}")
    print(
        f"[shape] batch={origin_runner.shape[0]} heads={origin_runner.shape[1]} "
        f"seqlen={origin_runner.shape[2]} head_dim={origin_runner.shape[3]} dtype={args.dtype}"
    )
    print(f"[timing] benchmark fields use {TIMING_METHOD}")
    print(f"[baseline-seq] {SEQUENCE_DESCRIPTIONS[op_name]['baseline']}")
    print(f"[origin-seq] {SEQUENCE_DESCRIPTIONS[op_name]['origin']}")
    print(f"[combined-seq] {SEQUENCE_DESCRIPTIONS[op_name]['combined']}")
    print(f"[origin-kernel] {origin_runner.kernel_name}")
    for index, kernel_name in enumerate(combined_runner.kernel_names, start=1):
        print(f"[combined-kernel{index}] {kernel_name}")

    if correctness is not None:
        _print_check("origin", correctness["origin_vs_baseline"])
        _print_check("combined", correctness["combined_vs_baseline"])

    print(
        f"[baseline] median={baseline_stats['median_ms']:.3f} ms "
        f"mean={baseline_stats['mean_ms']:.3f} ms"
    )
    print(
        f"[origin] median={origin_stats['median_ms']:.3f} ms "
        f"mean={origin_stats['mean_ms']:.3f} ms"
    )
    print(
        f"[combined] median={combined_stats['median_ms']:.3f} ms "
        f"mean={combined_stats['mean_ms']:.3f} ms"
    )
    print(f"[speedup baseline->origin] {baseline_to_origin:.4f}x")
    print(f"[speedup baseline->combined] {baseline_to_combined:.4f}x")
    print(f"[speedup origin->combined] {origin_to_combined:.4f}x")
    return result


def main() -> None:
    args = parse_args()
    if args.config_json:
        if args.combined_json and args.combined_json != args.config_json:
            raise ValueError("--config-json and --combined-json were both provided with different paths.")
        args.combined_json = args.config_json

    if args.op == "all" and (args.origin_json or args.combined_json):
        raise ValueError("--origin-json/--combined-json are only supported when --op is attn/gemma2/h2o.")

    if args.op == "all":
        results = []
        for index, op_name in enumerate(OP_SPECS.keys(), start=1):
            if index > 1:
                print("")
            print(f"========== [{index}/{len(OP_SPECS)}] {op_name} ==========")
            results.append(run_single_op(op_name, args))

        payload = {
            "mode": "all",
            "seqlen": args.seqlen,
            "dtype": args.dtype,
            "results": {item["op"]: item for item in results},
            "speedup_summary": {
                item["op"]: item["speedup"] for item in results
            },
        }

        print("")
        print("[summary]")
        for item in results:
            print(
                f"{item['op']}: baseline={item['baseline']['median_ms']:.3f} ms "
                f"origin={item['origin']['median_ms']:.3f} ms "
                f"combined={item['combined']['median_ms']:.3f} ms "
                f"baseline->combined={item['speedup']['baseline_to_combined']:.4f}x"
            )
    else:
        payload = run_single_op(args.op, args)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[output] wrote {output_path}")


if __name__ == "__main__":
    main()
