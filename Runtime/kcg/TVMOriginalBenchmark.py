#!/usr/bin/env python3
"""
Standalone TVM Relay benchmark for the original Attention / H2O / Gemma2 formulas.

This script measures:
1. eager original latency
2. TVM Relay compile latency
3. TVM Relay first-call latency
4. TVM Relay steady-state latency

Example:
  python Runtime/kcg/TVMOriginalBenchmark.py --op all --device cuda --dtype float16 --seqlen 4096
  python Runtime/kcg/TVMOriginalBenchmark.py --op h2o --json-out /tmp/h2o_relay.json
"""

import argparse
import importlib
import io
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the original Attention / H2O / Gemma2 formulas with TVM Relay."
    )
    parser.add_argument("--op", choices=("attn", "h2o", "gemma2", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--timer", choices=("cpu_wall", "cuda_event"), default="cpu_wall")
    parser.add_argument("--scheduler", choices=("metaschedule", "relay"), default="metaschedule")
    parser.add_argument("--num-trials-per-iter", type=int, default=4)
    parser.add_argument("--max-trials-per-task", type=int, default=128)
    parser.add_argument("--max-trials-global", type=int, default=-1)
    parser.add_argument("--runner-timeout-sec", type=float, default=30.0)
    parser.add_argument("--ms-cost-model", choices=("random", "xgb", "mlp", "none"), default="xgb")
    parser.add_argument("--disable-cublas", action="store_true")
    parser.add_argument("--input-init", choices=("ones", "randn"), default="randn")
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--tanh-scale", type=float, default=50.0)
    parser.add_argument("--onnx-opset", type=int, default=18)
    parser.add_argument("--work-dir-root", default="")
    parser.add_argument("--export-lib-dir", default="")
    parser.add_argument("--tvm-python-path", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def to_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    return mapping[name]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def causal_upper_mask(seqlen: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = torch.triu(
        torch.ones((seqlen, seqlen), device=device, dtype=torch.bool),
        diagonal=1,
    )
    return torch.where(
        base,
        torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype),
        torch.zeros((seqlen, seqlen), device=device, dtype=dtype),
    )


def make_qkv(
    bs: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    init_mode: str,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if init_mode == "ones":
        q = torch.ones((bs, heads, seqlen, head_dim), dtype=dtype, device=device)
        k = torch.ones((bs, heads, head_dim, seqlen), dtype=dtype, device=device)
        v = torch.ones((bs, heads, seqlen, head_dim), dtype=dtype, device=device)
        return q, k, v

    q = torch.randn((bs, heads, seqlen, head_dim), dtype=dtype, device=device) * input_scale
    k = torch.randn((bs, heads, head_dim, seqlen), dtype=dtype, device=device) * input_scale
    v = torch.randn((bs, heads, seqlen, head_dim), dtype=dtype, device=device) * input_scale
    return q, k, v


def flatten_tensors(obj: Any) -> List[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out: List[torch.Tensor] = []
        for item in obj:
            out.extend(flatten_tensors(item))
        return out
    raise TypeError(f"Unsupported output type: {type(obj)!r}")


def snapshot_output(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(snapshot_output(x) for x in obj)
    if isinstance(obj, list):
        return [snapshot_output(x) for x in obj]
    return obj


def default_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 1e-2, 1e-2


def compare_outputs(
    ref: Any,
    test: Any,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    ref_list = flatten_tensors(ref)
    test_list = flatten_tensors(test)
    if len(ref_list) != len(test_list):
        return {
            "ok": False,
            "reason": f"output_count_mismatch: {len(ref_list)} vs {len(test_list)}",
        }

    ok = True
    max_abs = 0.0
    max_rel = 0.0
    for lhs, rhs in zip(ref_list, test_list):
        close = torch.allclose(lhs, rhs, rtol=rtol, atol=atol)
        ok = ok and bool(close)
        diff = (lhs - rhs).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = rhs.abs() + 1e-12
        max_rel = max(max_rel, float((diff / denom).max().item()))
    return {
        "ok": ok,
        "max_abs": max_abs,
        "max_rel": max_rel,
    }


def format_exception(exc: Exception) -> str:
    msg = str(exc).strip()
    parts = [f"{type(exc).__name__}: {msg}" if msg else f"{type(exc).__name__}: {exc!r}"]
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_text = "".join(tb_lines).strip()
    if tb_text:
        parts.append(tb_text)
    return "\n".join(parts)


def summarize_times(times_ms: Sequence[float]) -> Dict[str, float]:
    return {
        "median_ms": float(statistics.median(times_ms)),
        "mean_ms": float(statistics.mean(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def time_once(
    fn: Any,
    args: Tuple[Any, ...],
    device: torch.device,
    timer: str,
    sync_fn: Any = None,
    event_stream: Any = None,
) -> Tuple[Any, float]:
    do_sync = sync_fn if sync_fn is not None else (lambda: sync_device(device))

    if timer == "cuda_event":
        do_sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record(event_stream)
            out = fn(*args)
            end.record(event_stream)
        do_sync()
        return out, float(start.elapsed_time(end))

    do_sync()
    st = time.perf_counter()
    with torch.inference_mode():
        out = fn(*args)
    do_sync()
    et = time.perf_counter()
    return out, (et - st) * 1000.0


def benchmark_fn(
    fn: Any,
    args: Tuple[Any, ...],
    device: torch.device,
    warmup: int,
    iters: int,
    timer: str,
    sync_fn: Any = None,
    event_stream: Any = None,
) -> Tuple[Any, Dict[str, float]]:
    last_out = None
    for _ in range(warmup):
        last_out, _ = time_once(fn, args, device, timer, sync_fn=sync_fn, event_stream=event_stream)

    times_ms: List[float] = []
    for _ in range(iters):
        last_out, ms = time_once(fn, args, device, timer, sync_fn=sync_fn, event_stream=event_stream)
        times_ms.append(ms)

    return last_out, summarize_times(times_ms)


class H2OOriginal(nn.Module):
    def __init__(self, seqlen: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        mask = causal_upper_mask(seqlen, device, dtype).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)
        self.scale = 1.0 / math.sqrt(float(head_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k) * self.scale + self.mask
        row_max = scores.max(dim=-1, keepdim=True).values
        p_exp = torch.exp(scores - row_max)
        denom = p_exp.sum(dim=-1, keepdim=True)
        p = p_exp / denom
        out = torch.matmul(p, v)
        row_sum = p.sum(dim=2, keepdim=False)
        return out, row_sum


class AttentionOriginal(nn.Module):
    def __init__(self, seqlen: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        mask = causal_upper_mask(seqlen, device, dtype).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)
        self.scale = 1.0 / math.sqrt(float(head_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k) * self.scale + self.mask
        row_max = scores.max(dim=-1, keepdim=True).values
        p_exp = torch.exp(scores - row_max)
        denom = p_exp.sum(dim=-1, keepdim=True)
        p = p_exp / denom
        return torch.matmul(p, v)


class Gemma2Original(nn.Module):
    def __init__(
        self,
        seqlen: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        tanh_scale: float,
    ):
        super().__init__()
        mask = causal_upper_mask(seqlen, device, dtype).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)
        self.scale = 1.0 / math.sqrt(float(head_dim))
        self.tanh_scale = tanh_scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k) * self.scale
        y = torch.tanh(scores / self.tanh_scale) * self.tanh_scale
        y = y + self.mask
        row_max = y.max(dim=-1, keepdim=True).values
        p_scaled = torch.exp(y - row_max)
        denom = p_scaled.sum(dim=-1, keepdim=True)
        p = p_scaled / denom
        return torch.matmul(p, v)


def build_original_module(
    op_name: str,
    seqlen: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    tanh_scale: float,
) -> nn.Module:
    if op_name == "attn":
        return AttentionOriginal(seqlen, head_dim, dtype, device)
    if op_name == "h2o":
        return H2OOriginal(seqlen, head_dim, dtype, device)
    if op_name == "gemma2":
        return Gemma2Original(seqlen, head_dim, dtype, device, tanh_scale)
    raise ValueError(f"Unsupported op: {op_name}")


def cleanup_runtime(device: torch.device) -> None:
    sync_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync_device(device)


def setup_import_paths(args: argparse.Namespace) -> List[str]:
    added: List[str] = []

    def add(path: str) -> None:
        path = path.strip()
        if not path:
            return
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)

    if args.tvm_python_path:
        for item in args.tvm_python_path.split(os.pathsep):
            add(item)

    env_tvm_python = os.environ.get("TVM_PYTHON_PATH", "")
    if env_tvm_python:
        for item in env_tvm_python.split(os.pathsep):
            add(item)

    tvm_home = os.environ.get("TVM_HOME", "")
    if tvm_home:
        add(os.path.join(tvm_home, "python"))

    return added


def import_module_or_error(name: str) -> Tuple[Any, str]:
    try:
        return importlib.import_module(name), ""
    except Exception as exc:
        return None, f"{name}: {type(exc).__name__}: {exc}"


def safe_find_spec(name: str) -> str:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return "None"
        return str(getattr(spec, "origin", None))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def import_relay_stack(args: argparse.Namespace) -> Dict[str, Any]:
    added_paths = setup_import_paths(args)

    onnx, onnx_err = import_module_or_error("onnx")
    tvm, tvm_err = import_module_or_error("tvm")
    relay, relay_err = import_module_or_error("tvm.relay")
    relay_frontend, relay_frontend_err = import_module_or_error("tvm.relay.frontend")
    graph_executor, graph_executor_err = import_module_or_error("tvm.contrib.graph_executor")
    tvm_runtime, tvm_runtime_err = import_module_or_error("tvm.runtime")
    ms = None
    ms_err = ""
    ms_runner = None
    ms_runner_err = ""
    if args.scheduler == "metaschedule":
        ms, ms_err = import_module_or_error("tvm.meta_schedule")
        ms_runner, ms_runner_err = import_module_or_error("tvm.meta_schedule.runner")

    failures = [
        err
        for err in (
            onnx_err,
            tvm_err,
            relay_err,
            relay_frontend_err,
            graph_executor_err,
            tvm_runtime_err,
            ms_err,
            ms_runner_err,
        )
        if err
    ]
    if failures:
        detail_lines = [
            "Import TVM Relay/ONNX failed.",
            f"python_executable={sys.executable}",
            f"python_version={sys.version.split()[0]}",
        ]
        if added_paths:
            detail_lines.append(f"added_sys_path={added_paths}")
        if tvm is not None:
            tvm_dir = os.path.dirname(getattr(tvm, "__file__", ""))
            relay_init_exists = os.path.exists(os.path.join(tvm_dir, "relay", "__init__.py"))
            graph_executor_exists = os.path.exists(
                os.path.join(tvm_dir, "contrib", "graph_executor.py")
            )
            detail_lines.append(f"tvm_file={getattr(tvm, '__file__', '')}")
            detail_lines.append(f"tvm_path={list(getattr(tvm, '__path__', []))}")
            detail_lines.append(f"find_spec_tvm.relay={safe_find_spec('tvm.relay')}")
            detail_lines.append(
                f"find_spec_tvm.contrib.graph_executor={safe_find_spec('tvm.contrib.graph_executor')}"
            )
            detail_lines.append(f"relay_init_exists={relay_init_exists}")
            detail_lines.append(f"graph_executor_exists={graph_executor_exists}")
        detail_lines.extend(failures)
        raise RuntimeError(" | ".join(detail_lines))

    from_onnx = getattr(relay_frontend, "from_onnx", None)
    if from_onnx is None:
        raise RuntimeError(
            "TVM Relay ONNX frontend imported but `from_onnx` was not found. "
            f"python_executable={sys.executable}"
        )

    return {
        "onnx": onnx,
        "tvm": tvm,
        "relay": relay,
        "from_onnx": from_onnx,
        "graph_executor": graph_executor,
        "tvm_runtime": tvm_runtime,
        "ms": ms,
        "ms_runner": ms_runner,
        "added_paths": added_paths,
    }


def get_tvm_target(tvm: Any, device: torch.device, use_cublas: bool) -> Any:
    dev_id = device.index if device.index is not None else 0
    tvm_dev = tvm.cuda(dev_id)
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    max_shared_memory_per_block = getattr(
        tvm_dev,
        "max_shared_memory_per_block",
        getattr(props, "shared_memory_per_block", 49152),
    )
    max_threads_per_block = getattr(
        tvm_dev,
        "max_threads_per_block",
        getattr(props, "max_threads_per_block", 1024),
    )
    warp_size = getattr(tvm_dev, "warp_size", getattr(props, "warp_size", 32))
    target_dict: Dict[str, Any] = {
        "kind": "cuda",
        "model": props.name,
        "arch": f"sm_{major}{minor}",
        "max_shared_memory_per_block": int(max_shared_memory_per_block),
        "max_threads_per_block": int(max_threads_per_block),
        # Some TVM code paths query max_num_threads, others query max_threads_per_block.
        "max_num_threads": int(max_threads_per_block),
        "thread_warp_size": int(warp_size),
    }
    if use_cublas:
        target_dict["libs"] = ["cublas"]
    host = None
    if sys.platform == "darwin" and platform.machine() == "arm64":
        host = {
            "kind": "llvm",
            "mtriple": "arm64-apple-macos",
            "mcpu": "generic",
        }
    return tvm.target.Target(target_dict, host=host)


def torch_module_to_relay(
    module: nn.Module,
    input_names: Sequence[str],
    inputs: Sequence[torch.Tensor],
    output_names: Sequence[str],
    onnx_opset: int,
    onnx: Any,
    from_onnx: Any,
) -> Tuple[Any, Dict[str, Any]]:
    shape_dict = {
        input_name: list(input_.shape)
        for input_name, input_ in zip(input_names, inputs)
    }
    export_kwargs = {
        "input_names": list(input_names),
        "output_names": list(output_names),
        "verbose": False,
        "opset_version": onnx_opset,
    }

    try:
        onnx_bytes = io.BytesIO()
        torch.onnx.export(
            module,
            args=tuple(inputs),
            f=onnx_bytes,
            **export_kwargs,
        )
        onnx_model = onnx.load_model_from_string(onnx_bytes.getvalue())
    except Exception as exc:
        # Large models can exceed protobuf's in-memory serialization limit.
        if "Failed to serialize proto" not in str(exc):
            raise
        with tempfile.TemporaryDirectory(prefix="tvm_onnx_export_") as temp_dir:
            onnx_path = Path(temp_dir) / "model.onnx"
            torch.onnx.export(
                module,
                args=tuple(inputs),
                f=str(onnx_path),
                external_data=True,
                **export_kwargs,
            )
            onnx_model = onnx.load(str(onnx_path), load_external_data=True)

    imported = from_onnx(onnx_model, shape_dict, freeze_params=True)
    if isinstance(imported, tuple) and len(imported) == 2:
        mod, params = imported
    else:
        mod, params = imported, {}
    return mod, params


def torch_tensor_to_tvm_nd(arg: torch.Tensor, tvm: Any, tvm_dev: Any = None) -> Any:
    tensor = arg.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    def _to_dlpack_capsule() -> Any:
        if tensor.device.type == "cuda":
            with torch.cuda.device(tensor.device):
                return tensor.__dlpack__()
        return tensor.__dlpack__()

    nd_ns = getattr(tvm, "nd", None)
    if nd_ns is not None and hasattr(nd_ns, "from_dlpack"):
        return nd_ns.from_dlpack(_to_dlpack_capsule())

    runtime_ns = getattr(tvm, "runtime", None)
    if runtime_ns is not None:
        ndarray_ns = getattr(runtime_ns, "ndarray", None)
        if ndarray_ns is not None and hasattr(ndarray_ns, "from_dlpack"):
            return ndarray_ns.from_dlpack(_to_dlpack_capsule())
        if ndarray_ns is not None and hasattr(ndarray_ns, "array"):
            np_arr = tensor.cpu().numpy()
            if tvm_dev is not None:
                return ndarray_ns.array(np_arr, device=tvm_dev)
            return ndarray_ns.array(np_arr)

    raise AttributeError(
        "Cannot find TVM tensor import API. Expected one of "
        "`tvm.nd.from_dlpack`, `tvm.runtime.ndarray.from_dlpack`, "
        "or `tvm.runtime.ndarray.array`."
    )


def tvm_nd_to_torch(obj: Any, device: torch.device) -> torch.Tensor:
    if hasattr(obj, "to_dlpack"):
        return torch.utils.dlpack.from_dlpack(obj)
    if hasattr(obj, "numpy"):
        return torch.from_numpy(obj.numpy()).to(device)
    raise TypeError(f"Unsupported TVM output type: {type(obj)!r}")


def maybe_export_library(lib: Any, export_lib_dir: str, op_name: str) -> str:
    if not export_lib_dir:
        return ""
    out_dir = Path(export_lib_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{op_name}.so"
    lib.export_library(str(out_path))
    return str(out_path)


def resolve_work_dir(root: str, tag: str, op_name: str) -> str:
    base = Path(root) if root else Path.cwd()
    stamp = f"{int(time.time() * 1000)}_{os.getpid()}"
    work_dir = base / f"{tag}_{op_name}_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def build_relay_runtime(
    built: Any,
    device: torch.device,
    stack: Dict[str, Any],
    output_count: int,
) -> Dict[str, Any]:
    tvm = stack["tvm"]
    relay = stack["relay"]
    graph_executor = stack["graph_executor"]
    dev = tvm.cuda(device.index if device.index is not None else 0)

    if isinstance(built, tuple) and len(built) == 3:
        graph_json, lib, params = built
        module = graph_executor.create(graph_json, lib, dev)
        if params:
            module.load_params(relay.save_param_dict(params))
        exportable_lib = lib
    else:
        module = graph_executor.GraphModule(built["default"](dev))
        exportable_lib = built

    def sync_all() -> None:
        dev.sync()
        sync_device(device)

    def prepare_inputs(args: Sequence[torch.Tensor], input_names: Sequence[str]) -> None:
        data_tvm = {
            name: torch_tensor_to_tvm_nd(arg, tvm, dev)
            for name, arg in zip(input_names, args)
        }
        module.set_input(**data_tvm)

    def run_raw() -> None:
        module.run()

    def fetch_outputs() -> Any:
        if output_count == 1:
            return tvm_nd_to_torch(module.get_output(0), device)
        return [tvm_nd_to_torch(module.get_output(i), device) for i in range(output_count)]

    return {
        "module": module,
        "dev": dev,
        "sync": sync_all,
        "prepare_inputs": prepare_inputs,
        "run_raw": run_raw,
        "fetch_outputs": fetch_outputs,
        "exportable_lib": exportable_lib,
    }


def compile_with_relay(
    module: nn.Module,
    input_names: Sequence[str],
    inputs: Sequence[torch.Tensor],
    output_names: Sequence[str],
    device: torch.device,
    args: argparse.Namespace,
    op_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stack = import_relay_stack(args)
    tvm = stack["tvm"]
    relay = stack["relay"]

    mod, params = torch_module_to_relay(
        module=module,
        input_names=input_names,
        inputs=inputs,
        output_names=output_names,
        onnx_opset=args.onnx_opset,
        onnx=stack["onnx"],
        from_onnx=stack["from_onnx"],
    )
    target = get_tvm_target(tvm, device, use_cublas=not args.disable_cublas)

    compile_meta: Dict[str, Any] = {
        "compiler_kind": "relay_graph_executor",
        "target": str(target),
        "python_executable": sys.executable,
        "scheduler": args.scheduler,
    }
    if stack["added_paths"]:
        compile_meta["added_sys_path"] = stack["added_paths"]
    if args.work_dir_root:
        compile_meta["ignored_work_dir_root"] = args.work_dir_root

    tik = time.perf_counter()
    if args.scheduler == "metaschedule":
        ms = stack["ms"]
        extracted_tasks = ms.relay_integration.extract_tasks(mod, target=target, params=params)
        task_count = len(extracted_tasks)
        compile_meta["task_count"] = task_count
        if task_count == 0:
            with tvm.transform.PassContext(opt_level=3):
                built = relay.build(mod, target=target, params=params)
            compile_meta["build_variant"] = "direct_no_tasks"
        else:
            work_dir = resolve_work_dir(args.work_dir_root, "relay_ms", op_name)
            max_trials_global = (
                args.max_trials_global
                if args.max_trials_global > 0
                else task_count * args.max_trials_per_task
            )
            runner = "local"
            runner_mod = stack["ms_runner"]
            runner_cls = getattr(runner_mod, "Runner", None) if runner_mod is not None else None
            if runner_cls is not None and hasattr(runner_cls, "create"):
                runner = runner_cls.create("local", timeout_sec=args.runner_timeout_sec)
                compile_meta["runner_mode"] = "Runner.create"
            else:
                compile_meta["runner_mode"] = "local_string"

            database = ms.relay_integration.tune_relay(
                mod=mod,
                params=params,
                target=target,
                work_dir=work_dir,
                max_trials_global=max_trials_global,
                max_trials_per_task=args.max_trials_per_task,
                num_trials_per_iter=args.num_trials_per_iter,
                runner=runner,
                cost_model=args.ms_cost_model,
            )
            built = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target,
                params=params,
                backend="graph",
            )
            compile_meta["work_dir"] = work_dir
            compile_meta["num_trials_per_iter"] = args.num_trials_per_iter
            compile_meta["max_trials_per_task"] = args.max_trials_per_task
            compile_meta["max_trials_global"] = max_trials_global
            compile_meta["ms_cost_model"] = args.ms_cost_model
            compile_meta["build_variant"] = "metaschedule"
    else:
        with tvm.transform.PassContext(opt_level=3):
            built = relay.build(mod, target=target, params=params)
        compile_meta["build_variant"] = "direct"
    compile_meta["compile_ms"] = (time.perf_counter() - tik) * 1000.0

    runtime = build_relay_runtime(
        built=built,
        device=device,
        stack=stack,
        output_count=len(output_names),
    )
    export_path = maybe_export_library(runtime["exportable_lib"], args.export_lib_dir, op_name)
    if export_path:
        compile_meta["exported_lib"] = export_path
    return runtime, compile_meta


def run_single_op(
    op_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    original_mod = build_original_module(
        op_name,
        args.seqlen,
        args.head_dim,
        q.dtype,
        device,
        args.tanh_scale,
    )
    original_mod.eval()
    inputs = (q, k, v)
    input_names = ["q", "k", "v"]
    output_names = ["out", "row_sum"] if op_name == "h2o" else ["out"]

    eager_original_out, eager_original_stats = benchmark_fn(
        original_mod,
        inputs,
        device,
        args.warmup,
        args.iters,
        args.timer,
    )
    eager_original_out = snapshot_output(eager_original_out)
    rtol, atol = default_tolerances(q.dtype)

    result: Dict[str, Any] = {
        "timer": args.timer,
        "eager": {
            "original": eager_original_stats,
        },
    }

    try:
        runtime, compile_meta = compile_with_relay(
            module=original_mod,
            input_names=input_names,
            inputs=inputs,
            output_names=output_names,
            device=device,
            args=args,
            op_name=op_name,
        )
        runtime["prepare_inputs"](inputs, input_names)

        _, relay_first_ms = time_once(
            runtime["run_raw"],
            tuple(),
            device,
            args.timer,
            sync_fn=runtime["sync"],
        )

        relay_out = snapshot_output(runtime["fetch_outputs"]())

        _, relay_stats = benchmark_fn(
            runtime["run_raw"],
            tuple(),
            device,
            args.warmup,
            args.iters,
            args.timer,
            sync_fn=runtime["sync"],
        )

        relay_vs_eager = compare_outputs(
            eager_original_out,
            relay_out,
            rtol=rtol,
            atol=atol,
        )

        result["tvm"] = {
            **compile_meta,
            "original": {
                "first_call_ms": relay_first_ms,
                **relay_stats,
                "relay_vs_eager": relay_vs_eager,
                "steady_speedup_vs_eager": (
                    eager_original_stats["median_ms"] / relay_stats["median_ms"]
                    if relay_stats["median_ms"] > 0
                    else 0.0
                ),
                "timed_region": "graph_executor_run_only",
                "timed_output_conversion": False,
            },
        }
    except Exception as exc:
        result["tvm"] = {
            "error": format_exception(exc),
        }

    del original_mod
    del eager_original_out
    cleanup_runtime(device)
    return result


def print_summary(op_name: str, result: Dict[str, Any]) -> None:
    eager = result["eager"]
    print(f"=== {op_name} ===")
    print(f"timer: {result['timer']}")
    print(f"eager original median: {eager['original']['median_ms']:.4f} ms")

    tvm_result = result.get("tvm", {})
    if "error" in tvm_result:
        print(f"TVM failed: {tvm_result['error']}")
        return

    print(f"TVM route: {tvm_result['compiler_kind']}")
    print(f"TVM target: {tvm_result['target']}")
    print(f"TVM scheduler: {tvm_result['scheduler']}")
    if "ms_cost_model" in tvm_result:
        print(f"TVM MS cost model: {tvm_result['ms_cost_model']}")
    print(f"TVM Relay compile: {tvm_result['compile_ms']:.4f} ms")
    print(
        f"TVM original median: {tvm_result['original']['median_ms']:.4f} ms "
        f"(first={tvm_result['original']['first_call_ms']:.4f} ms)"
    )
    print(f"TVM correctness: {tvm_result['original']['relay_vs_eager']}")
    print(
        f"TVM original vs eager: x{tvm_result['original']['steady_speedup_vs_eager']:.4f}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script currently supports CUDA devices only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    q, k, v = make_qkv(
        args.bs,
        args.heads,
        args.seqlen,
        args.head_dim,
        dtype,
        device,
        args.input_init,
        args.input_scale,
    )

    ops: Iterable[str] = ("attn", "h2o", "gemma2") if args.op == "all" else (args.op,)

    all_results: Dict[str, Any] = {
        "meta": {
            "torch_version": torch.__version__,
            "device": str(device),
            "dtype": args.dtype,
            "shape": {
                "bs": args.bs,
                "heads": args.heads,
                "seqlen": args.seqlen,
                "head_dim": args.head_dim,
            },
            "warmup": args.warmup,
            "iters": args.iters,
            "timer": args.timer,
            "compiler_kind": "relay_graph_executor",
            "scheduler": args.scheduler,
            "num_trials_per_iter": args.num_trials_per_iter,
            "max_trials_per_task": args.max_trials_per_task,
            "max_trials_global": args.max_trials_global,
            "runner_timeout_sec": args.runner_timeout_sec,
            "ms_cost_model": args.ms_cost_model,
            "measure_kind": "original_only",
            "input_init": args.input_init,
            "input_scale": args.input_scale,
            "tanh_scale": args.tanh_scale,
            "onnx_opset": args.onnx_opset,
            "use_cublas": not args.disable_cublas,
            "work_dir_root": args.work_dir_root,
            "export_lib_dir": args.export_lib_dir,
            "tvm_python_path": args.tvm_python_path,
            "seed": args.seed,
        },
        "results": {},
    }

    for op_name in ops:
        result = run_single_op(op_name, q, k, v, device, args)
        all_results["results"][op_name] = result
        print_summary(op_name, result)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved JSON results to {args.json_out}")

    print(json.dumps(all_results, indent=2))
    cleanup_runtime(device)


if __name__ == "__main__":
    main()
