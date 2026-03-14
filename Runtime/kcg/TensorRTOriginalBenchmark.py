#!/usr/bin/env python3
"""
Standalone TensorRT benchmark for the original Attention / H2O / Gemma2 formulas.

This script measures:
1. eager original latency
2. TensorRT engine build latency
3. TensorRT first-call latency
4. TensorRT steady-state latency

Example:
  python Runtime/kcg/TensorRTOriginalBenchmark.py --op all --device cuda --dtype float16 --seqlen 4096
  python Runtime/kcg/TensorRTOriginalBenchmark.py --op gemma2 --dtype float32 --timer cpu_wall --json-out /tmp/gemma2_trt.json
"""

import argparse
import io
import json
import math
import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the original Attention / H2O / Gemma2 formulas with TensorRT."
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
    parser.add_argument("--disable-tensor-core", action="store_true")
    parser.add_argument("--input-init", choices=("ones", "randn"), default="randn")
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--tanh-scale", type=float, default=50.0)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--workspace-gib", type=int, default=0)
    parser.add_argument("--save-engine", default="")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--verbose-trt", action="store_true")
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


def summarize_times(times_ms: Sequence[float]) -> Dict[str, float]:
    return {
        "median_ms": float(statistics.median(times_ms)),
        "mean_ms": float(statistics.mean(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def time_once(fn: Any, args: Tuple[Any, ...], device: torch.device, timer: str) -> Tuple[Any, float]:
    if timer == "cuda_event":
        if device.type != "cuda":
            raise ValueError("cuda_event timer requires a CUDA device.")
        sync_device(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record()
            out = fn(*args)
            end.record()
        sync_device(device)
        return out, float(start.elapsed_time(end))

    sync_device(device)
    st = time.perf_counter()
    with torch.inference_mode():
        out = fn(*args)
    sync_device(device)
    et = time.perf_counter()
    return out, (et - st) * 1000.0


def benchmark_fn(
    fn: Any,
    args: Tuple[Any, ...],
    device: torch.device,
    warmup: int,
    iters: int,
    timer: str,
) -> Tuple[Any, Dict[str, float]]:
    last_out = None
    for _ in range(warmup):
        last_out, _ = time_once(fn, args, device, timer)

    times_ms: List[float] = []
    for _ in range(iters):
        last_out, ms = time_once(fn, args, device, timer)
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


def prepare_io_names(op_name: str) -> Tuple[List[str], List[str]]:
    input_names = ["q", "k", "v"]
    if op_name == "attn":
        return input_names, ["out"]
    if op_name == "h2o":
        return input_names, ["out", "row_sum"]
    if op_name == "gemma2":
        return input_names, ["out"]
    raise ValueError(f"Unsupported op: {op_name}")


def torch_module_to_onnx(
    module: nn.Module,
    input_names: List[str],
    inputs: Sequence[torch.Tensor],
    output_names: List[str],
    opset: int,
    simplify: bool,
):
    import onnx

    onnx_bytes = io.BytesIO()
    try:
        torch.onnx.export(
            module,
            args=tuple(inputs),
            f=onnx_bytes,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
        )
    except Exception as exc:
        if opset < 18:
            raise RuntimeError(
                f"ONNX export failed for opset {opset}. Recent PyTorch ONNX export may first "
                "materialize newer reduce-ops semantics and then try to downgrade, which can "
                "fail on attention graphs. Please rerun with `--opset 18`."
            ) from exc
        raise
    onnx_model = onnx.load_model_from_string(onnx_bytes.getvalue())
    if simplify:
        import onnxsim

        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, "onnxsim simplify failed"
    return onnx_model


def trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int64: torch.int64,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
        trt.bfloat16: torch.bfloat16,
    }
    if dtype not in mapping:
        raise KeyError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


def trt_build_engine_from_onnx(
    onnx_model: Any,
    workspace_gib: int,
    engine_fn: str,
    verbose: bool,
    disable_tensor_core: bool,
) -> bytes:
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        trt_logger.min_severity = trt.Logger.Severity.VERBOSE

    trt.init_libnvinfer_plugins(trt_logger, namespace="")
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)
    if disable_tensor_core and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    if workspace_gib > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * (2**30))

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)
    onnx_buf = onnx_model.SerializeToString()
    if not parser.parse(onnx_buf):
        errs = []
        for i in range(parser.num_errors):
            errs.append(str(parser.get_error(i)))
        raise RuntimeError("Failed to parse ONNX for TensorRT:\n" + "\n".join(errs))

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT failed to build serialized engine.")

    if engine_fn:
        with open(engine_fn, "wb") as f:
            f.write(engine_bytes)

    return engine_bytes


def trt_build_independent_runtime(engine_bytes: bytes, verbose: bool):
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        trt_logger.min_severity = trt.Logger.Severity.VERBOSE

    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize engine.")

    input_specs = []
    output_specs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        spec = {
            "name": name,
            "dtype": trt_dtype_to_torch(engine.get_tensor_dtype(name)),
            "shape": tuple(engine.get_tensor_shape(name)),
        }
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_specs.append(spec)
        else:
            output_specs.append(spec)

    context = engine.create_execution_context()
    stream = torch.cuda.current_stream()

    def run(*args: torch.Tensor):
        device = args[0].device
        for spec, arg in zip(input_specs, args):
            context.set_tensor_address(spec["name"], arg.data_ptr())

        outputs = []
        for spec in output_specs:
            out = torch.empty(spec["shape"], dtype=spec["dtype"], device=device)
            context.set_tensor_address(spec["name"], out.data_ptr())
            outputs.append(out)

        ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return run


def run_single_op(
    op_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    module = build_original_module(
        op_name,
        args.seqlen,
        args.head_dim,
        q.dtype,
        device,
        args.tanh_scale,
    ).eval()
    inputs = (q, k, v)
    input_names, output_names = prepare_io_names(op_name)

    eager_out, eager_stats = benchmark_fn(
        module, inputs, device, args.warmup, args.iters, args.timer
    )

    build_start = time.perf_counter()
    onnx_model = torch_module_to_onnx(
        module=module,
        input_names=input_names,
        inputs=inputs,
        output_names=output_names,
        opset=args.opset,
        simplify=not args.no_simplify,
    )
    onnx_export_ms = (time.perf_counter() - build_start) * 1000.0

    build_start = time.perf_counter()
    engine_bytes = trt_build_engine_from_onnx(
        onnx_model=onnx_model,
        workspace_gib=args.workspace_gib,
        engine_fn=args.save_engine,
        verbose=args.verbose_trt,
        disable_tensor_core=args.disable_tensor_core,
    )
    engine_build_ms = (time.perf_counter() - build_start) * 1000.0

    runtime_start = time.perf_counter()
    trt_run = trt_build_independent_runtime(engine_bytes, verbose=args.verbose_trt)
    runtime_build_ms = (time.perf_counter() - runtime_start) * 1000.0

    trt_first_out, trt_first_ms = time_once(trt_run, inputs, device, args.timer)
    trt_first_out = snapshot_output(trt_first_out)
    _, trt_stats = benchmark_fn(
        trt_run, inputs, device, args.warmup, args.iters, args.timer
    )

    rtol, atol = default_tolerances(q.dtype)
    trt_vs_eager = compare_outputs(eager_out, trt_first_out, rtol=rtol, atol=atol)

    return {
        "timer": args.timer,
        "eager": {
            "original": eager_stats,
        },
        "trt": {
            "onnx_export_ms": onnx_export_ms,
            "engine_build_ms": engine_build_ms,
            "runtime_build_ms": runtime_build_ms,
            "first_call_ms": trt_first_ms,
            **trt_stats,
            "trt_vs_eager": trt_vs_eager,
            "steady_speedup_vs_eager": (
                eager_stats["median_ms"] / trt_stats["median_ms"]
                if trt_stats["median_ms"] > 0
                else 0.0
            ),
        },
    }


def print_summary(op_name: str, result: Dict[str, Any]) -> None:
    eager = result["eager"]
    trt = result["trt"]
    print(f"=== {op_name} ===")
    print(f"timer: {result['timer']}")
    print(f"eager original median: {eager['original']['median_ms']:.4f} ms")
    print(
        f"trt build: onnx={trt['onnx_export_ms']:.4f} ms | "
        f"engine={trt['engine_build_ms']:.4f} ms | runtime={trt['runtime_build_ms']:.4f} ms"
    )
    print(
        f"trt original median: {trt['median_ms']:.4f} ms "
        f"(first={trt['first_call_ms']:.4f} ms)"
    )
    print(f"trt correctness: {trt['trt_vs_eager']}")
    print(f"trt vs eager: x{trt['steady_speedup_vs_eager']:.4f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("TensorRT benchmark requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch build.")

    if args.disable_tensor_core:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

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
            "system": "tensorrt_original",
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
            "disable_tensor_core": args.disable_tensor_core,
            "input_init": args.input_init,
            "input_scale": args.input_scale,
            "tanh_scale": args.tanh_scale,
            "opset": args.opset,
            "workspace_gib": args.workspace_gib,
            "simplify": not args.no_simplify,
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


if __name__ == "__main__":
    main()
