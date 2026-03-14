#!/usr/bin/env python3
"""
Standalone torch.compile benchmark for the original Attention / H2O / Gemma2 formulas.

This script measures only the original PyTorch path for each op:
1. eager original latency
2. torch.compile original first-call latency
3. torch.compile original steady-state latency

Example:
  python Runtime/kcg/TorchCompileSplitCombinedBenchmark.py --op all --device cuda --dtype float32 --seqlen 4096
  python Runtime/kcg/TorchCompileSplitCombinedBenchmark.py --op h2o --device cuda --mode reduce-overhead --json-out /tmp/h2o_tc.json
"""

import argparse
import gc
import json
import math
import os
import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Avoid lingering inductor compile worker pools on process exit.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the original Attention / H2O / Gemma2 formulas with torch.compile."
    )
    parser.add_argument("--op", choices=("attn", "h2o", "gemma2", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--timer", choices=("cpu_wall", "cuda_event"), default="cpu_wall")
    parser.add_argument("--input-init", choices=("ones", "randn"), default="randn")
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--tanh-scale", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def to_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
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


def maybe_mark_cudagraph_step_begin(device: torch.device) -> None:
    if device.type != "cuda":
        return
    compiler_ns = getattr(torch, "compiler", None)
    if compiler_ns is None:
        return
    mark = getattr(compiler_ns, "cudagraph_mark_step_begin", None)
    if mark is None:
        return
    mark()


def time_once(fn: Any, args: Tuple[Any, ...], device: torch.device, timer: str) -> Tuple[Any, float]:
    if timer == "cuda_event":
        if device.type != "cuda":
            raise ValueError("cuda_event timer requires a CUDA device.")
        sync_device(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            maybe_mark_cudagraph_step_begin(device)
            start.record()
            out = fn(*args)
            end.record()
        sync_device(device)
        return out, float(start.elapsed_time(end))

    sync_device(device)
    st = time.perf_counter()
    with torch.inference_mode():
        maybe_mark_cudagraph_step_begin(device)
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
    # Keep this original baseline numerically aligned with TensorRTOriginalBenchmark.py.
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
    # Keep this original baseline numerically aligned with TensorRTOriginalBenchmark.py.
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
    # Keep this original baseline numerically aligned with TensorRTOriginalBenchmark.py.
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
    reset = getattr(torch, "_dynamo", None)
    if reset is not None and hasattr(reset, "reset"):
        torch._dynamo.reset()
    sync_device(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync_device(device)


def try_compile(module: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    return torch.compile(
        module,
        backend=args.backend,
        mode=args.mode,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
    )


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
    inputs = (q, k, v)

    eager_original_out, eager_original_stats = benchmark_fn(
        original_mod, inputs, device, args.warmup, args.iters, args.timer
    )
    rtol, atol = default_tolerances(q.dtype)

    result: Dict[str, Any] = {
        "timer": args.timer,
        "eager": {
            "original": eager_original_stats,
        }
    }

    try:
        compiled_baseline = try_compile(original_mod, args)

        compiled_original_first_out, compiled_original_first_ms = time_once(
            compiled_baseline, inputs, device, args.timer
        )
        compiled_original_first_out = snapshot_output(compiled_original_first_out)

        _, compiled_original_stats = benchmark_fn(
            compiled_baseline, inputs, device, args.warmup, args.iters, args.timer
        )

        compiled_vs_eager_original = compare_outputs(
            eager_original_out,
            compiled_original_first_out,
            rtol=rtol,
            atol=atol,
        )

        result["compiled"] = {
            "original": {
                "first_call_ms": compiled_original_first_ms,
                **compiled_original_stats,
                "compiled_vs_eager": compiled_vs_eager_original,
                "steady_speedup_vs_eager": (
                    eager_original_stats["median_ms"] / compiled_original_stats["median_ms"]
                    if compiled_original_stats["median_ms"] > 0
                    else 0.0
                ),
            }
        }
    except Exception as exc:
        result["compiled"] = {
            "error": str(exc),
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

    compiled = result.get("compiled", {})
    if "error" in compiled:
        print(f"torch.compile failed: {compiled['error']}")
        return

    print(
        f"compiled original median: {compiled['original']['median_ms']:.4f} ms "
        f"(first={compiled['original']['first_call_ms']:.4f} ms)"
    )
    print(f"compiled original correctness: {compiled['original']['compiled_vs_eager']}")
    print(
        f"compiled original vs eager: x{compiled['original']['steady_speedup_vs_eager']:.4f}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
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
            "compile_threads_env": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", ""),
            "shape": {
                "bs": args.bs,
                "heads": args.heads,
                "seqlen": args.seqlen,
                "head_dim": args.head_dim,
            },
            "warmup": args.warmup,
            "iters": args.iters,
            "backend": args.backend,
            "mode": args.mode,
            "timer": args.timer,
            "measure_kind": "original_only",
            "fullgraph": args.fullgraph,
            "dynamic": args.dynamic,
            "input_init": args.input_init,
            "input_scale": args.input_scale,
            "tanh_scale": args.tanh_scale,
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
