#!/usr/bin/env python3
"""
Benchmark a Llama-2-style decoder stack end-to-end with:
1. eager PyTorch
2. torch.compile

The model structure reuses Llama2E2EAttentionKernelBenchmark.py so the
attention formula, RoPE path, mask behavior, and model config logic stay
aligned with the existing end-to-end benchmark. This script only runs the
pure PyTorch attention path with eager and torch.compile execution.

Examples:
  python Runtime/kcg/TorchCompileLlama2E2EBenchmark.py --op attn --seqlen 512 --layers 2
  python Runtime/kcg/TorchCompileLlama2E2EBenchmark.py --op all --seqlen 2048 --dtype float16
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Keep the same default inductor worker behavior as the standalone compile
# benchmark to reduce compile-time noise and lingering worker pools.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

from Llama2E2EAttentionKernelBenchmark import (
    LlamaLikeModel,
    OP_CHOICES,
    OP_SPECS,
    build_model_config,
    dev_name,
    load_kernel_entry,
    parse_kernel_name,
    select_runtime_device,
    summarize_times,
    to_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Llama-2-style decoder stack using eager PyTorch vs torch.compile."
    )
    parser.add_argument("--op", choices=OP_CHOICES, default="attn")
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length to benchmark.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--layers",
        type=int,
        default=32,
        help="Number of decoder layers. Defaults to Llama-2-7B style.",
    )
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
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Optional override. Must match heads * head_dim from the selected kernel config.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Optional override. Must match the selected kernel config.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Optional override. Must match the selected kernel config.",
    )
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--mode", default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--timer", choices=("cpu_wall", "cuda_event"), default="cuda_event")
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def cleanup_runtime(device: torch.device) -> None:
    reset = getattr(torch, "_dynamo", None)
    if reset is not None and hasattr(reset, "reset"):
        torch._dynamo.reset()
    sync_device(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync_device(device)


def try_compile(module: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    return torch.compile(
        module,
        backend=args.backend,
        mode=args.mode,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
    )


def infer_kernel_shape(op_name: str, kernel_entry: Dict[str, Any]) -> Tuple[Tuple[int, int, int, int], List[str]]:
    required_names = OP_SPECS[op_name]["required_names"]
    kernel_names = [kernel_entry[key] for key in required_names]
    shapes = []
    for kernel_name in kernel_names:
        shape, _ = parse_kernel_name(kernel_name)
        shapes.append(shape)
    if len(set(shapes)) != 1:
        raise ValueError(f"Kernel shape mismatch: {shapes}")
    return shapes[0], kernel_names


def build_eager_model(
    op_name: str,
    args: argparse.Namespace,
    model_cfg,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> LlamaLikeModel:
    model = LlamaLikeModel(model_cfg, head_dim, op_name, kernel_runner=None).to(device=device, dtype=dtype)
    model.eval()
    model.set_attention_impl("torch")
    return model


def make_result_meta(
    op_name: str,
    args: argparse.Namespace,
    model_cfg,
    head_dim: int,
    runtime_device_id: int,
    ffn_hidden: int,
) -> Dict[str, Any]:
    return {
        "family": "llama2_style",
        "op": op_name,
        "requested_device_id": args.device_id,
        "runtime_device_id": runtime_device_id,
        "device": dev_name(runtime_device_id),
        "batch_size": model_cfg.batch_size,
        "seqlen": model_cfg.max_seq_len,
        "hidden_size": model_cfg.dim,
        "num_layers": model_cfg.n_layers,
        "num_heads": model_cfg.n_heads,
        "head_dim": head_dim,
        "ffn_hidden_size": ffn_hidden,
        "multiple_of": model_cfg.multiple_of,
        "with_output_head": model_cfg.with_output_head,
        "dtype": args.dtype,
    }


def print_result(result: Dict[str, Any]) -> None:
    print(f"[op] {result['op']}")
    print(f"[config] source={result['kernel_config_source']}")
    if result["model"] is None or result["eager_e2e"] is None:
        print(f"[run] failed before benchmark completed: {result.get('compile_error', 'unknown error')}")
        return
    print(
        f"[device] requested={result['model']['requested_device_id']} runtime={result['model']['runtime_device_id']}"
    )
    for idx, kernel_name in enumerate(result["kernel_names"], start=1):
        print(f"[kernel{idx}] {kernel_name}")
    print(
        f"[model] batch={result['model']['batch_size']} seqlen={result['model']['seqlen']} "
        f"dim={result['model']['hidden_size']} layers={result['model']['num_layers']} "
        f"heads={result['model']['num_heads']} head_dim={result['model']['head_dim']} "
        f"ffn={result['model']['ffn_hidden_size']} output_head={result['model']['with_output_head']}"
    )
    print(
        f"[compile config] backend={result['compile']['backend']} mode={result['compile']['mode']} "
        f"fullgraph={result['compile']['fullgraph']} dynamic={result['compile']['dynamic']} "
        f"timer={result['compile']['timer']}"
    )
    print(f"[eager] median={result['eager_e2e']['median_ms']:.3f} ms mean={result['eager_e2e']['mean_ms']:.3f} ms")

    if result.get("compile_error"):
        print(f"[compiled] failed: {result['compile_error']}")
        return

    print(f"[compiled first_call] {result['compiled_first_call_ms']:.3f} ms")
    print(
        f"[compiled] median={result['compiled_e2e']['median_ms']:.3f} ms "
        f"mean={result['compiled_e2e']['mean_ms']:.3f} ms"
    )
    print(f"[speedup compiled vs eager] {result['speedup']:.4f}x")


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    kernel_entry, kernel_source = load_kernel_entry(op_name, args.seqlen, args.config_json, args.fallback_root)
    kernel_shape, kernel_names = infer_kernel_shape(op_name, kernel_entry)
    runtime_device_id = select_runtime_device(args.device_id)
    model_cfg, head_dim = build_model_config(args, kernel_shape)

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(dev_name(runtime_device_id))
    tokens = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(model_cfg.batch_size, model_cfg.max_seq_len),
        dtype=torch.long,
        device=device,
    )

    eager_model = None
    compiled_model = None
    result: Dict[str, Any] = {
        "op": op_name,
        "kernel_config_source": kernel_source,
        "kernel_entry": kernel_entry,
        "kernel_names": kernel_names,
        "model": None,
        "compile": {
            "backend": args.backend,
            "mode": args.mode,
            "fullgraph": bool(args.fullgraph),
            "dynamic": bool(args.dynamic),
            "timer": args.timer,
            "compile_threads_env": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", ""),
        },
        "eager_e2e": None,
        "compiled_first_call_ms": None,
        "compiled_e2e": None,
        "speedup": None,
        "compile_error": None,
    }

    try:
        eager_model = build_eager_model(op_name, args, model_cfg, head_dim, dtype, device)
        _, eager_stats = benchmark_fn(eager_model, (tokens,), device, args.warmup, args.iters, args.timer)

        ffn_hidden = eager_model.layers[0].feed_forward.hidden_dim if eager_model.layers else 0
        result["model"] = make_result_meta(op_name, args, model_cfg, head_dim, runtime_device_id, ffn_hidden)
        result["eager_e2e"] = eager_stats

        compiled_model = try_compile(eager_model, args)
        _, compiled_first_ms = time_once(compiled_model, (tokens,), device, args.timer)

        _, compiled_stats = benchmark_fn(compiled_model, (tokens,), device, args.warmup, args.iters, args.timer)
        result["compiled_first_call_ms"] = float(compiled_first_ms)
        result["compiled_e2e"] = compiled_stats

        if compiled_stats["median_ms"] > 0:
            result["speedup"] = float(eager_stats["median_ms"] / compiled_stats["median_ms"])
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        del compiled_model
        del eager_model
        cleanup_runtime(device)

    print_result(result)
    return result


def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    if len(results) <= 1:
        return
    print("")
    print("[summary]")
    for op_name, item in results.items():
        if item.get("eager_e2e") is None:
            print(f"{op_name}: failed")
            continue
        if item.get("compile_error"):
            print(f"{op_name}: eager={item['eager_e2e']['median_ms']:.3f} ms compiled=failed")
            continue
        speedup_text = f"{item['speedup']:.4f}x" if item.get("speedup") is not None else "n/a"
        print(
            f"{op_name}: eager={item['eager_e2e']['median_ms']:.3f} ms "
            f"compiled={item['compiled_e2e']['median_ms']:.3f} ms "
            f"speedup={speedup_text}"
        )


def build_json_payload(args: argparse.Namespace, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "meta": {
            "torch_version": torch.__version__,
            "measure_kind": "llama2_e2e_torch_vs_torch_compile",
            "backend": args.backend,
            "mode": args.mode,
            "fullgraph": bool(args.fullgraph),
            "dynamic": bool(args.dynamic),
            "timer": args.timer,
            "compile_threads_env": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", ""),
            "seed": args.seed,
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    op_names: Iterable[str] = OP_SPECS.keys() if args.op == "all" else (args.op,)
    results: Dict[str, Dict[str, Any]] = {}
    for index, op_name in enumerate(op_names, start=1):
        if index > 1:
            print("")
        print(f"========== [{index}/{len(OP_SPECS) if args.op == 'all' else 1}] {op_name} ==========")
        results[op_name] = run_single_op(op_name, args)

    print_summary(results)

    if args.json_out:
        payload = build_json_payload(args, results)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[json] saved to {args.json_out}")


if __name__ == "__main__":
    main()
