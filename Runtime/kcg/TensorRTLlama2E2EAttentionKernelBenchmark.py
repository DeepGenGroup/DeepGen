#!/usr/bin/env python3
"""
Benchmark a Llama-2-style decoder stack end-to-end with:
1. pure PyTorch attention/model execution
2. a full-model TensorRT runtime

The PyTorch path and TensorRT path share the same ONNX-export-friendly model
definition.

Examples:
  python Runtime/kcg/TensorRTLlama2E2EAttentionKernelBenchmark.py --op attn --seqlen 512 --layers 2
  python Runtime/kcg/TensorRTLlama2E2EAttentionKernelBenchmark.py --op all --seqlen 2048 --dtype float16
"""

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from Llama2E2EAttentionKernelBenchmark import (
    OP_CHOICES,
    OP_SPECS,
    build_model_config,
    dev_name,
    load_kernel_entry,
    select_runtime_device,
    to_torch_dtype,
)
from TensorRTOriginalBenchmark import (
    torch_module_to_onnx,
    trt_build_engine_from_onnx,
    trt_build_independent_runtime,
)
from TVMLlama2E2EAttentionKernelBenchmark import build_torch_model, infer_kernel_shape
from TVMOriginalBenchmark import benchmark_fn, format_exception, time_once

TRT_TIMER = "cuda_event"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Llama-2-style decoder stack using pure PyTorch vs full-model TensorRT."
    )
    parser.set_defaults(disable_tensor_core=True)
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
        help="Optional override. Must match heads * head_dim from the selected kernel.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Optional override. Must match the selected kernel.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Optional override. Must match the selected kernel.",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--workspace-gib", type=int, default=0)
    parser.add_argument("--save-engine", default="")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--verbose-trt", action="store_true")
    parser.add_argument(
        "--disable-tensor-core",
        dest="disable_tensor_core",
        action="store_true",
        help="Disable Tensor Core / TF32 paths (default).",
    )
    parser.add_argument(
        "--enable-tensor-core",
        dest="disable_tensor_core",
        action="store_false",
        help="Enable Tensor Core / TF32 paths.",
    )
    return parser.parse_args()


def make_trt_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = dict(vars(args))
    payload["timer"] = TRT_TIMER
    return argparse.Namespace(**payload)


def resolve_engine_path(engine_path: str, op_name: str, multi_op: bool) -> str:
    if not engine_path or not multi_op:
        return engine_path

    path = Path(engine_path)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return str(path.with_name(f"{stem}_{op_name}{suffix}"))


def export_onnx_artifact(
    module: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    op_name: str,
) -> Tuple[Any, Optional[tempfile.TemporaryDirectory], bool, bool]:
    try:
        artifact = torch_module_to_onnx(
            module=module,
            input_names=["tokens"],
            inputs=inputs,
            output_names=["out"],
            opset=args.opset,
            simplify=not args.no_simplify,
        )
        return artifact, None, False, not args.no_simplify
    except Exception as exc:
        if "Failed to serialize proto" not in str(exc):
            raise

    temp_dir = tempfile.TemporaryDirectory(prefix=f"trt_llama2_onnx_{op_name}_")
    onnx_path = Path(temp_dir.name) / f"{op_name}.onnx"
    torch.onnx.export(
        module,
        args=tuple(inputs),
        f=str(onnx_path),
        input_names=["tokens"],
        output_names=["out"],
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
        external_data=True,
    )
    if not args.no_simplify:
        print(
            "[trt export note] large model detected; skip onnxsim and use external-data ONNX export",
            flush=True,
        )
    return str(onnx_path), temp_dir, True, False


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
        "timer": args.timer,
    }


def print_result(result: Dict[str, Any]) -> None:
    print(f"[op] {result['op']}")
    print(f"[config] source={result['config_source']}")
    print(
        f"[device] requested={result['model']['requested_device_id']} runtime={result['model']['runtime_device_id']}"
    )
    print(
        f"[model] batch={result['model']['batch_size']} seqlen={result['model']['seqlen']} "
        f"dim={result['model']['hidden_size']} layers={result['model']['num_layers']} "
        f"heads={result['model']['num_heads']} head_dim={result['model']['head_dim']} "
        f"ffn={result['model']['ffn_hidden_size']} output_head={result['model']['with_output_head']}"
    )
    print(
        f"[torch] median={result['torch_e2e']['median_ms']:.3f} ms "
        f"mean={result['torch_e2e']['mean_ms']:.3f} ms"
    )

    if result.get("trt_error"):
        print(f"[trt] failed: {result['trt_error']}")
        return

    print(f"[trt export] {result['trt_export_ms']:.3f} ms")
    print(f"[trt build] {result['trt_engine_build_ms']:.3f} ms")
    print(f"[trt runtime] {result['trt_runtime_build_ms']:.3f} ms")
    print(f"[trt first_call] {result['trt_first_call_ms']:.3f} ms")
    print(
        f"[trt] median={result['trt_e2e']['median_ms']:.3f} ms "
        f"mean={result['trt_e2e']['mean_ms']:.3f} ms"
    )
    if result["trt_speedup_vs_torch"] is not None:
        print(f"[trt speedup vs torch] {result['trt_speedup_vs_torch']:.4f}x")


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    kernel_entry, kernel_source = load_kernel_entry(op_name, args.seqlen, args.config_json, args.fallback_root)
    kernel_shape = infer_kernel_shape(op_name, kernel_entry)
    runtime_device_id = select_runtime_device(args.device_id)
    model_cfg, head_dim = build_model_config(args, kernel_shape)

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(dev_name(runtime_device_id))
    torch.cuda.set_device(runtime_device_id)
    tokens = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(model_cfg.batch_size, model_cfg.max_seq_len),
        dtype=torch.long,
        device=device,
    )

    torch_model = build_torch_model(op_name, model_cfg, head_dim, dtype, device)
    _, torch_stats = benchmark_fn(
        torch_model,
        (tokens,),
        device,
        args.warmup,
        args.iters,
        args.timer,
    )

    ffn_hidden = torch_model.layers[0].feed_forward.hidden_dim if torch_model.layers else 0
    result = {
        "op": op_name,
        "config_source": kernel_source,
        "kernel_entry": kernel_entry,
        "model": make_result_meta(op_name, args, model_cfg, head_dim, runtime_device_id, ffn_hidden),
        "torch_e2e": torch_stats,
        "trt_export_ms": None,
        "trt_engine_build_ms": None,
        "trt_runtime_build_ms": None,
        "trt_first_call_ms": None,
        "trt_e2e": None,
        "trt_meta": None,
        "trt_speedup_vs_torch": None,
        "trt_error": None,
    }

    try:
        export_start = time.perf_counter()
        onnx_artifact, onnx_temp_dir, used_external_data, simplify_applied = export_onnx_artifact(
            module=torch_model,
            inputs=(tokens,),
            args=args,
            op_name=op_name,
        )
        result["trt_export_ms"] = float((time.perf_counter() - export_start) * 1000.0)

        engine_path = resolve_engine_path(args.save_engine, op_name, args.op == "all")

        build_start = time.perf_counter()
        engine_bytes = trt_build_engine_from_onnx(
            onnx_model=onnx_artifact,
            workspace_gib=args.workspace_gib,
            engine_fn=engine_path,
            verbose=args.verbose_trt,
            disable_tensor_core=args.disable_tensor_core,
        )
        result["trt_engine_build_ms"] = float((time.perf_counter() - build_start) * 1000.0)

        runtime_start = time.perf_counter()
        trt_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(trt_stream):
            trt_run = trt_build_independent_runtime(engine_bytes, verbose=args.verbose_trt)
        result["trt_runtime_build_ms"] = float((time.perf_counter() - runtime_start) * 1000.0)
        result["trt_meta"] = {
            "opset": args.opset,
            "workspace_gib": args.workspace_gib,
            "simplify": simplify_applied,
            "external_data_export": used_external_data,
            "disable_tensor_core": args.disable_tensor_core,
            "save_engine": engine_path,
        }

        _, first_call_ms = time_once(
            trt_run,
            (tokens,),
            device,
            args.timer,
            sync_fn=trt_stream.synchronize,
            event_stream=trt_stream,
        )
        _, trt_stats = benchmark_fn(
            trt_run,
            (tokens,),
            device,
            args.warmup,
            args.iters,
            args.timer,
            sync_fn=trt_stream.synchronize,
            event_stream=trt_stream,
        )

        result["trt_first_call_ms"] = float(first_call_ms)
        result["trt_e2e"] = trt_stats

        if trt_stats["median_ms"] > 0 and torch_stats["median_ms"] > 0:
            result["trt_speedup_vs_torch"] = float(torch_stats["median_ms"] / trt_stats["median_ms"])
    except Exception as exc:
        result["trt_error"] = format_exception(exc)
    finally:
        if "onnx_temp_dir" in locals() and onnx_temp_dir is not None:
            onnx_temp_dir.cleanup()

    print_result(result)
    return result


def print_summary(results: Dict[str, Any]) -> None:
    if len(results) <= 1:
        return
    print("")
    print("[summary]")
    for op_name, item in results.items():
        if item.get("trt_error"):
            print(f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms trt=failed")
            continue
        speedup_text = (
            f"{item['trt_speedup_vs_torch']:.4f}x"
            if item.get("trt_speedup_vs_torch") is not None
            else "n/a"
        )
        print(
            f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms "
            f"trt={item['trt_e2e']['median_ms']:.3f} ms "
            f"trt_speedup_vs_torch={speedup_text}"
        )


def main() -> None:
    args = parse_args()
    trt_args = make_trt_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        raise RuntimeError("TensorRT benchmark requires CUDA and at least one visible GPU.")

    if args.disable_tensor_core:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    op_names = list(OP_SPECS.keys()) if args.op == "all" else [args.op]
    results: Dict[str, Any] = {}
    for index, op_name in enumerate(op_names, start=1):
        if index > 1:
            print("")
        print(f"========== [{index}/{len(op_names)}] {op_name} ==========")
        results[op_name] = run_single_op(op_name, trt_args)

    print_summary(results)


if __name__ == "__main__":
    main()
