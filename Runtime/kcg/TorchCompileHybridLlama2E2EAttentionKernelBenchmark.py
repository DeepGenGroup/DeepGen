#!/usr/bin/env python3
"""
Benchmark a Llama-2-style decoder stack end-to-end with:
1. pure PyTorch attention/model execution
2. torch.compile hybrid: the whole model is compiled by torch.compile while the
   attention core is delegated to the DeepGen runtime-compiled kernel through a
   torch.library custom op (analogous to the TensorRT custom plugin used by
   TensorRTHybridLlama2E2EAttentionKernelBenchmark.py).

Examples:
  python Runtime/kcg/TorchCompileHybridLlama2E2EAttentionKernelBenchmark.py --op attn --seqlen 512 --layers 2
  python Runtime/kcg/TorchCompileHybridLlama2E2EAttentionKernelBenchmark.py --op all --seqlen 2048 --dtype float16
"""

import argparse
import gc
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Match TorchCompileLlama2E2EBenchmark default: keep the inductor compile worker
# pool small to avoid lingering subprocesses between ops.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from Llama2E2EAttentionKernelBenchmark import (
    AttentionKernelRunner,
    FeedForward,
    GEMMA2_TANH_SCALE,
    LlamaLikeModel,
    OP_CHOICES,
    OP_SPECS,
    RMSNorm,
    benchmark_model,
    build_model_config,
    dev_name,
    load_kernel_entry,
    to_torch_dtype,
)
from TVMLlama2E2EAttentionKernelBenchmark import (
    apply_rotary_emb_real,
    infer_kernel_shape,
    precompute_rope_tables,
)
from TVMOriginalBenchmark import benchmark_fn, format_exception, time_once


COMPILE_TIMER = "cuda_event"
COMPILE_STAGE_ORDER = ("model",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Llama-2-style decoder stack using pure PyTorch vs torch.compile "
            "with a custom attention op wrapping runtime-compiled DeepGen kernels."
        )
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
    parser.add_argument("--backend", default="inductor", help="torch.compile backend.")
    parser.add_argument("--fullgraph", action="store_true", help="Pass fullgraph=True to torch.compile.")
    parser.add_argument("--dynamic", action="store_true", help="Pass dynamic=True to torch.compile.")
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


def make_compile_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = dict(vars(args))
    payload["timer"] = COMPILE_TIMER
    return argparse.Namespace(**payload)


def make_result_meta(
    op_name: str,
    args: argparse.Namespace,
    model_cfg,
    head_dim: int,
    runtime_device_id: int,
    ffn_hidden: int,
    parameter_count: int,
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
        "parameter_count": parameter_count,
        "dtype": args.dtype,
        "timer": args.timer,
    }


# Module-level registry that maps an integer handle to a runtime-compiled
# DeepGen attention kernel runner. The custom op only takes tensor / scalar /
# string arguments, so the runner itself is looked up via the handle inside the
# op body (the handle is captured as a Python int on the owning nn.Module and
# inlined as a constant during dynamo tracing).
_RUNNER_REGISTRY: Dict[int, AttentionKernelRunner] = {}
_RUNNER_HANDLE_COUNTER = 0


def register_runner(runner: AttentionKernelRunner) -> int:
    global _RUNNER_HANDLE_COUNTER
    _RUNNER_HANDLE_COUNTER += 1
    handle = _RUNNER_HANDLE_COUNTER
    _RUNNER_REGISTRY[handle] = runner
    return handle


def unregister_runner(handle: int) -> None:
    _RUNNER_REGISTRY.pop(handle, None)


def run_pretransposed_attention_core(
    runner: AttentionKernelRunner,
    qq: torch.Tensor,
    kk: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    batch_size, heads, head_dim, seq_len = qq.shape
    current_shape = (batch_size, heads, seq_len, head_dim)
    if current_shape != tuple(runner.shape):
        raise ValueError(
            f"Input shape {current_shape} does not match compiled kernel shape {runner.shape}"
        )
    if qq.dtype != runner.dtype or kk.dtype != runner.dtype or v.dtype != runner.dtype:
        raise TypeError(
            f"Input dtype must be {runner.dtype}, got {qq.dtype}, {kk.dtype}, {v.dtype}"
        )

    if runner.op_name == "h2o":
        em, denom, row_sum, out = runner._get_workspace(batch_size, heads, seq_len, head_dim)
        runner.kernels[0].run(qq, kk, em, denom)
        runner.kernels[1].run(kk, qq, em, denom, row_sum)
        runner.kernels[2].run(qq, kk, v, em, denom, out)
        return out

    em, denom, out = runner._get_workspace(batch_size, heads, seq_len, head_dim)
    runner.kernels[0].run(qq, kk, em, denom)
    runner.kernels[1].run(qq, kk, v, em, denom, out)
    return out


# Register a custom op so torch.compile keeps the attention core as a single
# opaque node in the compiled graph (mirroring how the TRT plugin keeps the
# DeepGen kernels behind a single TensorRT layer).
@torch.library.custom_op("deepgen::attention_core", mutates_args=())
def _deepgen_attention_core(
    query_t: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    runner_handle: int,
) -> torch.Tensor:
    runner = _RUNNER_REGISTRY[runner_handle]
    out = run_pretransposed_attention_core(runner, query_t, key_t, value)
    # The kernel writes into a cached workspace; clone it so downstream graph
    # captures (e.g. inductor CUDA graphs) do not alias our internal buffers.
    return out.clone()


@_deepgen_attention_core.register_fake
def _deepgen_attention_core_fake(
    query_t: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    runner_handle: int,
) -> torch.Tensor:
    runner = _RUNNER_REGISTRY[runner_handle]
    batch_size, heads, seq_len, head_dim = runner.shape
    return value.new_empty((batch_size, heads, seq_len, head_dim))


def apply_attention_custom_op(
    query_t: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    runner_handle: int,
) -> torch.Tensor:
    return torch.ops.deepgen.attention_core(query_t, key_t, value, runner_handle)


class HybridLlamaAttention(nn.Module):
    def __init__(
        self,
        config,
        head_dim: int,
        op_name: str,
        kernel_runner: Optional[AttentionKernelRunner],
        runner_handle: int,
    ):
        super().__init__()
        self.op_name = op_name
        self.n_heads = config.n_heads
        self.head_dim = head_dim
        self.impl_mode = "torch"
        self.kernel_runner = kernel_runner
        self.runner_handle = int(runner_handle)

        proj_dim = self.n_heads * self.head_dim
        self.wq = nn.Linear(config.dim, proj_dim, bias=False)
        self.wk = nn.Linear(config.dim, proj_dim, bias=False)
        self.wv = nn.Linear(config.dim, proj_dim, bias=False)
        self.wo = nn.Linear(proj_dim, config.dim, bias=False)

    def set_impl_mode(self, mode: str) -> None:
        if mode not in ("torch", "kernel"):
            raise ValueError(f"Unsupported attention impl mode: {mode}")
        self.impl_mode = mode

    def _torch_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.op_name == "gemma2":
            scores = torch.tanh(scores / GEMMA2_TANH_SCALE) * GEMMA2_TANH_SCALE
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores.float(), dim=-1).type_as(query)
        if self.op_name == "h2o":
            row_sum = probs.sum(dim=2, keepdim=False).unsqueeze(-1)
            output = torch.matmul(probs, values)
            return output + row_sum.type_as(output) * torch.finfo(output.dtype).tiny
        return torch.matmul(probs, values)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb_real(xq, xk, rope_cos=rope_cos, rope_sin=rope_sin)
        query = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        if self.impl_mode == "kernel":
            if self.kernel_runner is None:
                raise RuntimeError("Attention kernel runner is not available for kernel mode.")
            query_t = query.transpose(-1, -2).contiguous()
            key_t = keys.transpose(-1, -2).contiguous()
            output = apply_attention_custom_op(
                query_t, key_t, values.contiguous(), self.runner_handle
            )
        else:
            output = self._torch_attention(query, keys, values, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class HybridTransformerBlock(nn.Module):
    def __init__(
        self,
        config,
        head_dim: int,
        op_name: str,
        kernel_runner: Optional[AttentionKernelRunner],
        runner_handle: int,
    ):
        super().__init__()
        self.attention = HybridLlamaAttention(config, head_dim, op_name, kernel_runner, runner_handle)
        self.feed_forward = FeedForward(config.dim, config.multiple_of, config.ffn_dim_multiplier)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def set_attention_impl(self, mode: str) -> None:
        self.attention.set_impl_mode(mode)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), rope_cos, rope_sin, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class HybridLlamaLikeModel(nn.Module):
    def __init__(
        self,
        config,
        head_dim: int,
        op_name: str,
        kernel_runner: Optional[AttentionKernelRunner],
        runner_handle: int,
    ):
        super().__init__()
        self.config = config
        self.op_name = op_name
        self.head_dim = head_dim
        self.attention_impl = "torch"

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [
                HybridTransformerBlock(config, head_dim, op_name, kernel_runner, runner_handle)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False) if config.with_output_head else None

        rope_cos, rope_sin = precompute_rope_tables(head_dim, config.max_seq_len * 2)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def set_attention_impl(self, mode: str) -> None:
        self.attention_impl = mode
        for layer in self.layers:
            layer.set_attention_impl(mode)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        if self.attention_impl == "torch" and seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin, mask)

        h = self.norm(h)
        if self.output is not None:
            h = self.output(h)
        return h


def build_torch_model(
    op_name: str,
    model_cfg,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    kernel_runner: Optional[AttentionKernelRunner],
    runner_handle: int,
) -> HybridLlamaLikeModel:
    model = HybridLlamaLikeModel(model_cfg, head_dim, op_name, kernel_runner, runner_handle).to(
        device=device, dtype=dtype
    )
    model.eval()
    return model


class TorchCompilePluginRuntime:
    def __init__(
        self,
        model: HybridLlamaLikeModel,
        op_name: str,
        kernel_runner: AttentionKernelRunner,
        args: argparse.Namespace,
        sample_tokens: torch.Tensor,
    ):
        self.model = model
        self.op_name = op_name
        self.kernel_runner = kernel_runner
        self.args = args
        self.device = sample_tokens.device
        self.dtype = model.norm.weight.dtype
        self.multi_op = args.op == "all"

        self.run = None

        self.engine_count = 0
        self.stage_stats = {name: self._empty_stage_stats() for name in COMPILE_STAGE_ORDER}
        self.total_export_ms = 0.0
        self.total_engine_build_ms = 0.0
        self.total_runtime_build_ms = 0.0

        self._build(sample_tokens)

    @staticmethod
    def _empty_stage_stats() -> Dict[str, Any]:
        return {
            "count": 0,
            "export_ms": 0.0,
            "engine_build_ms": 0.0,
            "runtime_build_ms": 0.0,
        }

    def _record_artifact(self, stage_name: str, artifact: Dict[str, Any]) -> None:
        stats = self.stage_stats[stage_name]
        stats["count"] += 1
        stats["export_ms"] += artifact["export_ms"]
        stats["engine_build_ms"] += artifact["engine_build_ms"]
        stats["runtime_build_ms"] += artifact["runtime_build_ms"]

        self.engine_count += 1
        self.total_export_ms += artifact["export_ms"]
        self.total_engine_build_ms += artifact["engine_build_ms"]
        self.total_runtime_build_ms += artifact["runtime_build_ms"]

    def _build_compile_runtime(
        self,
        module: nn.Module,
        stage_name: str,
    ):
        module = module.to(device=self.device, dtype=self.dtype).eval()

        # torch.compile has no separate ONNX export step. Keep export_ms == 0
        # so the per-stage layout still matches the TRT hybrid script.
        export_start = time.perf_counter()
        export_ms = float((time.perf_counter() - export_start) * 1000.0)

        build_start = time.perf_counter()
        compiled = torch.compile(
            module,
            backend=self.args.backend,
            fullgraph=bool(self.args.fullgraph),
            dynamic=bool(self.args.dynamic),
        )
        engine_build_ms = float((time.perf_counter() - build_start) * 1000.0)

        # torch.compile is lazy: there is no separate runtime instantiation
        # step. The first invocation triggers tracing+compile (captured as
        # first_call_ms in run_single_op), so this stage stays at 0 ms.
        runtime_start = time.perf_counter()
        runtime_build_ms = float((time.perf_counter() - runtime_start) * 1000.0)

        artifact = {
            "run": compiled,
            "export_ms": export_ms,
            "engine_build_ms": engine_build_ms,
            "runtime_build_ms": runtime_build_ms,
        }
        self._record_artifact(stage_name, artifact)
        return artifact["run"]

    def _build(self, sample_tokens: torch.Tensor) -> None:
        del sample_tokens  # tracing happens lazily on first call
        self.model.set_attention_impl("kernel")
        self.run = self._build_compile_runtime(module=self.model, stage_name="model")

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.run(tokens)


def print_stage_stats(stage_stats: Dict[str, Dict[str, Any]]) -> None:
    for stage_name in COMPILE_STAGE_ORDER:
        item = stage_stats.get(stage_name)
        if item is None or item["count"] <= 0:
            continue
        print(
            f"[torch.compile hybrid {stage_name}] count={item['count']} "
            f"export={item['export_ms']:.3f} ms "
            f"build={item['engine_build_ms']:.3f} ms "
            f"runtime={item['runtime_build_ms']:.3f} ms"
        )


def print_result(result: Dict[str, Any]) -> None:
    print(f"[op] {result['op']}")
    print(f"[config] source={result['config_source']}")
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
    print(f"[timer] {result['model']['timer']}")
    print(
        f"[torch] median={result['torch_e2e']['median_ms']:.3f} ms "
        f"mean={result['torch_e2e']['mean_ms']:.3f} ms"
    )

    if result.get("hybrid_error"):
        print(f"[torch.compile hybrid] failed: {result['hybrid_error']}")
        return

    print(f"[kernel compile] {result['kernel_compile_ms']:.3f} ms")
    print("[plugin kernel source] runtime_compile_via_libdeepgen")
    print(f"[torch.compile hybrid export] {result['compiled_hybrid_export_ms']:.3f} ms")
    print(f"[torch.compile hybrid build] {result['compiled_hybrid_engine_build_ms']:.3f} ms")
    print(f"[torch.compile hybrid runtime] {result['compiled_hybrid_runtime_build_ms']:.3f} ms")
    meta = result["compiled_hybrid_meta"]
    print(f"[torch.compile hybrid engines] count={meta['engine_count']}")
    print_stage_stats(meta["stage_stats"])
    print(f"[torch.compile hybrid first_call] {result['compiled_hybrid_first_call_ms']:.3f} ms")
    print(
        f"[torch.compile hybrid] median={result['compiled_hybrid_e2e']['median_ms']:.3f} ms "
        f"mean={result['compiled_hybrid_e2e']['mean_ms']:.3f} ms"
    )
    if result["compiled_hybrid_speedup_vs_torch"] is not None:
        print(
            f"[torch.compile hybrid speedup vs torch] {result['compiled_hybrid_speedup_vs_torch']:.4f}x"
        )


def cleanup_runtime(device: torch.device) -> None:
    dynamo_ns = getattr(torch, "_dynamo", None)
    if dynamo_ns is not None and hasattr(dynamo_ns, "reset"):
        try:
            dynamo_ns.reset()
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    kernel_entry, kernel_source = load_kernel_entry(op_name, args.seqlen, args.config_json, args.fallback_root)
    kernel_shape = infer_kernel_shape(op_name, kernel_entry)
    model_cfg, head_dim = build_model_config(args, kernel_shape)

    runner = AttentionKernelRunner(op_name, kernel_entry, args.dtype, args.device_id)
    dtype = to_torch_dtype(args.dtype)
    device = torch.device(dev_name(runner.device_id))
    tokens = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(model_cfg.batch_size, model_cfg.max_seq_len),
        dtype=torch.long,
        device=device,
    )

    runner_handle = register_runner(runner)

    plugin_runtime = None
    hybrid_model = None
    torch_model = None
    try:
        hybrid_model = build_torch_model(
            op_name, model_cfg, head_dim, dtype, device, runner, runner_handle
        )

        torch_model = LlamaLikeModel(model_cfg, head_dim, op_name, runner).to(device=device, dtype=dtype)
        torch_model.load_state_dict(hybrid_model.state_dict(), strict=False)
        torch_model.eval()
        torch_model.set_attention_impl("torch")
        _, torch_stats = benchmark_model(torch_model, tokens, "torch", args.warmup, args.iters)

        ffn_hidden = hybrid_model.layers[0].feed_forward.hidden_dim if hybrid_model.layers else 0
        parameter_count = int(sum(p.numel() for p in hybrid_model.parameters()))
        result: Dict[str, Any] = {
            "op": op_name,
            "config_source": kernel_source,
            "kernel_entry": kernel_entry,
            "kernel_names": list(runner.kernel_names),
            "kernel_compile_ms": float(runner.compile_ms),
            "model": make_result_meta(
                op_name, args, model_cfg, head_dim, runner.device_id, ffn_hidden, parameter_count
            ),
            "torch_e2e": torch_stats,
            "compiled_hybrid_export_ms": None,
            "compiled_hybrid_engine_build_ms": None,
            "compiled_hybrid_runtime_build_ms": None,
            "compiled_hybrid_first_call_ms": None,
            "compiled_hybrid_e2e": None,
            "compiled_hybrid_meta": None,
            "compiled_hybrid_speedup_vs_torch": None,
            "hybrid_error": None,
        }

        try:
            plugin_runtime = TorchCompilePluginRuntime(
                model=hybrid_model,
                op_name=op_name,
                kernel_runner=runner,
                args=args,
                sample_tokens=tokens,
            )
            result["compiled_hybrid_export_ms"] = float(plugin_runtime.total_export_ms)
            result["compiled_hybrid_engine_build_ms"] = float(plugin_runtime.total_engine_build_ms)
            result["compiled_hybrid_runtime_build_ms"] = float(plugin_runtime.total_runtime_build_ms)
            result["compiled_hybrid_meta"] = {
                "engine_count": plugin_runtime.engine_count,
                "stage_stats": plugin_runtime.stage_stats,
                "backend": args.backend,
                "fullgraph": bool(args.fullgraph),
                "dynamic": bool(args.dynamic),
                "disable_tensor_core": args.disable_tensor_core,
                "compile_threads_env": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", ""),
            }

            _, first_call_ms = time_once(
                plugin_runtime,
                (tokens,),
                device,
                args.timer,
            )
            _, hybrid_stats = benchmark_fn(
                plugin_runtime,
                (tokens,),
                device,
                args.warmup,
                args.iters,
                args.timer,
            )

            result["compiled_hybrid_first_call_ms"] = float(first_call_ms)
            result["compiled_hybrid_e2e"] = hybrid_stats

            if hybrid_stats["median_ms"] > 0 and torch_stats["median_ms"] > 0:
                result["compiled_hybrid_speedup_vs_torch"] = float(
                    torch_stats["median_ms"] / hybrid_stats["median_ms"]
                )
        except Exception as exc:
            result["hybrid_error"] = format_exception(exc)
    finally:
        del plugin_runtime
        del torch_model
        del hybrid_model
        unregister_runner(runner_handle)
        cleanup_runtime(device)

    print_result(result)
    return result


def print_summary(results: Dict[str, Any]) -> None:
    if len(results) <= 1:
        return
    print("")
    print("[summary]")
    for op_name, item in results.items():
        if item.get("hybrid_error"):
            print(f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms torch.compile_hybrid=failed")
            continue
        speedup_text = (
            f"{item['compiled_hybrid_speedup_vs_torch']:.4f}x"
            if item.get("compiled_hybrid_speedup_vs_torch") is not None
            else "n/a"
        )
        print(
            f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms "
            f"torch.compile_hybrid={item['compiled_hybrid_e2e']['median_ms']:.3f} ms "
            f"torch.compile_hybrid_speedup_vs_torch={speedup_text}"
        )


def main() -> None:
    args = parse_args()
    compile_args = make_compile_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        raise RuntimeError("torch.compile hybrid benchmark requires CUDA and at least one visible GPU.")

    if args.disable_tensor_core:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    op_names = list(OP_SPECS.keys()) if args.op == "all" else [args.op]
    results: Dict[str, Any] = {}
    for index, op_name in enumerate(op_names, start=1):
        if index > 1:
            print("")
        print(f"========== [{index}/{len(op_names)}] {op_name} ==========")
        results[op_name] = run_single_op(op_name, compile_args)

    print_summary(results)


if __name__ == "__main__":
    main()
