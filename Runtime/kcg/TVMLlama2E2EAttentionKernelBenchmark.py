#!/usr/bin/env python3
"""
Benchmark a Llama-2-style decoder stack end-to-end with:
1. pure PyTorch attention/model execution
2. a full-model TVM Relay runtime

The PyTorch path and TVM path share the same ONNX-export-friendly model
definition.

Examples:
  python Runtime/kcg/TVMLlama2E2EAttentionKernelBenchmark.py --op attn --seqlen 512 --layers 2
  python Runtime/kcg/TVMLlama2E2EAttentionKernelBenchmark.py --op all --seqlen 2048 --dtype float16
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Llama2E2EAttentionKernelBenchmark import (
    FeedForward,
    GEMMA2_TANH_SCALE,
    OP_CHOICES,
    OP_SPECS,
    RMSNorm,
    build_model_config,
    dev_name,
    load_kernel_entry,
    parse_kernel_name,
    select_runtime_device,
    to_torch_dtype,
)
from TVMOriginalBenchmark import (
    benchmark_fn,
    compile_with_relay,
    format_exception,
    time_once,
)

TVM_DEFAULT_OPTIONS = {
    "timer": "cuda_event",
    "scheduler": "metaschedule",
    "num_trials_per_iter": 4,
    "max_trials_per_task": 128,
    "max_trials_global": -1,
    "runner_timeout_sec": 30.0,
    "ms_cost_model": "xgb",
    "disable_cublas": False,
    "onnx_opset": 18,
    "work_dir_root": "",
    "export_lib_dir": "",
    "tvm_python_path": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Llama-2-style decoder stack using pure PyTorch vs full-model TVM Relay."
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
    return parser.parse_args()


def make_tvm_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = dict(vars(args))
    payload.update(TVM_DEFAULT_OPTIONS)
    return argparse.Namespace(**payload)


def precompute_rope_tables(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(end).float()
    freqs = torch.outer(positions, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float().reshape(*x.shape[:-1], -1, 2)
    x_even = x_float[..., 0]
    x_odd = x_float[..., 1]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.reshape_as(x_float.reshape(*x.shape)).type_as(x)


def apply_rotary_emb_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = xq.shape[1]
    cos = rope_cos[:seq_len].to(device=xq.device, dtype=xq.dtype).view(1, seq_len, 1, -1)
    sin = rope_sin[:seq_len].to(device=xq.device, dtype=xq.dtype).view(1, seq_len, 1, -1)
    xq_out = xq * cos + rotate_half(xq) * sin
    xk_out = xk * cos + rotate_half(xk) * sin
    return xq_out, xk_out


class TVMLlamaAttention(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str):
        super().__init__()
        self.op_name = op_name
        self.n_heads = config.n_heads
        self.head_dim = head_dim

        proj_dim = self.n_heads * self.head_dim
        self.wq = nn.Linear(config.dim, proj_dim, bias=False)
        self.wk = nn.Linear(config.dim, proj_dim, bias=False)
        self.wv = nn.Linear(config.dim, proj_dim, bias=False)
        self.wo = nn.Linear(proj_dim, config.dim, bias=False)

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

        scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.op_name == "gemma2":
            scores = torch.tanh(scores / GEMMA2_TANH_SCALE) * GEMMA2_TANH_SCALE
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores.float(), dim=-1).type_as(query)
        output = torch.matmul(probs, values)
        if self.op_name == "h2o":
            # Keep the extra reduction in-graph without changing the final tensor.
            row_sum = probs.sum(dim=2, keepdim=False).unsqueeze(-1).type_as(output)
            zero = output[..., :1] - output[..., :1]
            output = output + row_sum * zero

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class TVMTransformerBlock(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str):
        super().__init__()
        self.attention = TVMLlamaAttention(config, head_dim, op_name)
        self.feed_forward = FeedForward(config.dim, config.multiple_of, config.ffn_dim_multiplier)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), rope_cos, rope_sin, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class TVMLlamaLikeModel(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TVMTransformerBlock(config, head_dim, op_name) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False) if config.with_output_head else None

        rope_cos, rope_sin = precompute_rope_tables(head_dim, config.max_seq_len * 2)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin, mask)

        h = self.norm(h)
        if self.output is not None:
            h = self.output(h)
        return h


def infer_kernel_shape(op_name: str, kernel_entry: Dict[str, Any]) -> Tuple[int, int, int, int]:
    shapes = []
    for key in OP_SPECS[op_name]["required_names"]:
        shape, _ = parse_kernel_name(kernel_entry[key])
        shapes.append(shape)
    if len(set(shapes)) != 1:
        raise ValueError(f"Kernel shape mismatch while inferring model config: {shapes}")
    return shapes[0]


def build_torch_model(
    op_name: str,
    model_cfg,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> TVMLlamaLikeModel:
    model = TVMLlamaLikeModel(model_cfg, head_dim, op_name).to(device=device, dtype=dtype)
    model.eval()
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
        "timer": TVM_DEFAULT_OPTIONS["timer"],
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

    if result.get("tvm_error"):
        print(f"[tvm] failed: {result['tvm_error']}")
        return

    print(f"[tvm compile] {result['tvm_compile_ms']:.3f} ms")
    print(f"[tvm first_call] {result['tvm_first_call_ms']:.3f} ms")
    print(
        f"[tvm] median={result['tvm_e2e']['median_ms']:.3f} ms "
        f"mean={result['tvm_e2e']['mean_ms']:.3f} ms"
    )
    if result["tvm_speedup_vs_torch"] is not None:
        print(f"[tvm speedup vs torch] {result['tvm_speedup_vs_torch']:.4f}x")


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    kernel_entry, kernel_source = load_kernel_entry(op_name, args.seqlen, args.config_json, args.fallback_root)
    kernel_shape = infer_kernel_shape(op_name, kernel_entry)
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
        "tvm_compile_ms": None,
        "tvm_first_call_ms": None,
        "tvm_e2e": None,
        "tvm_meta": None,
        "tvm_speedup_vs_torch": None,
        "tvm_error": None,
    }

    try:
        runtime, compile_meta = compile_with_relay(
            module=torch_model,
            input_names=("tokens",),
            inputs=(tokens,),
            output_names=("out",),
            device=device,
            args=args,
            op_name=f"llama2_{op_name}",
        )
        runtime["prepare_inputs"]((tokens,), ("tokens",))

        _, first_call_ms = time_once(
            runtime["run_raw"],
            tuple(),
            device,
            args.timer,
            sync_fn=runtime["sync"],
        )
        _, tvm_stats = benchmark_fn(
            runtime["run_raw"],
            tuple(),
            device,
            args.warmup,
            args.iters,
            args.timer,
            sync_fn=runtime["sync"],
        )

        result["tvm_compile_ms"] = float(compile_meta["compile_ms"])
        result["tvm_first_call_ms"] = float(first_call_ms)
        result["tvm_e2e"] = tvm_stats
        result["tvm_meta"] = compile_meta

        if tvm_stats["median_ms"] > 0 and torch_stats["median_ms"] > 0:
            result["tvm_speedup_vs_torch"] = float(torch_stats["median_ms"] / tvm_stats["median_ms"])
    except Exception as exc:
        result["tvm_error"] = format_exception(exc)

    print_result(result)
    return result


def print_summary(results: Dict[str, Any]) -> None:
    if len(results) <= 1:
        return
    print("")
    print("[summary]")
    for op_name, item in results.items():
        if item.get("tvm_error"):
            print(f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms tvm=failed")
            continue
        speedup_text = (
            f"{item['tvm_speedup_vs_torch']:.4f}x"
            if item.get("tvm_speedup_vs_torch") is not None
            else "n/a"
        )
        print(
            f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms "
            f"tvm={item['tvm_e2e']['median_ms']:.3f} ms "
            f"tvm_speedup_vs_torch={speedup_text}"
        )


def main() -> None:
    args = parse_args()
    tvm_args = make_tvm_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    op_names = list(OP_SPECS.keys()) if args.op == "all" else [args.op]
    results: Dict[str, Any] = {}
    for index, op_name in enumerate(op_names, start=1):
        if index > 1:
            print("")
        print(f"========== [{index}/{len(op_names)}] {op_name} ==========")
        results[op_name] = run_single_op(op_name, tvm_args)

    print_summary(results)


if __name__ == "__main__":
    main()
