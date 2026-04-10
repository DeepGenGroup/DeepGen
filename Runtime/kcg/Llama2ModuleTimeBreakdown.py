#!/usr/bin/env python3
"""
Profile a Llama-2-style decoder stack at module granularity for:
1. pure PyTorch attention
2. our selected attention kernel

The model skeleton matches the existing end-to-end benchmark. Only the
attention core switches between PyTorch and our kernel path; all other modules
stay in PyTorch.

Examples:
  python Runtime/kcg/Llama2ModuleTimeBreakdown.py --op attn
  python Runtime/kcg/Llama2ModuleTimeBreakdown.py --op all --seqlen 4096
  python Runtime/kcg/Llama2ModuleTimeBreakdown.py --op gemma2 --iters 10 --warmup 3
"""

import argparse
import math
import statistics
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Llama2E2EAttentionKernelBenchmark import (
    AttentionKernelRunner,
    GEMMA2_TANH_SCALE,
    OP_CHOICES,
    OP_SPECS,
    RMSNorm,
    apply_rotary_emb,
    build_model_config,
    dev_name,
    load_kernel_entry,
    precompute_freqs_cis,
    to_torch_dtype,
    torch_ns,
)


SECTION_ORDER = [
    "tok_embeddings",
    "mask_build",
    "attention_norm",
    "qkv_proj",
    "rope_and_layout",
    "attention_core",
    "o_proj",
    "attn_residual",
    "ffn_norm",
    "ffn_gate_proj",
    "ffn_up_proj",
    "ffn_act_mul",
    "ffn_down_proj",
    "ffn_residual",
    "final_norm",
    "output_head",
]

PER_LAYER_SECTIONS = {
    "attention_norm",
    "qkv_proj",
    "rope_and_layout",
    "attention_core",
    "o_proj",
    "attn_residual",
    "ffn_norm",
    "ffn_gate_proj",
    "ffn_up_proj",
    "ffn_act_mul",
    "ffn_down_proj",
    "ffn_residual",
}

GROUPS = {
    "embedding_block": ("tok_embeddings",),
    "mask_block": ("mask_build",),
    "attention_block": (
        "attention_norm",
        "qkv_proj",
        "rope_and_layout",
        "attention_core",
        "o_proj",
        "attn_residual",
    ),
    "ffn_block": (
        "ffn_norm",
        "ffn_gate_proj",
        "ffn_up_proj",
        "ffn_act_mul",
        "ffn_down_proj",
        "ffn_residual",
    ),
    "output_block": ("final_norm", "output_head"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile module-level time breakdown for a Llama-2-style decoder stack."
    )
    parser.add_argument("--op", choices=OP_CHOICES, default="all")
    parser.add_argument("--seqlen", type=int, default=4096, help="Sequence length to benchmark.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--multiple-of", type=int, default=256)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-output-head", action="store_true")
    parser.add_argument("--config-json", default="", help="Optional explicit config json path.")
    parser.add_argument(
        "--fallback-root",
        default=str(Path(__file__).resolve().parents[2] / "bench_attention"),
        help="Fallback root containing per-seqlen benchmark jsons.",
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    return parser.parse_args()


def summarize_times(times_ms: Iterable[float]) -> Dict[str, float]:
    values = list(times_ms)
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "sum_ms": float(sum(values)),
    }


class SectionProfiler:
    def __init__(self):
        self.total_ms_by_section: Dict[str, float] = defaultdict(float)
        self.iteration_sections: List[Dict[str, float]] = []
        self._pending_events: List[Tuple[str, Any, Any]] = []

    def begin_iteration(self) -> None:
        self._pending_events = []

    @contextmanager
    def section(self, name: str):
        start = torch_ns.Event(enable_timing=True)
        end = torch_ns.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending_events.append((name, start, end))

    def finish_iteration(self) -> Dict[str, float]:
        torch_ns.synchronize()
        totals: Dict[str, float] = defaultdict(float)
        for name, start, end in self._pending_events:
            elapsed = float(start.elapsed_time(end))
            totals[name] += elapsed
            self.total_ms_by_section[name] += elapsed
        self.iteration_sections.append(dict(totals))
        self._pending_events = []
        return dict(totals)


def section_scope(profiler: Optional[SectionProfiler], name: str):
    return profiler.section(name) if profiler is not None else nullcontext()


class ProfiledFeedForward(nn.Module):
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

    def forward(self, x: torch.Tensor, profiler: Optional[SectionProfiler] = None) -> torch.Tensor:
        with section_scope(profiler, "ffn_gate_proj"):
            gate = self.w1(x)
        with section_scope(profiler, "ffn_up_proj"):
            up = self.w3(x)
        with section_scope(profiler, "ffn_act_mul"):
            hidden = F.silu(gate) * up
        with section_scope(profiler, "ffn_down_proj"):
            return self.w2(hidden)


class ProfiledLlamaAttention(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str, kernel_runner: AttentionKernelRunner):
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
            _ = probs.sum(dim=2, keepdim=False)
        return torch.matmul(probs, values)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        profiler: Optional[SectionProfiler] = None,
    ) -> torch.Tensor:
        batch_size, seqlen, _ = x.shape

        with section_scope(profiler, "qkv_proj"):
            xq = self.wq(x).view(batch_size, seqlen, self.n_heads, self.head_dim)
            xk = self.wk(x).view(batch_size, seqlen, self.n_heads, self.head_dim)
            xv = self.wv(x).view(batch_size, seqlen, self.n_heads, self.head_dim)

        with section_scope(profiler, "rope_and_layout"):
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            query = xq.transpose(1, 2)
            keys = xk.transpose(1, 2)
            values = xv.transpose(1, 2)

        with section_scope(profiler, "attention_core"):
            if self.impl_mode == "kernel":
                output = self.kernel_runner(query, keys, values)
            else:
                output = self._torch_attention(query, keys, values, mask)

        with section_scope(profiler, "o_proj"):
            output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)
            output = self.wo(output)
        return output


class ProfiledTransformerBlock(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str, kernel_runner: AttentionKernelRunner):
        super().__init__()
        self.attention = ProfiledLlamaAttention(config, head_dim, op_name, kernel_runner)
        self.feed_forward = ProfiledFeedForward(config.dim, config.multiple_of, config.ffn_dim_multiplier)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def set_attention_impl(self, mode: str) -> None:
        self.attention.set_impl_mode(mode)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        profiler: Optional[SectionProfiler] = None,
    ) -> torch.Tensor:
        with section_scope(profiler, "attention_norm"):
            attn_in = self.attention_norm(x)
        attn_out = self.attention(attn_in, freqs_cis, mask, profiler=profiler)
        with section_scope(profiler, "attn_residual"):
            h = x + attn_out

        with section_scope(profiler, "ffn_norm"):
            ffn_in = self.ffn_norm(h)
        ffn_out = self.feed_forward(ffn_in, profiler=profiler)
        with section_scope(profiler, "ffn_residual"):
            return h + ffn_out


class ProfiledLlamaModel(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str, kernel_runner: AttentionKernelRunner):
        super().__init__()
        self.config = config
        self.attention_impl = "torch"

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [ProfiledTransformerBlock(config, head_dim, op_name, kernel_runner) for _ in range(config.n_layers)]
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

    def forward(self, tokens: torch.Tensor, profiler: Optional[SectionProfiler] = None) -> torch.Tensor:
        _, seqlen = tokens.shape
        with section_scope(profiler, "tok_embeddings"):
            h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)

        mask = None
        if self.attention_impl == "torch" and seqlen > 1:
            with section_scope(profiler, "mask_build"):
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device, dtype=h.dtype)
                mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask, profiler=profiler)

        with section_scope(profiler, "final_norm"):
            h = self.norm(h)
        if self.output is not None:
            with section_scope(profiler, "output_head"):
                h = self.output(h)
        return h


def benchmark_breakdown(
    model: ProfiledLlamaModel,
    tokens: torch.Tensor,
    mode: str,
    warmup: int,
    iters: int,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float], Dict[str, float]]:
    model.set_attention_impl(mode)
    profiler = SectionProfiler()
    total_times_ms: List[float] = []
    output = None

    with torch.inference_mode():
        for _ in range(warmup):
            output = model(tokens)
        torch_ns.synchronize()

        for _ in range(iters):
            profiler.begin_iteration()
            start = torch_ns.Event(enable_timing=True)
            end = torch_ns.Event(enable_timing=True)
            start.record()
            output = model(tokens, profiler=profiler)
            end.record()
            profiler.finish_iteration()
            total_times_ms.append(float(start.elapsed_time(end)))

    total_stats = summarize_times(total_times_ms)
    section_mean_ms = {
        name: total / iters for name, total in profiler.total_ms_by_section.items()
    }
    section_share_pct = {
        name: (100.0 * total / total_stats["sum_ms"]) if total_stats["sum_ms"] > 0 else 0.0
        for name, total in profiler.total_ms_by_section.items()
    }
    return output, total_stats, section_mean_ms, section_share_pct


def ordered_sections(section_mean_ms: Dict[str, float]) -> List[str]:
    seen = set()
    ordered = []
    for name in SECTION_ORDER:
        if name in section_mean_ms:
            ordered.append(name)
            seen.add(name)
    extras = sorted(name for name in section_mean_ms if name not in seen)
    ordered.extend(extras)
    return ordered


def grouped_mean_ms(section_mean_ms: Dict[str, float]) -> Dict[str, float]:
    grouped = {}
    for group_name, members in GROUPS.items():
        grouped[group_name] = float(sum(section_mean_ms.get(member, 0.0) for member in members))
    return grouped


def print_breakdown(
    label: str,
    total_stats: Dict[str, float],
    section_mean_ms: Dict[str, float],
    section_share_pct: Dict[str, float],
    n_layers: int,
) -> None:
    print(f"[{label}] median={total_stats['median_ms']:.3f} ms mean={total_stats['mean_ms']:.3f} ms")
    print(f"[{label} breakdown]")
    accounted_mean = 0.0
    for name in ordered_sections(section_mean_ms):
        mean_ms = section_mean_ms[name]
        share_pct = section_share_pct.get(name, 0.0)
        accounted_mean += mean_ms
        extra = ""
        if name in PER_LAYER_SECTIONS and n_layers > 0:
            extra = f" per_layer={mean_ms / n_layers:.3f} ms"
        print(f"  {name:<18} {mean_ms:>10.3f} ms  {share_pct:>7.2f}%{extra}")

    unaccounted_mean = total_stats["mean_ms"] - accounted_mean
    if abs(unaccounted_mean) < 1e-6:
        unaccounted_mean = 0.0
    unaccounted_pct = 100.0 * unaccounted_mean / total_stats["mean_ms"] if total_stats["mean_ms"] > 0 else 0.0
    print(f"  {'unaccounted':<18} {unaccounted_mean:>10.3f} ms  {unaccounted_pct:>7.2f}%")

    grouped = grouped_mean_ms(section_mean_ms)
    print(f"[{label} grouped]")
    for group_name in ("embedding_block", "mask_block", "attention_block", "ffn_block", "output_block"):
        mean_ms = grouped[group_name]
        share_pct = 100.0 * mean_ms / total_stats["mean_ms"] if total_stats["mean_ms"] > 0 else 0.0
        print(f"  {group_name:<18} {mean_ms:>10.3f} ms  {share_pct:>7.2f}%")


def print_comparison(
    torch_total: Dict[str, float],
    torch_sections: Dict[str, float],
    torch_shares: Dict[str, float],
    kernel_total: Dict[str, float],
    kernel_sections: Dict[str, float],
    kernel_shares: Dict[str, float],
) -> None:
    print("[module comparison]")
    section_names = ordered_sections({**torch_sections, **kernel_sections})
    for name in section_names:
        torch_ms = torch_sections.get(name, 0.0)
        kernel_ms = kernel_sections.get(name, 0.0)
        speedup = None
        if kernel_ms > 0 and torch_ms >= 0:
            speedup = torch_ms / kernel_ms
        speedup_text = f"{speedup:.4f}x" if speedup is not None and math.isfinite(speedup) else "n/a"
        print(
            f"  {name:<18} torch={torch_ms:>10.3f} ms ({torch_shares.get(name, 0.0):>6.2f}%)  "
            f"kernel={kernel_ms:>10.3f} ms ({kernel_shares.get(name, 0.0):>6.2f}%)  "
            f"speedup={speedup_text}"
        )

    total_speedup = None
    if kernel_total["median_ms"] > 0:
        total_speedup = torch_total["median_ms"] / kernel_total["median_ms"]
    total_speedup_text = f"{total_speedup:.4f}x" if total_speedup is not None and math.isfinite(total_speedup) else "n/a"
    print(
        f"[total] torch_median={torch_total['median_ms']:.3f} ms "
        f"kernel_median={kernel_total['median_ms']:.3f} ms speedup={total_speedup_text}"
    )


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    kernel_entry, kernel_source = load_kernel_entry(op_name, args.seqlen, args.config_json, args.fallback_root)
    runner = AttentionKernelRunner(op_name, kernel_entry, args.dtype, args.device_id)
    model_cfg, head_dim = build_model_config(args, runner.shape)

    dtype = to_torch_dtype(args.dtype)
    device = torch.device(dev_name(runner.device_id))

    model = ProfiledLlamaModel(model_cfg, head_dim, op_name, runner).to(device=device, dtype=dtype)
    model.eval()

    tokens = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(model_cfg.batch_size, model_cfg.max_seq_len),
        dtype=torch.long,
        device=device,
    )

    _, torch_total, torch_sections, torch_shares = benchmark_breakdown(
        model, tokens, "torch", args.warmup, args.iters
    )
    _, kernel_total, kernel_sections, kernel_shares = benchmark_breakdown(
        model, tokens, "kernel", args.warmup, args.iters
    )

    ffn_hidden = model.layers[0].feed_forward.hidden_dim if model.layers else 0

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
    print_breakdown("torch", torch_total, torch_sections, torch_shares, model_cfg.n_layers)
    print_breakdown("kernel", kernel_total, kernel_sections, kernel_shares, model_cfg.n_layers)
    print_comparison(torch_total, torch_sections, torch_shares, kernel_total, kernel_sections, kernel_shares)

    return {
        "op": op_name,
        "kernel_config_source": kernel_source,
        "kernel_entry": kernel_entry,
        "torch_total": torch_total,
        "torch_sections": torch_sections,
        "torch_shares": torch_shares,
        "kernel_total": kernel_total,
        "kernel_sections": kernel_sections,
        "kernel_shares": kernel_shares,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    op_names = list(OP_SPECS.keys()) if args.op == "all" else [args.op]
    results = []
    for index, op_name in enumerate(op_names, start=1):
        if index > 1:
            print("")
        print(f"========== [{index}/{len(op_names)}] {op_name} ==========")
        results.append(run_single_op(op_name, args))

    if len(results) > 1:
        print("")
        print("[summary]")
        for item in results:
            torch_ms = item["torch_total"]["median_ms"]
            kernel_ms = item["kernel_total"]["median_ms"]
            speedup = torch_ms / kernel_ms if kernel_ms > 0 else None
            speedup_text = f"{speedup:.4f}x" if speedup is not None and math.isfinite(speedup) else "n/a"
            print(
                f"{item['op']}: torch={torch_ms:.3f} ms kernel={kernel_ms:.3f} ms speedup={speedup_text}"
            )


if __name__ == "__main__":
    main()
