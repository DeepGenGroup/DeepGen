#!/usr/bin/env python3
"""
Benchmark a Llama-2-style decoder stack end-to-end with:
1. pure PyTorch attention/model execution
2. TensorRT subgraph engines with an in-graph custom attention plugin

The PyTorch path and the TensorRT path share the same ONNX-export-friendly
building blocks. Attention core is emitted as a TensorRT custom plugin rather
than being stitched from Python outside the engine. The plugin .so is only the
TensorRT layer wrapper; the underlying DeepGen attention kernels are still
compiled at runtime via libdeepgen and then loaded by the plugin.

Examples:
  python Runtime/kcg/TensorRTHybridLlama2E2EAttentionKernelBenchmark.py --op attn --seqlen 512 --layers 2
  python Runtime/kcg/TensorRTHybridLlama2E2EAttentionKernelBenchmark.py --op all --seqlen 2048 --dtype float16
"""

import argparse
import copy
import ctypes
import io
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from TensorRTOriginalBenchmark import trt_build_engine_from_onnx, trt_build_independent_runtime
from TVMLlama2E2EAttentionKernelBenchmark import (
    apply_rotary_emb_real,
    infer_kernel_shape,
    precompute_rope_tables,
)
from TVMOriginalBenchmark import benchmark_fn, format_exception, time_once


TRT_TIMER = "cuda_event"
TRT_PLUGIN_NAME = "DeepGenAttentionCorePlugin"
TRT_PLUGIN_NAMESPACE = ""
TRT_PLUGIN_VERSION = "1"
TRT_STAGE_ORDER = ("model",)
DEFAULT_PLUGIN_LIB = str(Path(__file__).resolve().parents[2] / "bin" / "libdeepgen_trt_attention_plugin.so")
DEFAULT_PLUGIN_BUILD_SCRIPT = str(Path(__file__).resolve().parents[2] / "build_tools" / "build_trt_attention_plugin.sh")
_LOADED_SHARED_LIBRARIES: Dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Llama-2-style decoder stack using pure PyTorch vs a single TensorRT engine with a custom attention plugin wrapper around runtime-compiled DeepGen kernels."
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
    parser.add_argument("--save-engine-dir", dest="save_engine", help=argparse.SUPPRESS)
    parser.add_argument(
        "--plugin-lib",
        default=DEFAULT_PLUGIN_LIB,
        help="Path to the TensorRT custom attention plugin shared library.",
    )
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


def resolve_engine_path(
    engine_path: str,
    op_name: str,
    artifact_name: Optional[str],
    multi_op: bool,
) -> str:
    if not engine_path:
        return ""

    path = Path(engine_path)
    if path.exists() and path.is_dir():
        root = path / op_name if multi_op else path
        root.mkdir(parents=True, exist_ok=True)
        filename = f"{artifact_name}.engine" if artifact_name else "model.engine"
        return str(root / filename)

    original_suffix = "".join(path.suffixes)
    suffix = original_suffix if original_suffix else ".engine"
    stem = path.name[: -len(original_suffix)] if original_suffix else path.name
    name_parts = [stem]
    if multi_op:
        name_parts.append(op_name)
    if artifact_name:
        name_parts.append(artifact_name)
    output_name = "_".join(part for part in name_parts if part) + suffix
    output_path = path.with_name(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def export_onnx_artifact(
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    args: argparse.Namespace,
    artifact_name: str,
) -> Tuple[Any, Optional[tempfile.TemporaryDirectory], bool, bool]:
    artifact_label = artifact_name or "model"

    def _contains_custom_plugin_op(onnx_model: Any) -> bool:
        return any(node.op_type == TRT_PLUGIN_NAME for node in onnx_model.graph.node)

    def _torch_onnx_export(target):
        export_kwargs = {
            "args": tuple(inputs),
            "f": target,
            "input_names": input_names,
            "output_names": output_names,
            "opset_version": args.opset,
            "do_constant_folding": True,
            "verbose": False,
        }
        try:
            torch.onnx.export(module, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(module, **export_kwargs)

    try:
        import onnx

        onnx_bytes = io.BytesIO()
        _torch_onnx_export(onnx_bytes)
        artifact = onnx.load_model_from_string(onnx_bytes.getvalue())
        should_simplify = not args.no_simplify and not _contains_custom_plugin_op(artifact)
        if not args.no_simplify and not should_simplify:
            print(
                f"[trt plugin export note] skip onnxsim for {artifact_label} because it contains a custom plugin op",
                flush=True,
            )
        if should_simplify:
            import onnxsim

            artifact, simplify_ok = onnxsim.simplify(artifact)
            if not simplify_ok:
                raise RuntimeError("onnxsim simplify failed")
        return artifact, None, False, should_simplify
    except Exception as exc:
        exc_text = str(exc)
        if "Failed to serialize proto" not in exc_text and "larger than the 2GiB limit" not in exc_text:
            raise

    temp_dir = tempfile.TemporaryDirectory(prefix=f"trt_hybrid_llama2_{artifact_name}_")
    onnx_path = Path(temp_dir.name) / f"{artifact_name}.onnx"
    try:
        torch.onnx.export(
            module,
            args=tuple(inputs),
            f=str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            verbose=False,
            external_data=True,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            module,
            args=tuple(inputs),
            f=str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            verbose=False,
            external_data=True,
        )
    if not args.no_simplify:
        print(
            f"[trt plugin export note] large artifact detected for {artifact_label}; "
            "skip onnxsim and use external-data ONNX export",
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


@dataclass
class KernelLaunchSpec:
    binary_path: str
    kernel_name: str
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    shm_bytes: int


@dataclass
class AttentionPluginSpec:
    op_name: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    num_kernels: int
    kernels: List[KernelLaunchSpec]


def _load_shared_library(path: str) -> Any:
    if path in _LOADED_SHARED_LIBRARIES:
        return _LOADED_SHARED_LIBRARIES[path]
    mode = getattr(ctypes, "RTLD_GLOBAL", None)
    handle = ctypes.CDLL(path, mode=mode) if mode is not None else ctypes.CDLL(path)
    _LOADED_SHARED_LIBRARIES[path] = handle
    return handle


def preload_tensorrt_shared_libraries() -> None:
    try:
        import tensorrt as trt  # noqa: F401
    except Exception:
        return

    package_dir = Path(trt.__file__).resolve().parent
    libs_dir = package_dir.parent / "tensorrt_libs"
    if not libs_dir.exists():
        return

    for lib_name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        lib_path = libs_dir / lib_name
        if lib_path.exists():
            _load_shared_library(str(lib_path))


def build_deepgen_trt_plugin_if_needed(plugin_path: Path) -> None:
    default_plugin_path = Path(DEFAULT_PLUGIN_LIB).expanduser().resolve()
    if plugin_path != default_plugin_path:
        return

    build_script = Path(DEFAULT_PLUGIN_BUILD_SCRIPT).expanduser().resolve()
    if not build_script.exists():
        return

    print(f"[plugin build] {plugin_path} not found, invoking {build_script}", flush=True)
    env = dict(os.environ)
    env.setdefault("Python_EXECUTABLE", sys.executable)
    build_result = subprocess.run(
        ["bash", str(build_script)],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
    )
    if build_result.returncode != 0:
        raise subprocess.CalledProcessError(build_result.returncode, build_result.args)


def load_deepgen_trt_plugin(plugin_lib: str) -> Any:
    if not plugin_lib:
        raise ValueError("A TensorRT plugin library path is required for plugin-mode benchmark.")
    plugin_path = Path(plugin_lib).expanduser().resolve()
    if not plugin_path.exists():
        try:
            build_deepgen_trt_plugin_if_needed(plugin_path)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Automatic plugin build failed via {DEFAULT_PLUGIN_BUILD_SCRIPT}. "
                "Please build the plugin manually or fix the build environment."
            ) from exc
    if not plugin_path.exists():
        raise FileNotFoundError(
            f"TensorRT attention plugin library not found: {plugin_path}. "
            "Build the plugin target first or pass --plugin-lib explicitly."
        )
    preload_tensorrt_shared_libraries()
    return _load_shared_library(str(plugin_path))


def build_attention_plugin_spec(runner: AttentionKernelRunner) -> AttentionPluginSpec:
    batch_size, num_heads, seq_len, head_dim = runner.shape
    actual_kernels = [
        KernelLaunchSpec(
            binary_path=kernel.kernelBinaryPath,
            kernel_name=kernel.kernelName,
            grid=tuple(int(v) for v in kernel.gridDims),
            block=tuple(int(v) for v in kernel.blockDims),
            shm_bytes=int(kernel.shmSize),
        )
        for kernel in runner.kernels
    ]
    kernels = list(actual_kernels)
    while len(kernels) < 3:
        kernels.append(
            KernelLaunchSpec(
                binary_path="",
                kernel_name="",
                grid=(0, 0, 0),
                block=(0, 0, 0),
                shm_bytes=0,
            )
        )
    return AttentionPluginSpec(
        op_name=runner.op_name,
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        seq_len=int(seq_len),
        head_dim=int(head_dim),
        num_kernels=len(actual_kernels),
        kernels=kernels[:3],
    )


def run_pretransposed_attention_core(
    runner: AttentionKernelRunner,
    qq: torch.Tensor,
    kk: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    batch_size, heads, head_dim, seq_len = qq.shape
    current_shape = (batch_size, heads, seq_len, head_dim)
    if current_shape != tuple(runner.shape):
        raise ValueError(f"Input shape {current_shape} does not match compiled kernel shape {runner.shape}")
    if qq.dtype != runner.dtype or kk.dtype != runner.dtype or v.dtype != runner.dtype:
        raise TypeError(f"Input dtype must be {runner.dtype}, got {qq.dtype}, {kk.dtype}, {v.dtype}")

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


class DeepGenAttentionCoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_t: torch.Tensor,
        key_t: torch.Tensor,
        value: torch.Tensor,
        plugin_spec: AttentionPluginSpec,
        runner: AttentionKernelRunner,
    ) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return value.clone()
        return run_pretransposed_attention_core(runner, query_t, key_t, value)

    @staticmethod
    def symbolic(
        g,
        query_t,
        key_t,
        value,
        plugin_spec: AttentionPluginSpec,
        runner: AttentionKernelRunner,
    ):
        attrs: Dict[str, Any] = {
            "plugin_namespace_s": TRT_PLUGIN_NAMESPACE,
            "plugin_version_s": TRT_PLUGIN_VERSION,
            "num_outputs_i": 1,
            "op_name_s": plugin_spec.op_name,
            "batch_size_i": plugin_spec.batch_size,
            "num_heads_i": plugin_spec.num_heads,
            "seq_len_i": plugin_spec.seq_len,
            "head_dim_i": plugin_spec.head_dim,
            "num_kernels_i": plugin_spec.num_kernels,
        }
        for index, spec in enumerate(plugin_spec.kernels[:3], start=1):
            prefix = f"k{index}"
            attrs[f"{prefix}_binary_path_s"] = spec.binary_path
            attrs[f"{prefix}_kernel_name_s"] = spec.kernel_name
            attrs[f"{prefix}_grid_x_i"] = spec.grid[0]
            attrs[f"{prefix}_grid_y_i"] = spec.grid[1]
            attrs[f"{prefix}_grid_z_i"] = spec.grid[2]
            attrs[f"{prefix}_block_x_i"] = spec.block[0]
            attrs[f"{prefix}_block_y_i"] = spec.block[1]
            attrs[f"{prefix}_block_z_i"] = spec.block[2]
            attrs[f"{prefix}_shm_bytes_i"] = spec.shm_bytes
        return g.op(TRT_PLUGIN_NAME, query_t, key_t, value, **attrs)


def apply_attention_plugin(
    query_t: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    plugin_spec: AttentionPluginSpec,
    runner: AttentionKernelRunner,
) -> torch.Tensor:
    return DeepGenAttentionCoreFunction.apply(query_t, key_t, value, plugin_spec, runner)


class HybridLlamaAttention(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str, kernel_runner: Optional[AttentionKernelRunner]):
        super().__init__()
        self.op_name = op_name
        self.n_heads = config.n_heads
        self.head_dim = head_dim
        self.impl_mode = "torch"
        self.kernel_runner = kernel_runner
        self.plugin_spec = build_attention_plugin_spec(kernel_runner) if kernel_runner is not None else None

        proj_dim = self.n_heads * self.head_dim
        self.wq = nn.Linear(config.dim, proj_dim, bias=False)
        self.wk = nn.Linear(config.dim, proj_dim, bias=False)
        self.wv = nn.Linear(config.dim, proj_dim, bias=False)
        self.wo = nn.Linear(proj_dim, config.dim, bias=False)

    def set_impl_mode(self, mode: str) -> None:
        if mode not in ("torch", "kernel", "plugin"):
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

        if self.impl_mode == "plugin":
            if self.kernel_runner is None or self.plugin_spec is None:
                raise RuntimeError("Attention plugin mode requires a compiled kernel runner.")
            query_t = query.transpose(-1, -2).contiguous()
            key_t = keys.transpose(-1, -2).contiguous()
            output = apply_attention_plugin(query_t, key_t, values.contiguous(), self.plugin_spec, self.kernel_runner)
        elif self.impl_mode == "kernel":
            if self.kernel_runner is None:
                raise RuntimeError("Attention kernel runner is not available for kernel mode.")
            output = self.kernel_runner(query, keys, values)
        else:
            output = self._torch_attention(query, keys, values, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class HybridTransformerBlock(nn.Module):
    def __init__(self, config, head_dim: int, op_name: str, kernel_runner: Optional[AttentionKernelRunner]):
        super().__init__()
        self.attention = HybridLlamaAttention(config, head_dim, op_name, kernel_runner)
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
    def __init__(self, config, head_dim: int, op_name: str, kernel_runner: Optional[AttentionKernelRunner]):
        super().__init__()
        self.config = config
        self.op_name = op_name
        self.head_dim = head_dim
        self.attention_impl = "torch"

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [HybridTransformerBlock(config, head_dim, op_name, kernel_runner) for _ in range(config.n_layers)]
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
) -> HybridLlamaLikeModel:
    model = HybridLlamaLikeModel(model_cfg, head_dim, op_name, kernel_runner).to(device=device, dtype=dtype)
    model.eval()
    return model


class EmbeddingTRTModule(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.tok_embeddings = copy.deepcopy(embedding)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tok_embeddings(tokens)


class AttnPreTRTModule(nn.Module):
    def __init__(self, block: HybridTransformerBlock, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        super().__init__()
        self.attention_norm = copy.deepcopy(block.attention_norm)
        self.wq = copy.deepcopy(block.attention.wq)
        self.wk = copy.deepcopy(block.attention.wk)
        self.wv = copy.deepcopy(block.attention.wv)
        self.n_heads = block.attention.n_heads
        self.head_dim = block.attention.head_dim
        self.register_buffer("rope_cos", rope_cos.detach().clone(), persistent=False)
        self.register_buffer("rope_sin", rope_sin.detach().clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        attn_in = self.attention_norm(x)
        xq = self.wq(attn_in).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(attn_in).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = self.wv(attn_in).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb_real(xq, xk, rope_cos=self.rope_cos, rope_sin=self.rope_sin)
        return xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)


class AttnPostTRTModule(nn.Module):
    def __init__(self, block: HybridTransformerBlock):
        super().__init__()
        self.wo = copy.deepcopy(block.attention.wo)

    def forward(self, attn_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = attn_out.shape
        output = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return residual + self.wo(output)


class FFNBlockTRTModule(nn.Module):
    def __init__(self, block: HybridTransformerBlock):
        super().__init__()
        self.ffn_norm = copy.deepcopy(block.ffn_norm)
        self.feed_forward = copy.deepcopy(block.feed_forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.feed_forward(self.ffn_norm(x))


class FinalTRTModule(nn.Module):
    def __init__(self, model: HybridLlamaLikeModel):
        super().__init__()
        self.norm = copy.deepcopy(model.norm)
        self.output = copy.deepcopy(model.output) if model.output is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.output is not None:
            h = self.output(h)
        return h


class LayerWithAttentionPluginTRTModule(nn.Module):
    def __init__(
        self,
        block: HybridTransformerBlock,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        plugin_spec: AttentionPluginSpec,
        kernel_runner: AttentionKernelRunner,
    ):
        super().__init__()
        self.attention_norm = copy.deepcopy(block.attention_norm)
        self.wq = copy.deepcopy(block.attention.wq)
        self.wk = copy.deepcopy(block.attention.wk)
        self.wv = copy.deepcopy(block.attention.wv)
        self.wo = copy.deepcopy(block.attention.wo)
        self.ffn_norm = copy.deepcopy(block.ffn_norm)
        self.feed_forward = copy.deepcopy(block.feed_forward)
        self.n_heads = block.attention.n_heads
        self.head_dim = block.attention.head_dim
        self.plugin_spec = plugin_spec
        self.kernel_runner = kernel_runner
        self.register_buffer("rope_cos", rope_cos.detach().clone(), persistent=False)
        self.register_buffer("rope_sin", rope_sin.detach().clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        attn_in = self.attention_norm(x)

        xq = self.wq(attn_in).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(attn_in).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = self.wv(attn_in).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb_real(xq, xk, rope_cos=self.rope_cos, rope_sin=self.rope_sin)

        query_t = xq.transpose(1, 2).transpose(-1, -2).contiguous()
        key_t = xk.transpose(1, 2).transpose(-1, -2).contiguous()
        value = xv.transpose(1, 2).contiguous()
        attn_out = apply_attention_plugin(query_t, key_t, value, self.plugin_spec, self.kernel_runner)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        h = residual + self.wo(attn_out)
        return h + self.feed_forward(self.ffn_norm(h))


class TensorRTPluginRuntime:
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
        self.stream = torch.cuda.Stream(device=self.device)
        self.plugin_handle = load_deepgen_trt_plugin(args.plugin_lib)

        self.run = None

        self.engine_count = 0
        self.external_data_engine_count = 0
        self.simplified_engine_count = 0
        self.engine_paths: List[str] = []
        self.stage_stats = {name: self._empty_stage_stats() for name in TRT_STAGE_ORDER}
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
        self.external_data_engine_count += int(bool(artifact["external_data_export"]))
        self.simplified_engine_count += int(bool(artifact["simplify"]))
        if artifact["engine_path"]:
            self.engine_paths.append(artifact["engine_path"])

    def _build_trt_runtime(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        input_names: List[str],
        output_names: List[str],
        artifact_name: Optional[str],
        stage_name: str,
    ):
        module = module.to(device=self.device, dtype=self.dtype).eval()
        engine_path = resolve_engine_path(
            self.args.save_engine,
            self.op_name,
            artifact_name,
            multi_op=self.multi_op,
        )
        onnx_temp_dir = None
        try:
            export_start = time.perf_counter()
            onnx_artifact, onnx_temp_dir, used_external_data, simplify_applied = export_onnx_artifact(
                module=module,
                inputs=inputs,
                input_names=input_names,
                output_names=output_names,
                args=self.args,
                artifact_name=artifact_name,
            )
            export_ms = float((time.perf_counter() - export_start) * 1000.0)

            build_start = time.perf_counter()
            engine_bytes = trt_build_engine_from_onnx(
                onnx_model=onnx_artifact,
                workspace_gib=self.args.workspace_gib,
                engine_fn=engine_path,
                verbose=self.args.verbose_trt,
                disable_tensor_core=self.args.disable_tensor_core,
            )
            engine_build_ms = float((time.perf_counter() - build_start) * 1000.0)

            runtime_start = time.perf_counter()
            with torch.cuda.stream(self.stream):
                run = trt_build_independent_runtime(engine_bytes, verbose=self.args.verbose_trt)
            runtime_build_ms = float((time.perf_counter() - runtime_start) * 1000.0)
        finally:
            if onnx_temp_dir is not None:
                onnx_temp_dir.cleanup()

        artifact = {
            "run": run,
            "export_ms": export_ms,
            "engine_build_ms": engine_build_ms,
            "runtime_build_ms": runtime_build_ms,
            "simplify": simplify_applied,
            "external_data_export": used_external_data,
            "engine_path": engine_path,
        }
        self._record_artifact(stage_name, artifact)
        return artifact["run"]

    def _build(self, sample_tokens: torch.Tensor) -> None:
        self.model.set_attention_impl("plugin")
        self.run = self._build_trt_runtime(
            module=self.model,
            inputs=(sample_tokens,),
            input_names=["tokens"],
            output_names=["out"],
            artifact_name=None,
            stage_name="model",
        )

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.cuda.stream(self.stream):
            return self.run(tokens)


def print_stage_stats(stage_stats: Dict[str, Dict[str, Any]]) -> None:
    for stage_name in TRT_STAGE_ORDER:
        item = stage_stats.get(stage_name)
        if item is None or item["count"] <= 0:
            continue
        print(
            f"[trt plugin {stage_name}] count={item['count']} "
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
        print(f"[trt plugin] failed: {result['hybrid_error']}")
        return

    print(f"[kernel compile] {result['kernel_compile_ms']:.3f} ms")
    print("[plugin kernel source] runtime_compile_via_libdeepgen")
    print(f"[trt plugin export] {result['trt_hybrid_export_ms']:.3f} ms")
    print(f"[trt plugin build] {result['trt_hybrid_engine_build_ms']:.3f} ms")
    print(f"[trt plugin runtime] {result['trt_hybrid_runtime_build_ms']:.3f} ms")
    meta = result["trt_hybrid_meta"]
    print(
        f"[trt plugin engines] count={meta['engine_count']} "
        f"simplified={meta['simplified_engine_count']} "
        f"external_data={meta['external_data_engine_count']}"
    )
    print_stage_stats(meta["stage_stats"])
    print(f"[trt plugin first_call] {result['trt_hybrid_first_call_ms']:.3f} ms")
    print(
        f"[trt plugin] median={result['trt_hybrid_e2e']['median_ms']:.3f} ms "
        f"mean={result['trt_hybrid_e2e']['mean_ms']:.3f} ms"
    )
    if result["trt_hybrid_speedup_vs_torch"] is not None:
        print(f"[trt plugin speedup vs torch] {result['trt_hybrid_speedup_vs_torch']:.4f}x")


def run_single_op(op_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    load_deepgen_trt_plugin(args.plugin_lib)
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

    hybrid_model = build_torch_model(op_name, model_cfg, head_dim, dtype, device, runner)

    torch_model = LlamaLikeModel(model_cfg, head_dim, op_name, runner).to(device=device, dtype=dtype)
    torch_model.load_state_dict(hybrid_model.state_dict(), strict=False)
    torch_model.eval()
    torch_model.set_attention_impl("torch")
    _, torch_stats = benchmark_model(torch_model, tokens, "torch", args.warmup, args.iters)

    ffn_hidden = hybrid_model.layers[0].feed_forward.hidden_dim if hybrid_model.layers else 0
    parameter_count = int(sum(p.numel() for p in hybrid_model.parameters()))
    result = {
        "op": op_name,
        "config_source": kernel_source,
        "kernel_entry": kernel_entry,
        "kernel_names": list(runner.kernel_names),
        "kernel_compile_ms": float(runner.compile_ms),
        "model": make_result_meta(op_name, args, model_cfg, head_dim, runner.device_id, ffn_hidden, parameter_count),
        "torch_e2e": torch_stats,
        "trt_hybrid_export_ms": None,
        "trt_hybrid_engine_build_ms": None,
        "trt_hybrid_runtime_build_ms": None,
        "trt_hybrid_first_call_ms": None,
        "trt_hybrid_e2e": None,
        "trt_hybrid_meta": None,
        "trt_hybrid_speedup_vs_torch": None,
        "hybrid_error": None,
    }

    try:
        plugin_runtime = TensorRTPluginRuntime(
            model=hybrid_model,
            op_name=op_name,
            kernel_runner=runner,
            args=args,
            sample_tokens=tokens,
        )
        result["trt_hybrid_export_ms"] = float(plugin_runtime.total_export_ms)
        result["trt_hybrid_engine_build_ms"] = float(plugin_runtime.total_engine_build_ms)
        result["trt_hybrid_runtime_build_ms"] = float(plugin_runtime.total_runtime_build_ms)
        result["trt_hybrid_meta"] = {
            "engine_count": plugin_runtime.engine_count,
            "external_data_engine_count": plugin_runtime.external_data_engine_count,
            "simplified_engine_count": plugin_runtime.simplified_engine_count,
            "stage_stats": plugin_runtime.stage_stats,
            "save_engine": args.save_engine,
            "plugin_lib": args.plugin_lib,
            "disable_tensor_core": args.disable_tensor_core,
            "engine_paths": plugin_runtime.engine_paths,
        }

        _, first_call_ms = time_once(
            plugin_runtime,
            (tokens,),
            device,
            args.timer,
            sync_fn=plugin_runtime.stream.synchronize,
            event_stream=plugin_runtime.stream,
        )
        _, hybrid_stats = benchmark_fn(
            plugin_runtime,
            (tokens,),
            device,
            args.warmup,
            args.iters,
            args.timer,
            sync_fn=plugin_runtime.stream.synchronize,
            event_stream=plugin_runtime.stream,
        )

        result["trt_hybrid_first_call_ms"] = float(first_call_ms)
        result["trt_hybrid_e2e"] = hybrid_stats

        if hybrid_stats["median_ms"] > 0 and torch_stats["median_ms"] > 0:
            result["trt_hybrid_speedup_vs_torch"] = float(torch_stats["median_ms"] / hybrid_stats["median_ms"])
    except Exception as exc:
        result["hybrid_error"] = format_exception(exc)

    print_result(result)
    return result


def print_summary(results: Dict[str, Any]) -> None:
    if len(results) <= 1:
        return
    print("")
    print("[summary]")
    for op_name, item in results.items():
        if item.get("hybrid_error"):
            print(f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms trt_plugin=failed")
            continue
        speedup_text = (
            f"{item['trt_hybrid_speedup_vs_torch']:.4f}x"
            if item.get("trt_hybrid_speedup_vs_torch") is not None
            else "n/a"
        )
        print(
            f"{op_name}: torch={item['torch_e2e']['median_ms']:.3f} ms "
            f"trt_plugin={item['trt_hybrid_e2e']['median_ms']:.3f} ms "
            f"trt_plugin_speedup_vs_torch={speedup_text}"
        )


def main() -> None:
    args = parse_args()
    trt_args = make_trt_args(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        raise RuntimeError("TensorRT hybrid benchmark requires CUDA and at least one visible GPU.")

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
