import time
import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Callable

# ----------------------------------------
# 模型配置
# ----------------------------------------
class ModelArgs:
    dim = 4096
    n_layers = 8
    n_heads = 8
    vocab_size = 8192
    max_seq_len = 2048
    device = 'npu'

# ----------------------------------------
# 定义 RMSNorm
# ----------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

# ----------------------------------------
# 定义多头注意力
# ----------------------------------------
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * head_dim, bias=False).npu()
        self.wk = nn.Linear(args.dim, args.n_heads * head_dim, bias=False).npu()
        self.wv = nn.Linear(args.dim, args.n_heads * head_dim, bias=False).npu()
        self.wo = nn.Linear(args.n_heads * head_dim, args.dim, bias=False).npu()

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.args.dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.wo(context)

# ----------------------------------------
# 定义前馈网络
# ----------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

# ----------------------------------------
# 定义一个 Transformer 块
# ----------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim)
        self.attn = Attention(args)
        self.ffn_norm = RMSNorm(args.dim)
        self.ffn = FeedForward(args.dim)

    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        return h + self.ffn(self.ffn_norm(h))

# ----------------------------------------
# 定义整体 LlamaOld 模型
# ----------------------------------------
class LlamaOld(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
            h = self.norm(h)
        return self.head(h)

def evaluate_npu_time(f: Callable, warmup: int = 1, runs: int = 5) -> tuple[torch.Tensor, float]:
    # NPU 专用
    # 1) 预热几次
    for _ in range(warmup):
        _ = f()
        torch.npu.synchronize()

    times = []
    out = None
    for _ in range(runs):
        ev_st = torch_npu.npu.Event(enable_timing=True)
        ev_et = torch_npu.npu.Event(enable_timing=True)
        ev_st.record()
        out = f()
        ev_et.record()
        torch.npu.synchronize()          # 确保所有核函数都完成
        times.append(ev_st.elapsed_time(ev_et))
    return out, float(np.median(times))

def evaluate_cpu_time(f: Callable, runs: int = 5) -> tuple[torch.Tensor, float]:
    # 纯 CPU 专用
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = f()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # 转为 ms
    return out, float(np.median(times))

if __name__ == "__main__":
    args = ModelArgs()

    # --- NPU ---
    model_npu = LlamaOld(args).to(args.device)
    input_npu = torch.randint(0, args.vocab_size, (1, args.max_seq_len)).to(args.device)
    f_base = lambda: model_npu(input_npu)

    # --- CPU ---
    model_cpu = LlamaOld(args).to('cpu')
    input_cpu = input_npu.to('cpu')
    f_benchmark = lambda: model_cpu(input_cpu)

    out_npu, t_npu = evaluate_npu_time(f_base)
    out_cpu, t_cpu = evaluate_cpu_time(f_benchmark)

    print(f"NPU 运行时间（中位数，剔除预热）: {t_npu:.2f} ms")
    print(f"CPU 运行时间（中位数）:           {t_cpu:.2f} ms")
