import math
import time

import numpy as np
import click
import torch
import torch.nn as nn
import torch.nn.functional as F

import flashinfer

class Attn(nn.Module):
  def __init__(self, kv_head_num, head_num, head_dim):
    super().__init__()
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.group_size = self.head_num // self.kv_head_num
  
  def forward(self, q, k, v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    if self.group_size > 1:
      k = k[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)
      v = v[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    probs = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()
    return out.view(batch_size, q_len, self.head_num, self.head_dim)
  
  def forward_flashinfer(self, q, k, v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    assert batch_size == 1
 
    q = q.view(q_len, self.head_num, self.head_dim)
    k = k.view(kv_len, self.kv_head_num, self.head_dim)
    v = v.view(kv_len, self.kv_head_num, self.head_dim)
    out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    return out.view(batch_size, q_len, self.head_num, self.head_dim)



class H2O(nn.Module):
  def __init__(self, kv_head_num, head_num, head_dim):
    super().__init__()
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.group_size = self.head_num // self.kv_head_num
  
  def forward(self, q, k, v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    if self.group_size > 1:
      k = k[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)
      v = v[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    probs = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()

    # h2o
    if self.group_size > 1:
      h2o_score = probs.reshape(batch_size, self.kv_head_num, self.group_size, q_len, kv_len).mean(dim=2)
    else:
      h2o_score = probs 
    h2o_score = h2o_score.sum(dim=2)

    return out.view(batch_size, q_len, self.head_num, self.head_dim), h2o_score.view(batch_size, self.kv_head_num, kv_len)

class RoCo(nn.Module):
  def __init__(self, kv_head_num, head_num, head_dim):
    super().__init__()
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.group_size = self.head_num // self.kv_head_num
  
  def forward(self, q, k, v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    if self.group_size > 1:
      k = k[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)
      v = v[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    probs = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()

    # roco
    if self.group_size > 1:
      probs = probs.reshape(batch_size, self.kv_head_num, self.group_size, q_len, kv_len).mean(dim=2)

    roco_score = probs 
    roco_sq_score = probs ** 2

    roco_score = roco_score.sum(dim=2)
    roco_sq_score = roco_sq_score.sum(dim=2)

    return out.view(batch_size, q_len, self.head_num, self.head_dim), roco_score.view(batch_size, self.kv_head_num, kv_len), roco_sq_score.view(batch_size, self.kv_head_num, kv_len)


class KeyFormer(nn.Module):
  def __init__(self, tau, kv_head_num, head_num, head_dim):
    super().__init__()
    self.tau = tau
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.group_size = self.head_num // self.kv_head_num
  
  def forward(self, q, k, v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    if self.group_size > 1:
      k = k[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)
      v = v[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    probs = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()

    # keyformer
    kf_score = F.gumbel_softmax(scores.float(), tau=self.tau, hard=False, dim=-1)
    if self.group_size > 1:
      kf_score = kf_score.reshape(batch_size, self.kv_head_num, self.group_size, q_len, kv_len).mean(dim=2)
    kf_score = kf_score.sum(dim=2)

    return out.view(batch_size, q_len, self.head_num, self.head_dim), kf_score


class Snapkv(nn.Module):
  def __init__(self, kernel_size, kv_head_num, head_num, head_dim):
    super().__init__()
    self.kernel_size = kernel_size
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.group_size = self.head_num // self.kv_head_num
  
  def forward(self, q, k, v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    if self.group_size > 1:
      k = k[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)
      v = v[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    probs = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()

    # snapkv
    if self.group_size > 1:
      snapkv_score = probs.reshape(batch_size, self.kv_head_num, self.group_size, q_len, kv_len).mean(dim=2)
    else:
      snapkv_score = probs 
    snapkv_score = snapkv_score.sum(dim=2)
    snapkv_score = F.avg_pool1d(snapkv_score, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

    return out.view(batch_size, q_len, self.head_num, self.head_dim), snapkv_score.view(batch_size, self.kv_head_num, kv_len)


class Corm(nn.Module):
  def __init__(self, kv_head_num, head_num, head_dim):
    super().__init__()
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.group_size = self.head_num // self.kv_head_num

  def forward(self, q, k, v, corm_mask):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    if self.group_size > 1:
      k = k[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)
      v = v[:, :, :, None, :].expand(batch_size, kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(batch_size, kv_len, self.head_num, self.head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    probs = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()

    # corm
    if self.group_size > 1:
      corm_score = probs.reshape(batch_size, self.kv_head_num, self.group_size, q_len, kv_len).mean(dim=2)
    else:
      corm_score = probs 
    corm_score = probs >= corm_mask
    corm_score = corm_score.any(dim=2)

    return out.view(batch_size, q_len, self.head_num, self.head_dim), corm_score.view(batch_size, self.kv_head_num, kv_len)




def membound(f, args):
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.empty_cache()
  start_mem_gib = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
  start_peak_mem_gib = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
  f(*args)
  free_gpu_mem, total_gpu_mem = torch.cuda.mem_get_info()
  end_mem_gib = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
  end_peak_mem_gib = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

  return end_peak_mem_gib

def perf(f, args, gflops):
  warmup = 10
  repeat = 10
  torch.cuda.synchronize()
  for _ in range(warmup):
    f(*args)

  ms = []
  torch.cuda.synchronize()
  for _ in range(repeat):
    torch.cuda.synchronize()
    tik = time.time()
    f(*args)
    torch.cuda.synchronize()
    tok = time.time()
    ms.append((tok - tik) * 1000.0)
  avg_ms = np.mean(ms)

  return gflops / (avg_ms / 1000.0)
 
  


@click.command()
@click.option('--model_name', '-m', type=str, default="attn")
def main(model_name):
  for seqlen in [128, 256, 512, 1024, 2048, 4096, 8192]:
    batch_size = 1
    head_num = 32
    kv_head_num = 32
    head_dim = 128
    dtype = torch.float16
    device = torch.cuda.current_device()

    q = torch.randn(batch_size, seqlen, head_num, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, kv_head_num, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, kv_head_num, head_dim, dtype=dtype, device=device)

    tau = 1.5
    kernel_size = 5
    model_map = {
      'attn': Attn(head_num, kv_head_num, head_dim).eval().cuda().forward,
      'attn_fa': Attn(head_num, kv_head_num, head_dim).eval().cuda().forward_flashinfer,
      'h2o': H2O(head_num, kv_head_num, head_dim).eval().cuda().forward,
      'roco': RoCo(head_num, kv_head_num, head_dim).eval().cuda().forward,
      'keyformer': KeyFormer(tau, head_num, kv_head_num, head_dim).eval().cuda().forward,
      'snapkv': Snapkv(kernel_size, head_num, kv_head_num, head_dim).eval().cuda().forward,
      'corm': Corm(head_num, kv_head_num, head_dim).eval().cuda().forward,
    }

    model = model_map[model_name]

    args = [q, k, v]
    if 'corm' == model_name:
      corm_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device)
      for i in range(seqlen):
        corm_mask[i] /= i + 1
      args.append(corm_mask)
  
    assert head_num == kv_head_num
    # causal
    gflops = 4 * batch_size * head_num * seqlen * seqlen * head_dim * 1e-9 / 2
    peak_mem_gib = membound(model, args)
    gflops_s = perf(model, args, gflops)
    tflops_s = gflops_s / 1000
    print(f"model_name: {model_name}")
    print(f"seqlen: {seqlen}")
    print(f"peak_mem_gib: {peak_mem_gib}")
    print(f"tflops_s: {tflops_s}")
 
if __name__ == "__main__":
  main()

