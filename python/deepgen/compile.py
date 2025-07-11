import torch
import torch.nn.functional as F
import math
from utils import getGPUInfo, perf
from tqdm import tqdm
from runtime import Runtime
from autotune import buildSapce
import statistics
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


def compile(kernel, inputs, target, system="our"):
  if system == "our":
    print(f"=== Kernel Name: {kernel} ===")
    print("Input Shapes:", [inp.shape for inp in inputs])
    # get gpu info
    _, gpu_info = getGPUInfo(target)  # smem, cu/sm, reg and others
    # return kernel add into torch model
    cfgs = buildSapce(kernel, inputs, gpu_info)
    # auto tuning...
    rt = Runtime(target=target, arch=gpu_info.arch)
    best_cfg, best_t = {}, 9999
    inputs_ = [inp.data_ptr() for inp in inputs]
    for cfg in tqdm(cfgs, desc=f" Start Auto Tuning "):
    # for cfg in cfgs:
      try:
        func = rt.compile(kernel, cfg)
        t = perf(kernelFunc=func, inputs=inputs_)
      except Exception as e:
        print(f"This config have error: {cfg}")
        print(e)
        continue
      if t < 10:
        print(f"runtime launch failed cfg: {cfg}")
      elif t < best_t:
        best_t = t
        best_cfg = cfg
    return best_cfg, best_t


def torch_attn(Q, K, V, O):
  d = Q.shape[1] * Q.shape[3]
  P = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d)
  S = F.softmax(P, dim=-1)
  O = torch.matmul(S, V)
  return O
  

def test_attn(Q, K, V, O):
  inputs = [Q, K, V, O]
  our_inputs = [Q.transpose(2, 3).contiguous(), K.transpose(2, 3).contiguous(), V, O]
  # perf torch
  t = perf(torch_attn, inputs=inputs)
  print(f"torch time cost: {t} ms")
  # auto tune
  cfg, tt = compile("attention", inputs=our_inputs, target="cuda")
  print(f"our best config: {cfg}")
  print(f"our best time cost: {tt} ms")


if __name__ == "__main__":
  # A = torch.randn(128, 1024, 128, dtype=torch.float32, device='cuda')
  # B = torch.randn(128, 128, 1024, dtype=torch.float32, device='cuda')
  # C = torch.empty((128, 1024, 1024), dtype=torch.float32, device='cuda')
  # compile("matmul", [A.transpose(-1, -2).contiguous(), B, C], target="rocm")
  # print(C)
  # print(torch.matmul(A, B))

  Q = torch.randn(1, 32, 4096, 128, dtype=torch.float32, device='cuda')
  K = torch.randn(1, 32, 4096, 128, dtype=torch.float32, device='cuda')
  V = torch.randn(1, 32, 4096, 128, dtype=torch.float32, device='cuda')
  O = torch.empty((1, 32, 4096, 128), dtype=torch.float32, device='cuda')
  test_attn(Q, K, V, O)
  
