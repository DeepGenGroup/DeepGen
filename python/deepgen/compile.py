import torch
import torch.nn.functional as F
import os
from utils import getGPUInfo
from tqdm import tqdm
from runtime import Runtime
from autotune import buildSapce
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
    for cfg in tqdm(cfgs, desc=f" Start Auto Tuning "):
      func = rt.compile(kernel, cfg)
      func(*[inp.data_ptr() for inp in inputs])

if __name__ == "__main__":
  # A = torch.randn(128, 1024, 128, dtype=torch.float32, device='cuda')
  # B = torch.randn(128, 128, 1024, dtype=torch.float32, device='cuda')
  # C = torch.empty((128, 1024, 1024), dtype=torch.float32, device='cuda')
  # compile("matmul", [A.transpose(-1, -2).contiguous(), B, C], target="cuda")
  # print(C)
  # print(torch.matmul(A, B))

  Q = torch.randn(1, 32, 4096, 128, dtype=torch.float32, device='cuda')
  K = torch.randn(1, 32, 4096, 128, dtype=torch.float32, device='cuda')
  V = torch.randn(1, 32, 4096, 128, dtype=torch.float32, device='cuda')
  O = torch.empty((1, 32, 4096, 128), dtype=torch.float32, device='cuda')
  compile("attention", [Q.transpose(2, 3).contiguous(), K.transpose(2, 3).contiguous(), V, O], target="cuda")
  print(O)
  P = F.softmax(torch.matmul(Q, K.transpose(2, 3)), dim=-1)
  print(torch.matmul(P, V))