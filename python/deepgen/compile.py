import torch
import os
from utils import getGPUInfo
from tqdm import tqdm
from runtime import Runtime
from autotune import buildSapce


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
  A = torch.randn(1024, 128, dtype=torch.float32, device='cuda')
  B = torch.randn(128, 1024, dtype=torch.float32, device='cuda')
  C = torch.empty((1024, 1024), dtype=torch.float32, device='cuda')
  compile("matmul", [A.t().contiguous(), B, C], target="rocm")
  print(C)
  print(torch.mm(A, B))