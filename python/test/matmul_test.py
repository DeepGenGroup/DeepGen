import torch
import torch.nn.functional as F
import math
import statistics
from deepgen.autotune import buildSapce
from deepgen.runtime import Runtime
from deepgen.utils import getGPUInfo, perf

def matmul_test(inputs):
  print("Input Shapes:", [inp.shape for inp in inputs])
  # get gpu info
  _, gpu_info = getGPUInfo(target="cuda")  # smem, cu/sm, reg and others
  # return kernel add into torch model
  cfgs = buildSapce("matmul", inputs, gpu_info)
  # auto tuning...
  rt = Runtime(target="cuda", arch=gpu_info.arch)
  inputs_ = [inp.data_ptr() for inp in inputs]
  # for cfg in tqdm(cfgs, desc=f" Start Auto Tuning "):
  func = rt.compile("matmul", cfgs[0])
  # t = perf(kernelFunc=func, inputs=inputs_)
  func(*inputs_)
  

if __name__ == "__main__":
  M = 1024
  N = 1024
  K = 1024
  A = torch.randn(M, K, dtype=torch.float32, device='cuda')
  B = torch.randn(K, N, dtype=torch.float32, device='cuda')
  C = torch.empty(M, N, dtype=torch.float32, device='cuda')
  
  # torch
  C_ = torch.matmul(A, B)

  # our
  inputs = [A.transpose(0, 1).contiguous(), B, C]
  matmul_test(inputs)
  
  if torch.allclose(C, C_, 1e-3, 1e-3) :
    print('test corect!')
  else:
    print('test error!')