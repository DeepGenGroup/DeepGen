import torch
import torch.nn.functional as F
import math
import statistics
from deepgen.autotune import buildSapce
from deepgen.runtime import Runtime
from deepgen.utils import getGPUInfo, perf
from tqdm import tqdm


def test(inputs, target="cuda"):
  print("Input Shapes:", [inp.shape for inp in inputs])
  _, gpu_info = getGPUInfo(target=target)
  # print(gpu_info.shared_mem_per_block, gpu_info.compute_units)
  cfgs = buildSapce("attention", inputs, gpu_info)
  # print(cfgs)
  rt = Runtime(target=target, arch=gpu_info.arch)
  inputs_ = [inp.data_ptr() for inp in inputs]
  
  print(cfgs[1])
  func = rt.compile("attention", cfgs[1])
  func(*inputs_)


def attnFunc(Q, K, V):
  P = torch.matmul(Q, K.transpose(2, 3))
  S = F.softmax(P, dim=-1)
  O = torch.matmul(S, V)
  return O
  

if __name__ == "__main__":
  bs = 1
  hn = 32
  sl = 2048
  hd = 128
  Q = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
  K = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
  V = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
  O = torch.empty(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
  inputs = [Q.transpose(2, 3).contiguous(), K.transpose(2, 3).contiguous(), V, O]
  test(inputs=inputs)
  print(O)
  O_ = attnFunc(Q, K, V)
  print(O_)
  
  if torch.allclose(O,O_,1e-3,1e-3) :
    print('test corect!')
  else:
    print('test error!')
