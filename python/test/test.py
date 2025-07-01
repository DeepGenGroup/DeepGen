import torch
import torch.nn.functional as F
import math
import statistics
from deepgen.autotune import buildSapce
from deepgen.runtime import Runtime
from deepgen.utils import getGPUInfo, perf
from tqdm import tqdm

def autoTuneAttn(inputs):
  print("Input Shapes:", [inp.shape for inp in inputs])
  # get gpu info
  _, gpu_info = getGPUInfo(target="rocm")  # smem, cu/sm, reg and others
  # return kernel add into torch model
  cfgs = buildSapce("attention", inputs, gpu_info)
  # auto tuning...
  rt = Runtime(target="rocm", arch=gpu_info.arch)
  best_cfg, best_t = {}, 9999
  inputs_ = [inp.data_ptr() for inp in inputs]
  # for cfg in tqdm(cfgs, desc=f" Start Auto Tuning "):
  for i in tqdm(range(0, len(cfgs)), desc=f" Start Auto Tuning "):
    func = rt.compile("attention", cfgs[i])
    t = perf(kernelFunc=func, inputs=inputs_)
    if t < 10:
      print(f"runtime launch failed cfg: {cfgs[i]}")
    elif t < best_t:
      best_t = t
      best_cfg = cfgs[i]
    if i % 100 == 0:
      print(f"iter: {i+1}  best time cost: {best_t} ms")
  return best_cfg, best_t

def testTorch(inputs):
  def attnFunc(Q, K, V, O):
    d = Q.shape[1] * Q.shape[3]
    P = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d)
    S = F.softmax(P, dim=-1)
    O = torch.matmul(S, V)
    return O
  t = perf(attnFunc, inputs)
  return t
  

if __name__ == "__main__":
  bs = 1
  hn = 32
  sl = 2048
  hd = 128
  Q = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
  K = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
  V = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
  O = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device='cuda')

  inputs = [Q.transpose(2, 3).contiguous(), K.transpose(2, 3).contiguous(), V, O]
  inputs_ = [Q, K, V, O]
  t = testTorch(inputs=inputs_)
  print(f"torch time cost: {t} ms")

  cfg, t = autoTuneAttn(inputs=inputs)
  print(f"best config: {cfg}")
  print(f"best time cost: {t} ms")
