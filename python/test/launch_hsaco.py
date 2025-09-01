import torch
import torch.nn.functional as F
import math
import statistics
from deepgen.autotune import buildSapce
from deepgen.runtime import Runtime, makeHostSrc
from deepgen.utils import compileModuleFromSrc
from deepgen.utils import getGPUInfo, perf
from tqdm import tqdm

bs = 1
hn = 32
sl = 2048
hd = 128
Q = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
K = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
V = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
O = torch.empty(bs, hn, sl, hd, dtype=torch.float32, device='cuda')
inputs = [Q.transpose(2, 3).contiguous(), K.transpose(2, 3).contiguous(), V, O]

kernel_dir = "/home/xiebaokang/projects/DeepGen/_TempCodes/rocmAttnTest/attn.hsaco"
src = makeHostSrc("attention", 4, [64, 32, 1], [128, 1, 1], 18816, kernel_dir)
print(src)
mod = compileModuleFromSrc("launch", src)
mod.launch(*inputs)
print(O)

def attnFunc(Q, K, V):
  P = torch.matmul(Q, K.transpose(2, 3))
  S = F.softmax(P, dim=-1)
  O = torch.matmul(S, V)
  return O
O_ = attnFunc(Q, K, V)
print(O_)


if torch.allclose(O,O_,1e-3,1e-3) :
  print('test corect!')
else:
  print('test error!')