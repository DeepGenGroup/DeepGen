import torch
import torch.nn.functional as F
import math
import statistics
from deepgen.autotune import buildSapce
from deepgen.runtime import Runtime
from deepgen.utils import getGPUInfo, perf
from tqdm import tqdm


# def test(inputs, target="rocm"):
#   print("Input Shapes:", [inp.shape for inp in inputs])
#   _, gpu_info = getGPUInfo(target=target)
#   # print(gpu_info.shared_mem_per_block, gpu_info.compute_units)
#   cfgs = buildSapce("attention", inputs, gpu_info)
#   # print(cfgs)
#   rt = Runtime(target=target, arch=gpu_info.arch)
#   inputs_ = [inp.data_ptr() for inp in inputs]
  
#   print(cfgs[1])
#   func = rt.compile("attention", cfgs[1])
#   func(*inputs_)

def test(inputs, target="rocm"):
  print("Input Shapes:", [inp.shape for inp in inputs])
  _, gpu_info = getGPUInfo(target=target)

  # print(cfgs)
  rt = Runtime(target=target, arch=gpu_info.arch)
  inputs_ = [inp.data_ptr() for inp in inputs]
  cfg = {
    'shape': [1, 1, 128, 128], 
    'type': ['fp32', 'fp32', 'fp32', 'fp32'], 
    'grid': [4, 1, 1], 
    'block': [128, 1, 1], 
    'smem': 18816, 
    'config': {
      'attention': {
        'Br': 32, 'Bc': 64, 'Hd': 128, 'Slice1': 16, 'Slice2': 8, 
        'PTr': 4, 'PTc': 4, 'OTr': 4, 'OTc': 8,
         
        'GLOB_LOAD_WIDTH_Q': 4, 'GLOB_LOAD_WIDTH_K': 4, 'GLOB_LOAD_WIDTH_V': 4, 
        
        'BLOCK_LAYOUT_P_Y': 2, 'BLOCK_LAYOUT_P_X': 1, 'WARP_LAYOUT_P_Y': 4, 'WARP_LAYOUT_P_X': 16, 
        'BLOCK_SCATTER_WIDTH_Q': 4, 'BLOCK_SCATTER_WIDTH_K': 4, 'WARP_SCATTER_WIDTH_Q': 4, 'WARP_SCATTER_WIDTH_K': 4, 
        
        'BLOCK_LAYOUT_O_Y': 1, 'BLOCK_LAYOUT_O_X': 2, 'WARP_LAYOUT_O_Y': 8, 'WARP_LAYOUT_O_X': 8, 
        'BLOCK_SCATTER_WIDTH_P': 4, 'BLOCK_SCATTER_WIDTH_V': 4, 'WARP_SCATTER_WIDTH_P': 4, 'WARP_SCATTER_WIDTH_V': 4, 
        
        'UNROLL_NUM': 16, 'WARP_SIZE': 64, 
        'LOAD_CONTINUOUS_P': 1, 'LOAD_CONTINUOUS_O': 1, 
        'SHARED_PREFETCH_P': 0, 'REG_PREFETCH_P': 0, 'REG_PREFETCH_O': 0
      }
    }
  }
  func = rt.compile("attention", cfg)
  func(*inputs_)


def attnFunc(Q, K, V):
  P = torch.matmul(Q, K.transpose(2, 3))
  S = F.softmax(P, dim=-1)
  O = torch.matmul(S, V)
  return O
  

if __name__ == "__main__":
  bs = 1
  hn = 1
  sl = 128
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
