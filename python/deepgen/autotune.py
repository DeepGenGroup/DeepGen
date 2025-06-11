import torch
from typing import Any, Dict, List


# class createMmConfigSpace:
#   @staticmethod
#   def 

#   @staticmethod
#   def buildSpace(inputs):
#     pass


# class createAttnConfigSpace:
#   @staticmethod
#   def buildSpace(inputs):
#     pass
  

def buildSapce(kernel: str, inputs, gpu_info):
  # if kernel == "matmul":
  #   return createMmConfigSpace.buildSpace(inputs)
  # cfg["grid"], cfg["block"], cfg["smem"]
  tmp_dict = {
    "shape": [1024, 1024, 128], 
    "type": ["fp32", "fp32", "fp32"], 
    "grid": [64, 1, 1], 
    "block": [256, 1, 1],
    "smem": 16384,
    "config": {
      "matmul": {
        "BLOCK_SIZE_M": 128, "THREAD_SIZE_M": 8, "BLOCK_SIZE_N": 128, "THREAD_SIZE_N": 8,
        "LOCAL_SPLIT_U": 1, "BLOCK_SIZE_K": 16, 
        "GLOB_LOAD_WIDTH_A": 4, "GLOB_LOAD_WIDTH_B": 4, "GLOB_STORE_WIDTH": 4,
        "BLOCK_LAYOUT_Y": 4, "BLOCK_LAYOUT_X": 2, "WARP_LAYOUT_Y": 4, "WARP_LAYOUT_X": 8,
        "BLOCK_SCATTER_WIDTH_M": 8, "WARP_SCATTER_WIDTH_M": 8, "BLOCK_SCATTER_WIDTH_N": 4, "WARP_SCATTER_WIDTH_N": 4,
        "WARP_SIZE": 32, "LOAD_CONTINUOUS": 1, "STORE_CONTINUOUS": 1, "SHARED_PREFETCH": 0, "REG_PREFETCH": 0, 
        "BLOCK_MAPPING": 4, "UNROLL_NUM": 16
      }
    }
  }
  return [tmp_dict]
    
  tuneConfig = {
    "shape": [1, 32, 4096, 128], 
    "type": ["fp32", "fp32", "fp32"], 
    "grid": [64, 1, 1], 
    "block": [256, 1, 1],
    "smem": 34304,
    "config": {
      "attention": {
        "Br": 64, "Bc": 64, "Hd": 128, "Slice1": 16, "Slice2": 16, "PTr": 2, "PTc": 8, "OTr": 4, "OTc": 8, 

        "GLOB_LOAD_WIDTH_Q": 4, "GLOB_LOAD_WIDTH_K": 4, "GLOB_LOAD_WIDTH_V": 4, 
        "LOAD_CONTINUOUS_P": 1, "LOAD_CONTINUOUS_O": 1, 

        "SHARED_PREFETCH_P": 0, "REG_PREFETCH_P": 0, "REG_PREFETCH_O": 0,

        "BLOCK_LAYOUT_P_Y": 8, "BLOCK_LAYOUT_P_X": 1, "WARP_LAYOUT_P_Y": 4, "WARP_LAYOUT_P_X": 8,
        "BLOCK_SCATTER_WIDTH_Q": 2, "BLOCK_SCATTER_WIDTH_K": 2, "WARP_SCATTER_WIDTH_Q": 1, "WARP_SCATTER_WIDTH_K": 1,

        "BLOCK_LAYOUT_O_Y": 2, "BLOCK_LAYOUT_O_X": 4, "WARP_LAYOUT_O_Y": 8, "WARP_LAYOUT_O_X": 4,
        "BLOCK_SCATTER_WIDTH_P": 2, "BLOCK_SCATTER_WIDTH_V": 2, "WARP_SCATTER_WIDTH_P": 1, "WARP_SCATTER_WIDTH_V": 1,
        "WARP_SIZE": 32, "UNROLL_NUM": 16
      }
    }
  }
  

