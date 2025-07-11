import os, math
from enum import Enum
import subprocess
import tempfile

dirname = os.path.dirname(os.path.realpath(__file__))

class Kernel(Enum):
  Attention = 1

def makeAttnSrc(args):
  code = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <iostream>
#include <float.h>
template<int VecLen>
__device__ __forceinline__ void VecCpy(float* a, float* b);

template<>
__device__ __forceinline__ void VecCpy<4>(float* a, float* b) {{
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
}}

template<>
__device__ __forceinline__ void VecCpy<2>(float* a, float* b) {{
  (reinterpret_cast<float2*>(a)[0]) = (reinterpret_cast<float2*>(b)[0]);
}}

template<>
__device__ __forceinline__ void VecCpy<1>(float* a, float* b) {{
  (reinterpret_cast<float*>(a)[0]) = (reinterpret_cast<float*>(b)[0]);
}} 
extern "C" __global__
void attention(float* Q, float* K, float* V, float* O) {{
  const int warp_id = threadIdx.x / {args["WarpSize"]};
  const int lane_id = threadIdx.x % {args["WarpSize"]};
  const int warp_y = warp_id / {args["BlockLayoutXP"]};
  const int warp_x = warp_id % {args["BlockLayoutXP"]};
  const int lane_y = lane_id / {args["WarpLayoutXP"]};
  const int lane_x = lane_id % {args["WarpLayoutXP"]};
  const int warp_y_ = warp_id / {args["BlockLayoutXO"]};
  const int warp_x_ = warp_id % {args["BlockLayoutXO"]};
  const int lane_y_ = lane_id / {args["WarpLayoutXO"]};
  const int lane_x_ = lane_id % {args["WarpLayoutXO"]};
  __shared__ float LDSMemory[{args["LDS_SZIE"]}];
  float* smQ = (float *)(LDSMemory);
  float* smK = (float *)(smQ + ({args["Br"]} * {args["SliceP"]}));
  float* smP = (float *)(smK + ({args["Bc"]} * {args["SliceP"]}));
  float* smV = (float *)(smP + ({args["Br"]} * {args["Bc"]}));
  float* smMax = (float *)(smV + ({args["Hd"]} * {args["SliceO"]}));
  float* smSum = (float *)(smMax + ({args["Br"]}));
  float* smFactor = (float *)(smSum + ({args["Br"]}));
  #pragma unroll
  for (int i=0; i<{args["Br"]}; i+={args["THREAD_NUM"]}) {{
    if (i + threadIdx.x < {args["Br"]}) {{
      smSum[i + threadIdx.x] = 0.0f;
      smMax[i + threadIdx.x] = -FLT_MAX;
    }}
  }}
  float tileO[{args["BrTileO"]} * {args["HdTile"]}] = {{0.0f}};
  Q = Q + blockIdx.z * {args["head_num"]} * {args["seq_len"]} * {args["Hd"]} + blockIdx.y * {args["seq_len"]} * {args["Hd"]};
  K = K + blockIdx.z * {args["head_num"]} * {args["seq_len"]} * {args["Hd"]} + blockIdx.y * {args["seq_len"]} * {args["Hd"]};
  V = V + blockIdx.z * {args["head_num"]} * {args["seq_len"]} * {args["Hd"]} + blockIdx.y * {args["seq_len"]} * {args["Hd"]};
  O = O + blockIdx.z * {args["head_num"]} * {args["seq_len"]} * {args["Hd"]} + blockIdx.y * {args["seq_len"]} * {args["Hd"]};
  Q = Q + blockIdx.x * {args["Br"]};
  O = O + blockIdx.x * {args["Br"]} * {args["Hd"]};
  for (int bx=0; bx<{args["seq_len"]}; bx+={args["Bc"]}) {{
    float regQ[{args["BrTileP"]}], regK[{args["BcTileP"]}], tileP[{args["BrTileP"]}*{args["BcTileP"]}] = {{0.0f}};
    float regV[{args["HdTile"]}], regP[{args["BrTileO"]}];
    float rowSum[{args["BrTileP"]}] = {{0.0f}};
    float rowMax[{args["BrTileP"]}] = {{-FLT_MAX}};
    for (int k=0; k<{args["Hd"]}; k+={args["SliceP"]}) {{
      #pragma unroll
      for (int i=0; i<{args["GLOB_LOAD_NUM_Q"]}; i++) {{
        int x = i * {args["GLOB_LOAD_ROW_WIDTH_Q"]} + threadIdx.x * {args["GlobLoadWidthQ"]};
        VecCpy<{args["GlobLoadWidthQ"]}>(&smQ[x], &Q[(x/{args["Br"]} + k) * {args["seq_len"]} + x%{args["Br"]}]);
      }}
      #pragma unroll
      for (int i=0; i<{args["GLOB_LOAD_NUM_K"]}; i++) {{
        int x = i * {args["GLOB_LOAD_ROW_WIDTH_K"]} + threadIdx.x * {args["GlobLoadWidthK"]};
        VecCpy<{args["GlobLoadWidthK"]}>(&smK[x], &K[(x/{args["Bc"]} + k) * {args["seq_len"]} + bx + x%{args["Bc"]}]);
      }}
      __syncthreads();
      #pragma unroll
      for (int bk=0; bk<{args["SliceP"]}; bk++) {{
        #pragma unroll
        for (int i=0; i<{args["BLOCK_REPEAT_Y_P"]}; i++) {{
          #pragma unroll
          for (int j=0; j<{args["WARP_REPEAT_Y_P"]}; j++) {{
            int idx = (i * {args["BlockLayoutYP"]} + warp_y) * {args["WarpLayoutYP"]} * {args["BlockScatterWidthYP"]} + (j * {args["WarpLayoutYP"]} + lane_y) * {args["WarpScatterWidthYP"]};
            VecCpy<{args["WarpScatterWidthYP"]}>(&regQ[i * {args["BlockScatterWidthYP"]} + j * {args["WarpScatterWidthYP"]}], &smQ[bk * {args["Br"]} + idx]);
          }}
        }}
        #pragma unroll
        for (int i=0; i<{args["BLOCK_REPEAT_X_P"]}; i++) {{
          #pragma unroll
          for (int j=0; j<{args["WARP_REPEAT_X_P"]}; j++) {{
            int idx = (i * {args["BlockLayoutXP"]} + warp_x) * {args["WarpLayoutXP"]} * {args["BlockScatterWidthXP"]} + (j * {args["WarpLayoutXP"]} + lane_x) * {args["WarpScatterWidthXP"]};
            VecCpy<{args["WarpScatterWidthXP"]}>(&regK[i * {args["BlockScatterWidthXP"]} + j * {args["WarpScatterWidthXP"]}], &smK[bk * {args["Bc"]} + idx]);
          }}
        }}
        #pragma unroll
        for (int cy=0; cy<{args["BrTileP"]}; cy++) {{
          #pragma unroll
          for (int cx=0; cx<{args["BcTileP"]}; cx++) {{
            tileP[cy * {args["BcTileP"]} + cx] += regQ[cy] * regK[cx] * {args["Scale"]};
          }}
        }}
      }}
      __syncthreads();
    }}
    #pragma unroll
    for (int i=0; i<{args["BrTileP"]}; i++) {{
      #pragma unroll
      for (int j=0; j<{args["BcTileP"]}; j++) {{
        float oldMax = rowMax[i];
        rowMax[i] = fmaxf(oldMax, tileP[i * {args["BcTileP"]} + j]);
        rowSum[i] = rowSum[i] * __expf(oldMax - rowMax[i]) + __expf(tileP[i * {args["BcTileP"]} + j] - rowMax[i]);
      }}
    }}
    #pragma unroll
    for (int i=0; i<{args["BrTileP"]}; i++) {{
      #pragma unroll
      for (int pos=1; pos<{args["BLOCK_X"]}; pos*=2) {{
        float oldMax = __shfl_down(rowMax[i], pos, {args["BLOCK_X"]});
        float oldSum = __shfl_down(rowSum[i], pos, {args["BLOCK_X"]});
        float newMax = fmaxf(oldMax, rowMax[i]);
        rowSum[i] = oldSum * __expf(oldMax - newMax) + rowSum[i] * __expf(rowMax[i] - newMax);
        rowMax[i] = newMax;
      }}
    }}
    if (threadIdx.x % {args["BLOCK_X"]} == 0) {{
      #pragma unroll
      for (int i=0; i<{args["BLOCK_REPEAT_Y_P"]}; i++) {{
        #pragma unroll
        for (int j=0; j<{args["WARP_REPEAT_Y_P"]}; j++) {{
          #pragma unroll
          for (int k=0; k<{args["WarpScatterWidthYP"]}; k++) {{
            int idx = (i * {args["BlockLayoutYP"]} + warp_y) * {args["WarpLayoutYP"]} * {args["BlockScatterWidthYP"]} + (j * {args["WarpLayoutYP"]} + lane_y) * {args["WarpScatterWidthYP"]} + k;
            int ii = i * {args["BlockScatterWidthYP"]} + j * {args["WarpScatterWidthYP"]} + k;
            float oldMax = smMax[idx];
            float oldSum = smSum[idx];
            float newMax = fmaxf(oldMax, rowMax[ii]);
            float factor = __expf(oldMax - newMax);
            smMax[idx] = newMax;
            smFactor[idx] = factor;
            smSum[idx] = oldSum * factor + rowSum[ii] * __expf(rowMax[ii] - newMax);
            rowMax[ii] = newMax;
          }}
        }}
      }}
    }}
    #pragma unroll
    for (int i=0; i<{args["BrTileP"]}; i++) {{
      rowMax[i] = __shfl(rowMax[i], 0, {args["BLOCK_X"]});
    }}
    #pragma unroll
    for (int i=0; i<{args["BrTileP"]}; i++) {{
      #pragma unroll
      for (int j=0; j<{args["BcTileP"]}; j++) {{
        tileP[i * {args["BcTileP"]} + j] = __expf(tileP[i * {args["BcTileP"]} + j] - rowMax[i]);
      }}
    }}
    #pragma unroll
    for (int i0=0; i0<{args["BLOCK_REPEAT_Y_P"]}; i0++) {{
      #pragma unroll
      for (int i1=0; i1<{args["BLOCK_REPEAT_X_P"]}; i1++) {{
        #pragma unroll
        for (int j0=0; j0<{args["WARP_REPEAT_Y_P"]}; j0++) {{
          #pragma unroll
          for (int j1=0; j1<{args["WARP_REPEAT_X_P"]}; j1++) {{
            #pragma unroll 
            for (int k=0; k<{args["WarpScatterWidthYP"]}; k++) {{
              VecCpy<{args["WarpScatterWidthXP"]}>(&smP[((i0 * {args["BlockLayoutYP"]} + warp_y) * {args["WarpLayoutYP"]} * {args["BlockScatterWidthYP"]} + (j0 * {args["WarpLayoutYP"]} + lane_y) * {args["WarpScatterWidthYP"]} + k) * {args["Bc"]} + 
                                              (i1 * {args["BlockLayoutXP"]} + warp_x) * {args["WarpLayoutXP"]} * {args["BlockScatterWidthXP"]} + (j1 * {args["WarpLayoutXP"]} + lane_x) * {args["WarpScatterWidthXP"]}], 
                                        &tileP[(i0 * {args["BlockScatterWidthYP"]} + j0 * {args["WarpScatterWidthYP"]} + k) * {args["BcTileP"]} + 
                                              i1 * {args["BlockScatterWidthXP"]} + j1 * {args["WarpScatterWidthXP"]}]);
            }}
          }}
        }}
      }}
    }}
    __syncthreads();
    float rowFactor[{args["BrTileO"]}];
    #pragma unroll
    for (int i=0; i<{args["BLOCK_REPEAT_Y_O"]}; i++) {{
      #pragma unroll
      for (int j=0; j<{args["WARP_REPEAT_Y_O"]}; j++) {{
        int idx = (i * {args["BlockLayoutYO"]} + warp_y_) * {args["WarpLayoutYO"]} * {args["BlockScatterWidthYO"]} + (j * {args["WarpLayoutYO"]} + lane_y_) * {args["WarpScatterWidthYO"]};
        VecCpy<{args["WarpScatterWidthYO"]}>(&rowFactor[i * {args["BlockScatterWidthYO"]} + j * {args["WarpScatterWidthYO"]}], &smFactor[idx]);
      }}
    }}
    #pragma unroll
    for (int i=0; i<{args["BrTileO"]}; i++) {{
      #pragma unroll
      for (int j=0; j<{args["HdTile"]}; j++) {{
        tileO[i * {args["HdTile"]} + j] *= rowFactor[i];
      }}
    }}
    for (int k=0; k<{args["Bc"]}; k+={args["SliceO"]}) {{
      #pragma unroll
      for (int i=0; i<{args["GLOB_LOAD_NUM_V"]}; i++) {{
        int x = i * {args["GLOB_LOAD_ROW_WIDTH_V"]} + threadIdx.x * {args["GlobLoadWidthV"]};
        VecCpy<{args["GlobLoadWidthV"]}>(&smV[x], &V[(x/{args["Hd"]} + bx + k) * {args["Hd"]} + x%{args["Hd"]}]);
      }}
      __syncthreads();
      #pragma unroll
      for (int bk=0; bk<{args["SliceO"]}; bk++) {{
        #pragma unroll
        for (int i=0; i<{args["BLOCK_REPEAT_Y_O"]}; i++) {{
          #pragma unroll
          for (int j=0; j<{args["WARP_REPEAT_Y_O"]}; j++) {{
            #pragma unroll
            for (int kk=0; kk<{args["WarpScatterWidthYO"]}; kk++) {{
              int idx = (i * {args["BlockLayoutYO"]} + warp_y_) * {args["WarpLayoutYO"]} * {args["BlockScatterWidthYO"]} + (j * {args["WarpLayoutYO"]} + lane_y_) * {args["WarpScatterWidthYO"]} + kk;
              VecCpy<1>(&regP[i * {args["BlockScatterWidthYO"]} + j * {args["WarpScatterWidthYO"]} + kk], &smP[idx * {args["Bc"]} + k + bk]);
            }}
          }}
        }}
        #pragma unroll
        for (int i=0; i<{args["BLOCK_REPEAT_X_O"]}; i++) {{
          #pragma unroll
          for (int j=0; j<{args["WARP_REPEAT_X_O"]}; j++) {{
            int idx = (i * {args["BlockLayoutXO"]} + warp_x_) * {args["WarpLayoutXO"]} * {args["BlockScatterWidthXO"]} + (j * {args["WarpLayoutXO"]} + lane_x_) * {args["WarpScatterWidthXO"]};
            VecCpy<{args["WarpScatterWidthXO"]}>(&regV[i * {args["BlockScatterWidthXO"]} + j * {args["WarpScatterWidthXO"]}], &smV[bk * {args["Hd"]} + idx]);
          }}
        }}
        #pragma unroll
        for (int cy=0; cy<{args["BrTileO"]}; cy++) {{
          #pragma unroll
          for (int cx=0; cx<{args["HdTile"]}; cx++) {{
            tileO[cy * {args["HdTile"]} + cx] += regP[cy] * regV[cx];
          }}
        }}
      }}
      __syncthreads();
    }}
  }}
  float rowSum_[{args["BrTileO"]}];
  #pragma unroll
  for (int i=0; i<{args["BLOCK_REPEAT_Y_O"]}; i++) {{
    #pragma unroll
    for (int j=0; j<{args["WARP_REPEAT_Y_O"]}; j++) {{
      int idx = (i * {args["BlockLayoutYO"]} + warp_y_) * {args["WarpLayoutYO"]} * {args["BlockScatterWidthYO"]} + (j * {args["WarpLayoutYO"]} + lane_y_) * {args["WarpScatterWidthYO"]};
      VecCpy<{args["WarpScatterWidthYO"]}>(&rowSum_[i * {args["BlockScatterWidthYO"]} + j * {args["WarpScatterWidthYO"]}], &smSum[idx]);
    }}
  }}
  #pragma unroll
  for (int i=0; i<{args["BrTileO"]}; i++) {{
    #pragma unroll
    for (int j=0; j<{args["HdTile"]}; j++) {{
      tileO[i * {args["HdTile"]} + j] /= rowSum_[i];
    }}
  }}
  #pragma unroll
  for (int i0=0; i0<{args["BLOCK_REPEAT_Y_O"]}; i0++) {{
    #pragma unroll
    for (int i1=0; i1<{args["BLOCK_REPEAT_X_O"]}; i1++) {{
      #pragma unroll
      for (int j0=0; j0<{args["WARP_REPEAT_Y_O"]}; j0++) {{
        #pragma unroll
        for (int j1=0; j1<{args["WARP_REPEAT_X_O"]}; j1++) {{
          #pragma unroll 
          for (int kk=0; kk<{args["WarpScatterWidthYO"]}; kk++) {{
            VecCpy<{args["WarpScatterWidthXO"]}>(&O[((i0 * {args["BlockLayoutYO"]} + warp_y_) * {args["WarpLayoutYO"]} * {args["BlockScatterWidthYO"]} + (j0 * {args["WarpLayoutYO"]} + lane_y_) * {args["WarpScatterWidthYO"]} + kk) * {args["Hd"]} + 
                                           (i1 * {args["BlockLayoutXO"]} + warp_x_) * {args["WarpLayoutXO"]} * {args["BlockScatterWidthXO"]} + (j1 * {args["WarpLayoutXO"]} + lane_x_) * {args["WarpScatterWidthXO"]}], 
                                      &tileO[(i0 * {args["BlockScatterWidthYO"]} + j0 * {args["WarpScatterWidthYO"]} + kk) * {args["HdTile"]} + 
                                              i1 * {args["BlockScatterWidthXO"]} + j1 * {args["WarpScatterWidthXO"]}]);
          }}
        }}
      }}
    }}
  }}
}}
"""
  return code

class HIPCompiler:
  def __init__(self, arch="906"):
    self.arch = f"gfx{arch}:sramecc+:xnack-"
  
  def generate(self, kernel, shape, config):
    if kernel == Kernel.Attention:
      block_y = int(config["Br"] / config["PTr"])
      block_x = int(config["Bc"] / config["PTc"])
      thread_num = block_y * block_x
      smem_size = config["Br"]*config["Slice1"] + config["Bc"]*config["Slice1"] + \
                  config["Br"]*config["Bc"] + shape[3]*config["Slice2"] + 3*config["Br"]
      args = {
        "head_num": shape[1],
        "seq_len": shape[2],
        "Br": config["Br"],
        "Bc": config["Bc"],
        "Hd": shape[3],
        "SliceP": config["Slice1"],
        "SliceO": config["Slice2"],
        "BrTileP": config["PTr"],
        "BcTileP": config["PTc"],
        "BrTileO": config["OTr"],
        "HdTile": config["OTc"],
        "GlobLoadWidthQ": config["GLOB_LOAD_WIDTH_Q"],
        "GlobLoadWidthK": config["GLOB_LOAD_WIDTH_K"],
        "GlobLoadWidthV": config["GLOB_LOAD_WIDTH_V"],
        "BlockLayoutYP": config["BLOCK_LAYOUT_P_Y"],
        "BlockLayoutXP": config["BLOCK_LAYOUT_P_X"],
        "WarpLayoutYP": config["WARP_LAYOUT_P_Y"],
        "WarpLayoutXP": config["WARP_LAYOUT_P_X"],
        "BlockScatterWidthYP": config["BLOCK_SCATTER_WIDTH_Q"],
        "BlockScatterWidthXP": config["BLOCK_SCATTER_WIDTH_K"],
        "WarpScatterWidthYP": config["WARP_SCATTER_WIDTH_Q"],
        "WarpScatterWidthXP": config["WARP_SCATTER_WIDTH_K"],
        "BlockLayoutYO": config["BLOCK_LAYOUT_O_Y"],
        "BlockLayoutXO": config["BLOCK_LAYOUT_O_X"],
        "WarpLayoutYO": config["WARP_LAYOUT_O_Y"],
        "WarpLayoutXO": config["WARP_LAYOUT_O_X"],
        "BlockScatterWidthYO": config["BLOCK_SCATTER_WIDTH_P"],
        "BlockScatterWidthXO": config["BLOCK_SCATTER_WIDTH_V"],
        "WarpScatterWidthYO": config["WARP_SCATTER_WIDTH_P"],
        "WarpScatterWidthXO": config["WARP_SCATTER_WIDTH_V"],
        "WarpSize": config["WARP_SIZE"],
        "BLOCK_X": block_x,
        "THREAD_NUM": thread_num,
        "BLOCK_REPEAT_Y_P": int(config["PTr"] / config["BLOCK_SCATTER_WIDTH_Q"]),
        "BLOCK_REPEAT_X_P": int(config["PTc"] / config["BLOCK_SCATTER_WIDTH_K"]),
        "WARP_REPEAT_Y_P": int(config["BLOCK_SCATTER_WIDTH_Q"] / config["WARP_SCATTER_WIDTH_Q"]),
        "WARP_REPEAT_X_P": int(config["BLOCK_SCATTER_WIDTH_K"] / config["WARP_SCATTER_WIDTH_K"]),
        "BLOCK_REPEAT_Y_O": int(config["OTr"] / config["BLOCK_SCATTER_WIDTH_P"]),
        "BLOCK_REPEAT_X_O": int(config["OTc"] / config["BLOCK_SCATTER_WIDTH_V"]),
        "WARP_REPEAT_Y_O": int(config["BLOCK_SCATTER_WIDTH_P"] / config["WARP_SCATTER_WIDTH_P"]),
        "WARP_REPEAT_X_O": int(config["BLOCK_SCATTER_WIDTH_V"] / config["WARP_SCATTER_WIDTH_V"]),
        "GLOB_LOAD_NUM_Q": int(config["Br"] * config["Slice1"] / thread_num / config["GLOB_LOAD_WIDTH_Q"]),
        "GLOB_LOAD_NUM_K": int(config["Bc"] * config["Slice1"] / thread_num / config["GLOB_LOAD_WIDTH_K"]),
        "GLOB_LOAD_NUM_V": int(shape[3] * config["Slice2"] / thread_num / config["GLOB_LOAD_WIDTH_V"]),
        "GLOB_LOAD_ROW_WIDTH_Q": thread_num * config["GLOB_LOAD_WIDTH_Q"],
        "GLOB_LOAD_ROW_WIDTH_K": thread_num * config["GLOB_LOAD_WIDTH_K"],
        "GLOB_LOAD_ROW_WIDTH_V": thread_num * config["GLOB_LOAD_WIDTH_V"],
        "LDS_SZIE": smem_size,
        "Scale": 1 / math.sqrt(shape[1] * shape[3])
      }
      # print(args)
      return makeAttnSrc(args=args)
    
  def build(self, kernel, shape, config):
    code = self.generate( kernel, shape, config)
    print(code)
    with tempfile.TemporaryDirectory() as tmpdir:
      src_path = os.path.join(tmpdir, "main.cpp")
      with open(src_path, "w") as f:
        f.write(code)
      hsaco = os.path.join(dirname, "tmp/attn.hsaco")  # 这修改hsaco生成地址
      hipcc = "/opt/dtk/hip/bin/hipcc"
      if 'HIP_PATH' in os.environ:
        # print(os.getenv("HIP_PATH"))
        hipcc = os.path.join(os.getenv("HIP_PATH"), "bin/hipcc")
      cmd = [hipcc, 
             "-c", 
             "--genco", 
             f"--offload-arch={self.arch}", 
             "-O3", 
             "-ffast-math", 
             "-g0", 
             "-o", hsaco, src_path]
      ret = subprocess.check_call(cmd)
      if ret == 0:
        return hsaco
    return None


if __name__ == "__main__":
  config = {
    'Br': 32, 'Bc': 64, 'Hd': 128, 'Slice1': 16, 'Slice2': 8, 
    'PTr': 4, 'PTc': 4, 'OTr': 4, 'OTc': 8, 
    'GLOB_LOAD_WIDTH_Q': 4, 'GLOB_LOAD_WIDTH_K': 4, 'GLOB_LOAD_WIDTH_V': 4, 
    'BLOCK_LAYOUT_P_Y': 4, 'BLOCK_LAYOUT_P_X': 1, 'WARP_LAYOUT_P_Y': 2, 'WARP_LAYOUT_P_X': 16,
    'BLOCK_SCATTER_WIDTH_Q': 4, 'BLOCK_SCATTER_WIDTH_K': 2, 'WARP_SCATTER_WIDTH_Q': 2, 'WARP_SCATTER_WIDTH_K': 1,
    'BLOCK_LAYOUT_O_Y': 1, 'BLOCK_LAYOUT_O_X': 4, 'WARP_LAYOUT_O_Y': 8, 'WARP_LAYOUT_O_X': 4, 
    'BLOCK_SCATTER_WIDTH_P': 2, 'BLOCK_SCATTER_WIDTH_V': 4, 'WARP_SCATTER_WIDTH_P': 2, 'WARP_SCATTER_WIDTH_V': 2, 
    'WARP_SIZE': 32
  }
  compile = HIPCompiler()
  path = compile.build(Kernel.Attention, [1, 32, 2048, 128], config)
  print(path)