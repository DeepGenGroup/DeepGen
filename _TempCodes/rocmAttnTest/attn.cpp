#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

template<int VecLen>
__device__ __forceinline__ void VecCpy(float* a, float* b);

template<>
__device__ __forceinline__ void VecCpy<4>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<2>(float* a, float* b) {
  (reinterpret_cast<float2*>(a)[0]) = (reinterpret_cast<float2*>(b)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<1>(float* a, float* b) {
  (reinterpret_cast<float*>(a)[0]) = (reinterpret_cast<float*>(b)[0]);
} 


extern "C" __global__ void  FlashAttention_FP32(float* Q, float* K, float* V, float* O) {
  int head_num=32, seq_len=2048;
  constexpr int Br=32;
  constexpr int Bc=64;
  constexpr int Hd=128;
  constexpr int SliceP=16;
  constexpr int SliceO=8;
  constexpr int BrTileP=4;
  constexpr int BcTileP=4;
  constexpr int BrTileO=4;
  constexpr int HdTile=8;

  constexpr int GlobLoadWidthQ=4;
  constexpr int GlobLoadWidthK=4;
  constexpr int GlobLoadWidthV=4;

  constexpr int BlockLayoutYP=2;
  constexpr int BlockLayoutXP=1;
  constexpr int WarpLayoutYP=4;
  constexpr int WarpLayoutXP=16;

  constexpr int BlockScatterWidthYP=4;
  constexpr int BlockScatterWidthXP=2;
  constexpr int WarpScatterWidthYP=4;
  constexpr int WarpScatterWidthXP=2;

  constexpr int BlockLayoutYO=1;
  constexpr int BlockLayoutXO=2;
  constexpr int WarpLayoutYO=8;
  constexpr int WarpLayoutXO=8;

  constexpr int BlockScatterWidthYO=4;
  constexpr int BlockScatterWidthXO=4;
  constexpr int WarpScatterWidthYO=4;
  constexpr int WarpScatterWidthXO=4;

  constexpr int WarpSize=64;
  constexpr int BLOCK_Y = Br / BrTileP;
  constexpr int BLOCK_X = Bc / BcTileP;
  constexpr int THREAD_NUM = BLOCK_X * BLOCK_Y;
  const int tid = threadIdx.x;
  const int by = blockIdx.x;
  const int batch = blockIdx.z;
  const int head = blockIdx.y;

  // thread mapping
  const int warp_id = tid / WarpSize;
  const int lane_id = tid % WarpSize;
  const int warp_y = warp_id / BlockLayoutXP;
  const int warp_x = warp_id % BlockLayoutXP;
  const int lane_y = lane_id / WarpLayoutXP;
  const int lane_x = lane_id % WarpLayoutXP;

  const int warp_y_ = warp_id / BlockLayoutXO;
  const int warp_x_ = warp_id % BlockLayoutXO;
  const int lane_y_ = lane_id / WarpLayoutXO;
  const int lane_x_ = lane_id % WarpLayoutXO;

  // split number
  constexpr int BLOCK_REPEAT_Y = BrTileP / BlockScatterWidthYP;
  constexpr int BLOCK_REPEAT_X = BcTileP / BlockScatterWidthXP;
  constexpr int WARP_REPEAT_Y = BlockScatterWidthYP / WarpScatterWidthYP;
  constexpr int WARP_REPEAT_X = BlockScatterWidthXP / WarpScatterWidthXP;

  constexpr int BLOCK_REPEAT_Y_ = BrTileO / BlockScatterWidthYO;
  constexpr int BLOCK_REPEAT_X_ = HdTile / BlockScatterWidthXO;
  constexpr int WARP_REPEAT_Y_ = BlockScatterWidthYO / WarpScatterWidthYO;
  constexpr int WARP_REPEAT_X_ = BlockScatterWidthXO / WarpScatterWidthXO;

  // global to shread args
  constexpr int GLOB_LOAD_TOTAL_WIDTH_Q = Br * SliceP / THREAD_NUM;
  constexpr int GLOB_LOAD_TOTAL_WIDTH_K = Bc * SliceP / THREAD_NUM;
  constexpr int GLOB_LOAD_TOTAL_WIDTH_V = Hd * SliceO / THREAD_NUM;
  constexpr int GLOB_LOAD_NUM_Q = GLOB_LOAD_TOTAL_WIDTH_Q / GlobLoadWidthQ;
  constexpr int GLOB_LOAD_NUM_K = GLOB_LOAD_TOTAL_WIDTH_K / GlobLoadWidthK;
  constexpr int GLOB_LOAD_NUM_V = GLOB_LOAD_TOTAL_WIDTH_V / GlobLoadWidthV;
  constexpr int GLOB_LOAD_ROW_WIDTH_Q = THREAD_NUM * GlobLoadWidthQ;
  constexpr int GLOB_LOAD_ROW_WIDTH_K = THREAD_NUM * GlobLoadWidthK;
  constexpr int GLOB_LOAD_ROW_WIDTH_V = THREAD_NUM * GlobLoadWidthV;

  const int LDS_SZIE = (Br*SliceP + Bc*SliceP + Br*Bc + Hd*SliceO + 3*Br);
  __shared__ float LDSMemory[LDS_SZIE];
  // sm offset
  float* smQ = (float *)(LDSMemory);
  float* smK = (float *)(smQ + (Br * SliceP));
  float* smP = (float *)(smK + (Bc * SliceP));
  float* smV = (float *)(smP + (Br * Bc));
  float* smMax = (float *)(smV + (Hd * SliceO));
  float* smSum = (float *)(smMax + (Br));
  float* smFactor = (float *)(smSum + (Br));
  // init smMax and smSum
  #pragma unroll
  for (int i=0; i<Br; i+=THREAD_NUM) {
    if (i + tid < Br) {
      smSum[i + tid] = 0.0f;
      smMax[i + tid] = -FLT_MAX;
    }
  }
  // tileO
  float tileO[BrTileO * HdTile] = {0.0f};

  // batch offset
  Q = Q + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  K = K + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  V = V + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  O = O + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  // offset
  Q = Q + by * Br;
  O = O + by * Br * Hd;

  for (int bx=0; bx<seq_len; bx+=Bc) {
    float regQ[BrTileP], regK[BcTileP], tileP[BrTileP*BcTileP] = {0.0f};
    float regV[HdTile], regP[BrTileO];
      // max and sum
    float rowSum[BrTileP] = {0.0f};
    float rowMax[BrTileP] = {-FLT_MAX};

    for (int k=0; k<Hd; k+=SliceP) {  // for K
      // globQ to sharedQ
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_Q; i++) {
        int idx = i * GLOB_LOAD_ROW_WIDTH_Q + tid * GlobLoadWidthQ;
        int ty = idx / Br, tx = idx % Br;
        VecCpy<GlobLoadWidthQ>(&smQ[idx], &Q[(ty + k) * seq_len + tx]);
      }
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_K; i++) {
        int idx = i * GLOB_LOAD_ROW_WIDTH_K + tid * GlobLoadWidthK;
        int ty = idx / Bc, tx = idx % Bc;
        VecCpy<GlobLoadWidthK>(&smK[idx], &K[(ty + k) * seq_len + bx + tx]);
      }
      __syncthreads();

      #pragma unroll
      for (int bk=0; bk<SliceP; bk++) {   // 外积
        // sharedA to regA
        #pragma unroll
        for (int i=0; i<BLOCK_REPEAT_Y; i++) {
          #pragma unroll
          for (int j=0; j<WARP_REPEAT_Y; j++) {
            int idx = (i * BlockLayoutYP + warp_y) * WarpLayoutYP * BlockScatterWidthYP + (j * WarpLayoutYP + lane_y) * WarpScatterWidthYP;
            VecCpy<WarpScatterWidthYP>(&regQ[i * BlockScatterWidthYP + j * WarpScatterWidthYP], &smQ[bk * Br + idx]);
          }
        }

        // sharedB to regB
        #pragma unroll
        for (int i=0; i<BLOCK_REPEAT_X; i++) {
          #pragma unroll
          for (int j=0; j<WARP_REPEAT_X; j++) {
            int idx = (i * BlockLayoutXP + warp_x) * WarpLayoutXP * BlockScatterWidthXP + (j * WarpLayoutXP + lane_x) * WarpScatterWidthXP;
            VecCpy<WarpScatterWidthXP>(&regK[i * BlockScatterWidthXP + j * WarpScatterWidthXP], &smK[bk * Bc + idx]);
          }
        }

        // computing result
        #pragma unroll
        for (int cy=0; cy<BrTileP; cy++) {
          #pragma unroll
          for (int cx=0; cx<BcTileP; cx++) {
            tileP[cy * BcTileP + cx] += regQ[cy] * regK[cx];
          }
        }
      }
      __syncthreads();
    }

    // thread level
    for (int i=0; i<BrTileP; i++) {
      for (int j=0; j<BcTileP; j++) {
        float oldMax = rowMax[i];
        rowMax[i] = max(oldMax, tileP[i * BcTileP + j]);
        rowSum[i] = rowSum[i] * exp(oldMax - rowMax[i]) + exp(tileP[i * BcTileP + j] - rowMax[i]);
      }
    }
    // warp level
    for (int i=0; i<BrTileP; i++) {
      for (int pos=1; pos<BLOCK_X; pos*=2) {
        float oldMax = __shfl_down(rowMax[i], pos, BLOCK_X);  // Bc必须由一个warp计算 -> blockLayoutX必须是1
        float oldSum = __shfl_down(rowSum[i], pos, BLOCK_X);
        // float oldMax = rowMax[i];
        // float oldSum = rowSum[i];
        float newMax = max(oldMax, rowMax[i]);
        rowSum[i] = oldSum * exp(oldMax - newMax) + rowSum[i] * exp(rowMax[i] - newMax);
        rowMax[i] = newMax;
      }
    }
    // block level
    if (tid % BLOCK_X == 0) {
      for (int i=0; i<BLOCK_REPEAT_Y; i++) {
        for (int j=0; j<WARP_REPEAT_Y; j++) {
          for (int k=0; k<WarpScatterWidthYP; k++) {
            int idx = (i * BlockLayoutYP + warp_y) * WarpLayoutYP * BlockScatterWidthYP + (j * WarpLayoutYP + lane_y) * WarpScatterWidthYP + k;
            int ii = i * BlockScatterWidthYP + j * WarpScatterWidthYP + k;
            float oldMax = smMax[idx];
            float oldSum = smSum[idx];
            float newMax = max(oldMax, rowMax[ii]);
            float factor = exp(oldMax - newMax);
            smMax[idx] = newMax;
            smFactor[idx] = factor;
            smSum[idx] = oldSum * factor + rowSum[ii] * exp(rowMax[ii] - newMax);
            rowMax[ii] = newMax;
          }
        }
      }
    }
    // __syncthreads();
    // broadcast
    #pragma unroll
    for (int i=0; i<BrTileP; i++) {
      rowMax[i] = __shfl(rowMax[i], 0, BLOCK_X);
      // rowMax[i] = rowMax[i];
    }
    // update tilep
    for (int i=0; i<BrTileP; i++) {
      for (int j=0; j<BcTileP; j++) {
        tileP[i * BcTileP + j] = exp(tileP[i * BcTileP + j] - rowMax[i]);
      }
    }
    // tileP to smP
    #pragma unroll
    for (int i0=0; i0<BLOCK_REPEAT_Y; i0++) {
      #pragma unroll
      for (int i1=0; i1<BLOCK_REPEAT_X; i1++) {
        #pragma unroll
        for (int j0=0; j0<WARP_REPEAT_Y; j0++) {
          #pragma unroll
          for (int j1=0; j1<WARP_REPEAT_X; j1++) {
            #pragma unroll 
            for (int k=0; k<WarpScatterWidthYP; k++) {
              VecCpy<WarpScatterWidthXP>(&smP[((i0 * BlockLayoutYP + warp_y) * WarpLayoutYP * BlockScatterWidthYP + (j0 * WarpLayoutYP + lane_y) * WarpScatterWidthYP + k) * Bc + 
                                               (i1 * BlockLayoutXP + warp_x) * WarpLayoutXP * BlockScatterWidthXP + (j1 * WarpLayoutXP + lane_x) * WarpScatterWidthXP], 
                                        &tileP[(i0 * BlockScatterWidthYP + j0 * WarpScatterWidthYP + k) * BcTileP + 
                                                i1 * BlockScatterWidthXP + j1 * WarpScatterWidthXP]);
            }
          }
        }
      }
    }
    __syncthreads();

    // load smFactor to rowFactor
    float rowFactor[BrTileO];
    for (int i=0; i<BLOCK_REPEAT_Y_; i++) {
      for (int j=0; j<WARP_REPEAT_Y_; j++) {
        int idx = (i * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j * WarpLayoutYO + lane_y_) * WarpScatterWidthYO;
        VecCpy<WarpScatterWidthYO>(&rowFactor[i * BlockScatterWidthYO + j * WarpScatterWidthYO], &smFactor[idx]);
      }
    }
    // tileo * rowFactor
    for (int i=0; i<BrTileO; i++) {
      for (int j=0; j<HdTile; j++) {
        tileO[i * HdTile + j] *= rowFactor[i];
      }
    }
    // inner For K (o = p * v)
    for (int k=0; k<Bc; k+=SliceO) {  // for K
      // globQ to sharedQ
      __syncthreads();
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_V; i++) {
        int idx = i * GLOB_LOAD_ROW_WIDTH_V + tid * GlobLoadWidthV;
        int ty = idx / Hd, tx = idx % Hd;
        VecCpy<GlobLoadWidthV>(&smV[idx], &V[(ty + bx + k) * Hd + tx]);
      }
      __syncthreads();

      #pragma unroll
      for (int bk=0; bk<SliceO; bk++) {   // 外积
        // sharedA to regA
        #pragma unroll
        for (int i=0; i<BLOCK_REPEAT_Y_; i++) {
          #pragma unroll
          for (int j=0; j<WARP_REPEAT_Y_; j++) {
            #pragma unroll
            for (int kk=0; kk<WarpScatterWidthYO; kk++) {
              int idx = (i * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j * WarpLayoutYO + lane_y_) * WarpScatterWidthYO + kk;
              VecCpy<1>(&regP[i * BlockScatterWidthYO + j * WarpScatterWidthYO + kk], &smP[idx * Bc + k + bk]);
            }
          }
        }

        // sharedB to regB
        #pragma unroll
        for (int i=0; i<BLOCK_REPEAT_X_; i++) {
          #pragma unroll
          for (int j=0; j<WARP_REPEAT_X_; j++) {
            int idx = (i * BlockLayoutXO + warp_x_) * WarpLayoutXO * BlockScatterWidthXO + (j * WarpLayoutXO + lane_x_) * WarpScatterWidthXO;
            VecCpy<WarpScatterWidthXO>(&regV[i * BlockScatterWidthXO + j * WarpScatterWidthXO], &smV[bk * Hd + idx]);
          }
        }

        // computing result
        #pragma unroll
        for (int cy=0; cy<BrTileO; cy++) {
          #pragma unroll
          for (int cx=0; cx<HdTile; cx++) {
            tileO[cy * HdTile + cx] += regP[cy] * regV[cx];
          }
        }
      }
    }
  }
  // load smSum
  float rowSum_[BrTileO];
  for (int i=0; i<BLOCK_REPEAT_Y_; i++) {
    for (int j=0; j<WARP_REPEAT_Y_; j++) {
      int idx = (i * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j * WarpLayoutYO + lane_y_) * WarpScatterWidthYO;
      VecCpy<WarpScatterWidthYO>(&rowSum_[i * BlockScatterWidthYO + j * WarpScatterWidthYO], &smSum[idx]);
    }
  }
  // // update tileo
  for (int i=0; i<BrTileO; i++) {
    for (int j=0; j<HdTile; j++) {
      tileO[i * HdTile + j] /= rowSum_[i];
    }
  }
  // store O
  #pragma unroll
  for (int i0=0; i0<BLOCK_REPEAT_Y_; i0++) {
    #pragma unroll
    for (int i1=0; i1<BLOCK_REPEAT_X_; i1++) {
      #pragma unroll
      for (int j0=0; j0<WARP_REPEAT_Y_; j0++) {
        #pragma unroll
        for (int j1=0; j1<WARP_REPEAT_X_; j1++) {
          #pragma unroll 
          for (int kk=0; kk<WarpScatterWidthYO; kk++) {
            VecCpy<WarpScatterWidthXO>(&O[((i0 * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j0 * WarpLayoutYO + lane_y_) * WarpScatterWidthYO + kk) * Hd + 
                                           (i1 * BlockLayoutXO + warp_x_) * WarpLayoutXO * BlockScatterWidthXO + (j1 * WarpLayoutXO + lane_x_) * WarpScatterWidthXO], 
                                      &tileO[(i0 * BlockScatterWidthYO + j0 * WarpScatterWidthYO + kk) * HdTile + 
                                              i1 * BlockScatterWidthXO + j1 * WarpScatterWidthXO]);
          }
        }
      }
    }
  }
}
