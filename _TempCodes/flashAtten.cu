#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

template<int VecLen>
__device__ __forceinline__ void VecCpy(float* a, float* b);

template<>
__device__ __forceinline__ void VecCpy<8>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
  (reinterpret_cast<float4*>(a+4)[0]) = (reinterpret_cast<float4*>(b+4)[0]);
}

template<>
__device__ __forceinline__ void VecCpy<6>(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
  (reinterpret_cast<float2*>(a+4)[0]) = (reinterpret_cast<float2*>(b+4)[0]);
}

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


template <typename T>
void display(T *host, int len) {
    // 打印
    int mid = len / 2;
    int start = (rand() % (mid - 1)) + 1;
    int end = (rand() % (mid - 1)) + mid + 1;
    printf("{%.7f, ..., %.7f, ..., %.7f, ..., %.7f, ..., %.7f}\n", host[0], host[start], host[mid], host[end], host[len - 1]);
}

template <
  const int Br,
  const int Bc,
  const int Hd,
  const int SliceP,
  const int SliceO,
  const int BrTileP,
  const int BcTileP,
  const int BrTileO,
  const int HdTile,

  const int GlobLoadWidthQ,
  const int GlobLoadWidthK,
  const int GlobLoadWidthV,

  const int BlockLayoutYP,
  const int BlockLayoutXP,
  const int WarpLayoutYP,
  const int WarpLayoutXP,

  const int BlockScatterWidthYP,
  const int BlockScatterWidthXP,
  const int WarpScatterWidthYP,
  const int WarpScatterWidthXP,

  const int BlockLayoutYO,
  const int BlockLayoutXO,
  const int WarpLayoutYO,
  const int WarpLayoutXO,

  const int BlockScatterWidthYO,
  const int BlockScatterWidthXO,
  const int WarpScatterWidthYO,
  const int WarpScatterWidthXO,

  const int WarpSize>
__global__ void  FlashAttention_FP32_1(float* Q, float* K, float* V, float* O, int head_num, int seq_len) {
  // load_continuous == 0
  const int BLOCK_Y = Br / BrTileP;
  const int BLOCK_X = Bc / BcTileP;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y;
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

  const int warp_y_ = warp_id / BlockLayoutXO;  // [0, 1]
  const int warp_x_ = warp_id % BlockLayoutXO;
  const int lane_y_ = lane_id / WarpLayoutXO;   // [0, 8]
  const int lane_x_ = lane_id % WarpLayoutXO;

  // split number
  const int BLOCK_REPEAT_Y = BrTileP / BlockScatterWidthYP;
  const int BLOCK_REPEAT_X = BcTileP / BlockScatterWidthXP;
  const int WARP_REPEAT_Y = BlockScatterWidthYP / WarpScatterWidthYP;
  const int WARP_REPEAT_X = BlockScatterWidthXP / WarpScatterWidthXP;

  const int BLOCK_REPEAT_Y_ = BrTileO / BlockScatterWidthYO;
  const int BLOCK_REPEAT_X_ = HdTile / BlockScatterWidthXO;
  const int WARP_REPEAT_Y_ = BlockScatterWidthYO / WarpScatterWidthYO;
  const int WARP_REPEAT_X_ = BlockScatterWidthXO / WarpScatterWidthXO;

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_Q = Br * SliceP / THREAD_NUM;
  const int GLOB_LOAD_TOTAL_WIDTH_K = Bc * SliceP / THREAD_NUM;
  const int GLOB_LOAD_TOTAL_WIDTH_V = Hd * SliceO / THREAD_NUM;
  const int GLOB_LOAD_NUM_Q = GLOB_LOAD_TOTAL_WIDTH_Q / GlobLoadWidthQ;
  const int GLOB_LOAD_NUM_K = GLOB_LOAD_TOTAL_WIDTH_K / GlobLoadWidthK;
  const int GLOB_LOAD_NUM_V = GLOB_LOAD_TOTAL_WIDTH_V / GlobLoadWidthV;
  const int GLOB_LOAD_ROW_WIDTH_Q = THREAD_NUM / SliceP * GlobLoadWidthQ;
  const int GLOB_LOAD_ROW_WIDTH_K = THREAD_NUM / SliceP * GlobLoadWidthK;
  const int GLOB_LOAD_ROW_WIDTH_V = THREAD_NUM / SliceO * GlobLoadWidthV;

  const int ldty = tid / (THREAD_NUM / SliceP);
  const int ldtx = tid % (THREAD_NUM / SliceP);
  const int ldty_ = tid / (THREAD_NUM / SliceO);
  const int ldtx_ = tid % (THREAD_NUM / SliceO);

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

  for (int bx=0; bx<seq_len; bx+=Br) {
    float regQ[BrTileP], regK[BcTileP], tileP[BrTileP*BcTileP] = {0.0f};
    float regV[HdTile], regP[BrTileO];
      // max and sum
    float rowSum[BrTileP] = {0.0f};
    float rowMax[BrTileP] = {-FLT_MAX};

    for (int k=0; k<Hd; k+=SliceP) {  // for K
      // globQ to sharedQ
      __syncthreads();
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_Q; i++) {
        int x = i * GLOB_LOAD_ROW_WIDTH_Q + ldtx * GlobLoadWidthQ;
        VecCpy<GlobLoadWidthQ>(&smQ[ldty * Br + x], &Q[(ldty + k) * seq_len + x]);
      }
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_K; i++) {
        int x = i * GLOB_LOAD_ROW_WIDTH_K + ldtx * GlobLoadWidthK;
        VecCpy<GlobLoadWidthK>(&smK[ldty * Bc + x], &K[(ldty + k) * seq_len + bx + x]);
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
        float oldMax = __shfl_down_sync(0xffffffff, rowMax[i], pos, BLOCK_X);  // Bc必须由一个warp计算 -> blockLayoutX必须是1
        float oldSum = __shfl_down_sync(0xffffffff, rowSum[i], pos, BLOCK_X);
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
      rowMax[i] = __shfl_sync(0xffffffff, rowMax[i], 0, BLOCK_X);
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
        int x = i * GLOB_LOAD_ROW_WIDTH_V + ldtx_ * GlobLoadWidthV;
        VecCpy<GlobLoadWidthQ>(&smV[ldty_ * Hd + x], &V[(ldty_ + bx + k) * Hd + x]);
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
  // update tileo
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


template <
  const int Br = 64,
  const int Bc = 64,
  const int Hd = 128,
  const int SliceP = 16,
  const int SliceO = 16,
  const int BrTileP = 4,
  const int BcTileP = 4,
  const int BrTileO = 4,
  const int HdTile = 8,

  const int GlobLoadWidthQ = 4,
  const int GlobLoadWidthK = 4,
  const int GlobLoadWidthV = 4,

  const int BlockLayoutYP = 8,
  const int BlockLayoutXP = 1,
  const int WarpLayoutYP = 2,
  const int WarpLayoutXP = 16,
  const int BlockScatterWidthYP = 2,
  const int BlockScatterWidthXP = 2,
  const int WarpScatterWidthYP = 1,
  const int WarpScatterWidthXP = 1,

  const int BlockLayoutYO = 2,
  const int BlockLayoutXO = 4,
  const int WarpLayoutYO = 8,
  const int WarpLayoutXO = 4,
  const int BlockScatterWidthYO = 2,
  const int BlockScatterWidthXO = 2,
  const int WarpScatterWidthYO = 1,
  const int WarpScatterWidthXO = 1,
  const int WarpSize = 32>
__global__ void  FlashAttention_FP32_2(float* Q, float* K, float* V, float* O, int head_num, int seq_len) {
  // load_continuous == 1
  const int BLOCK_Y = Br / BrTileP;
  const int BLOCK_X = Bc / BcTileP;
  const int THREAD_NUM = BLOCK_X * BLOCK_Y;
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

  const int warp_y_ = warp_id / BlockLayoutXO;  // [0, 1]
  const int warp_x_ = warp_id % BlockLayoutXO;
  const int lane_y_ = lane_id / WarpLayoutXO;   // [0, 8]
  const int lane_x_ = lane_id % WarpLayoutXO;

  // split number
  const int BLOCK_REPEAT_Y = BrTileP / BlockScatterWidthYP;
  const int BLOCK_REPEAT_X = BcTileP / BlockScatterWidthXP;
  const int WARP_REPEAT_Y = BlockScatterWidthYP / WarpScatterWidthYP;
  const int WARP_REPEAT_X = BlockScatterWidthXP / WarpScatterWidthXP;

  const int BLOCK_REPEAT_Y_ = BrTileO / BlockScatterWidthYO;
  const int BLOCK_REPEAT_X_ = HdTile / BlockScatterWidthXO;
  const int WARP_REPEAT_Y_ = BlockScatterWidthYO / WarpScatterWidthYO;
  const int WARP_REPEAT_X_ = BlockScatterWidthXO / WarpScatterWidthXO;

  // global to shread args
  const int GLOB_LOAD_TOTAL_WIDTH_Q = Br * SliceP / THREAD_NUM;
  const int GLOB_LOAD_TOTAL_WIDTH_K = Bc * SliceP / THREAD_NUM;
  const int GLOB_LOAD_TOTAL_WIDTH_V = Hd * SliceO / THREAD_NUM;
  const int GLOB_LOAD_NUM_Q = GLOB_LOAD_TOTAL_WIDTH_Q / GlobLoadWidthQ;
  const int GLOB_LOAD_NUM_K = GLOB_LOAD_TOTAL_WIDTH_K / GlobLoadWidthK;
  const int GLOB_LOAD_NUM_V = GLOB_LOAD_TOTAL_WIDTH_V / GlobLoadWidthV;
  const int GLOB_LOAD_ALL_WIDTH_Q = THREAD_NUM * GlobLoadWidthQ;
  const int GLOB_LOAD_ALL_WIDTH_K = THREAD_NUM * GlobLoadWidthK;
  const int GLOB_LOAD_ALL_WIDTH_V = THREAD_NUM * GlobLoadWidthV;

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

  for (int bx=0; bx<seq_len; bx+=Br) {
    float regQ[BrTileP], regK[BcTileP], tileP[BrTileP*BcTileP] = {0.0f};
    float regV[HdTile], regP[BrTileO];
      // max and sum
    float rowSum[BrTileP] = {0.0f};
    float rowMax[BrTileP] = {-FLT_MAX};

    for (int k=0; k<Hd; k+=SliceP) {  // for K
      // globQ to sharedQ
      __syncthreads();
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_Q; i++) {
        int idx = i * GLOB_LOAD_ALL_WIDTH_Q + tid * GlobLoadWidthQ;
        int ldty = idx / Br, ldtx = idx % Br;
        VecCpy<GlobLoadWidthQ>(&smQ[ldty * Br + ldtx], &Q[(ldty + k) * seq_len + ldtx]);
      }
      // globB to sharedB
      #pragma unroll
      for (int i=0; i<GLOB_LOAD_NUM_K; i++) {
        int idx = i * GLOB_LOAD_ALL_WIDTH_K + tid * GlobLoadWidthK;
        int ldty = idx / Bc, ldtx = idx % Bc;
        VecCpy<GlobLoadWidthK>(&smK[ldty * Bc + ldtx], &K[(ldty + k) * seq_len + bx + ldtx]);
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
    }

    // // thread level
    for (int i=0; i<BrTileP; i++) {
      for (int j=0; j<BcTileP; j++) {
        float oldMax = rowMax[i];
        rowMax[i] = max(oldMax, tileP[i * BcTileP + j]);
        rowSum[i] = rowSum[i] * exp(oldMax - rowMax[i]) + exp(tileP[i * BcTileP + j] - rowMax[i]);
      }
    }
    // // warp level
    for (int i=0; i<BrTileP; i++) {
      for (int pos=1; pos<BLOCK_X; pos*=2) {
        float oldMax = __shfl_down_sync(0xffffffff, rowMax[i], pos, BLOCK_X);  // Bc必须由一个warp计算 -> blockLayoutX必须是1
        float oldSum = __shfl_down_sync(0xffffffff, rowSum[i], pos, BLOCK_X);
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
      rowMax[i] = __shfl_sync(0xffffffff, rowMax[i], 0, BLOCK_X);
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
        int idx = i * GLOB_LOAD_ALL_WIDTH_V + tid * GlobLoadWidthV;
        int ldty_ = idx / Hd, ldtx_ = idx % Hd;
        VecCpy<GlobLoadWidthQ>(&smV[ldty_ * Hd + ldtx_], &V[(ldty_ + bx + k) * Hd + ldtx_]);
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
  // update tileo
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

__global__ void FlashAttention_FP32_MLIR_16(float* arg0, float* arg1, float* arg2, float* arg3) {
  __shared__ float array0[16][64];
  __shared__ float array1[16][64];
  __shared__ float array2[16][128];
  __shared__ float array3[64][64];
  __shared__ float array4[64];
  int expr0 = (blockIdx.x * 64);
  float array7[4];
  __shared__ float array5[64];
  __shared__ float array6[64];
  for (int iter0 = 0; iter0 < 64; iter0 += 256) {
    if ((((threadIdx.x * -1) + (iter0 * -1)) + 63) >= 0 &&  true) {
      constexpr float const0th = -FLT_MAX;
      array5[(iter0 + threadIdx.x)] = const0th;
      constexpr float const1th = 0;
      array6[(iter0 + threadIdx.x)] = const1th;
    }
  }
  float array8[4][8];
  for (int iter1 = 0; iter1 < 4; iter1 += 1) {
    for (int iter2 = 0; iter2 < 8; iter2 += 1) {
      constexpr float const2th = 0;
      array8[iter1][iter2] = const2th;
    }
  }
  for (int iter3 = 0; iter3 < 128; iter3 += 64) {
    float array9[4];
    float array10[4];
    float array11[4];
    for (int iter4 = 0; iter4 < 4; iter4 += 1) {
      constexpr float const3th = -FLT_MAX;
      array10[iter4] = const3th;
      constexpr float const4th = 0;
      array11[iter4] = const4th;
    }
    float array12[4];
    float array13[4];
    float array14[8];
    float array15[4];
    float array16[4];
    float array17[4];
    float array18[8];
    float array19[4][4];
    for (int iter5 = 0; iter5 < 4; iter5 += 1) {
      for (int iter6 = 0; iter6 < 4; iter6 += 1) {
        constexpr float const5th = 0;
        array19[iter5][iter6] = const5th;
      }
    }
    for (int iter7 = 0; iter7 < 128; iter7 += 16) {
      __syncthreads();
      for (int iter8 = 0; iter8 < 1; iter8 += 1) {
        auto vec0 = (reinterpret_cast<float4*>(&(arg0[blockIdx.z * 16384 + blockIdx.y * 16384 + ((iter7 + (iter8 * 16)) + ((threadIdx.x * 4) / 64)) * 128 + (expr0 + ((threadIdx.x * 4) % 64)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array12[(iter8 * 4)]))[0]) = vec0;
      }
      for (int iter9 = 0; iter9 < 1; iter9 += 1) {
        auto vec1 = (reinterpret_cast<float4*>(&(arg1[blockIdx.z * 16384 + blockIdx.y * 16384 + ((iter7 + (iter9 * 16)) + ((threadIdx.x * 4) / 64)) * 128 + (iter3 + ((threadIdx.x * 4) % 64)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array13[(iter9 * 4)]))[0]) = vec1;
      }
      for (int iter10 = 0; iter10 < 1; iter10 += 1) {
        auto vec2 = (reinterpret_cast<float4*>(&(array12[(iter10 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array0[((iter10 * 16) + ((threadIdx.x * 4) / 64))][((threadIdx.x * 4) % 64)]))[0]) = vec2;
      }
      for (int iter11 = 0; iter11 < 1; iter11 += 1) {
        auto vec3 = (reinterpret_cast<float4*>(&(array13[(iter11 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array1[((iter11 * 16) + ((threadIdx.x * 4) / 64))][((threadIdx.x * 4) % 64)]))[0]) = vec3;
      }
      __syncthreads();
      for (int iter12 = 0; iter12 < 16; iter12 += 1) {
        for (int iter13 = 0; iter13 < 2; iter13 += 1) {
          for (int iter14 = 0; iter14 < 2; iter14 += 1) {
            auto vec4 = (reinterpret_cast<float1*>(&(array0[iter12][(((((iter13 * 8) + (threadIdx.x / 32)) * 4) + (iter14 * 2)) + ((threadIdx.x % 32) / 16))]))[0]);
            (reinterpret_cast<float1*>(&(array15[((iter13 * 2) + iter14)]))[0]) = vec4;
          }
        }
        for (int iter15 = 0; iter15 < 2; iter15 += 1) {
          for (int iter16 = 0; iter16 < 2; iter16 += 1) {
            auto vec5 = (reinterpret_cast<float1*>(&(array1[iter12][(((iter15 * 32) + (iter16 * 16)) + (threadIdx.x % 16))]))[0]);
            (reinterpret_cast<float1*>(&(array16[((iter15 * 2) + iter16)]))[0]) = vec5;
          }
        }
        for (int iter17 = 0; iter17 < 4; iter17 += 1) {
          for (int iter18 = 0; iter18 < 4; iter18 += 1) {
            auto R0 = array19[iter17][iter18];
            auto R1 = array15[iter17];
            auto R2 = array16[iter18];
            auto temp0 = R1 * R2;
            auto temp14 = temp0 + R0;
            array19[iter17][iter18] = temp14;
          }
        }
      }
    }
    for (int iter19 = 0; iter19 < 4; iter19 += 1) {
      for (int iter20 = 0; iter20 < 4; iter20 += 1) {
        auto R3 = array19[iter19][iter20];
        auto R4 = array10[iter19];
        auto R5 = array11[iter19];
        auto temp22 = max(R4 , R3);
        auto temp28 = R4 - temp22;
        auto temp42 = exp(temp28);
        auto temp1 = temp42 * R5;
        auto temp29 = R3 - temp22;
        auto temp43 = exp(temp29);
        auto temp15 = temp1 + temp43;
        array10[iter19] = temp22;
        array11[iter19] = temp15;
      }
    }
    for (int iter21 = 0; iter21 < 4; iter21 += 1) {
      constexpr int32_t const6th = 16;
      constexpr int32_t const7th = 1;
      auto R6 = array10[iter21];
      auto temp55 =  __shfl_down_sync(0xffffffff, R6, const7th, const6th);
      auto R7 = array11[iter21];
      auto temp56 =  __shfl_down_sync(0xffffffff, R7, const7th, const6th);
      auto temp23 = max(R6 , temp55);
      auto temp30 = R6 - temp23;
      auto temp44 = exp(temp30);
      auto temp31 = temp55 - temp23;
      auto temp45 = exp(temp31);
      auto temp2 = R7 * temp44;
      auto temp3 = temp56 * temp45;
      auto temp16 = temp2 + temp3;
      array10[iter21] = temp23;
      array11[iter21] = temp16;
      constexpr int32_t const8th = 2;
      auto R8 = array10[iter21];
      auto temp57 =  __shfl_down_sync(0xffffffff, R8, const8th, const6th);
      auto R9 = array11[iter21];
      auto temp58 =  __shfl_down_sync(0xffffffff, R9, const8th, const6th);
      auto temp24 = max(R8 , temp57);
      auto temp32 = R8 - temp24;
      auto temp46 = exp(temp32);
      auto temp33 = temp57 - temp24;
      auto temp47 = exp(temp33);
      auto temp4 = R9 * temp46;
      auto temp5 = temp58 * temp47;
      auto temp17 = temp4 + temp5;
      array10[iter21] = temp24;
      array11[iter21] = temp17;
      constexpr int32_t const9th = 4;
      auto R10 = array10[iter21];
      auto temp59 =  __shfl_down_sync(0xffffffff, R10, const9th, const6th);
      auto R11 = array11[iter21];
      auto temp60 =  __shfl_down_sync(0xffffffff, R11, const9th, const6th);
      auto temp25 = max(R10 , temp59);
      auto temp34 = R10 - temp25;
      auto temp48 = exp(temp34);
      auto temp35 = temp59 - temp25;
      auto temp49 = exp(temp35);
      auto temp6 = R11 * temp48;
      auto temp7 = temp60 * temp49;
      auto temp18 = temp6 + temp7;
      array10[iter21] = temp25;
      array11[iter21] = temp18;
      constexpr int32_t const10th = 8;
      auto R12 = array10[iter21];
      auto temp61 =  __shfl_down_sync(0xffffffff, R12, const10th, const6th);
      auto R13 = array11[iter21];
      auto temp62 =  __shfl_down_sync(0xffffffff, R13, const10th, const6th);
      auto temp26 = max(R12 , temp61);
      auto temp36 = R12 - temp26;
      auto temp50 = exp(temp36);
      auto temp37 = temp61 - temp26;
      auto temp51 = exp(temp37);
      auto temp8 = R13 * temp50;
      auto temp9 = temp62 * temp51;
      auto temp19 = temp8 + temp9;
      array10[iter21] = temp26;
      array11[iter21] = temp19;
    }
    if ((threadIdx.x % 16) == 0 &&  true) {
      for (int iter22 = 0; iter22 < 4; iter22 += 2) {
        for (int iter23 = 0; iter23 < 2; iter23 += 1) {
          for (int iter24 = 0; iter24 < 1; iter24 += 1) {
            auto R14 = array5[((((((iter22 * 8) + ((threadIdx.x / 32) * 2)) * 2) + (iter23 * 2)) + ((threadIdx.x % 32) / 16)) + iter24)];
            auto R15 = array10[((iter22 + iter23) + iter24)];
            auto R16 = array6[((((((iter22 * 8) + ((threadIdx.x / 32) * 2)) * 2) + (iter23 * 2)) + ((threadIdx.x % 32) / 16)) + iter24)];
            auto R17 = array11[((iter22 + iter23) + iter24)];
            auto temp27 = max(R15 , R14);
            auto temp38 = R15 - temp27;
            auto temp52 = exp(temp38);
            auto temp39 = R14 - temp27;
            auto temp53 = exp(temp39);
            auto temp10 = R17 * temp52;
            auto temp11 = R16 * temp53;
            auto temp20 = temp10 + temp11;
            array5[((((((iter22 * 8) + ((threadIdx.x / 32) * 2)) * 2) + (iter23 * 2)) + ((threadIdx.x % 32) / 16)) + iter24)] = temp27;
            array6[((((((iter22 * 8) + ((threadIdx.x / 32) * 2)) * 2) + (iter23 * 2)) + ((threadIdx.x % 32) / 16)) + iter24)] = temp20;
            array4[((((((iter22 * 8) + ((threadIdx.x / 32) * 2)) * 2) + (iter23 * 2)) + ((threadIdx.x % 32) / 16)) + iter24)] = temp53;
            array10[((iter22 + iter23) + iter24)] = temp27;
          }
        }
      }
    }
    for (int iter25 = 0; iter25 < 4; iter25 += 1) {
      auto R18 = array10[iter25];
      constexpr int32_t const11th = 16;
      constexpr int32_t const12th = 0;
      auto temp63 =  __shfl_sync(0xffffffff, R18, const12th, const11th);
      array10[iter25] = temp63;
    }
    for (int iter26 = 0; iter26 < 4; iter26 += 1) {
      for (int iter27 = 0; iter27 < 4; iter27 += 1) {
        auto R19 = array19[iter26][iter27];
        auto R20 = array10[iter26];
        auto temp40 = R19 - R20;
        auto temp54 = exp(temp40);
        array19[iter26][iter27] = temp54;
      }
    }
    for (int iter28 = 0; iter28 < 4; iter28 += 2) {
      for (int iter29 = 0; iter29 < 4; iter29 += 2) {
        for (int iter30 = 0; iter30 < 2; iter30 += 1) {
          for (int iter31 = 0; iter31 < 2; iter31 += 1) {
            for (int iter32 = 0; iter32 < 1; iter32 += 1) {
              for (int iter33 = 0; iter33 < 1; iter33 += 1) {
                auto vec6 = (reinterpret_cast<float1*>(&(array19[((iter28 + iter30) + iter32)][((iter29 + iter31) + iter33)]))[0]);
                (reinterpret_cast<float1*>(&(array3[((((((iter28 * 8) + ((threadIdx.x / 32) * 2)) * 2) + (iter30 * 2)) + ((threadIdx.x % 32) / 16)) + iter32)][((((iter29 * 16) + (iter31 * 16)) + (threadIdx.x % 16)) + iter33)]))[0]) = vec6;
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int iter34 = 0; iter34 < 2; iter34 += 1) {
      for (int iter35 = 0; iter35 < 2; iter35 += 1) {
        auto vec7 = (reinterpret_cast<float1*>(&(array4[(((((iter34 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter35 * 8)) + ((threadIdx.x % 32) / 4))]))[0]);
        (reinterpret_cast<float1*>(&(array9[((iter34 * 2) + iter35)]))[0]) = vec7;
      }
    }
    for (int iter36 = 0; iter36 < 4; iter36 += 1) {
      for (int iter37 = 0; iter37 < 8; iter37 += 1) {
        auto R21 = array9[iter36];
        auto R22 = array8[iter36][iter37];
        auto temp12 = R22 * R21;
        array8[iter36][iter37] = temp12;
      }
    }
    for (int iter38 = 0; iter38 < 64; iter38 += 16) {
      __syncthreads();
      for (int iter39 = 0; iter39 < 2; iter39 += 1) {
        auto vec8 = (reinterpret_cast<float4*>(&(arg2[blockIdx.z * 16384 + blockIdx.y * 16384 + (((iter3 + iter38) + (iter39 * 8)) + ((threadIdx.x * 4) / 128)) * 128 + ((threadIdx.x * 4) % 128) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array14[(iter39 * 4)]))[0]) = vec8;
      }
      for (int iter40 = 0; iter40 < 2; iter40 += 1) {
        auto vec9 = (reinterpret_cast<float4*>(&(array14[(iter40 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array2[((iter40 * 8) + ((threadIdx.x * 4) / 128))][((threadIdx.x * 4) % 128)]))[0]) = vec9;
      }
      __syncthreads();
      for (int iter41 = 0; iter41 < 16; iter41 += 1) {
        for (int iter42 = 0; iter42 < 2; iter42 += 1) {
          for (int iter43 = 0; iter43 < 2; iter43 += 1) {
            for (int iter44 = 0; iter44 < 1; iter44 += 1) {
              auto vec10 = (reinterpret_cast<float1*>(&(array3[((((((iter42 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter43 * 8)) + ((threadIdx.x % 32) / 4)) + iter44)][(iter38 + iter41)]))[0]);
              (reinterpret_cast<float1*>(&(array17[(((iter42 * 2) + iter43) + iter44)]))[0]) = vec10;
            }
          }
        }
        for (int iter45 = 0; iter45 < 4; iter45 += 1) {
          for (int iter46 = 0; iter46 < 2; iter46 += 1) {
            auto vec11 = (reinterpret_cast<float1*>(&(array2[iter41][(((((iter45 * 4) + ((threadIdx.x / 32) % 4)) * 8) + (iter46 * 4)) + (threadIdx.x % 4))]))[0]);
            (reinterpret_cast<float1*>(&(array18[((iter45 * 2) + iter46)]))[0]) = vec11;
          }
        }
        for (int iter47 = 0; iter47 < 4; iter47 += 1) {
          for (int iter48 = 0; iter48 < 8; iter48 += 1) {
            auto R23 = array8[iter47][iter48];
            auto R24 = array17[iter47];
            auto R25 = array18[iter48];
            auto temp13 = R24 * R25;
            auto temp21 = temp13 + R23;
            array8[iter47][iter48] = temp21;
          }
        }
      }
    }
  }
  for (int iter49 = 0; iter49 < 2; iter49 += 1) {
    for (int iter50 = 0; iter50 < 2; iter50 += 1) {
      auto vec12 = (reinterpret_cast<float1*>(&(array6[(((((iter49 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter50 * 8)) + ((threadIdx.x % 32) / 4))]))[0]);
      (reinterpret_cast<float1*>(&(array7[((iter49 * 2) + iter50)]))[0]) = vec12;
    }
  }
  for (int iter51 = 0; iter51 < 4; iter51 += 1) {
    for (int iter52 = 0; iter52 < 8; iter52 += 1) {
      auto R26 = array7[iter51];
      auto R27 = array8[iter51][iter52];
      auto temp41 = R27 / R26;
      array8[iter51][iter52] = temp41;
    }
  }
  for (int iter53 = 0; iter53 < 4; iter53 += 2) {
    for (int iter54 = 0; iter54 < 8; iter54 += 2) {
      for (int iter55 = 0; iter55 < 2; iter55 += 1) {
        for (int iter56 = 0; iter56 < 2; iter56 += 1) {
          for (int iter57 = 0; iter57 < 1; iter57 += 1) {
            for (int iter58 = 0; iter58 < 1; iter58 += 1) {
              auto vec13 = (reinterpret_cast<float1*>(&(array8[((iter53 + iter55) + iter57)][((iter54 + iter56) + iter58)]))[0]);
              (reinterpret_cast<float1*>(&(arg3[blockIdx.z * 16384 + blockIdx.y * 16384 + ((((expr0 + (((iter53 * 2) + (((threadIdx.x / 32) / 4) * 2)) * 8)) + (iter55 * 8)) + ((threadIdx.x % 32) / 4)) + iter57) * 128 + ((((((iter54 * 4) + (((threadIdx.x / 32) % 4) * 2)) * 4) + (iter56 * 4)) + (threadIdx.x % 4)) + iter58) * 1 + 0]))[0]) = vec13;
            }
          }
        }
      }
    }
  }
}

__global__ void FlashAttention_FP32_MLIR_32(float* arg0, float* arg1, float* arg2, float* arg3) {
  __shared__ float array0[16][64];
  __shared__ float array1[16][64];
  __shared__ float array2[16][128];
  __shared__ float array3[64][64];
  __shared__ float array4[64];
  int expr0 = (blockIdx.x * 64);
  float array7[4];
  __shared__ float array5[64];
  __shared__ float array6[64];
  for (int iter0 = 0; iter0 < 64; iter0 += 256) {
    if ((((threadIdx.x * -1) + (iter0 * -1)) + 63) >= 0 &&  true) {
      constexpr float const0th = -FLT_MAX;
      array5[(iter0 + threadIdx.x)] = const0th;
      constexpr float const1th = 0;
      array6[(iter0 + threadIdx.x)] = const1th;
    }
  }
  float array8[4][8];
  for (int iter1 = 0; iter1 < 4; iter1 += 1) {
    for (int iter2 = 0; iter2 < 8; iter2 += 1) {
      constexpr float const2th = 0;
      array8[iter1][iter2] = const2th;
    }
  }
  for (int iter3 = 0; iter3 < 128; iter3 += 64) {
    float array9[4];
    float array10[8];
    float array11[8];
    for (int iter4 = 0; iter4 < 8; iter4 += 1) {
      constexpr float const3th = -FLT_MAX;
      array10[iter4] = const3th;
      constexpr float const4th = 0;
      array11[iter4] = const4th;
    }
    float array12[4];
    float array13[4];
    float array14[8];
    float array15[8];
    float array16[2];
    float array17[4];
    float array18[8];
    float array19[8][2];
    for (int iter5 = 0; iter5 < 8; iter5 += 1) {
      for (int iter6 = 0; iter6 < 2; iter6 += 1) {
        constexpr float const5th = 0;
        array19[iter5][iter6] = const5th;
      }
    }
    for (int iter7 = 0; iter7 < 128; iter7 += 16) {
      __syncthreads();
      for (int iter8 = 0; iter8 < 1; iter8 += 1) {
        auto vec0 = (reinterpret_cast<float4*>(&(arg0[blockIdx.z * 16384 + blockIdx.y * 16384 + ((iter7 + (iter8 * 16)) + ((threadIdx.x * 4) / 64)) * 128 + (expr0 + ((threadIdx.x * 4) % 64)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array12[(iter8 * 4)]))[0]) = vec0;
      }
      for (int iter9 = 0; iter9 < 1; iter9 += 1) {
        auto vec1 = (reinterpret_cast<float4*>(&(arg1[blockIdx.z * 16384 + blockIdx.y * 16384 + ((iter7 + (iter9 * 16)) + ((threadIdx.x * 4) / 64)) * 128 + (iter3 + ((threadIdx.x * 4) % 64)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array13[(iter9 * 4)]))[0]) = vec1;
      }
      for (int iter10 = 0; iter10 < 1; iter10 += 1) {
        auto vec2 = (reinterpret_cast<float4*>(&(array12[(iter10 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array0[((iter10 * 16) + ((threadIdx.x * 4) / 64))][((threadIdx.x * 4) % 64)]))[0]) = vec2;
      }
      for (int iter11 = 0; iter11 < 1; iter11 += 1) {
        auto vec3 = (reinterpret_cast<float4*>(&(array13[(iter11 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array1[((iter11 * 16) + ((threadIdx.x * 4) / 64))][((threadIdx.x * 4) % 64)]))[0]) = vec3;
      }
      __syncthreads();
      for (int iter12 = 0; iter12 < 16; iter12 += 1) {
        for (int iter13 = 0; iter13 < 2; iter13 += 1) {
          for (int iter14 = 0; iter14 < 2; iter14 += 1) {
            auto vec4 = (reinterpret_cast<float2*>(&(array0[iter12][((((iter13 * 8) + (threadIdx.x / 32)) * 4) + ((iter14 + ((threadIdx.x % 32) / 32)) * 2))]))[0]);
            (reinterpret_cast<float2*>(&(array15[((iter13 * 4) + (iter14 * 2))]))[0]) = vec4;
          }
        }
        for (int iter15 = 0; iter15 < 1; iter15 += 1) {
          for (int iter16 = 0; iter16 < 2; iter16 += 1) {
            auto vec5 = (reinterpret_cast<float1*>(&(array1[iter12][(((iter15 * 64) + (iter16 * 32)) + (threadIdx.x % 32))]))[0]);
            (reinterpret_cast<float1*>(&(array16[((iter15 * 2) + iter16)]))[0]) = vec5;
          }
        }
        for (int iter17 = 0; iter17 < 8; iter17 += 1) {
          for (int iter18 = 0; iter18 < 2; iter18 += 1) {
            auto R0 = array19[iter17][iter18];
            auto R1 = array15[iter17];
            auto R2 = array16[iter18];
            auto temp0 = R1 * R2;
            auto temp16 = temp0 + R0;
            array19[iter17][iter18] = temp16;
          }
        }
      }
    }
    for (int iter19 = 0; iter19 < 8; iter19 += 1) {
      for (int iter20 = 0; iter20 < 2; iter20 += 1) {
        auto R3 = array19[iter19][iter20];
        auto R4 = array10[iter19];
        auto R5 = array11[iter19];
        auto temp25 = max(R4 , R3);
        auto temp32 = R4 - temp25;
        auto temp48 = exp(temp32);
        auto temp1 = temp48 * R5;
        auto temp33 = R3 - temp25;
        auto temp49 = exp(temp33);
        auto temp17 = temp1 + temp49;
        array10[iter19] = temp25;
        array11[iter19] = temp17;
      }
    }
    for (int iter21 = 0; iter21 < 8; iter21 += 1) {
      constexpr int32_t const6th = 32;
      constexpr int32_t const7th = 1;
      auto R6 = array10[iter21];
      auto temp63 =  __shfl_down_sync(0xffffffff, R6, const7th, const6th);
      auto R7 = array11[iter21];
      auto temp64 =  __shfl_down_sync(0xffffffff, R7, const7th, const6th);
      auto temp26 = max(R6 , temp63);
      auto temp34 = R6 - temp26;
      auto temp50 = exp(temp34);
      auto temp35 = temp63 - temp26;
      auto temp51 = exp(temp35);
      auto temp2 = R7 * temp50;
      auto temp3 = temp64 * temp51;
      auto temp18 = temp2 + temp3;
      array10[iter21] = temp26;
      array11[iter21] = temp18;
      constexpr int32_t const8th = 2;
      auto R8 = array10[iter21];
      auto temp65 =  __shfl_down_sync(0xffffffff, R8, const8th, const6th);
      auto R9 = array11[iter21];
      auto temp66 =  __shfl_down_sync(0xffffffff, R9, const8th, const6th);
      auto temp27 = max(R8 , temp65);
      auto temp36 = R8 - temp27;
      auto temp52 = exp(temp36);
      auto temp37 = temp65 - temp27;
      auto temp53 = exp(temp37);
      auto temp4 = R9 * temp52;
      auto temp5 = temp66 * temp53;
      auto temp19 = temp4 + temp5;
      array10[iter21] = temp27;
      array11[iter21] = temp19;
      constexpr int32_t const9th = 4;
      auto R10 = array10[iter21];
      auto temp67 =  __shfl_down_sync(0xffffffff, R10, const9th, const6th);
      auto R11 = array11[iter21];
      auto temp68 =  __shfl_down_sync(0xffffffff, R11, const9th, const6th);
      auto temp28 = max(R10 , temp67);
      auto temp38 = R10 - temp28;
      auto temp54 = exp(temp38);
      auto temp39 = temp67 - temp28;
      auto temp55 = exp(temp39);
      auto temp6 = R11 * temp54;
      auto temp7 = temp68 * temp55;
      auto temp20 = temp6 + temp7;
      array10[iter21] = temp28;
      array11[iter21] = temp20;
      constexpr int32_t const10th = 8;
      auto R12 = array10[iter21];
      auto temp69 =  __shfl_down_sync(0xffffffff, R12, const10th, const6th);
      auto R13 = array11[iter21];
      auto temp70 =  __shfl_down_sync(0xffffffff, R13, const10th, const6th);
      auto temp29 = max(R12 , temp69);
      auto temp40 = R12 - temp29;
      auto temp56 = exp(temp40);
      auto temp41 = temp69 - temp29;
      auto temp57 = exp(temp41);
      auto temp8 = R13 * temp56;
      auto temp9 = temp70 * temp57;
      auto temp21 = temp8 + temp9;
      array10[iter21] = temp29;
      array11[iter21] = temp21;
      constexpr int32_t const11th = 16;
      auto R14 = array10[iter21];
      auto temp71 =  __shfl_down_sync(0xffffffff, R14, const11th, const6th);
      auto R15 = array11[iter21];
      auto temp72 =  __shfl_down_sync(0xffffffff, R15, const11th, const6th);
      auto temp30 = max(R14 , temp71);
      auto temp42 = R14 - temp30;
      auto temp58 = exp(temp42);
      auto temp43 = temp71 - temp30;
      auto temp59 = exp(temp43);
      auto temp10 = R15 * temp58;
      auto temp11 = temp72 * temp59;
      auto temp22 = temp10 + temp11;
      array10[iter21] = temp30;
      array11[iter21] = temp22;
    }
    if ((threadIdx.x % 32) == 0 &&  true) {
      for (int iter22 = 0; iter22 < 8; iter22 += 4) {
        for (int iter23 = 0; iter23 < 4; iter23 += 2) {
          for (int iter24 = 0; iter24 < 2; iter24 += 1) {
            auto R16 = array5[(((((iter22 * 8) + ((threadIdx.x / 32) * 4)) + iter23) + (((threadIdx.x % 32) / 32) * 2)) + iter24)];
            auto R17 = array10[((iter22 + iter23) + iter24)];
            auto R18 = array6[(((((iter22 * 8) + ((threadIdx.x / 32) * 4)) + iter23) + (((threadIdx.x % 32) / 32) * 2)) + iter24)];
            auto R19 = array11[((iter22 + iter23) + iter24)];
            auto temp31 = max(R17 , R16);
            auto temp44 = R17 - temp31;
            auto temp60 = exp(temp44);
            auto temp45 = R16 - temp31;
            auto temp61 = exp(temp45);
            auto temp12 = R19 * temp60;
            auto temp13 = R18 * temp61;
            auto temp23 = temp12 + temp13;
            array5[(((((iter22 * 8) + ((threadIdx.x / 32) * 4)) + iter23) + (((threadIdx.x % 32) / 32) * 2)) + iter24)] = temp31;
            array6[(((((iter22 * 8) + ((threadIdx.x / 32) * 4)) + iter23) + (((threadIdx.x % 32) / 32) * 2)) + iter24)] = temp23;
            array4[(((((iter22 * 8) + ((threadIdx.x / 32) * 4)) + iter23) + (((threadIdx.x % 32) / 32) * 2)) + iter24)] = temp61;
            array10[((iter22 + iter23) + iter24)] = temp31;
          }
        }
      }
    }
    for (int iter25 = 0; iter25 < 8; iter25 += 1) {
      auto R20 = array10[iter25];
      constexpr int32_t const12th = 32;
      constexpr int32_t const13th = 0;
      auto temp73 =  __shfl_sync(0xffffffff, R20, const13th, const12th);
      array10[iter25] = temp73;
    }
    for (int iter26 = 0; iter26 < 8; iter26 += 1) {
      for (int iter27 = 0; iter27 < 2; iter27 += 1) {
        auto R21 = array19[iter26][iter27];
        auto R22 = array10[iter26];
        auto temp46 = R21 - R22;
        auto temp62 = exp(temp46);
        array19[iter26][iter27] = temp62;
      }
    }
    for (int iter28 = 0; iter28 < 8; iter28 += 4) {
      for (int iter29 = 0; iter29 < 2; iter29 += 2) {
        for (int iter30 = 0; iter30 < 4; iter30 += 2) {
          for (int iter31 = 0; iter31 < 2; iter31 += 1) {
            for (int iter32 = 0; iter32 < 2; iter32 += 1) {
              for (int iter33 = 0; iter33 < 1; iter33 += 1) {
                auto vec6 = (reinterpret_cast<float1*>(&(array19[((iter28 + iter30) + iter32)][((iter29 + iter31) + iter33)]))[0]);
                (reinterpret_cast<float1*>(&(array3[(((((iter28 * 8) + ((threadIdx.x / 32) * 4)) + iter30) + (((threadIdx.x % 32) / 32) * 2)) + iter32)][((((iter29 * 32) + (iter31 * 32)) + (threadIdx.x % 32)) + iter33)]))[0]) = vec6;
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int iter34 = 0; iter34 < 2; iter34 += 1) {
      for (int iter35 = 0; iter35 < 2; iter35 += 1) {
        auto vec7 = (reinterpret_cast<float1*>(&(array4[(((((iter34 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter35 * 8)) + ((threadIdx.x % 32) / 4))]))[0]);
        (reinterpret_cast<float1*>(&(array9[((iter34 * 2) + iter35)]))[0]) = vec7;
      }
    }
    for (int iter36 = 0; iter36 < 4; iter36 += 1) {
      for (int iter37 = 0; iter37 < 8; iter37 += 1) {
        auto R23 = array9[iter36];
        auto R24 = array8[iter36][iter37];
        auto temp14 = R24 * R23;
        array8[iter36][iter37] = temp14;
      }
    }
    for (int iter38 = 0; iter38 < 64; iter38 += 16) {
      __syncthreads();
      for (int iter39 = 0; iter39 < 2; iter39 += 1) {
        auto vec8 = (reinterpret_cast<float4*>(&(arg2[blockIdx.z * 16384 + blockIdx.y * 16384 + (((iter3 + iter38) + (iter39 * 8)) + ((threadIdx.x * 4) / 128)) * 128 + ((threadIdx.x * 4) % 128) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array14[(iter39 * 4)]))[0]) = vec8;
      }
      for (int iter40 = 0; iter40 < 2; iter40 += 1) {
        auto vec9 = (reinterpret_cast<float4*>(&(array14[(iter40 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array2[((iter40 * 8) + ((threadIdx.x * 4) / 128))][((threadIdx.x * 4) % 128)]))[0]) = vec9;
      }
      __syncthreads();
      for (int iter41 = 0; iter41 < 16; iter41 += 1) {
        for (int iter42 = 0; iter42 < 2; iter42 += 1) {
          for (int iter43 = 0; iter43 < 2; iter43 += 1) {
            for (int iter44 = 0; iter44 < 1; iter44 += 1) {
              auto vec10 = (reinterpret_cast<float1*>(&(array3[((((((iter42 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter43 * 8)) + ((threadIdx.x % 32) / 4)) + iter44)][(iter38 + iter41)]))[0]);
              (reinterpret_cast<float1*>(&(array17[(((iter42 * 2) + iter43) + iter44)]))[0]) = vec10;
            }
          }
        }
        for (int iter45 = 0; iter45 < 4; iter45 += 1) {
          for (int iter46 = 0; iter46 < 2; iter46 += 1) {
            auto vec11 = (reinterpret_cast<float1*>(&(array2[iter41][(((((iter45 * 4) + ((threadIdx.x / 32) % 4)) * 8) + (iter46 * 4)) + (threadIdx.x % 4))]))[0]);
            (reinterpret_cast<float1*>(&(array18[((iter45 * 2) + iter46)]))[0]) = vec11;
          }
        }
        for (int iter47 = 0; iter47 < 4; iter47 += 1) {
          for (int iter48 = 0; iter48 < 8; iter48 += 1) {
            auto R25 = array8[iter47][iter48];
            auto R26 = array17[iter47];
            auto R27 = array18[iter48];
            auto temp15 = R26 * R27;
            auto temp24 = temp15 + R25;
            array8[iter47][iter48] = temp24;
          }
        }
      }
    }
  }
  for (int iter49 = 0; iter49 < 2; iter49 += 1) {
    for (int iter50 = 0; iter50 < 2; iter50 += 1) {
      auto vec12 = (reinterpret_cast<float1*>(&(array6[(((((iter49 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter50 * 8)) + ((threadIdx.x % 32) / 4))]))[0]);
      (reinterpret_cast<float1*>(&(array7[((iter49 * 2) + iter50)]))[0]) = vec12;
    }
  }
  for (int iter51 = 0; iter51 < 4; iter51 += 1) {
    for (int iter52 = 0; iter52 < 8; iter52 += 1) {
      auto R28 = array7[iter51];
      auto R29 = array8[iter51][iter52];
      auto temp47 = R29 / R28;
      array8[iter51][iter52] = temp47;
    }
  }
  for (int iter53 = 0; iter53 < 4; iter53 += 2) {
    for (int iter54 = 0; iter54 < 8; iter54 += 2) {
      for (int iter55 = 0; iter55 < 2; iter55 += 1) {
        for (int iter56 = 0; iter56 < 2; iter56 += 1) {
          for (int iter57 = 0; iter57 < 1; iter57 += 1) {
            for (int iter58 = 0; iter58 < 1; iter58 += 1) {
              auto vec13 = (reinterpret_cast<float1*>(&(array8[((iter53 + iter55) + iter57)][((iter54 + iter56) + iter58)]))[0]);
              (reinterpret_cast<float1*>(&(arg3[blockIdx.z * 16384 + blockIdx.y * 16384 + ((((expr0 + (((iter53 * 2) + (((threadIdx.x / 32) / 4) * 2)) * 8)) + (iter55 * 8)) + ((threadIdx.x % 32) / 4)) + iter57) * 128 + ((((((iter54 * 4) + (((threadIdx.x / 32) % 4) * 2)) * 4) + (iter56 * 4)) + (threadIdx.x % 4)) + iter58) * 1 + 0]))[0]) = vec13;
            }
          }
        }
      }
    }
  }
}

__global__ void FlashAttention_FP32_MLIR_8(float* arg0, float* arg1, float* arg2, float* arg3) {
  __shared__ float array0[16][64];
  __shared__ float array1[16][64];
  __shared__ float array2[16][128];
  __shared__ float array3[64][64];
  __shared__ float array4[64];
  int expr0 = (blockIdx.x * 64);
  float array7[4];
  __shared__ float array5[64];
  __shared__ float array6[64];
  for (int iter0 = 0; iter0 < 64; iter0 += 256) {
    if ((((threadIdx.x * -1) + (iter0 * -1)) + 63) >= 0 &&  true) {
      constexpr float const0th = -FLT_MAX;
      array5[(iter0 + threadIdx.x)] = const0th;
      constexpr float const1th = 0;
      array6[(iter0 + threadIdx.x)] = const1th;
    }
  }
  float array8[4][8];
  for (int iter1 = 0; iter1 < 4; iter1 += 1) {
    for (int iter2 = 0; iter2 < 8; iter2 += 1) {
      constexpr float const2th = 0;
      array8[iter1][iter2] = const2th;
    }
  }
  for (int iter3 = 0; iter3 < 128; iter3 += 64) {
    float array9[4];
    float array10[2];
    float array11[2];
    for (int iter4 = 0; iter4 < 2; iter4 += 1) {
      constexpr float const3th = -FLT_MAX;
      array10[iter4] = const3th;
      constexpr float const4th = 0;
      array11[iter4] = const4th;
    }
    float array12[4];
    float array13[4];
    float array14[8];
    float array15[2];
    float array16[8];
    float array17[4];
    float array18[8];
    float array19[2][8];
    for (int iter5 = 0; iter5 < 2; iter5 += 1) {
      for (int iter6 = 0; iter6 < 8; iter6 += 1) {
        constexpr float const5th = 0;
        array19[iter5][iter6] = const5th;
      }
    }
    for (int iter7 = 0; iter7 < 128; iter7 += 16) {
      __syncthreads();
      for (int iter8 = 0; iter8 < 1; iter8 += 1) {
        auto vec0 = (reinterpret_cast<float4*>(&(arg0[blockIdx.z * 16384 + blockIdx.y * 16384 + ((iter7 + (iter8 * 16)) + ((threadIdx.x * 4) / 64)) * 128 + (expr0 + ((threadIdx.x * 4) % 64)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array12[(iter8 * 4)]))[0]) = vec0;
      }
      for (int iter9 = 0; iter9 < 1; iter9 += 1) {
        auto vec1 = (reinterpret_cast<float4*>(&(arg1[blockIdx.z * 16384 + blockIdx.y * 16384 + ((iter7 + (iter9 * 16)) + ((threadIdx.x * 4) / 64)) * 128 + (iter3 + ((threadIdx.x * 4) % 64)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array13[(iter9 * 4)]))[0]) = vec1;
      }
      for (int iter10 = 0; iter10 < 1; iter10 += 1) {
        auto vec2 = (reinterpret_cast<float4*>(&(array12[(iter10 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array0[((iter10 * 16) + ((threadIdx.x * 4) / 64))][((threadIdx.x * 4) % 64)]))[0]) = vec2;
      }
      for (int iter11 = 0; iter11 < 1; iter11 += 1) {
        auto vec3 = (reinterpret_cast<float4*>(&(array13[(iter11 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array1[((iter11 * 16) + ((threadIdx.x * 4) / 64))][((threadIdx.x * 4) % 64)]))[0]) = vec3;
      }
      __syncthreads();
      for (int iter12 = 0; iter12 < 16; iter12 += 1) {
        for (int iter13 = 0; iter13 < 0; iter13 += 1) {
          for (int iter14 = 0; iter14 < 2; iter14 += 1) {
            auto vec4 = (reinterpret_cast<float2*>(&(array0[iter12][((((iter13 * 8) + (threadIdx.x / 32)) * 16) + (((iter14 * 4) + ((threadIdx.x % 32) / 8)) * 2))]))[0]);
            (reinterpret_cast<float2*>(&(array15[((iter13 * 4) + (iter14 * 2))]))[0]) = vec4;
          }
        }
        for (int iter15 = 0; iter15 < 4; iter15 += 1) {
          for (int iter16 = 0; iter16 < 2; iter16 += 1) {
            auto vec5 = (reinterpret_cast<float1*>(&(array1[iter12][(((iter15 * 16) + (iter16 * 8)) + (threadIdx.x % 8))]))[0]);
            (reinterpret_cast<float1*>(&(array16[((iter15 * 2) + iter16)]))[0]) = vec5;
          }
        }
        for (int iter17 = 0; iter17 < 2; iter17 += 1) {
          for (int iter18 = 0; iter18 < 8; iter18 += 1) {
            auto R0 = array19[iter17][iter18];
            auto R1 = array15[iter17];
            auto R2 = array16[iter18];
            auto temp0 = R1 * R2;
            auto temp12 = temp0 + R0;
            array19[iter17][iter18] = temp12;
          }
        }
      }
    }
    for (int iter19 = 0; iter19 < 2; iter19 += 1) {
      for (int iter20 = 0; iter20 < 8; iter20 += 1) {
        auto R3 = array19[iter19][iter20];
        auto R4 = array10[iter19];
        auto R5 = array11[iter19];
        auto temp19 = max(R4 , R3);
        auto temp24 = R4 - temp19;
        auto temp36 = exp(temp24);
        auto temp1 = temp36 * R5;
        auto temp25 = R3 - temp19;
        auto temp37 = exp(temp25);
        auto temp13 = temp1 + temp37;
        array10[iter19] = temp19;
        array11[iter19] = temp13;
      }
    }
    for (int iter21 = 0; iter21 < 2; iter21 += 1) {
      constexpr int32_t const6th = 8;
      constexpr int32_t const7th = 1;
      auto R6 = array10[iter21];
      auto temp47 =  __shfl_down_sync(0xffffffff, R6, const7th, const6th);
      auto R7 = array11[iter21];
      auto temp48 =  __shfl_down_sync(0xffffffff, R7, const7th, const6th);
      auto temp20 = max(R6 , temp47);
      auto temp26 = R6 - temp20;
      auto temp38 = exp(temp26);
      auto temp27 = temp47 - temp20;
      auto temp39 = exp(temp27);
      auto temp2 = R7 * temp38;
      auto temp3 = temp48 * temp39;
      auto temp14 = temp2 + temp3;
      array10[iter21] = temp20;
      array11[iter21] = temp14;
      constexpr int32_t const8th = 2;
      auto R8 = array10[iter21];
      auto temp49 =  __shfl_down_sync(0xffffffff, R8, const8th, const6th);
      auto R9 = array11[iter21];
      auto temp50 =  __shfl_down_sync(0xffffffff, R9, const8th, const6th);
      auto temp21 = max(R8 , temp49);
      auto temp28 = R8 - temp21;
      auto temp40 = exp(temp28);
      auto temp29 = temp49 - temp21;
      auto temp41 = exp(temp29);
      auto temp4 = R9 * temp40;
      auto temp5 = temp50 * temp41;
      auto temp15 = temp4 + temp5;
      array10[iter21] = temp21;
      array11[iter21] = temp15;
      constexpr int32_t const9th = 4;
      auto R10 = array10[iter21];
      auto temp51 =  __shfl_down_sync(0xffffffff, R10, const9th, const6th);
      auto R11 = array11[iter21];
      auto temp52 =  __shfl_down_sync(0xffffffff, R11, const9th, const6th);
      auto temp22 = max(R10 , temp51);
      auto temp30 = R10 - temp22;
      auto temp42 = exp(temp30);
      auto temp31 = temp51 - temp22;
      auto temp43 = exp(temp31);
      auto temp6 = R11 * temp42;
      auto temp7 = temp52 * temp43;
      auto temp16 = temp6 + temp7;
      array10[iter21] = temp22;
      array11[iter21] = temp16;
    }
    if ((threadIdx.x % 8) == 0 &&  true) {
      for (int iter22 = 0; iter22 < 2; iter22 += 4) {
        for (int iter23 = 0; iter23 < 4; iter23 += 2) {
          for (int iter24 = 0; iter24 < 2; iter24 += 1) {
            auto R12 = array5[((((((iter22 * 8) + ((threadIdx.x / 32) * 4)) * 4) + (iter23 * 4)) + (((threadIdx.x % 32) / 8) * 2)) + iter24)];
            auto R13 = array10[((iter22 + iter23) + iter24)];
            auto R14 = array6[((((((iter22 * 8) + ((threadIdx.x / 32) * 4)) * 4) + (iter23 * 4)) + (((threadIdx.x % 32) / 8) * 2)) + iter24)];
            auto R15 = array11[((iter22 + iter23) + iter24)];
            auto temp23 = max(R13 , R12);
            auto temp32 = R13 - temp23;
            auto temp44 = exp(temp32);
            auto temp33 = R12 - temp23;
            auto temp45 = exp(temp33);
            auto temp8 = R15 * temp44;
            auto temp9 = R14 * temp45;
            auto temp17 = temp8 + temp9;
            array5[((((((iter22 * 8) + ((threadIdx.x / 32) * 4)) * 4) + (iter23 * 4)) + (((threadIdx.x % 32) / 8) * 2)) + iter24)] = temp23;
            array6[((((((iter22 * 8) + ((threadIdx.x / 32) * 4)) * 4) + (iter23 * 4)) + (((threadIdx.x % 32) / 8) * 2)) + iter24)] = temp17;
            array4[((((((iter22 * 8) + ((threadIdx.x / 32) * 4)) * 4) + (iter23 * 4)) + (((threadIdx.x % 32) / 8) * 2)) + iter24)] = temp45;
            array10[((iter22 + iter23) + iter24)] = temp23;
          }
        }
      }
    }
    for (int iter25 = 0; iter25 < 2; iter25 += 1) {
      auto R16 = array10[iter25];
      constexpr int32_t const10th = 8;
      constexpr int32_t const11th = 0;
      auto temp53 =  __shfl_sync(0xffffffff, R16, const11th, const10th);
      array10[iter25] = temp53;
    }
    for (int iter26 = 0; iter26 < 2; iter26 += 1) {
      for (int iter27 = 0; iter27 < 8; iter27 += 1) {
        auto R17 = array19[iter26][iter27];
        auto R18 = array10[iter26];
        auto temp34 = R17 - R18;
        auto temp46 = exp(temp34);
        array19[iter26][iter27] = temp46;
      }
    }
    for (int iter28 = 0; iter28 < 2; iter28 += 4) {
      for (int iter29 = 0; iter29 < 8; iter29 += 2) {
        for (int iter30 = 0; iter30 < 4; iter30 += 2) {
          for (int iter31 = 0; iter31 < 2; iter31 += 1) {
            for (int iter32 = 0; iter32 < 2; iter32 += 1) {
              for (int iter33 = 0; iter33 < 1; iter33 += 1) {
                auto vec6 = (reinterpret_cast<float1*>(&(array19[((iter28 + iter30) + iter32)][((iter29 + iter31) + iter33)]))[0]);
                (reinterpret_cast<float1*>(&(array3[((((((iter28 * 8) + ((threadIdx.x / 32) * 4)) * 4) + (iter30 * 4)) + (((threadIdx.x % 32) / 8) * 2)) + iter32)][((((iter29 * 8) + (iter31 * 8)) + (threadIdx.x % 8)) + iter33)]))[0]) = vec6;
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int iter34 = 0; iter34 < 2; iter34 += 1) {
      for (int iter35 = 0; iter35 < 2; iter35 += 1) {
        auto vec7 = (reinterpret_cast<float1*>(&(array4[(((((iter34 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter35 * 8)) + ((threadIdx.x % 32) / 4))]))[0]);
        (reinterpret_cast<float1*>(&(array9[((iter34 * 2) + iter35)]))[0]) = vec7;
      }
    }
    for (int iter36 = 0; iter36 < 4; iter36 += 1) {
      for (int iter37 = 0; iter37 < 8; iter37 += 1) {
        auto R19 = array9[iter36];
        auto R20 = array8[iter36][iter37];
        auto temp10 = R20 * R19;
        array8[iter36][iter37] = temp10;
      }
    }
    for (int iter38 = 0; iter38 < 64; iter38 += 16) {
      __syncthreads();
      for (int iter39 = 0; iter39 < 2; iter39 += 1) {
        auto vec8 = (reinterpret_cast<float4*>(&(arg2[blockIdx.z * 16384 + blockIdx.y * 16384 + (((iter3 + iter38) + (iter39 * 8)) + ((threadIdx.x * 4) / 128)) * 128 + ((threadIdx.x * 4) % 128) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array14[(iter39 * 4)]))[0]) = vec8;
      }
      for (int iter40 = 0; iter40 < 2; iter40 += 1) {
        auto vec9 = (reinterpret_cast<float4*>(&(array14[(iter40 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array2[((iter40 * 8) + ((threadIdx.x * 4) / 128))][((threadIdx.x * 4) % 128)]))[0]) = vec9;
      }
      __syncthreads();
      for (int iter41 = 0; iter41 < 16; iter41 += 1) {
        for (int iter42 = 0; iter42 < 2; iter42 += 1) {
          for (int iter43 = 0; iter43 < 2; iter43 += 1) {
            for (int iter44 = 0; iter44 < 1; iter44 += 1) {
              auto vec10 = (reinterpret_cast<float1*>(&(array3[((((((iter42 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter43 * 8)) + ((threadIdx.x % 32) / 4)) + iter44)][(iter38 + iter41)]))[0]);
              (reinterpret_cast<float1*>(&(array17[(((iter42 * 2) + iter43) + iter44)]))[0]) = vec10;
            }
          }
        }
        for (int iter45 = 0; iter45 < 4; iter45 += 1) {
          for (int iter46 = 0; iter46 < 2; iter46 += 1) {
            auto vec11 = (reinterpret_cast<float1*>(&(array2[iter41][(((((iter45 * 4) + ((threadIdx.x / 32) % 4)) * 8) + (iter46 * 4)) + (threadIdx.x % 4))]))[0]);
            (reinterpret_cast<float1*>(&(array18[((iter45 * 2) + iter46)]))[0]) = vec11;
          }
        }
        for (int iter47 = 0; iter47 < 4; iter47 += 1) {
          for (int iter48 = 0; iter48 < 8; iter48 += 1) {
            auto R21 = array8[iter47][iter48];
            auto R22 = array17[iter47];
            auto R23 = array18[iter48];
            auto temp11 = R22 * R23;
            auto temp18 = temp11 + R21;
            array8[iter47][iter48] = temp18;
          }
        }
      }
    }
  }
  for (int iter49 = 0; iter49 < 2; iter49 += 1) {
    for (int iter50 = 0; iter50 < 2; iter50 += 1) {
      auto vec12 = (reinterpret_cast<float1*>(&(array6[(((((iter49 * 2) + ((threadIdx.x / 32) / 4)) * 16) + (iter50 * 8)) + ((threadIdx.x % 32) / 4))]))[0]);
      (reinterpret_cast<float1*>(&(array7[((iter49 * 2) + iter50)]))[0]) = vec12;
    }
  }
  for (int iter51 = 0; iter51 < 4; iter51 += 1) {
    for (int iter52 = 0; iter52 < 8; iter52 += 1) {
      auto R24 = array7[iter51];
      auto R25 = array8[iter51][iter52];
      auto temp35 = R25 / R24;
      array8[iter51][iter52] = temp35;
    }
  }
  for (int iter53 = 0; iter53 < 4; iter53 += 2) {
    for (int iter54 = 0; iter54 < 8; iter54 += 2) {
      for (int iter55 = 0; iter55 < 2; iter55 += 1) {
        for (int iter56 = 0; iter56 < 2; iter56 += 1) {
          for (int iter57 = 0; iter57 < 1; iter57 += 1) {
            for (int iter58 = 0; iter58 < 1; iter58 += 1) {
              auto vec13 = (reinterpret_cast<float1*>(&(array8[((iter53 + iter55) + iter57)][((iter54 + iter56) + iter58)]))[0]);
              (reinterpret_cast<float1*>(&(arg3[blockIdx.z * 16384 + blockIdx.y * 16384 + ((((expr0 + (((iter53 * 2) + (((threadIdx.x / 32) / 4) * 2)) * 8)) + (iter55 * 8)) + ((threadIdx.x % 32) / 4)) + iter57) * 128 + ((((((iter54 * 4) + (((threadIdx.x / 32) % 4) * 2)) * 4) + (iter56 * 4)) + (threadIdx.x % 4)) + iter58) * 1 + 0]))[0]) = vec13;
            }
          }
        }
      }
    }
  }
}

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);
  int device_id;
  cudaGetDevice(&device_id);
  cudaSetDevice(device_id);

  // 定义矩阵参数
  const int batch_size = 1;
  const int head_num = 1;
  const int seq_len = 128;
  const int head_dim = 128;
  const int len = batch_size * head_num * seq_len * head_dim;

  const int Br = 64;
  const int Bc = 64;
  const int Hd = 128;
  const int SliceP = 16;
  const int SliceO = 16;
  const int BrTileP = 2;
  const int BcTileP = 8;
  const int BrTileO = 4;
  const int HdTile = 8;

  const int GlobLoadWidthQ = 4;
  const int GlobLoadWidthK = 4;
  const int GlobLoadWidthV = 4;

  const int BlockLayoutYP = 8;
  const int BlockLayoutXP = 1;
  const int WarpLayoutYP = 4;
  const int WarpLayoutXP = 8;
  const int BlockScatterWidthYP = 2;
  const int BlockScatterWidthXP = 2;
  const int WarpScatterWidthYP = 1;
  const int WarpScatterWidthXP = 1;

  const int BlockLayoutYO = 2;
  const int BlockLayoutXO = 4;
  const int WarpLayoutYO = 8;
  const int WarpLayoutXO = 4;
  const int BlockScatterWidthYO = 2;
  const int BlockScatterWidthXO = 2;
  const int WarpScatterWidthYO = 1;
  const int WarpScatterWidthXO = 1;
  const int WarpSize = 32;

  // 分配主机内存
  float* h_Q = new float[len];
  float* h_K = new float[len];
  float* h_V = new float[len];
  float* h_O = new float[len];

  // 初始化输入矩阵
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    h_Q[i] = (std::rand() % 1000) * 0.01f;
    h_K[i] = (std::rand() % 1000) * 0.01f;
    h_V[i] = (std::rand() % 1000) * 0.01f;
    // h_Q[i] = 1.0f;
    // h_K[i] = 2.0f;
    // h_V[i] = 3.0f;
  }

  // 分配GPU内存
  float* d_Q, *d_K, *d_V, *d_O;
  cudaMalloc(&d_Q, len * sizeof(float));
  cudaMalloc(&d_K, len * sizeof(float));
  cudaMalloc(&d_V, len * sizeof(float));
  cudaMalloc(&d_O, len * sizeof(float));

  // 将数据从CPU复制到GPU
  cudaMemcpy(d_Q, h_Q, len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K, len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, len * sizeof(float), cudaMemcpyHostToDevice);

  dim3 gridSize1(seq_len/Br, head_num, batch_size);  // bx, by, bz
  dim3 blockSize1((Br/BrTileP) * (Bc/BcTileP));

  // FlashAttention_FP32_MLIR_32<<<gridSize1, blockSize1>>>(d_Q, d_K, d_V, d_O);
  // FlashAttention_FP32_MLIR_16<<<gridSize1, blockSize1>>>(d_Q, d_K, d_V, d_O);
  FlashAttention_FP32_MLIR_8<<<gridSize1, blockSize1>>>(d_Q, d_K, d_V, d_O);
  cudaMemcpy(h_O, d_O, len * sizeof(float), cudaMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<128; i++) {
    for (int j=0; j<128; j++) {
      printf("%.7f ", h_O[i * 128 + j]);
    }
    printf("\n");
  }
  printf("\n\n\n");

  // FlashAttention_FP32_1<Br, Bc, Hd, SliceP, SliceO, BrTileP, BcTileP, BrTileO, HdTile, 
  //   GlobLoadWidthQ, GlobLoadWidthK, GlobLoadWidthV, 
  //   BlockLayoutYP, BlockLayoutXP, WarpLayoutYP, WarpLayoutXP, 
  //   BlockScatterWidthYP, BlockScatterWidthXP, WarpScatterWidthYP, WarpScatterWidthXP, 
  //   BlockLayoutYO, BlockLayoutXO, WarpLayoutYO, WarpLayoutXO, 
  //   BlockScatterWidthYO, BlockScatterWidthXO, WarpScatterWidthYO, WarpScatterWidthXO, 
  //   WarpSize><<<gridSize1, blockSize1>>>(d_Q, d_K, d_V, d_O, head_num, seq_len);
  // cudaMemcpy(h_O, d_O, len * sizeof(float), cudaMemcpyDeviceToHost);
  // display(h_O, len);


  FlashAttention_FP32_2<Br, Bc, Hd, SliceP, SliceO, BrTileP, BcTileP, BrTileO, HdTile, 
    GlobLoadWidthQ, GlobLoadWidthK, GlobLoadWidthV, 
    BlockLayoutYP, BlockLayoutXP, WarpLayoutYP, WarpLayoutXP, 
    BlockScatterWidthYP, BlockScatterWidthXP, WarpScatterWidthYP, WarpScatterWidthXP, 
    BlockLayoutYO, BlockLayoutXO, WarpLayoutYO, WarpLayoutXO, 
    BlockScatterWidthYO, BlockScatterWidthXO, WarpScatterWidthYO, WarpScatterWidthXO, 
    WarpSize><<<gridSize1, blockSize1>>>(d_Q, d_K, d_V, d_O, head_num, seq_len);
  cudaMemcpy(h_O, d_O, len * sizeof(float), cudaMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<128; i++) {
    for (int j=0; j<128; j++) {
      printf("%.7f ", h_O[i * 128 + j]);
    }
    printf("\n");
  }
  printf("\n\n\n");

  // 释放主机内存
  delete[] h_Q;
  delete[] h_K;
  delete[] h_V;
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);

  return 0;
}