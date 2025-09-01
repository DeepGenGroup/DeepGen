#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <float.h>

#define M  128
#define N  128
#define BM 32
#define BN 64
#define TM 4
#define TN 4
#define WARPSIZE 64

#define BLOCK_LAYOUT_Y 2
#define BLOCK_LAYOUT_X 1
#define WARP_LAYOUT_Y 4
#define WARP_LAYOUT_X 16
#define THREAD_NUM 128

__device__ __forceinline__ void copy(float* a, float* b) {
  (reinterpret_cast<float4*>(a)[0]) = (reinterpret_cast<float4*>(b)[0]);
}


extern "C"
__global__ void softmax(float* A) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARPSIZE;
  const int lane_id = tid % WARPSIZE;
  const int warp_y = warp_id / BLOCK_LAYOUT_X;
  const int warp_x = warp_id % BLOCK_LAYOUT_X;
  const int lane_y = lane_id / WARP_LAYOUT_X;
  const int lane_x = lane_id % WARP_LAYOUT_X;

  constexpr int WM = WARP_LAYOUT_Y * TM;
  constexpr int WN = WARP_LAYOUT_X * TN;

  const int bid_y = blockIdx.x;
  const int by = bid_y * BM;

  __shared__ float smFactor[BM];
  __shared__ float smMax[BM];
  __shared__ float smSum[BM];
  #pragma unroll
  for (int i=0; i<BM; i+=THREAD_NUM) {
    if (i + tid < BM) {
      smSum[i + tid] = 0.0f;
      smMax[i + tid] = -FLT_MAX;
    }
  }

  for (int bx=0; bx<M; bx+=BM) {
    float tile[TM*TN];
    float rowSum[TM] = {0.0f};
    float rowMax[TM] = {-FLT_MAX};
    // load
    for (int i=0; i<TM; i+=4) {
      for (int j=0; j<TN; j+=4) {
        for (int ldy=0; ldy<4; ldy++) {
          copy(&tile[(i + ldy) * TN + j], &A[(by + warp_y*WM + lane_y*TM + i + ldy) * N + (bx + warp_x*WN + lane_x*TN + j)]);
        }
      }
    }
    // compute
    // thread level
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j++) {
        float newMax = max(tile[i * TN + j], rowMax[i]);
        rowSum[i] = rowSum[i] * exp(rowMax[i] - newMax) + exp(tile[i * TN + j] - newMax);
        rowMax[i] = newMax;
      }
    }
    // warp level
    for (int i=0; i<TM; i++) {
      for (int pos=1; pos<WARP_LAYOUT_X; pos*=2) {
        float rightMax = __shfl_down(rowMax[i], pos, WARP_LAYOUT_X);
        float rightSum = __shfl_down(rowSum[i], pos, WARP_LAYOUT_X);
        float newMax = max(rightMax, rowMax[i]);
        rowSum[i] = rightSum * exp(rightMax - newMax) + rowSum[i] * exp(rowMax[i] - newMax);
        rowMax[i] = newMax;
      }
    }
    // block level
    if (tid % WARP_LAYOUT_X == 0) {
      for (int i=0; i<TM; i++) {
        int idx = warp_y*WM + lane_y*TM + i;
        float oldMax = smMax[idx];
        float oldSum = smSum[idx];
        float newMax = max(oldMax, rowMax[i]);
        float factor = exp(oldMax - newMax);
        smMax[idx] = newMax;
        smFactor[idx] = factor;
        smSum[idx] = oldSum * factor + rowSum[i] * exp(rowMax[i] - newMax);
        rowMax[i] = newMax;
      }
    }
    // braodcast
    #pragma unroll
    for (int i=0; i<TM; i++) {
      rowMax[i] = __shfl(rowMax[i], 0, WARP_LAYOUT_X);
    }
    // compute part element exp
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j++) {
        tile[i * TN + j] = exp(tile[i * TN + j] - rowMax[i]);
      }
    }
    // store
    for (int i=0; i<TM; i+=4) {
      for (int j=0; j<TN; j+=4) {
        for (int ldy=0; ldy<4; ldy++) {
          copy(&A[(by + warp_y*WM + lane_y*TM + i + ldy) * N + (bx + warp_x*WN + lane_x*TN + j)], &tile[(i + ldy) * TN + j]);
        }
      }
    }
  }
}

