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

template <typename T>
void display(T *host, int len) {
    // 打印
    int mid = len / 2;
    int start = (rand() % (mid - 1)) + 1;
    int end = (rand() % (mid - 1)) + mid + 1;
    printf("{%.7f, ..., %.7f, ..., %.7f, ..., %.7f, ..., %.7f}\n", host[0], host[start], host[mid], host[end], host[len - 1]);
}

















template <
  const int Br=32,
  const int Bc=64,
  const int Hd=128,
  const int SliceP=16,
  const int SliceO=8,
  const int BrTileP=4,
  const int BcTileP=4,
  const int BrTileO=4,
  const int HdTile=8,

  const int GlobLoadWidthQ=4,
  const int GlobLoadWidthK=4,
  const int GlobLoadWidthV=4,

  const int BlockLayoutYP=2,
  const int BlockLayoutXP=1,
  const int WarpLayoutYP=4,
  const int WarpLayoutXP=16,

  const int BlockScatterWidthYP=4,
  const int BlockScatterWidthXP=4,
  const int WarpScatterWidthYP=4,
  const int WarpScatterWidthXP=4,

  const int BlockLayoutYO=1,
  const int BlockLayoutXO=2,
  const int WarpLayoutYO=8,
  const int WarpLayoutXO=8,

  const int BlockScatterWidthYO=4,
  const int BlockScatterWidthXO=4,
  const int WarpScatterWidthYO=4,
  const int WarpScatterWidthXO=4,

  const int WarpSize=64>
__global__ void  FlashAttention_FP32(float* Q, float* K, float* V, float* O, int head_num, int seq_len) {
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
  float tileO[BrTileO * HdTile];
  for (int i=0; i<BrTileO; i++) {
    for (int j=0; j<HdTile; j++) {
      tileO[i * HdTile + j] = 0.0f;
    }
  }

  // batch offset
  Q = Q + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  K = K + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  V = V + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  O = O + batch * head_num * seq_len * Hd + head * seq_len * Hd;
  // offset
  Q = Q + by * Br;
  O = O + by * Br * Hd;

  for (int bx=0; bx<seq_len; bx+=Bc) {
    float regQ[BrTileP], regK[BcTileP], tileP[BrTileP*BcTileP];
    float regV[HdTile], regP[BrTileO];
      // max and sum
    float rowSum[BrTileP] = {0.0f};
    float rowMax[BrTileP];
    for (int i=0; i<BrTileP; i++) {
      rowSum[i] = 0.0f;
      rowMax[i] = -FLT_MAX;
      for (int j=0; j<BcTileP; j++) {
        tileP[i * BcTileP + j] = 0.0f;
      }
    }

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
        // rowMax[i] = oldMax - tileP[i * BcTileP + j];
        // rowSum[i] = rowSum[i] * exp(oldMax - rowMax[i]) + exp(tileP[i * BcTileP + j] - rowMax[i]);
        rowSum[i] = rowSum[i] * (oldMax - rowMax[i]) + (tileP[i * BcTileP + j] - rowMax[i]);
      }
    }
    // // warp level
    // for (int i=0; i<BrTileP; i++) {
    //   for (int pos=1; pos<BLOCK_X; pos*=2) {
    //     float oldMax = __shfl_down(rowMax[i], pos, BLOCK_X);  // Bc必须由一个warp计算 -> blockLayoutX必须是1
    //     float oldSum = __shfl_down(rowSum[i], pos, BLOCK_X);
    //     // float oldMax = rowMax[i];
    //     // float oldSum = rowSum[i];
    //     float newMax = max(oldMax, rowMax[i]);
    //     rowSum[i] = oldSum * exp(oldMax - newMax) + rowSum[i] * exp(rowMax[i] - newMax);
    //     rowMax[i] = newMax;
    //   }
    // }
    // // block level
    // if (tid % BLOCK_X == 0) {
    //   for (int i=0; i<BLOCK_REPEAT_Y; i++) {
    //     for (int j=0; j<WARP_REPEAT_Y; j++) {
    //       for (int k=0; k<WarpScatterWidthYP; k++) {
    //         int idx = (i * BlockLayoutYP + warp_y) * WarpLayoutYP * BlockScatterWidthYP + (j * WarpLayoutYP + lane_y) * WarpScatterWidthYP + k;
    //         int ii = i * BlockScatterWidthYP + j * WarpScatterWidthYP + k;
    //         float oldMax = smMax[idx];
    //         float oldSum = smSum[idx];
    //         float newMax = max(oldMax, rowMax[ii]);
    //         float factor = exp(oldMax - newMax);
    //         smMax[idx] = newMax;
    //         smFactor[idx] = factor;
    //         smSum[idx] = oldSum * factor + rowSum[ii] * exp(rowMax[ii] - newMax);
    //         rowMax[ii] = newMax;
    //       }
    //     }
    //   }
    // }
    // // __syncthreads();
    // // broadcast
    // #pragma unroll
    // for (int i=0; i<BrTileP; i++) {
    //   // rowMax[i] = __shfl(rowMax[i], 0, BLOCK_X);
      // rowMax[i] = rowMax[i];
    // }
    // update tilep
    for (int i=0; i<BrTileP; i++) {
      for (int j=0; j<BcTileP; j++) {
        tileP[i * BcTileP + j] = exp(tileP[i * BcTileP + j] - rowMax[i]);
        // tileP[i * BcTileP + j] = rowMax[i];
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
              VecCpy<WarpScatterWidthXP>(&O[((i0 * BlockLayoutYP + warp_y) * WarpLayoutYP * BlockScatterWidthYP + (j0 * WarpLayoutYP + lane_y) * WarpScatterWidthYP + k) * Hd + 
                                               (i1 * BlockLayoutXP + warp_x) * WarpLayoutXP * BlockScatterWidthXP + (j1 * WarpLayoutXP + lane_x) * WarpScatterWidthXP + bx], 
              // VecCpy<WarpScatterWidthXP>(&smP[((i0 * BlockLayoutYP + warp_y) * WarpLayoutYP * BlockScatterWidthYP + (j0 * WarpLayoutYP + lane_y) * WarpScatterWidthYP + k) * Bc + 
              //                                  (i1 * BlockLayoutXP + warp_x) * WarpLayoutXP * BlockScatterWidthXP + (j1 * WarpLayoutXP + lane_x) * WarpScatterWidthXP],
                                        &tileP[(i0 * BlockScatterWidthYP + j0 * WarpScatterWidthYP + k) * BcTileP + 
                                                i1 * BlockScatterWidthXP + j1 * WarpScatterWidthXP]);
            }
          }
        }
      }
    }
    // __syncthreads();

    // load smFactor to rowFactor
    // float rowFactor[BrTileO];
    // for (int i=0; i<BLOCK_REPEAT_Y_; i++) {
    //   for (int j=0; j<WARP_REPEAT_Y_; j++) {
    //     int idx = (i * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j * WarpLayoutYO + lane_y_) * WarpScatterWidthYO;
    //     VecCpy<WarpScatterWidthYO>(&rowFactor[i * BlockScatterWidthYO + j * WarpScatterWidthYO], &smFactor[idx]);
    //   }
    // }
    // // tileo * rowFactor
    // for (int i=0; i<BrTileO; i++) {
    //   for (int j=0; j<HdTile; j++) {
    //     tileO[i * HdTile + j] *= rowFactor[i];
    //   }
    // }
    // inner For K (o = p * v)
    // for (int k=0; k<Bc; k+=SliceO) {  // for K
    //   // globQ to sharedQ
    //   __syncthreads();
    //   #pragma unroll
    //   for (int i=0; i<GLOB_LOAD_NUM_V; i++) {
    //     int idx = i * GLOB_LOAD_ROW_WIDTH_V + tid * GlobLoadWidthV;
    //     int ty = idx / Hd, tx = idx % Hd;
    //     VecCpy<GlobLoadWidthV>(&smV[idx], &V[(ty + bx + k) * Hd + tx]);
    //   }
    //   __syncthreads();

    //   #pragma unroll
    //   for (int bk=0; bk<SliceO; bk++) {   // 外积
    //     // sharedA to regA
    //     #pragma unroll
    //     for (int i=0; i<BLOCK_REPEAT_Y_; i++) {
    //       #pragma unroll
    //       for (int j=0; j<WARP_REPEAT_Y_; j++) {
    //         #pragma unroll
    //         for (int kk=0; kk<WarpScatterWidthYO; kk++) {
    //           int idx = (i * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j * WarpLayoutYO + lane_y_) * WarpScatterWidthYO + kk;
    //           VecCpy<1>(&regP[i * BlockScatterWidthYO + j * WarpScatterWidthYO + kk], &smP[idx * Bc + k + bk]);
    //         }
    //       }
    //     }

    //     // sharedB to regB
    //     #pragma unroll
    //     for (int i=0; i<BLOCK_REPEAT_X_; i++) {
    //       #pragma unroll
    //       for (int j=0; j<WARP_REPEAT_X_; j++) {
    //         int idx = (i * BlockLayoutXO + warp_x_) * WarpLayoutXO * BlockScatterWidthXO + (j * WarpLayoutXO + lane_x_) * WarpScatterWidthXO;
    //         VecCpy<WarpScatterWidthXO>(&regV[i * BlockScatterWidthXO + j * WarpScatterWidthXO], &smV[bk * Hd + idx]);
    //       }
    //     }

    //     // computing result
    //     #pragma unroll
    //     for (int cy=0; cy<BrTileO; cy++) {
    //       #pragma unroll
    //       for (int cx=0; cx<HdTile; cx++) {
    //         tileO[cy * HdTile + cx] += regP[cy] * regV[cx];
    //       }
    //     }
    //   }
    // }
  }
  // load smSum
  // float rowSum_[BrTileO];
  // for (int i=0; i<BLOCK_REPEAT_Y_; i++) {
  //   for (int j=0; j<WARP_REPEAT_Y_; j++) {
  //     int idx = (i * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j * WarpLayoutYO + lane_y_) * WarpScatterWidthYO;
  //     VecCpy<WarpScatterWidthYO>(&rowSum_[i * BlockScatterWidthYO + j * WarpScatterWidthYO], &smSum[idx]);
  //   }
  // }
  // // // update tileo
  // for (int i=0; i<BrTileO; i++) {
  //   for (int j=0; j<HdTile; j++) {
  //     tileO[i * HdTile + j] /= rowSum_[i];
  //   }
  // }
  // store O
  // #pragma unroll
  // for (int i0=0; i0<BLOCK_REPEAT_Y_; i0++) {
  //   #pragma unroll
  //   for (int i1=0; i1<BLOCK_REPEAT_X_; i1++) {
  //     #pragma unroll
  //     for (int j0=0; j0<WARP_REPEAT_Y_; j0++) {
  //       #pragma unroll
  //       for (int j1=0; j1<WARP_REPEAT_X_; j1++) {
  //         #pragma unroll 
  //         for (int kk=0; kk<WarpScatterWidthYO; kk++) {
  //           VecCpy<WarpScatterWidthXO>(&O[((i0 * BlockLayoutYO + warp_y_) * WarpLayoutYO * BlockScatterWidthYO + (j0 * WarpLayoutYO + lane_y_) * WarpScatterWidthYO + kk) * Hd + 
  //                                          (i1 * BlockLayoutXO + warp_x_) * WarpLayoutXO * BlockScatterWidthXO + (j1 * WarpLayoutXO + lane_x_) * WarpScatterWidthXO], 
  //                                     &tileO[(i0 * BlockScatterWidthYO + j0 * WarpScatterWidthYO + kk) * HdTile + 
  //                                             i1 * BlockScatterWidthXO + j1 * WarpScatterWidthXO]);
  //         }
  //       }
  //     }
  //   }
  // }
}











int main(int argc, char* argv[]) {
  int device_count;
  hipGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "没有找到可用的HIP设备" << std::endl;
    return 1;
  }
  int device_id;
  hipGetDevice(&device_id);
  hipSetDevice(device_id);

  // 加载HSACO文件作为模块
  hipModule_t module;
  hipModuleLoad(&module, argv[1]);
  // 获取内核函数
  hipFunction_t kernel;
  hipError_t error = hipModuleGetFunction(&kernel, module, "attention");
  if (error == hipSuccess) {
    std::cout << "Successfully obtained the function handle." << std::endl;
  } else {
    std::cerr << "Error in obtaining the function handle: " << hipGetErrorString(error) << std::endl;
    return 0;
  }

  // 定义矩阵参数
  const int batch_size = 1;
  const int head_num = 1;
  const int seq_len = 128;
  const int head_dim = 128;
  const int len = batch_size * head_num * seq_len * head_dim;

  // 分配主机内存
  float* h_Q = new float[len];
  float* h_K = new float[len];
  float* h_V = new float[len];
  float* h_O = new float[len];
  float* h_O_ = new float[len];

  // 初始化输入矩阵
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    h_Q[i] = (std::rand() % 10);
    h_K[i] = (std::rand() % 10);
    h_V[i] = (std::rand() % 10);

    // h_Q[i] = 1.0f;
    // h_K[i] = 1.0f;
    // h_V[i] = 1.0f;
  }

  // for (int i=0; i<head_dim; i++) {
  //   for (int j=0; j<seq_len; j++) {
  //     if (j < 64) {
  //       h_K[i * seq_len + j] = 1.0f;
  //     } else {
  //       h_K[i * seq_len + j] = 2.0f;
  //     }
  //   }
  // }


  // printf("K: \n");
  // for (int i=0; i<head_dim; i++) {
  //   for (int j=0; j<seq_len; j++) {
  //     printf("%.1f ", h_K[i * seq_len + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n\n");


  // 分配GPU内存
  float* d_Q, *d_K, *d_V, *d_O, *d_O_;
  hipMalloc(&d_Q, len * sizeof(float));
  hipMalloc(&d_K, len * sizeof(float));
  hipMalloc(&d_V, len * sizeof(float));
  hipMalloc(&d_O, len * sizeof(float));
  hipMalloc(&d_O_, len * sizeof(float));

  // 将数据从CPU复制到GPU
  hipMemcpy(d_Q, h_Q, len * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_K, h_K, len * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_V, h_V, len * sizeof(float), hipMemcpyHostToDevice);
  
  // hsaco
  void* args[] = {&d_Q, &d_K, &d_V, &d_O};
  hipModuleLaunchKernel(kernel, 4, 1, 1, 128, 1, 1, 18816, 0, args, NULL);
  hipMemcpy(h_O, d_O, len * sizeof(float), hipMemcpyDeviceToHost);
  for (int i=0; i<head_dim; i++) {
    for (int j=0; j<seq_len; j++) {
      if (j == 31) printf(" | ");
      printf("%.6f ", h_O[i * head_dim + j]);
    }
    printf("\n");
  }
  // display(h_O, len);

  // cpp1
  printf("\n\n\n");
  dim3 gridSize(seq_len/32, head_num, batch_size);  // bx, by, bz
  dim3 blockSize(128);
  FlashAttention_FP32<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_O_, head_num, seq_len);
  hipMemcpy(h_O_, d_O_, len * sizeof(float), hipMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<head_dim; i++) {
    for (int j=0; j<seq_len; j++) {
      if (j == 31) printf(" | ");
      printf("%.6f ", h_O_[i * head_dim + j]);
    }
    printf("\n");
  }

  // 同步设备
  hipModuleUnload(module);

  // 释放主机内存
  delete[] h_Q;
  delete[] h_K;
  delete[] h_V;
  delete[] h_O;
  delete[] h_O_;
  hipFree(d_Q);
  hipFree(d_K);
  hipFree(d_V);
  hipFree(d_O);
  hipFree(d_O_);

  return 0;
}    