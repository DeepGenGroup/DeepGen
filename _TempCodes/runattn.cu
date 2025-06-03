#include <cuda_runtime.h>
#include <cuda.h>
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
  const int Br=64,
  const int Bc=64,
  const int Hd=128,
  const int SliceP=16,
  const int SliceO=16,
  const int BrTileP=4,
  const int BcTileP=4,
  const int BrTileO=4,
  const int HdTile=8,

  const int GlobLoadWidthQ=4,
  const int GlobLoadWidthK=4,
  const int GlobLoadWidthV=4,

  const int BlockLayoutYP=8,
  const int BlockLayoutXP=1,
  const int WarpLayoutYP=2,
  const int WarpLayoutXP=16,

  const int BlockScatterWidthYP=2,
  const int BlockScatterWidthXP=2,
  const int WarpScatterWidthYP=1,
  const int WarpScatterWidthXP=1,

  const int BlockLayoutYO=2,
  const int BlockLayoutXO=4,
  const int WarpLayoutYO=8,
  const int WarpLayoutXO=4,

  const int BlockScatterWidthYO=2,
  const int BlockScatterWidthXO=2,
  const int WarpScatterWidthYO=1,
  const int WarpScatterWidthXO=1,
  // const int Br=64,
  // const int Bc=64,
  // const int Hd=128,
  // const int SliceP=16,
  // const int SliceO=16,
  // const int BrTileP=8,
  // const int BcTileP=2,
  // const int BrTileO=4,
  // const int HdTile=8,

  // const int GlobLoadWidthQ=4,
  // const int GlobLoadWidthK=4,
  // const int GlobLoadWidthV=4,

  // const int BlockLayoutYP=8,
  // const int BlockLayoutXP=1,
  // const int WarpLayoutYP=1,
  // const int WarpLayoutXP=32,

  // const int BlockScatterWidthYP=4,
  // const int BlockScatterWidthXP=2,
  // const int WarpScatterWidthYP=2,
  // const int WarpScatterWidthXP=1,

  // const int BlockLayoutYO=2,
  // const int BlockLayoutXO=4,
  // const int WarpLayoutYO=8,
  // const int WarpLayoutXO=4,

  // const int BlockScatterWidthYO=2,
  // const int BlockScatterWidthXO=2,
  // const int WarpScatterWidthYO=1,
  // const int WarpScatterWidthXO=1,

  const int WarpSize=32>
__global__ void  FlashAttention_FP32(float* Q, float* K, float* V, float* O, int head_num, int seq_len) {
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



__global__ void verify_kernel(float* A, float* B, int m, int n) {
  int batch = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float sub = A[batch*m*n + row * n + col] - B[batch*m*n + col * m + row];
    if (sub >= 0.0000001f || sub <= -0.0000001f) {
      // printf("%d %d\n", row, col);
      printf("error!\nindex: (y=%d, x=%d)\nmine: %f  verify: %.8f\nsub: %.8f\n", row, col, A[batch*m*n + row * n + col], B[batch*m*n + col * m + row], sub);
    }
  }
}

int main() {
  int device_count;
  CUresult cuResult = cuInit(0);
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "没有找到可用的CUDA设备" << std::endl;
    return 1;
  }
  int device_id;
  cudaGetDevice(&device_id);
  cudaSetDevice(device_id);

  CUcontext context;
  cuResult = cuCtxCreate(&context, 0, device_id);
  if (cuResult != CUDA_SUCCESS) {
    std::cerr << "Failed to create CUDA context: " << cuResult << std::endl;
    return 1;
  }

  // 加载.cubin模块
  CUmodule module;
  CUresult result = cuModuleLoad(&module, "/tmp/compile-ptx-src-ebe279.cubin");
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to load module: " << result << std::endl;
    return -1;
  }

  // 获取内核函数句柄
  CUfunction kernel;
  result = cuModuleGetFunction(&kernel, module, "attention1");
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to get kernel function" << std::endl;
    return -1;
  }

  // 定义矩阵参数
  const int batch_size = 1;
  const int head_num = 32;
  const int seq_len = 2048 ;
  const int head_dim = 128;
  const int len = batch_size * head_num * seq_len * head_dim;

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
    // h_K[i] = 1.0f;
    // h_V[i] = 1.0f;
  }
  // for (int i=0; i<128; i++) {
  //   for (int j=0; j<128; j++) {
  //     printf("%.7f ", h_Q[i*128+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n\n");
  // for (int i=0; i<128; i++) {
  //   for (int j=0; j<128; j++) {
  //     printf("%.7f ", h_K[i*128+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n\n");
  // for (int i=0; i<128; i++) {
  //   for (int j=0; j<128; j++) {
  //     printf("%.7f ", h_V[i*128+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n\n");

  // 分配GPU内存
  float* d_Q, *d_K, *d_V, *d_O, *d_O_;
  cudaMalloc(&d_Q, len * sizeof(float));
  cudaMalloc(&d_K, len * sizeof(float));
  cudaMalloc(&d_V, len * sizeof(float));
  cudaMalloc(&d_O, len * sizeof(float));
  cudaMalloc(&d_O_, len * sizeof(float));

  // 将数据从CPU复制到GPU
  cudaMemcpy(d_Q, h_Q, len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K, len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, len * sizeof(float), cudaMemcpyHostToDevice);

  cudaError_t err;
  void* args[] = {&d_Q, &d_K, &d_V, &d_O_};
  cuLaunchKernel(kernel, 32, 32, 1, 256, 1, 1, 29440, 0, args, NULL);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      printf("设备同步失败: %s\n", cudaGetErrorString(err));
      return 1;
  }
  cudaMemcpy(h_O, d_O_, len * sizeof(float), cudaMemcpyDeviceToHost);
  display(h_O, len);
  // for (int i=0; i<128; i++) {
  //   for (int j=0; j<128; j++) {
  //     printf("%.7f ", h_O[i*128+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n\n");

  // dim3 gridSize1(seq_len/64, head_num, batch_size);  // bx, by, bz
  // dim3 blockSize1(256);
  // FlashAttention_FP32<<<gridSize1, blockSize1>>>(d_Q, d_K, d_V, d_O, head_num, seq_len);
  // cudaMemcpy(h_O, d_O, len * sizeof(float), cudaMemcpyDeviceToHost);
  // display(h_O, len);
  // for (int i=0; i<128; i++) {
  //   for (int j=0; j<128; j++) {
  //     printf("%.7f ", h_O[i*128+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n\n");

  // 同步设备
  cuModuleUnload(module);
  cuCtxDestroy(context);

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