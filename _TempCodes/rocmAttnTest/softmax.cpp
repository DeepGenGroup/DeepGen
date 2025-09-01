#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <float.h>

#define M  128
#define N  128
#define BM 32
#define BN 64
#define WM 16
#define WN 64
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

__global__ void softmax(float *A, float* B) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARPSIZE;
  const int lane_id = tid % WARPSIZE;
  const int wy = (warp_id / BLOCK_LAYOUT_X) * WM;
  const int wx = (warp_id % BLOCK_LAYOUT_X) * WN;
  const int ty = (lane_id / WARP_LAYOUT_X) * TM;
  const int tx = (lane_id % WARP_LAYOUT_X) * TN;

  const int bid = blockIdx.x;
  const int by = bid * BM;

  __shared__ float smMax[BM];
  __shared__ float smSum[BM];

  #pragma unroll
  for (int i=0; i<BM; i+=THREAD_NUM) {
    if (i + tid < BM) {
      smSum[i + tid] = 0.0f;
      smMax[i + tid] = -FLT_MAX;
    }
  }
  float tile[TM*TN], rowMax[TM], rowSum[TM];
  // compute max and sum
  for (int bx=0; bx<N; bx+=BN) {
    // init
    for (int i=0; i<TM; i++) {
      rowMax[i] = -FLT_MAX;
      rowSum[i] = 0.0f;
    }
    // load
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j+=4) {
        copy(&tile[i * TN + j], &A[(by + wy + ty + i) * N + (bx + wx + tx)]);
      }
    }
    // compute
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j++) {
          float newMax = max(rowMax[i], tile[i * TN + j]);
          rowSum[i] = rowSum[i] * exp(rowMax[i] - newMax) + exp(tile[i * TN + j] - newMax);
          rowMax[i] = newMax;
      }
    }
    // warp level
    for (int i=0; i<TM; i++) {
      for (int pos=1; pos<WARP_LAYOUT_X; pos*=2) {
        float oldMax = __shfl_down(rowMax[i], pos, WARP_LAYOUT_X);  // Bc必须由一个warp计算 -> blockLayoutX必须是1
        float oldSum = __shfl_down(rowSum[i], pos, WARP_LAYOUT_X);
        float newMax = max(oldMax, rowMax[i]);
        rowSum[i] = oldSum * exp(oldMax - newMax) + rowSum[i] * exp(rowMax[i] - newMax);
        rowMax[i] = newMax;
      }
    }
    // block level
    if (tid % WARP_LAYOUT_X == 0) {
      for (int i=0; i<TM; i++) {
        int idx = wy + ty + i;
        float oldMax = smMax[idx];
        float oldSum = smSum[idx];
        float newMax = max(oldMax, rowMax[i]);
        smMax[idx] = newMax;
        smSum[idx] = oldSum * exp(oldMax - newMax) + rowSum[i] * exp(rowMax[i] - newMax);
      }
    }
  }
  // compute result and store
  for (int bx=0; bx<N; bx+=BN) {
    // load
    #pragma unroll
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j+=4) {
        copy(&tile[i * TN + j], &A[(by + wy + ty + i) * N + (bx + wx + tx)]);
      }
    }
    // sm to reg
    for (int i=0; i<TM; i+=4) {
      int idx = wy + ty + i;
      copy(rowMax, &smMax[idx]);
      copy(rowSum, &smSum[idx]);
    }
    // x - max
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j++) {
        // tile[i * TN + j] = exp(tile[i * TN + j] - rowMax[i]) / rowSum[i];
        // tile[i * TN + j] = exp(tile[i * TN + j] - rowMax[i]);
        // tile[i * TN + j] = rowSum[i];
        // tile[i * TN + j] = rowMax[i];
        tile[i * TN + j] = exp(tile[i * TN + j]);
      }
    }
    // store
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j+=4) {
        copy(&B[(by + wy + ty + i) * N + (bx + wx + tx)], &tile[i * TN + j]);
      }
    }
  }
}

void launch_hsaco(const char *path, float *h1, float *d1, float *h2, float *d2, int len) {
  int device_count;
  hipGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "没有找到可用的HIP设备" << std::endl;
  }
  int device_id;
  hipGetDevice(&device_id);
  hipSetDevice(device_id);
  // 加载HSACO文件作为模块
  hipModule_t module;
  hipModuleLoad(&module, path);
  // 获取内核函数
  hipFunction_t kernel;
  hipError_t error = hipModuleGetFunction(&kernel, module, "softmax");
  if (error == hipSuccess) {
    std::cout << "Successfully obtained the function handle." << std::endl;
  } else {
    std::cerr << "Error in obtaining the function handle: " << hipGetErrorString(error) << std::endl;
  }

  void* args[] = {&d1, &d2};
  hipModuleLaunchKernel(kernel, 4, 1, 1, 128, 1, 1, 256, 0, args, NULL);
  hipMemcpy(h2, d2, len * sizeof(float), hipMemcpyDeviceToHost);
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%.6f ", h2[i * N + j]);
      if (j == 31) printf(" | ");
    }
    printf("\n");
  }

  hipModuleUnload(module);
}

void launch_kernel(float *h1, float *d1, float *h2, float *d2, int len) {
  dim3 gridSize(M/BM);
  dim3 blockSize(128);
  softmax<<<gridSize, blockSize>>>(d1, d2);
  hipMemcpy(h2, d2, len * sizeof(float), hipMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%.6f ", h2[i * N + j]);
      if (j == 31) printf(" | ");
    }
    printf("\n");
  }
}

int main(int argc, char* argv[]) {
  constexpr int len = M * N;
  float *a1 = new float[len];
  float *a2 = new float[len];

  float *a3 = new float[len];
  float *a4 = new float[len];
  // init 
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    float tmp = (std::rand() % 100) * 0.1f;
    a1[i] = tmp;
    a2[i] = tmp;
  }
  // show input
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%.6f ", a1[i * N + j]);
      if (j == 31) printf(" | ");
    }
    printf("\n");
  }
  printf("\n\n\n");
  // malloc and copy
  float *d1, *d2, *d3, *d4;
  hipMalloc(&d1, len * sizeof(float));
  hipMalloc(&d2, len * sizeof(float));
  hipMalloc(&d3, len * sizeof(float));
  hipMalloc(&d4, len * sizeof(float));
  hipMemcpy(d1, a1, len * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d2, a2, len * sizeof(float), hipMemcpyHostToDevice);

  // launch
  launch_hsaco(argv[1], a1, d1, a3, d3, len);
  printf("\n\n\n");
  launch_kernel(a2, d2, a4, d4, len);

  // free
  delete[] a1;
  delete[] a2;
  hipFree(d1);
  hipFree(d2);
  delete[] a3;
  delete[] a4;
  hipFree(d3);
  hipFree(d4);
  return 0;
}