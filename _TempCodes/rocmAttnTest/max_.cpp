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

__global__ void max_(float *A) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARPSIZE;
  const int lane_id = tid % WARPSIZE;
  const int wy = (warp_id / BLOCK_LAYOUT_X) * WM;
  const int wx = (warp_id % BLOCK_LAYOUT_X) * WN;
  const int ty = (lane_id / WARP_LAYOUT_X) * TM;
  const int tx = (lane_id % WARP_LAYOUT_X) * TN;

  const int bid = blockIdx.x;
  const int by = bid * BM;

  for (int bx=0; bx<N; bx+=BN) {
    float tile[TM*TN];

    // load
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j+=4) {
        copy(&tile[i * TN + j], &A[(by + wy + ty + i) * N + (bx + wx + tx)]);
      }
    }
    // compute
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j++) {
        tile[i * TM + j] = max(tile[i * TM + j], 1.0);
      }
    }
    // store
    for (int i=0; i<TM; i++) {
      for (int j=0; j<TN; j+=4) {
        copy(&A[(by + wy + ty + i) * N + (bx + wx + tx)], &tile[i * TN + j]);
      }
    }
  }
}

void launch_hsaco(const char *path, float *h, float *d, int len) {
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
  hipError_t error = hipModuleGetFunction(&kernel, module, "max_");
  if (error == hipSuccess) {
    std::cout << "Successfully obtained the function handle." << std::endl;
  } else {
    std::cerr << "Error in obtaining the function handle: " << hipGetErrorString(error) << std::endl;
  }

  void* args[] = {&d};
  hipModuleLaunchKernel(kernel, 4, 1, 1, 128, 1, 1, 0, 0, args, NULL);
  hipMemcpy(h, d, len * sizeof(float), hipMemcpyDeviceToHost);
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%.6f ", h[i * N + j]);
      if (j == 31) printf(" | ");
    }
    printf("\n");
  }

  hipModuleUnload(module);
}

void launch_kernel(float *h, float *d, int len) {
  dim3 gridSize(M/BM);
  dim3 blockSize(128);
  max_<<<gridSize, blockSize>>>(d);
  hipMemcpy(h, d, len * sizeof(float), hipMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%.6f ", h[i * N + j]);
      if (j == 31) printf(" | ");
    }
    printf("\n");
  }
}

int main(int argc, char* argv[]) {
  constexpr int len = M * N;
  float *a1 = new float[len];
  float *a2 = new float[len];
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
  float *d1, *d2;
  hipMalloc(&d1, len * sizeof(float));
  hipMalloc(&d2, len * sizeof(float));
  hipMemcpy(d1, a1, len * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d2, a2, len * sizeof(float), hipMemcpyHostToDevice);

  // launch
  launch_hsaco(argv[1], a1, d1, len);
  printf("\n\n\n");
  launch_kernel(a2, d2, len);

  // free
  delete[] a1;
  delete[] a2;
  hipFree(d1);
  hipFree(d2);
  return 0;
}