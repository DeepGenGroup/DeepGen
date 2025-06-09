#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

__global__ void reduce(float* input, float* output) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  float elem = input[(bid + (tid / 16) * 16) * 16 + tid % 16];
  for (int i=1; i<16; i*=2) {
    elem += __shfl_down_sync(0xffffffff, elem, i, 16);
  }
  output[(bid + (tid / 16) * 16) * 16 + tid % 16] = elem;
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
  CUresult result = cuModuleLoad(&module, "/home/xiebaokang/projects/mlir/amendDeepGen/build/reduce.cubin");
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to load module: " << result << std::endl;
    return -1;
  }

  // 获取内核函数句柄
  CUfunction kernel;
  result = cuModuleGetFunction(&kernel, module, "reduce");
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to get kernel function" << std::endl;
    return -1;
  }


  // 分配主机内存
  int len = 32*16;
  float* h_input = new float[len];
  float* h_output = new float[len];

  // 初始化输入矩阵
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    h_input[i] = (std::rand() % 1000) * 0.01f;
    // h_input[i] = 1.0f;
  }

  // 分配GPU内存
  float *d_input, *d_output;
  cudaMalloc(&d_input, len * sizeof(float));
  cudaMalloc(&d_output, len * sizeof(float));

  // 将数据从CPU复制到GPU
  cudaMemcpy(d_input, h_input, len * sizeof(float), cudaMemcpyHostToDevice);

  cudaError_t err;
  void* args[] = {&d_input, &d_output};
  cuLaunchKernel(kernel, 16, 1, 1, 32, 1, 1, 0, 0, args, NULL);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      printf("设备同步失败: %s\n", cudaGetErrorString(err));
      return 1;
  }
  cudaMemcpy(h_output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<32; i++) {
    printf("%.7f ", h_output[i*16]);
  }
  printf("\n\n");

  dim3 gridSize1(16);  // bx, by, bz
  dim3 blockSize1(32);
  reduce<<<gridSize1, blockSize1>>>(d_input, d_output);
  cudaMemcpy(h_output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<32; i++) {
    printf("%.7f ", h_output[i*16]);
  }
  printf("\n\n");

  // 同步设备
  cuModuleUnload(module);
  cuCtxDestroy(context);

  // 释放主机内存
  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}    