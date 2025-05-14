#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

__global__ void broadcast_4(float* output) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  float elem = output[(bid + (tid / 4) * 4) * 4 + tid % 4];
  float first_elem = __shfl_sync(0xffffffff, elem, 0, 4);
  output[(bid + (tid / 4) * 4) * 4 + tid % 4] = first_elem;
}

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "没有找到可用的CUDA设备" << std::endl;
    return 1;
  }
  int device_id;
  cudaGetDevice(&device_id);
  cudaSetDevice(device_id);

  // 分配主机内存
  int len = 32*4;
  float* h_output = new float[len];

  // 初始化输入矩阵
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    h_output[i] = (std::rand() % 1000) * 0.01f;
  }
  for (int i=0; i<32; i++) {
    for (int j=0; j<4; j++) {
      printf("%.7f ", h_output[i * 4 + j]);
    }
    printf("\n");
  }
  printf("\n\n");

  // 分配GPU内存
  float *d_output;
  cudaMalloc(&d_output, len * sizeof(float));

  // 将数据从CPU复制到GPU
  cudaMemcpy(d_output, h_output, len * sizeof(float), cudaMemcpyHostToDevice);

  dim3 gridSize1(4);  // bx, by, bz
  dim3 blockSize1(32);
  broadcast_4<<<gridSize1, blockSize1>>>(d_output);
  cudaMemcpy(h_output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);
  // display(h_O, len);
  for (int i=0; i<32; i++) {
    for (int j=0; j<4; j++) {
      printf("%.7f ", h_output[i * 4 + j]);
    }
    printf("\n");
  }
  printf("\n\n");

  // 释放主机内存
  delete[] h_output;
  cudaFree(d_output);

  return 0;
}    