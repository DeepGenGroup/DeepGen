#include <hip/hip_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

// hipcc -S -o ./reduce_16.s reduce_16.cpp --amdgpu-target=gfx906

__global__ void broadcast_16(float* output) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int offset = bid * 64;
  float elem = output[offset + ((tid / 16) * 16 + tid % 16)];
  float first_elem = __shfl(elem, 0, 16);  // HIP的shuffle函数
  output[offset + ((tid / 16) * 16 + tid % 16)] = first_elem;
}

int main() {
  // 初始化HIP设备
  int device_count;
  hipGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "没有找到可用的HIP设备" << std::endl;
    return 1;
  }
  int device_id;
  hipGetDevice(&device_id);
  hipSetDevice(device_id);

  // 分配主机内存
  int len = 8 * 16;
  float* h_output = new float[len];

  // 初始化输入矩阵
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    h_output[i] = (std::rand() % 1000) * 0.01f;
  }
  for (int i=0; i<8; i++) {
    for (int j=0; j<16; j++) {
      printf("%.7f ", h_output[i * 16 + j]);
    }
    printf("\n");
  }
  printf("\n\n");

  // 分配GPU内存
  float *d_output;
  hipMalloc(&d_output, len * sizeof(float));

  // 数据拷贝
  hipMemcpy(d_output, h_output, len * sizeof(float), hipMemcpyHostToDevice);

  // 启动内核
  dim3 gridSize1(2);  // bx, by, bz
  dim3 blockSize1(64);
  hipLaunchKernelGGL(broadcast_16, gridSize1, blockSize1, 0, 0, d_output); 
  
  // 结果回传
  hipMemcpy(h_output, d_output, len * sizeof(float), hipMemcpyDeviceToHost);
  
  // 打印结果
  for (int i=0; i<8; i++) {
    for (int j=0; j<16; j++) {
      printf("%.7f ", h_output[i * 16 + j]);
    }
    printf("\n");
  }
  printf("\n\n");

  // 释放资源
  delete[] h_output;
  hipFree(d_output);

  return 0;
}