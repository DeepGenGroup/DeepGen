#include <hip/hip_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

__global__ void reduce(float* input, float* output) {
  int tid = threadIdx.x;
  float elem = input[(tid / 4) * 4 + tid % 4];
  for (int i=1; i<4; i*=2) {
    elem += __shfl_down(elem, i, 4);  // HIP的shuffle函数
  }
  output[(tid / 4) * 4 + tid % 4] = elem;
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
  int len = 16 * 4;
  float* h_input = new float[len];
  float* h_output = new float[len];

  // 初始化输入矩阵
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    h_input[i] = (std::rand() % 1000) * 0.01f;
  }
  for (int i=0; i<16; i++) {
    for (int j=0; j<4; j++) {
      printf("%.7f ", h_input[i * 4 + j]);
    }
    printf("\n");
  }
  printf("\n\n");

  // 分配GPU内存
  float *d_input, *d_output;
  hipMalloc(&d_input, len * sizeof(float));
  hipMalloc(&d_output, len * sizeof(float));

  // 数据拷贝
  hipMemcpy(d_input, h_input, len * sizeof(float), hipMemcpyHostToDevice);

  // 启动内核
  dim3 gridSize1(1);  // bx, by, bz
  dim3 blockSize1(64);
  hipLaunchKernelGGL(reduce, gridSize1, blockSize1, 0, 0, d_input, d_output); 
  
  // 结果回传
  hipMemcpy(h_output, d_output, len * sizeof(float), hipMemcpyDeviceToHost);
  
  // 打印结果
  for (int i=0; i<16; i++) {
    printf("%.7f ", d_output[i * 4]);
  }
  printf("\n\n");

  // 释放资源
  delete[] h_input;
  hipFree(d_input);
  delete[] h_output;
  hipFree(d_output);

  return 0;
}