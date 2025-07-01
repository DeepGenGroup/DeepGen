#include <hip/hip_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

__global__ void reduce(float* input, float* output) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int offset = bid * 64;
  float elem = input[offset + ((tid / 16) * 16 + tid % 16)];
  for (int i=1; i<16; i*=2) {
    elem += __shfl_down(elem, i, 64);  // HIP的shuffle函数
  }
  output[offset + ((tid / 16) * 16 + tid % 16)] = elem;
}

int main() {
  // 初始化HIP设备
  int device_count;
  hipGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No available HIP devices found" << std::endl;
    return 1;
  }
  
  int device_id;
  hipGetDevice(&device_id);
  hipSetDevice(device_id);

  // 创建HIP上下文
  hipCtx_t context;
  hipError_t ctxResult = hipCtxCreate(&context, 0, device_id);
  if (ctxResult != hipSuccess) {
    std::cerr << "Failed to create HIP context: " << hipGetErrorString(ctxResult) << std::endl;
    return 1;
  }

  // 加载HIP模块（使用.code替代.cubin）
  hipModule_t module;
  hipError_t result = hipModuleLoad(&module, "/tmp/kcg_kernel-a1b0df/kcg_kernel-a1b0df.hsaco");
  if (result != hipSuccess) {
    std::cerr << "Failed to load module: " << hipGetErrorString(result) << std::endl;
    hipCtxDestroy(context);
    return -1;
  }

  // 获取内核函数句柄
  hipFunction_t kernel;
  result = hipModuleGetFunction(&kernel, module, "reduce");
  if (result != hipSuccess) {
    std::cerr << "Failed to get kernel function: " << hipGetErrorString(result) << std::endl;
    hipModuleUnload(module);
    hipCtxDestroy(context);
    return -1;
  }

  // 分配主机内存
  const int len = 8 * 16;
  float* h_input = new float[len];
  float* h_output = new float[len];

  // 初始化输入矩阵
  std::srand(1);
  for (int i = 0; i < len; ++i) {
    h_input[i] = (std::rand() % 1000) * 0.01f;
  }

  // 打印初始数据
  for (int i=0; i<8; i++) {
    for (int j=0; j<16; j++) {
      printf("%.7f ", h_input[i*16 + j]);
    }
    printf("\n");
  }
  printf("\n\n");

  // 分配GPU内存
  float *d_input, *d_output;
  hipMalloc(&d_input, len * sizeof(float));
  hipMalloc(&d_output, len * sizeof(float));

  // 将数据从CPU复制到GPU
  hipMemcpy(d_input, h_input, len * sizeof(float), hipMemcpyHostToDevice);

  // 启动预编译内核
  void* args[] = {&d_input, &d_output};
  hipModuleLaunchKernel(kernel, 
                      2, 1, 1,    // gridDim
                      64, 1, 1,     // blockDim
                      0, 0,         // sharedMemBytes, stream
                      args, nullptr); // extra
  
  // 同步设备
  hipError_t syncResult = hipDeviceSynchronize();
  if (syncResult != hipSuccess) {
    printf("Device synchronization failed: %s\n", hipGetErrorString(syncResult));
    hipModuleUnload(module);
    hipCtxDestroy(context);
    return 1;
  }

  // 获取结果
  hipMemcpy(h_output, d_output, len * sizeof(float), hipMemcpyDeviceToHost);
  printf("llvm compile result:\n");
  for (int i=0; i<8; i++) {
    printf("%.7f ", h_output[i*16]);
  }
  printf("\n\n");

  // 启动动态编译内核
  dim3 gridSize1(2);
  dim3 blockSize1(64);
  hipLaunchKernelGGL(reduce, 
                    gridSize1, blockSize1, 
                    0, 0,      // sharedMemBytes, stream
                    d_input, d_output);
  
  // 再次同步和获取结果
  hipDeviceSynchronize();
  hipMemcpy(h_output, d_output, len * sizeof(float), hipMemcpyDeviceToHost);
  printf("hip compile result:\n");
  for (int i=0; i<8; i++) {
    printf("%.7f ", h_output[i*16]);
  }
  printf("\n\n");

  // 清理资源
  hipModuleUnload(module);
  hipCtxDestroy(context);
  delete[] h_input;
  delete[] h_output;
  hipFree(d_input);
  hipFree(d_output);

  return 0;
}