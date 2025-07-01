#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <iostream>

template <typename T>
void display(T *host, int len) {
    // 打印
    int mid = len / 2;
    int start = (rand() % (mid - 1)) + 1;
    int end = (rand() % (mid - 1)) + mid + 1;
    std::cout << "{" << host[0] << ", ..., " << host[start] << ", ..., "  << host[mid] << ", ..., "  << host[end] << ", ..., " << host[len - 1] << "}\n";
}

void _launch(CUdeviceptr arg0, CUdeviceptr arg1, CUdeviceptr arg2) {
  void* args[] = { &arg0, &arg1, &arg2 };
  CUmodule module;
  CUfunction kernel_fn;
  cuModuleLoad(&module, "/tmp/compile-ptx-src-123704.cubin");
  cuModuleGetFunction(&kernel_fn, module, "matmul");
  cuLaunchKernel(
    kernel_fn,
    64, 1, 1,
    256, 1, 1,
    16384, nullptr, args, nullptr
  );
}

int main() {
  const int M = 1024;
  const int N = 1024;
  const int K = 128;
  float *A = new float[M * K];
  float *B = new float[N * K];
  float *C = new float[N * M];

  for (int i = 0; i < M * K; i++) {
      A[i] = (rand() % 1000) * 0.01f;
      // A[i] = 1.0f;
  } 
  for (int i = 0; i < N * K; i++) {
      B[i] = (rand() % 1000) * 0.01f;
      // B[i] = 1.0f;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  _launch(reinterpret_cast<CUdeviceptr>(d_A), 
          reinterpret_cast<CUdeviceptr>(d_B),
          reinterpret_cast<CUdeviceptr>(d_C));
  
  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
  display(C, M*N);

}