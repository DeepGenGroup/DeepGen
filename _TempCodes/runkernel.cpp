#include <iostream>
#include <fstream>
#include <sstream>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>

template <typename T>
void display(T *host, int len) {
    // 打印
    int mid = len / 2;
    int start = (rand() % (mid - 1)) + 1;
    int end = (rand() % (mid - 1)) + mid + 1;
    std::cout << "{" << host[0] << ", ..., " << host[start] << ", ..., "  << host[mid] << ", ..., "  << host[end] << ", ..., " << host[len - 1] << "}\n";
}

__global__ void gemm_kernel(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[i*m+row] * B[i*n+col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void verify_kernel(float* C, float* D, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
      float sub = C[row * n + col] - D[row * n + col];
      if (sub >= 0.01f || sub <= -0.01f) {
        printf("error index: (y=%d, x=%d)\nerrer mine: %f   error verify: %f\nsub: %f\n", row, col, C[row * n + col], D[row * n + col], C[row * n + col]-D[row * n + col]);
      }
    }
}

void freeBufs(std::vector<float*> hostBufs, std::vector<float*> deviceBufs) {
  // 清理内存
  for (int i=0; i<hostBufs.size(); i++) {
    hipFree(deviceBufs[i]);
    delete[] hostBufs[i];
  }
}


int main(int argc, char** argv) {

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
  hipModuleLoad(&module, argv[6]);
  // 获取内核函数
  hipFunction_t kernel;
  hipError_t error = hipModuleGetFunction(&kernel, module, argv[7]);
  if (error == hipSuccess) {
    std::cout << "Successfully obtained the function handle." << std::endl;
  } else {
    std::cerr << "Error in obtaining the function handle: " << hipGetErrorString(error) << std::endl;
    return 0;
  }

  int M = std::stoi(argv[1]);
  int N = std::stoi(argv[2]);
  int K = std::stoi(argv[3]);

  float *A = new float[M * K];
  float *B = new float[N * K];
  float *C = new float[N * M];
  float *D = new float[N * M];
  for (int i = 0; i < M * K; i++) {
      A[i] = (rand() % 1000) * 0.01f;
      // A[i] = 1.0f;
  } 
  for (int i = 0; i < N * K; i++) {
      B[i] = (rand() % 1000) * 0.01f;
      // B[i] = 1.0f;
  }

  float *d_A, *d_B, *d_C, *d_D;
  hipMalloc(&d_A, M * K * sizeof(float));
  hipMalloc(&d_B, K * N * sizeof(float));
  hipMalloc(&d_C, M * N * sizeof(float));
  hipMalloc(&d_D, M * N * sizeof(float));

  hipMemcpy(d_A, A, M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice);

  void* args[] = {&d_A, &d_B, &d_C};
  dim3 dimBlock = {16, 16};
  dim3 dimGrid = {(N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y};

  for(int i=0;i<7;++i){
    hipModuleLaunchKernel(kernel, std::stoi(argv[4]), 1, 1, std::stoi(argv[5]), 1, 1, 2048, 0, args, NULL);  
  }
  gemm_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_D, M, N, K);
  verify_kernel<<<dimGrid, dimBlock>>>(d_C, d_D, M, N);

  freeBufs({A, B, C, D}, {d_A, d_B, d_C, d_D});
  hipModuleUnload(module);
  return 0;
}