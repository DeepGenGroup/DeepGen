#include <iostream>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <algorithm>
#include <vector>
#include <nvrtc.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <cuda.h>

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

// nvcc -o ./bin/runkernel runkernel.cu -lcuda
int main(int argc, char** argv) {
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
    CUresult result = cuModuleLoad(&module, argv[6]);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to load module: " << result << std::endl;
        return -1;
    }

    // 获取内核函数句柄
    CUfunction kernel;
    result = cuModuleGetFunction(&kernel, module, argv[7]);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to get kernel function" << std::endl;
        return -1;
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
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_D, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    void* args[] = {&d_A, &d_B, &d_C};
    dim3 dimBlock = {16, 16};
    dim3 dimGrid = {(N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y};

    // 调用核函数
    if (!cuLaunchKernel(kernel, std::stoi(argv[4]), 1, 1, std::stoi(argv[5]), 1, 1, 16384, 0, args, NULL)) {
        return 1;
    }
    gemm_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_D, M, N, K);
    verify_kernel<<<dimGrid, dimBlock>>>(d_C, d_D, M, N);

    // 同步设备
    cuModuleUnload(module);
    cuCtxDestroy(context);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;

    return 0;
}