#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

// cublas gemm
cublasStatus_t cublasMatMulTransA(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K, bool isTranA, bool isTranB) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status;
    bool handleCreated = false;
    if (!handle) {
        status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
        handleCreated = true;
    }
    cublasOperation_t tranA, tranB;
    int lda, ldb;
    if (isTranA && isTranB) {
        tranA = CUBLAS_OP_N; tranB = CUBLAS_OP_N;
        lda = M; ldb = K;
    } else if (!isTranA && isTranB) {
        tranA = CUBLAS_OP_T; tranB = CUBLAS_OP_N;
        lda = K; ldb = K;
    } else if (isTranA && !isTranB) {
        tranA = CUBLAS_OP_N; tranB = CUBLAS_OP_T;
        lda = M; ldb = N;
    } else {
        tranA = CUBLAS_OP_T; tranB = CUBLAS_OP_T;
        lda = K; ldb = N;
    }
    status = cublasSgemm(handle,
                        tranA,   // A不转置
                        tranB,   // B转置
                        M,             // 结果矩阵行数
                        N,             // 结果矩阵列数
                        K,             // 公共维度
                        &alpha,
                        A, lda,          // A的维度M×K，lda=M
                        B, ldb,          // B的维度NxK，ldb=N
                        &beta,
                        C, M);         // C的维度M×N，ldc=M

    // 清理临时创建的句柄
    if (handleCreated) {
        cublasDestroy(handle);
    }
    return status;
}

// gpu 验证
__global__ void verify_kernel(float* C, float* D, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
      float sub = C[row * n + col] - D[col * m + row];
      if (sub >= 0.0000001f || sub <= -0.0000001f) {
        // printf("%d %d\n", row, col);
        printf("error!\nindex: (y=%d, x=%d)\nmine: %f  verify: %.8f\nsub: %.8f\n", row, col, C[row * n + col], D[col * m + row], sub);
      }
    }
}

// nvcc -o ./bin/runkernel runkernel.cu -lcuda -lcublas -arch=sm_80
int main(int argc, char** argv) {
    if (argc <= 8){
        std::cout << "Usage : M N K gridDims blockDims shmBytes cubinPath cubinFunc" << std::endl;
        return 1;
    }
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
    CUresult result = cuModuleLoad(&module, argv[7]);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to load module: " << result << std::endl;
        return -1;
    }

    // 获取内核函数句柄
    CUfunction kernel;
    result = cuModuleGetFunction(&kernel, module, argv[8]);
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
    dim3 dimGrid = {N  / dimBlock.x, M  / dimBlock.y};

    // 调用核函数
    cudaError_t err;
    cuLaunchKernel(kernel, std::stoi(argv[4]), 1, 1, std::stoi(argv[5]), 1, 1, std::stoi(argv[6]), 0, args, NULL);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("设备同步失败: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasMatMulTransA(handle, d_A, d_B, d_D, M, N, K, true, false);
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