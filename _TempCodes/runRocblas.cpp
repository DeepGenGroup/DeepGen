#include <iostream>
#include <vector>
#include <cstdlib>      // rand, srand
#include <ctime>        // time
#include <cmath>        // fabs
#include <algorithm>    // std::sort
#include <hip/hip_runtime.h>
#include <rocblas.h>

// 简单的检查宏
#define CHECK_ROCBLAS_ERROR(status)                                \
    if (status != rocblas_status_success) {                        \
        std::cerr << "rocBLAS error: " << status << std::endl;     \
        exit(EXIT_FAILURE);                                        \
    }

#define CHECK_HIP_ERROR(status)                                    \
    if (status != hipSuccess) {                                    \
        std::cerr << "HIP error: " << status << std::endl;         \
        exit(EXIT_FAILURE);                                        \
    }

// 在CPU端进行列优先的矩阵乘法 C = alpha*A*B + beta*C
// A的尺寸(M,K)，B的尺寸(K,N)，C的尺寸(M,N)，都采用列优先
// lda=M, ldb=K, ldc=M
void cpu_gemm_colmajor(const float* A, const float* B, float* C,
                       int M, int N, int K,
                       float alpha, float beta)
{
    // C_ij = alpha * sum_k( A_ik * B_kj ) + beta * C_ij
    // 注意列优先索引: A_ik 在A中的位置: k*M + i
    //                  B_kj 在B中的位置: j*K + k
    //                  C_ij 在C中的位置: j*M + i
    for(int j = 0; j < N; j++)
    {
        for(int i = 0; i < M; i++)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; k++)
            {
                sum += A[k * M + i] * B[j * K + k];
            }
            // 原C_ij上再乘 beta
            C[j * M + i] = alpha * sum + beta * C[j * M + i];
        }
    }
}

int main()
{
    // 这里示范矩阵维度
    // A: M x K
    // B: K x N
    // C: M x N
    // 令 M=N=K=1024
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // 创建 rocblas 句柄
    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    // 申请 Host 端空间 (列优先)
    // A, B 初始化为随机值，C 初始化为0
    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);
    std::vector<float> hC(M * N, 0.0f);      // 存放GPU计算结果
    std::vector<float> hC_cpu(M * N, 0.0f);  // 存放CPU计算结果做对比

    // 用时间做随机种子
    srand(static_cast<unsigned>(time(NULL)));

    // 随机初始化 A 和 B (列优先)，值范围 [0,1)
    for(int col = 0; col < K; col++)
    {
        for(int row = 0; row < M; row++)
        {
            hA[col * M + row] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for(int col = 0; col < N; col++)
    {
        for(int row = 0; row < K; row++)
        {
            hB[col * K + row] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // CPU 参考结果: C_cpu = A * B (列优先)
    // alpha=1, beta=0
    std::cout << "Computing reference result on CPU (naive O(M*N*K) gemm) ...\n";
    cpu_gemm_colmajor(hA.data(), hB.data(), hC_cpu.data(), M, N, K, 1.0f, 0.0f);

    // Device 端指针
    float *dA, *dB, *dC;

    // 为 A, B, C 分别分配显存
    CHECK_HIP_ERROR(hipMalloc(&dA, M * K * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dB, K * N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dC, M * N * sizeof(float)));

    // 拷贝数据到 Device 端
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), M * N * sizeof(float), hipMemcpyHostToDevice));

    // alpha, beta
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 列优先时leading dimension
    rocblas_operation transA = rocblas_operation_none;
    rocblas_operation transB = rocblas_operation_none;
    int lda = M;  // 因为列优先，A 的 leading dimension = M
    int ldb = K;  // B 的 leading dimension = K
    int ldc = M;  // C 的 leading dimension = M

    //------------------------------------------------------------------------------
    // Warm-up 一次
    //------------------------------------------------------------------------------
    std::cout << "Warm-up (1 time) ...\n";
    CHECK_ROCBLAS_ERROR(
        rocblas_sgemm(
            handle,
            transA,
            transB,
            M,    // number of rows of op(A) and C
            N,    // number of columns of op(B) and C
            K,    // number of columns of op(A) and rows of op(B)
            &alpha,
            dA, lda,
            dB, ldb,
            &beta,
            dC, ldc
        )
    );
    // 保证Warm-up执行完毕
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    //------------------------------------------------------------------------------
    // 正式测试 20 次，记录耗时 (单位 ms)
    //------------------------------------------------------------------------------
    const int numRuns = 20;
    std::vector<float> times(numRuns, 0.0f);

    // 创建HIP事件用于计时
    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    std::cout << "Benchmark (20 times) using hipEventRecord ...\n";
    for(int i = 0; i < numRuns; i++)
    {
        // C清零（或可在循环外部只清零一次，然后Beta=0即可覆盖旧值，但这里更直观）
        CHECK_HIP_ERROR(hipMemset(dC, 0, M * N * sizeof(float)));

        // 记录起始事件
        CHECK_HIP_ERROR(hipEventRecord(start, 0));

        // 调用 sgemm: C = alpha * A * B + beta * C
        CHECK_ROCBLAS_ERROR(
            rocblas_sgemm(
                handle,
                transA,
                transB,
                M,
                N,
                K,
                &alpha,
                dA, lda,
                dB, ldb,
                &beta,
                dC, ldc
            )
        );

        // 记录结束事件
        CHECK_HIP_ERROR(hipEventRecord(stop, 0));
        // 等待事件结束
        CHECK_HIP_ERROR(hipEventSynchronize(stop));

        // 计算耗时
        float elapsedMs = 0.0f;
        CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedMs, start, stop));
        times[i] = elapsedMs;
    }

    // 取中位数
    std::sort(times.begin(), times.end());
    float medianTime = (times.size() % 2 == 1)
                       ? times[times.size() / 2]
                       : 0.5f * (times[times.size()/2 - 1] + times[times.size()/2]);

    std::cout << "Median of 20 runs: " << medianTime << " ms\n";

    // 此时 dC 中是最后一次 sgemm 的结果
    // 拷回GPU结果到 hC
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // 销毁句柄与事件，释放显存
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    //------------------------------------------------------------------------------
    // 评估 GPU 结果 与 CPU 结果 的差异
    //------------------------------------------------------------------------------
    double maxError = 0.0;
    double sumError = 0.0;
    double sumRef   = 0.0;

    for(int i = 0; i < M * N; i++)
    {
        float cpuVal = hC_cpu[i];
        float gpuVal = hC[i];
        double diff  = std::fabs(cpuVal - gpuVal);
        if(diff > maxError)
            maxError = diff;
        sumError += diff;
        sumRef   += std::fabs(cpuVal);
    }

    double avgError = sumError / (M * N);
    double relativeError = (sumRef > 1e-12) ? sumError / sumRef : 0.0;

    std::cout << "\nComparison of CPU and GPU results:\n";
    std::cout << "  Max absolute error:      " << maxError << "\n";
    std::cout << "  Average absolute error:  " << avgError << "\n";
    std::cout << "  Relative error (sumError/sumRef): " << relativeError << "\n";

    // 如果想要简单判断是否“正确”，可以设定一个阈值:
    if(maxError < 1e-3f)
        std::cout << "Results are probably correct (maxError < 1e-3)\n";
    else
        std::cout << "Results differ too much from CPU reference.\n";

    return 0;
}
