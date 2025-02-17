#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define CHECK_HIP(err) (hipErrorCheck(err, __FILE__, __LINE__))
#define CHECK_ROCBLAS(err) (rocblasCheckError(err, __FILE__, __LINE__))

inline void hipErrorCheck(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void rocblasCheckError(rocblas_status status, const char* file, int line) {
    if (status != rocblas_status_success) {
        std::cerr << "rocBLAS Error: " << status << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int size = 1024;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 创建rocBLAS句柄
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    // 分配主机内存
    std::vector<float> h_A(size * size, 1.0f);  // 初始化为1
    std::vector<float> h_B(size * size, 1.0f);  // 初始化为1
    std::vector<float> h_C(size * size, 0.0f);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size * size * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_B, size * size * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C, size * size * sizeof(float)));

    // 拷贝数据到设备
    CHECK_HIP(hipMemcpy(d_A, h_A.data(), size * size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), size * size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_C, h_C.data(), size * size * sizeof(float), hipMemcpyHostToDevice));

    // 执行GEMM运算
    for(int i=0;i<7;++i){

    CHECK_ROCBLAS(rocblas_sgemm(handle,
                               transA,
                               transB,
                               size,    // m
                               size,    // n
                               size,    // k
                               &alpha,
                               d_A,     // A矩阵
                               size,    // lda
                               d_B,     // B矩阵
                               size,    // ldb
                               &beta,
                               d_C,     // C矩阵
                               size));  // ldc
    }

    // 等待计算完成
    CHECK_HIP(hipDeviceSynchronize());

    // 拷贝结果回主机
    CHECK_HIP(hipMemcpy(h_C.data(), d_C, size * size * sizeof(float), hipMemcpyDeviceToHost));

    // 清理资源
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    CHECK_ROCBLAS(rocblas_destroy_handle(handle));

    std::cout << "GEMM计算完成！" << std::endl;

    // 简单验证结果（此处可扩展为更详细的验证）
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "C[1023][1023] = " << h_C.back() << std::endl;

    return 0;
}