
#include <stdio.h>
#include <stdlib.h>
#define __USE_GNU     //使用RTLD_DEFAULT和RTLD_NEXT宏需定义
#include <dlfcn.h>
 
typedef size_t (*strlen_t)(const char *); 
strlen_t strlen_f = NULL, strlen_f1 = NULL;
 
#define M 1024
#define N 1024
#define K 1024

// g++   -ldl -L/home/xushilong/rocm-llvm-install/lib -lmlir_cuda_runtime  main.cpp -o caller
int main(int argc, char **argv)
{
    // libhello.so是我们自己封装的一个测试的共享库文件
    // RTLD_LAZY 表示在对符号引用时才解析符号，但只对函数符号引用有效，而对于变量符号的引用总是在加载该动态库的时候立即绑定解析
    void *handle = dlopen("/home/xushilong/DeepGen/_TempCodes/output.so", RTLD_LAZY);  
    if(!handle) {
        printf("open failed: %s\n", dlerror());
        return 1;
    }   
    void *p = dlsym(handle, "GEMM_testKernel");  //argv[2]对应输入需获取地址的符号名
    if(!p) {
        printf("load failed: %s\n", dlerror());
        return 1;
    }
    float* a = new float[M*K];
    float* b = new float[K*N];
    float* c = new float[M*N];
    void (*fp)(float*,float*,float*) = (void (*)(float*,float*,float*))p;
    fp(a,b,c);
    dlclose(handle);
    return 0;
}