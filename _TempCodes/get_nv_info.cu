#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", 
               deviceProp.major, deviceProp.minor);
        printf("  Shared Memory Per SM (bytes): %zu\n", 
               deviceProp.sharedMemPerMultiprocessor);
        printf("  Shared Memory Per Block (bytes): %zu\n", 
               deviceProp.sharedMemPerBlock);
        
        // 获取最大动态共享内存大小（如果设备支持）
        int max_dynamic_shared_size;
        cudaDeviceGetAttribute(&max_dynamic_shared_size, 
                              cudaDevAttrMaxSharedMemoryPerBlockOptin, 
                              dev);
        printf("  Max Dynamic Shared Memory Per Block (bytes): %d\n", 
               max_dynamic_shared_size);
    }
    
    return 0;
}
