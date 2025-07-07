#include <hip/hip_runtime.h>

__global__ void kernel() {
    int lane_id = threadIdx.x % 64;  // AMD wavefront=64
    int value = 10 + lane_id;

    // 从 lane_id=2 的线程读取数据
    int shuffled = __shfl(value, 2, 64); 
}