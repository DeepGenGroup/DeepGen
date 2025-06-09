griddim=1024
blockdim=64
shm=4096
kernel=/tmp/kcg_kernel-7e9215/kcg_kernel-7e9215.hsaco
func=kcg_MM_b1M1024N1024K1024isAT1W64_BM32BN32BK16TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM1BSWN1WSWM1WSWN1LSU1Map8GSW0UN8RP0SP0LC0RC0
# ./RunKernel 1024 1024 1024 $griddim $blockdim $kernel $func  $shm
./RunKernel 1024 1024 1024 256 64 /tmp/kcg_kernel-032c0b.hsaco GEMM_testKernel 4096
# [lib] ============ 7
# blockdims = (64, 1, 1)
# griddims = (1024, 1, 1)
# ========= hsacoPath =  /tmp/kcg_kernel-7e9215/kcg_kernel-7e9215.hsaco
# ========= kernelName =  kcg_MM_b1M1024N1024K1024isAT1W64_BM32BN32BK16TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM1BSWN1WSWM1WSWN1LSU1Map8GSW0UN8RP0SP0LC0RC0
# ==== backend is HIP
# ==== shmBytes is 4096