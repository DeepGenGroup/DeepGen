# blockdims = (64, 1, 1)
# griddims = (1024, 1, 1)
# ========= hsacoPath =  /tmp/compile-ptx-src-37f8ee.cubin
# ========= kernelName =  GEMM_bMNK1x1024x1024x1024_DTabcfloat32xfloat32xfloat32_AT1_TTmn4x4_BTmnk32x32x8_BLmn1x1_WLmn8x8_GLWab4x4_GSW2_WSWab2x2_TSWab2x2_LSU1_BM16_UNROLL4_REGP0_SHMP0_LC0_RC0_

m=1024
n=1024
k=1024
griddim=1024
blockdim=64
shmBytes=4096  # (bm + bn)*2*bk * sizeof(float)=  64*2*8*4 
func=GEMM_bMNK1x1024x1024x1024_DTabcfloat32xfloat32xfloat32_AT1_TTmn4x4_BTmnk32x32x8_BLmn1x1_WLmn8x8_GLWab4x4_GSW2_WSWab2x2_TSWab2x2_LSU1_BM16_UNROLL4_REGP0_SHMP0_LC0_RC0_
path=/tmp/compile-ptx-src-37f8ee.cubin

/home/xushilong/DeepGen/_TempCodes/RunKernelCUDA $m $n $k $griddim $blockdim $shmBytes $path  $func