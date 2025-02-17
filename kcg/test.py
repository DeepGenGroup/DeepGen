# from Runtime.python.kcg.Loader import driver
# if __name__ == "__main__":
#     print("hello")
#     print(driver)
#     print(driver.loader)

from typing import List,Type
from kcg.Utils import *
from kcg.Kernel import *
from kcg.CompiledKernelFactory import *
from kcg.Operators import matmul
import sys
import numpy as np
from kcg.KCGCompiler import KCGCompiler
# import pymlir

############### User config ###############


def case_normal_0(kp_matmul: KernelArgMatmul) :
    #   {KEY_BLOCK_SIZE_M, 64}, {KEY_BLOCK_SIZE_N, 48}, {KEY_BLOCK_SIZE_K, 32}, {KEY_THREAD_SIZE_M, 4}, {KEY_THREAD_SIZE_N, 6}, 
    #   {KEY_GLOB_LOAD_WIDTH_A, 2}, {KEY_GLOB_LOAD_WIDTH_B, 2}, 
    #   {KEY_BLOCK_LAYOUT_M, 2}, {KEY_BLOCK_LAYOUT_N, 1}, {KEY_WARP_LAYOUT_M, 8}, {KEY_WARP_LAYOUT_N, 8},
    #   {KEY_WARP_SCATTER_WIDTH_A, 2}, {KEY_WARP_SCATTER_WIDTH_B, 2}, {KEY_THREAD_SCATTER_WIDTH_A, 2}, {KEY_THREAD_SCATTER_WIDTH_B, 2}, 
    #   {KEY_LOCAL_SPLIT_U, 2}, {KEY_BLOCK_MAPPING, 8}, {KEY_WARP_SIZE, 64}, {KEY_GLOB_STORE_WIDTH, 2}, 
    #   {KEY_DTYPE_A, (int)KcgDtype::float32},
    #   {KEY_DTYPE_B, (int)KcgDtype::float32},
    #   {KEY_DTYPE_C, (int)KcgDtype::float32},
    #   {KEY_M, 1024},{KEY_N, 1056},{KEY_K, 1024}, 
    #   {KEY_IS_A_TRANSPOSE, 1}
    
    kp_matmul.BLOCK_SIZE_M = 64
    kp_matmul.BLOCK_SIZE_N = 48
    kp_matmul.BLOCK_SIZE_K = 32
    kp_matmul.THREAD_SIZE_M = 4
    kp_matmul.THREAD_SIZE_N = 6
    kp_matmul.GLOB_LOAD_WIDTH_A = 2
    kp_matmul.GLOB_LOAD_WIDTH_B = 2
    kp_matmul.BLOCK_LAYOUT_M = 2
    kp_matmul.BLOCK_LAYOUT_N = 1
    kp_matmul.WARP_LAYOUT_M = 8
    kp_matmul.WARP_LAYOUT_N = 8
    kp_matmul.WARP_SCATTER_WIDTH_A = 2
    kp_matmul.WARP_SCATTER_WIDTH_B = 2
    kp_matmul.THREAD_SCATTER_WIDTH_A = 2
    kp_matmul.THREAD_SCATTER_WIDTH_B = 2
    kp_matmul.LOCAL_SPLIT_U = 2
    kp_matmul.BLOCK_MAPPING = 8
    kp_matmul.WARP_SIZE = 64
    kp_matmul.GLOB_STORE_WIDTH = 2
    kp_matmul.isATranspose = 1
    global m_len
    global n_len
    global k_len   
    kp_matmul.M = 1024
    kp_matmul.N = 1056
    kp_matmul.K = 1024


def case_normal_1(kp_matmul: KernelArgMatmul) :
    #   {KEY_BLOCK_SIZE_M, 64}, {KEY_BLOCK_SIZE_N, 48}, {KEY_BLOCK_SIZE_K, 32}, {KEY_THREAD_SIZE_M, 4}, {KEY_THREAD_SIZE_N, 6}, 
    #   {KEY_GLOB_LOAD_WIDTH_A, 2}, {KEY_GLOB_LOAD_WIDTH_B, 2}, 
    #   {KEY_BLOCK_LAYOUT_M, 2}, {KEY_BLOCK_LAYOUT_N, 1}, {KEY_WARP_LAYOUT_M, 8}, {KEY_WARP_LAYOUT_N, 8},
    #   {KEY_WARP_SCATTER_WIDTH_A, 2}, {KEY_WARP_SCATTER_WIDTH_B, 2}, {KEY_THREAD_SCATTER_WIDTH_A, 2}, {KEY_THREAD_SCATTER_WIDTH_B, 2}, 
    #   {KEY_LOCAL_SPLIT_U, 2}, {KEY_BLOCK_MAPPING, 8}, {KEY_WARP_SIZE, 64}, {KEY_GLOB_STORE_WIDTH, 2}, 
    #   {KEY_DTYPE_A, (int)KcgDtype::float32},
    #   {KEY_DTYPE_B, (int)KcgDtype::float32},
    #   {KEY_DTYPE_C, (int)KcgDtype::float32},
    #   {KEY_M, 1024},{KEY_N, 1056},{KEY_K, 1024}, 
    #   {KEY_IS_A_TRANSPOSE, 1}
    
    kp_matmul.BLOCK_SIZE_M = 64
    kp_matmul.BLOCK_SIZE_N = 64
    kp_matmul.BLOCK_SIZE_K = 32
    kp_matmul.THREAD_SIZE_M = 4
    kp_matmul.THREAD_SIZE_N = 4
    kp_matmul.GLOB_LOAD_WIDTH_A = 4
    kp_matmul.GLOB_LOAD_WIDTH_B = 4
    kp_matmul.BLOCK_LAYOUT_M = 4
    kp_matmul.BLOCK_LAYOUT_N = 1
    kp_matmul.WARP_LAYOUT_M = 4
    kp_matmul.WARP_LAYOUT_N = 16
    kp_matmul.WARP_SCATTER_WIDTH_A = 2
    kp_matmul.WARP_SCATTER_WIDTH_B = 2
    kp_matmul.THREAD_SCATTER_WIDTH_A = 2
    kp_matmul.THREAD_SCATTER_WIDTH_B = 2
    kp_matmul.LOCAL_SPLIT_U = 1
    kp_matmul.BLOCK_MAPPING = 8
    kp_matmul.WARP_SIZE = 64
    kp_matmul.GLOB_STORE_WIDTH = 4
    kp_matmul.isATranspose = 1
    kp_matmul.M = 1024
    kp_matmul.N = 1024
    kp_matmul.K = 1024


def case_bad_0(kp_matmul: KernelArgMatmul) :
    # Cijk_Ailk_Bljk_SB_MT256x32x8_SN_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL1_DTL0_ETSP_EPS1_FL0_GRVW4_GSU1_GSUAMB_ISA906_IU1_K1_KLA_LPA0_LPB0_LDL1_LRVW4_MAC_MDA2_NLCA1_NLCB1_ONLL1_PK0_PGR1_PLR1_RK0_SU32_SUM0_SUS256_SVW4_SNLL0_TT8_4_USFGROn1_VAW1_VSn1_VW4_WG32_8_1_WGM1
    kp_matmul.BLOCK_SIZE_M= 256  # 256
    kp_matmul.BLOCK_SIZE_N= 32 # 32
    kp_matmul.BLOCK_SIZE_K= 8  # 8
    kp_matmul.THREAD_SIZE_M= 8
    kp_matmul.THREAD_SIZE_N= 4  # th=32*8
    kp_matmul.VECTORIZE_WIDTH= 4  # 数字的个数
    kp_matmul.BLOCK_LAYOUT_M= 2
    kp_matmul.BLOCK_LAYOUT_N= 2
    kp_matmul.WARP_LAYOUT_M= 16
    kp_matmul.WARP_LAYOUT_N= 4
    kp_matmul.WARP_SIZE= 64
    kp_matmul.isATranspose = 0

m_len=1024  # 16 blocks
n_len=1024  # 16 blocks
k_len=1024   # 8 blocks
kp_matmul = KernelArgMatmul(m_len,n_len,k_len,
    EnumKernelDType.float32 ,
    EnumKernelDType.float32 ,
    EnumKernelDType.float32
    )

# case_normal_1(kp_matmul)
case_normal_0(kp_matmul)
# case_bad_0(kp_matmul)

def compare_with_error(tensor1, tensor2, abs_error=1e-2, rel_error=1e-2):
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor1) + 1e-5)  # 避免除以零的情况

    # 比较绝对误差和相对误差
    error_mask = (abs_diff > abs_error) & (rel_diff > rel_error)
    diff_elements = torch.sum(error_mask).item()
    max_error = torch.max(torch.abs(tensor1 - tensor2))
    return diff_elements, max_error

def test_correctness(kpm : KernelArgMatmul):
    kernelCompiler = KCGCompiler()
    kernelCompiler.set_config_json_path('/home/xushilong/CodeGenDemo/.vscode/c_cpp_properties.json')
    hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ = kernelCompiler.compileKernel(kpm)
    
    print("===== test info ========")
    print("hsacoPath = ", hsacoPath)
    print("kernelName = ", kernelName)
    print("gridDimX = ", gridDimX)
    print("gridDimY = ", gridDimY)
    print("gridDimZ = ", gridDimZ)
    print("blockDimX = ", blockDimX)
    print("blockDimY = ", blockDimY)
    print("blockDimZ = ", blockDimZ)    
    
    print("========= hsacoPath = ",hsacoPath)
    print("========= kernelName = ",kernelName)
    ###  ====  DBG：使用外部mlir调试   
    # funName = 'Matmul_m1024n1024k1024'
    # kernelName = 'GEMM_mnk1024x1024x512_f32f32f32_TTmn4x4_BTmnk64x64x16'

    inConfig = UserInputs(hsacoPath,kernelName,kpm)
    inConfig.m_gridDims = [gridDimX,gridDimY,gridDimZ]
    inConfig.m_blockDims = [blockDimX,blockDimY,blockDimZ]
    inConfig.operatorKind = EnumOperator.Matmul
    packedKernel = CompiledKernelFactory.getKernel(inConfig)
    
    a = torch.randn(kpm.M,kpm.K,dtype=inConfig.kernelParam.dtypeTorch('A'),device='cuda:0')
    b = torch.randn(kpm.K,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('B'),device='cuda:0')
    c = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda:0')
    d = torch.empty(kpm.M,kpm.N,dtype=inConfig.kernelParam.dtypeTorch('C'),device='cuda:0')
    atrans = torch.transpose(a,1,0).contiguous()  # 转置会令底层存储不连续，导致失败。必须使其连续
    assert(a.is_contiguous())
    assert(b.is_contiguous())
    assert(atrans.is_contiguous())
    print(inConfig.kernelParam.dtypeTorch('A'))
    print(inConfig.kernelParam.dtypeTorch('B'))
    print(inConfig.kernelParam.dtypeTorch('C'))
    M, K = a.shape
    K, N = b.shape
    print(f"python: M,N,K = {M},{N},{K}")
    print("conti: a",a.is_contiguous())
    print("conti: b",b.is_contiguous())
    print("conti: atrans",atrans.is_contiguous())
    res = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    benchmarkCount = 1
    
    if kpm.isATranspose :
        print('transpose test')
        # packedKernel.run(atrans,b,c) # warm up
        for i in range(0,benchmarkCount) : # benchmark
            start_event.record()
            packedKernel.run(atrans,b,c)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            res.append(elapsed_time)
    else:
        print('normal test')
        # packedKernel.run(a,b,c) # warm up
        for i in range(0,benchmarkCount) : # benchmark
            start_event.record()
            packedKernel.run(a,b,c)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            res.append(elapsed_time)
    print(f"codegenDemo median time: {np.median(res)} ms")
    # o.run(a,b,c,
    #       M,N,K, a.stride(0), a.stride(1),  
    #         b.stride(0), b.stride(1),  
    #         c.stride(0), c.stride(1),  
    #     )
    res1 = []
    # torch.matmul(a,b) # warm up
    for i in range(0,benchmarkCount) : # benchmark
        start_event.record()
        d = torch.matmul(a,b)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        res1.append(elapsed_time)
    print(f"rocblas median time: {np.median(res1)} ms")
    print(f"speed up: {np.median(res1)/np.median(res)}")
    print("c=",c)
    print("d=",d)
    if torch.allclose(c,d,atol=1e-2,rtol=1e-2):
        print('test correct!')
    else:
        diff,max_error= compare_with_error(d,c)
        print(f'test fail! maxerror={max_error}, diff=[{diff} / {M*N}], diffrate={diff/(M*N)}')
        # hipprof --pmc python ./test.py 

kp_matmul.check()
test_correctness(kp_matmul)
