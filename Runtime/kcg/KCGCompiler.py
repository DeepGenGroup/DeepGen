
from kcg.Utils import PathManager
from Operators.matmul import *


class KCGCompiler :
    def __init__(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("KCGCompiler", PathManager.kcg_compiler_path())
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.__compile_kernel_matmul = mod.compile_kernel_matmul
    
    def compileKernel(self, param : KernelArgMatmul) -> list:
        return self.__compile_kernel_matmul(
            param.BLOCK_SIZE_M,
            param.BLOCK_SIZE_N,
            param.BLOCK_SIZE_K,
            param.THREAD_SIZE_M,
            param.THREAD_SIZE_N,
            param.WARP_SIZE,
            param.BLOCK_LAYOUT_M,
            param.BLOCK_LAYOUT_N,
            param.WARP_LAYOUT_M,
            param.WARP_LAYOUT_N,
            param.GLOB_LOAD_WIDTH_A,
            param.GLOB_LOAD_WIDTH_B,
            param.WARP_SCATTER_WIDTH_A,
            param.WARP_SCATTER_WIDTH_B,
            param.THREAD_SCATTER_WIDTH_A,
            param.THREAD_SCATTER_WIDTH_B,
            param.LOCAL_SPLIT_U,
            param.BLOCK_MAPPING,
            param.GLOB_STORE_WIDTH,
            param.UNROLL_NUM,
            param.REG_PREFETCH,
            param.SHARED_PREFETCH,
            param.LOAD_CONTINUOUS,
            param.REDUCE_C_CONTINUOUS,
            param.dtype('A'),
            param.dtype('B'),
            param.dtype('C'),
            param.M,param.N,param.K,
            param.isATranspose
        )