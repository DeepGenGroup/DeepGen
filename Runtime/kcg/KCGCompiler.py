
# from kcg.Utils import PathManager
# from Operators.matmul import *
# from attn_FP32_test import *

# class KCGCompiler :
    
    
#     def __init__(self):
#         import importlib.util
#         spec = importlib.util.spec_from_file_location("KCGCompiler", PathManager.kcg_compiler_path())
#         mod = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(mod)
#         self.__compile_kernel_matmul = mod.compile_kernel_matmul
#         self.set_platform = mod.set_platform

#         attn_spec = importlib.util.spec_from_file_location("attention", PathManager.kcg_compiler_attention_path())
#         attn_mod = importlib.util.module_from_spec(attn_spec)
#         attn_spec.loader.exec_module(attn_mod)
#         self.__compile_kernel_FA = attn_mod.compile_attn
    
#     def compileKernel_FA(self, shape, config ) -> list:
#         hsacoPath = self.__compile_kernel_FA(shape,config)
#         kernelName = "attention1"
        
#         gridDimX,gridDimY,gridDimZ = [int(shape[2]/config[1]), shape[1], shape[0]]  # bx, by, bz
#         blockSize = [config[-1][0]]  # tx
#         shmBytes = config[-1][1]  # shared memroy size
#         return [(hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockSize,1,1,shmBytes)]
    
#     def compileKernel(self, param : MatmulTuningArgs) -> list:
#         return self.__compile_kernel_matmul(
#             param.BLOCK_SIZE_M,
#             param.BLOCK_SIZE_N,
#             param.BLOCK_SIZE_K,
#             param.THREAD_SIZE_M,
#             param.THREAD_SIZE_N,

#             param.WARP_SIZE,

#             param.BLOCK_LAYOUT_M,
#             param.BLOCK_LAYOUT_N,
#             param.WARP_LAYOUT_M,
#             param.WARP_LAYOUT_N,

#             param.GLOB_LOAD_WIDTH_A,
#             param.GLOB_LOAD_WIDTH_B,

#             param.WARP_SCATTER_WIDTH_A,
#             param.WARP_SCATTER_WIDTH_B,
#             param.THREAD_SCATTER_WIDTH_A,
#             param.THREAD_SCATTER_WIDTH_B,

#             param.LOCAL_SPLIT_U,
#             param.BLOCK_MAPPING,
#             param.GLOB_STORE_WIDTH,
#             param.UNROLL_NUM,
#             param.REG_PREFETCH,
#             param.SHARED_PREFETCH,
#             param.LOAD_CONTINUOUS,
#             param.REDUCE_C_CONTINUOUS,
            
#             param.dtype('A'),
#             param.dtype('B'),
#             param.dtype('C'),
#             param.M,param.N,param.K,param.batch,
#             param.isATranspose
#         )