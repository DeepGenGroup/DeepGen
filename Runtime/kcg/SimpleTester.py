from kcg.Utils import *
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Operators import matmul


        
def RunKernelBinary(op : OpInterface, binpath : str, kernelFuncName : str, dev : int, args : List[torch.Tensor]) :
    dtypes = []
    for tensor in args :
        dtypes.append(tensor.dtype)
    backend = EnumBackendType.CUDA
    if is_hip():
        backend = EnumBackendType.HIP
    kernelConfig = KernelConfigs(binpath,kernelFuncName,dtypes,backend)
    devId = args[0].get_device()
    kernel = op.GetCompiledKernel(kernelConfig,devId)
    kernel.run(*args)
    
    
def init_cuda(_devId) :
    DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
    print("init_cuda devid=",_devId)
    DeviceInfo.set_visible_devices([_devId])
    DeviceInfo.set_current_device(_devId)  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
    if not torch.cuda.is_available() :
        torch.cuda.init()
        torch.cuda.empty_cache()
        
if __name__ == '__main__' :
    # blockdims = (64, 1, 1)
    # griddims = (1024, 1, 1)
    # ========= hsacoPath =  /tmp/compile-ptx-src-b27d15.cubin
    # ========= kernelName =  GEMM_bMNK1x1024x1024x1024_DTabcfloat32xfloat32xfloat32_AT1_TTmn4x4_BTmnk32x32x8_BLmn1x1_WLmn8x8_GLWab4x4_GSW2_WSWab2x2_TSWab2x1_LSU1_BM16_UNROLL4_REGP0_SHMP0_LC0_RC0_
    # ==== backend is CUDA
    # ==== shmBytes is 2048
    init_cuda(7)
    OP = matmul.MatmulOp()
    binPath = "/tmp/compile-ptx-src-b27d15.cubin"
    kernelName = "GEMM_bMNK1x1024x1024x1024_DTabcfloat32xfloat32xfloat32_AT1_TTmn4x4_BTmnk32x32x8_BLmn1x1_WLmn8x8_GLWab4x4_GSW2_WSWab2x2_TSWab2x1_LSU1_BM16_UNROLL4_REGP0_SHMP0_LC0_RC0_"
    dev = f'cuda:{DeviceInfo.get_current_device()}'
    a = torch.randn(1024,1024,dtype=torch.float32,device=dev)
    b = torch.randn(1024,1024,dtype=torch.float32,device=dev)
    c = torch.empty(1024,1024,dtype=torch.float32,device=dev)
    
    c = torch.matmul(a,b)
    RunKernelBinary(OP,binPath,kernelName,7,[a.transpose(0,1).contiguous(),b,c])
    print(c)
    