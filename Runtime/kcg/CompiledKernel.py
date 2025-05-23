from kcg.Kernel import *
from kcg.Utils import *
from kcg.Loader import HIPLoaderST,CudaLoaderST,is_hip
from kcg.HIPLauncher import HIPLauncher
from kcg.CUDALauncher import CUDALauncher

class CompiledKernel:
    def __init__(self,
                 backend : EnumBackendType,
                 kernelBinaryPath:str, 
                 kernelName:str, shmSize:int, 
                 kernel_signature,
                 gridDims:list,
                 blockDims:list,
                 device = 0):
        self.signature = kernel_signature
        self.m_loader = None
        self.m_launcher = None
        if backend.value == EnumBackendType.HIP.value :
            # print(f"[D] gridDims={gridDims} , blockDims={blockDims}, device ={device}")
            self.m_loader = HIPLoaderST()
            self.m_launcher = HIPLauncher(kernelBinaryPath,kernelName,shmSize,self.signature,gridDims,blockDims,device)
        elif backend.value == EnumBackendType.CUDA.value :
            # print(f"[D] gridDims={gridDims} , blockDims={blockDims}, device ={device}")
            self.m_loader = CudaLoaderST()
            self.m_launcher = CUDALauncher(kernelBinaryPath,kernelName,shmSize,self.signature,gridDims,blockDims,device)
        else:
            assert False, f"Invalid backend value {backend.value}"
        
    def deleteBinary(self):
        self.m_launcher.releaseAndDeleteBinary()

    def setDevice(self,devId : int) :
        self.m_launcher.m_kernelLib.m_device = devId
    
    def run(self,*args):
        try:
            if self.m_launcher is not None:
                self.m_launcher.launchKernel(*args)
                return True
        except Exception as e :
            print("CompilerKernelRunErr:",e)
        return False