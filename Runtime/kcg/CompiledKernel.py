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
        if backend is EnumBackendType.HIP :
            # print(f"[D] gridDims={gridDims} , blockDims={blockDims}, device ={device}")
            self.m_loader = HIPLoaderST()
            self.m_launcher = HIPLauncher(kernelBinaryPath,kernelName,shmSize,self.signature,gridDims,blockDims,device)
        if backend is EnumBackendType.CUDA :
            # print(f"[D] gridDims={gridDims} , blockDims={blockDims}, device ={device}")
            self.m_loader = CudaLoaderST()
            self.m_launcher = CUDALauncher(kernelBinaryPath,kernelName,shmSize,self.signature,gridDims,blockDims,device)
            
    def deleteBinary(self):
        if os.path.exists(self.m_launcher.m_kernelLib.m_filePath) :
            os.remove(self.m_launcher.m_kernelLib.m_filePath)
            # print(f"deleted {self.m_launcher.m_kernelLib.m_filePath}")

    def setDevice(self,devId : int) :
        self.m_launcher.m_kernelLib.m_device = devId
    
    def run(self,*args):
        if self.m_launcher is not None:
            self.m_launcher.launchKernel(*args)
    