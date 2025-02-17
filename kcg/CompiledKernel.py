from kcg.Kernel import *
from kcg.Utils import *
from kcg.Loader import HIPLoaderST,CudaLoaderST,is_hip
from kcg.HIPLauncher import HIPLauncher

class CompiledKernel:
    def __init__(self,
                 kernelBinaryPath:str, 
                 kernelName:str, shmSize:int, 
                 kernel_signature,
                 gridDims:list,
                 blockDims:list,
                 device = DeviceInfo.get_current_device()):
        self.signature = kernel_signature
        self.device = device
        print(f"[D] gridDims={gridDims} , blockDims={blockDims}, device ={device}")
        self.m_loader = HIPLoaderST()
        self.m_launcher = HIPLauncher(kernelBinaryPath,kernelName,shmSize,self.signature,gridDims,blockDims,device)
        
    def run(self,*args,**kwargs):
        if self.m_launcher is not None:
            # print("[D] run")
            self.m_launcher.launchKernel(int(self.device), *args)
        
    