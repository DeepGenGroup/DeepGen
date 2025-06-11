from kcg.Utils import *
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Operators import matmul, attention


class OpInjector :
    def __init__(self, devId : int = 0):
        self.devId = devId
    
    def parseOp(self, pklPath : str, OpTy: Type[OpInterface]):
        kernelCfg : KernelConfigs = deserialize_from_file(pklPath)
        kernel = OpTy().GetCompiledKernel(kernelCfg, self.devId)
        def kernelRun(*args) :
            print(f"---- run custom {OpTy.__name__} ---------")
            kernel.run(*args)
        return kernelRun

class OpProxy :
    @staticmethod
    def f_matmul(a : torch.Tensor, b : torch.Tensor) :
        m = a.shape[-2]
        k = a.shape[-1]
        n = b.shape[-1]
        batch = a.shape[0:-2] 
        if [m,n,k] == [1024,1024,1024]: 
            c = torch.empty((m,n),dtype=torch.float32, device='cuda:7')
            def _f() :
                f = OpInjector().parseOp('/home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp.pkl',matmul.MatmulOp)
                aT = a.transpose(-1,-2).contiguous()
                f(aT,b,c)
                return c
            return _f()
        elif [m,n,k] == [256,256,512] : 
            c = torch.empty((m,n),dtype=torch.float32, device='cuda:7')
            def _f() :
                f = OpInjector().parseOp('/home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp_[]:256:256:512:4:.pkl',matmul.MatmulOp)
                aT = a.transpose(-1,-2).contiguous()
                f(aT,b,c)
                return c
            return _f()
        else:
            print("shapeA = ",a.shape, 'shapeB =',b.shape)
            return torch.matmul(a,b)
    
    @staticmethod
    def f_attention(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor) :
        r = torch.empty((1024,1024),dtype=torch.float32, device='cuda:7')
        def _f() :
            f = OpInjector().parseOp('/home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp.pkl',matmul.MatmulOp)
            f(q,k,v,r)
            return r
        return _f()
    
