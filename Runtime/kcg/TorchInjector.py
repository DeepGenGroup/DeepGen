from kcg.Utils import *
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Operators import matmul, attention

# 注入器。通过反序列化 compile过程得到的最佳 kernelConfig，构造该kernel的调用器
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

# op信息收集器。用于首次执行model时，采集需要编译的kenrel信息
class OpBaseArgsCollector :
    def __init__(self):
        self.infoList = []
        self.__hashTable = []
    
    def addInfo(self, OpTy : Type[OpInterface], baseArgs : List, dtype : torch.dtype) :
        obj = {
            "optype" : OpTy.__name__,
            "baseArgs" : baseArgs,
            "dtype" : dtype
        }
        hashkey = str(obj)
        if hashkey not in self.__hashTable :
            self.__hashTable.append(hashkey)
            self.infoList.append(obj)
    
    
    
# operator 代理人。f_matmul f_attention 用于代替 model中的对应算子。 collector用于在首次执行model时，收集需要编译的kernel信息（基础形状、dtype、op类型等）。
# collector收集的信息可用于后续的 tuning空间生产、编译
# f_ 开头的静态方法，参数列表和torch的保持相同，便于model直接替换对应算子。内部的 _f()通过调整参数，和 packedKernel的调用形式适配（如定义c）
class OpProxy :
    collector = OpBaseArgsCollector()
    @staticmethod
    def GetCollectedKernelArgs() -> List :
        return OpProxy.collector.infoList
    
    @staticmethod
    def f_matmul(a : torch.Tensor, b : torch.Tensor) :
        m = a.shape[-2]
        k = a.shape[-1]
        n = b.shape[-1]
        batch = a.shape[0:-2] 
        ret = torch.matmul(a,b)
        try:
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
                OpProxy.collector.addInfo(matmul.MatmulOp,[batch,m,n,k], a.dtype)
        except Exception as e:
            print(e)
        except IOError as e:
            print(e)
        return ret
    
    @staticmethod
    def f_attention(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor) :
        r = torch.empty((1024,1024),dtype=torch.float32, device='cuda:7')
        def _f() :
            f = OpInjector().parseOp('/home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp.pkl',matmul.MatmulOp)
            f(q,k,v,r)
            return r
        return _f()
    
