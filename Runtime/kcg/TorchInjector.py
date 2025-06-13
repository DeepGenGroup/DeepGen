from kcg.Operators import matmul, attention
from kcg.Kernel import *
from kcg.KernelTuneUtils import TuneResult
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
        self.opCounter = dict()
    
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
            if OpTy.__name__ not in self.opCounter.keys() :
                self.opCounter[OpTy.__name__] = 0
        self.opCounter[OpTy.__name__] += 1
        
    def getInfo(self) :
        for obj in self.infoList :
            info : List = obj['baseArgs']
            info.append(obj['dtype']) 
            ty = None
            if obj['optype'] == matmul.MatmulOp.__name__ :
                ty = matmul.MatmulOp
            elif obj['optype'] == attention.AttentionOp.__name__ :
                ty = attention.AttentionOp
            yield (ty,info)
            
    def getOpCallCount(self) -> Dict :
        return self.opCounter
# operator 代理人。f_matmul f_attention 用于代替 model中的对应算子。 collector用于在首次执行model时，收集需要编译的kernel信息（基础形状、dtype、op类型等）。
# collector收集的信息可用于后续的 tuning空间生产、编译
# f_ 开头的静态方法，参数列表和torch的保持相同，便于model直接替换对应算子。内部的 _f()通过调整参数，和 packedKernel的调用形式适配（如定义c）
class OpProxy :
    collector = OpBaseArgsCollector()
    __registedKernels_mm : List[TuneResult] = []
    __registedKernels_att : List[TuneResult] = []
    n_matmulCallCount = 0
    @staticmethod
    def GetCollectedKernelArgs() -> Generator :
        return OpProxy.collector.getInfo()
    
    @staticmethod
    def GetOpCallCounts() -> Dict :
        return OpProxy.collector.getOpCallCount()
    
    @staticmethod
    def registKernel(tr : TuneResult) :
        if tr is None or tr.bestConfigPkl is None:
            return
        if tr.bestSpeedup > 1 :
            if tr.OpTy is matmul.MatmulOp :
                OpProxy.__registedKernels_mm.append(tr)
                print(f"regist OK : {tr.OpTy.__name__} ,acc={tr.bestSpeedup}")
            elif tr.OpTy is attention.AttentionOp :
                OpProxy.__registedKernels_att.append(tr)
            else :
                print(f"registerKernel failed : {tr.OpTy.__name__} is unsupport")
        else:
            print(f"registerKernel : acc <= 1 ,skip ")
            
        
    @staticmethod
    def __select_matmul(a : torch.Tensor, b : torch.Tensor , batch,m,n,k) :
        default = torch._C._VariableFunctions.matmul(a, b)
        dev = a.get_device()
        for tr in OpProxy.__registedKernels_mm :
            bb,mm,nn,kk = tr.bestKernelBaseArg[0:-1]
            dt = tr.bestKernelBaseArg[-1]
            print(f'__select_matmul : [bbmmnnkk]={bb,mm,nn,kk}, [bmnk]={batch,m,n,k}')
            isBatchEqual = True
            if len(batch) == 1:
                if batch[0] == 1 and len(bb) == 0:
                    isBatchEqual = True
            else:
                isBatchEqual = bool(bb == batch)
            if isBatchEqual and [m,n,k] == [mm,nn,kk]: 
                def _f() :
                    shapeC = bb + [m,n]
                    c = torch.empty(shapeC,dtype=ToTorchType(EnumKernelDType(dt)), device=f'cuda:{dev}')
                    f = OpInjector(dev).parseOp(tr.bestConfigPkl, matmul.MatmulOp)
                    aT = a.transpose(-1,-2).contiguous()
                    f(aT,b,c)
                    return c
                return _f()
        OpProxy.collector.addInfo(matmul.MatmulOp,[batch,m,n,k], a.dtype)
        print("select default matmul")
        return default
    
    @staticmethod
    def f_matmul(a : torch.Tensor, b : torch.Tensor) :
        m = int(a.shape[-2])
        k = int(a.shape[-1])
        n = int(b.shape[-1])
        batch = [ int(x)  for x in a.shape[0:-2] ]
        print(f"b,m,n,k = {batch,m,n,k}")
        ret = torch._C._VariableFunctions.matmul(a, b)
        return OpProxy.__select_matmul(a,b,batch,m,n,k)
# /home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp_[]:256:256:512:4:.pkl
# /home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp_[]:256:512:256:4:.pkl

        try:
            if [m,n,k] == [256,256,512]: 
                c = torch.empty((m,n),dtype=torch.float32, device='cuda:7')
                def _f() :
                    f = OpInjector().parseOp('/home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp_[]:256:256:512:4:.pkl',matmul.MatmulOp)
                    aT = a.transpose(-1,-2).contiguous()
                    f(aT,b,c)
                    return c
                return _f()
            elif [m,n,k] == [256,512,256] : 
                c = torch.empty((m,n),dtype=torch.float32, device='cuda:7')
                def _f() :
                    f = OpInjector().parseOp('/home/xushilong/DeepGen/_tmp/bestConfig_MatmulOp_[]:256:512:256:4:.pkl',matmul.MatmulOp)
                    aT = a.transpose(-1,-2).contiguous()
                    f(aT,b,c)
                    return c
                return _f()
            else:
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
    
