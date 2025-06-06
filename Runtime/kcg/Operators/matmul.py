import numpy as np
import torch
from kcg.CompiledKernel import *
from kcg.Kernel import *
from kcg.Utils import *


@kcg_kernel
def _matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr
):
    'DUMP CODES'
    pass
 
# Call hook. 在这里带入实参并调用
def _matmul(a : torch.Tensor, b : torch.Tensor, c : torch.Tensor):
    dimsizeA = len(a.shape) 
    dimsizeB = len(b.shape)
    dimsizeC = len(c.shape)
    assert dimsizeA == dimsizeB == dimsizeC, "ABC must with same dim size"
    return _matmul_kernel(
        a, b, c
    )

# 基础参数
class MatmulBaseArgs(OpBaseArgs) :
    def __init__(self):
        super().__init__()
        self.operatorKind = EnumOperator.Matmul
        self.argDict = {
            "kind" : self.operatorKind,
            "b" : 0,
            "m" : 0,
            "n" : 0,
            "k" : 0,
            "dtype" : 0
        }
        # self.intValues : [b,m,n,k, dtypeInt]
    
    def getEnumDType(self) -> EnumKernelDType:
        dtInt = self.intValues[-1]
        return EnumKernelDType(dtInt)        
    
    def getIntDatalist(self) -> List[int] :
        return self.intValues[0:4] + [self.intValues[-1]]
    
    def parseFromTemplateDict(self,templateDict : Dict):
        batch = templateDict[ConfigKeywords.KEY_BATCH][0]
        m = templateDict[ConfigKeywords.KEY_M][0]
        n = templateDict[ConfigKeywords.KEY_N][0]
        k = templateDict[ConfigKeywords.KEY_K][0]
        dtype : int = templateDict[ConfigKeywords.KEY_DTYPE_C][0]
        self.intValues = [batch,m,n,k,dtype]
        
    def parseFromJsonfile(self,path : str):
        import json
        obj = None
        with open(path) as f :
            obj = json.load(f)
        self.intValues = [obj['b'],obj['m'],obj['n'],obj['k'], obj['dtype']]
        print(f"[ matmul ] b,m,n,k,dt = {self.intValues}")
        # self.operatorKind = obj['kind']
        
    def dumpToJson(self,path : str):
        import json
        self.argDict["kind"] = self.operatorKind
        self.argDict["b"] = self.intValues[0]
        self.argDict["m"] = self.intValues[1]
        self.argDict["n"] = self.intValues[2]
        self.argDict["k"] = self.intValues[3]
        self.argDict["dtype"] = self.intValues[4]
        with open(path,'w') as f:
            json.dump(self.argDict,f)
        

# 调优参数
class MatmulTuningArgs(TuningArgsInterface) :
    def __init__(self,batch = 1, m = 0,n = 0,k = 0, enumDType : EnumKernelDType = EnumKernelDType.float32):
        super().__init__()
        self.BLOCK_SIZE_M : int = 64
        self.BLOCK_SIZE_N : int = 64
        self.BLOCK_SIZE_K : int = 16
        self.THREAD_SIZE_M : int = 4
        self.THREAD_SIZE_N : int = 4
        self.WARP_SIZE : int = 64 
        self.BLOCK_LAYOUT_M : int = 4
        self.BLOCK_LAYOUT_N : int = 1
        self.WARP_LAYOUT_M : int = 16
        self.WARP_LAYOUT_N : int = 4
        self.dtA : EnumKernelDType = enumDType
        self.dtB : EnumKernelDType = enumDType
        self.dtC : EnumKernelDType = enumDType
        self.M : int = m
        self.N : int = n
        self.K : int = k
        self.batch : int = batch
        self.isATranspose : int = 1
        self.GLOB_LOAD_WIDTH_A : int = 0
        self.GLOB_LOAD_WIDTH_B : int = 0
        self.WARP_SCATTER_WIDTH_A : int = 0
        self.WARP_SCATTER_WIDTH_B : int = 0
        self.THREAD_SCATTER_WIDTH_A : int = 0
        self.THREAD_SCATTER_WIDTH_B : int = 0
        self.LOCAL_SPLIT_U : int = 0
        self.BLOCK_MAPPING : int = 0
        self.GLOB_STORE_WIDTH : int = 0
        
        self.UNROLL_NUM : int = 1
        self.REG_PREFETCH : int = 0
        self.SHARED_PREFETCH : int = 0
        self.LOAD_CONTINUOUS : int = 0
        self.REDUCE_C_CONTINUOUS : int = 0
    
    def assignWithList(self, *args):
        self.BLOCK_SIZE_M = args[0]
        self.BLOCK_SIZE_N = args[1]
        self.BLOCK_SIZE_K = args[2]
        self.THREAD_SIZE_M = args[3]
        self.THREAD_SIZE_N = args[4]
        self.GLOB_LOAD_WIDTH_A  = args[5]
        self.GLOB_LOAD_WIDTH_B  = args[6]
        self.BLOCK_LAYOUT_M = args[7]
        self.BLOCK_LAYOUT_N = args[8]
        self.WARP_LAYOUT_M = args[9]
        self.WARP_LAYOUT_N = args[10]
        self.WARP_SCATTER_WIDTH_A  = args[11]
        self.WARP_SCATTER_WIDTH_B  = args[12]
        self.THREAD_SCATTER_WIDTH_A  = args[13]
        self.THREAD_SCATTER_WIDTH_B  = args[14]
        self.SHARED_PREFETCH  = args[15]
        self.REG_PREFETCH  = args[16]
        self.LOAD_CONTINUOUS  = args[17]
        self.LOCAL_SPLIT_U  = args[18]
        self.GLOB_STORE_WIDTH   = args[19]
        self.REDUCE_C_CONTINUOUS  = args[20]
        self.BLOCK_MAPPING  = args[21]
        self.UNROLL_NUM  = args[22]
        self.WARP_SIZE = args[23]
        self.isATranspose  = args[24]

    
    def jsonfy(self) : 
        obj = {
            str(ConfigKeywords.KEY_BLOCK_SIZE_M) : (self.BLOCK_SIZE_M),
            str(ConfigKeywords.KEY_BLOCK_SIZE_N) : (self.BLOCK_SIZE_N),
            str(ConfigKeywords.KEY_BLOCK_SIZE_K) : (self.BLOCK_SIZE_K),
            str(ConfigKeywords.KEY_THREAD_SIZE_M) : (self.THREAD_SIZE_M),
            str(ConfigKeywords.KEY_THREAD_SIZE_N) : (self.THREAD_SIZE_N),
            str(ConfigKeywords.KEY_WARP_SIZE) : (self.WARP_SIZE),
            str(ConfigKeywords.KEY_BLOCK_LAYOUT_M) : (self.BLOCK_LAYOUT_M),
            str(ConfigKeywords.KEY_BLOCK_LAYOUT_N) : (self.BLOCK_LAYOUT_N),
            str(ConfigKeywords.KEY_WARP_LAYOUT_M) : (self.WARP_LAYOUT_M),
            str(ConfigKeywords.KEY_WARP_LAYOUT_N) : (self.WARP_LAYOUT_N),
            str(ConfigKeywords.KEY_DTYPE_A) : int(self.dtA),
            str(ConfigKeywords.KEY_DTYPE_B) : int(self.dtB), 
            str(ConfigKeywords.KEY_DTYPE_C) : int(self.dtC), 
            str(ConfigKeywords.KEY_M) : (self.M) ,
            str(ConfigKeywords.KEY_N) : (self.N) ,
            str(ConfigKeywords.KEY_K) : (self.K) ,
            str(ConfigKeywords.KEY_BATCH) : (self.batch) ,
            str(ConfigKeywords.KEY_IS_A_TRANSPOSE) : (self.isATranspose) ,
            str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A) : (self.GLOB_LOAD_WIDTH_A) ,
            str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B) : (self.GLOB_LOAD_WIDTH_B) ,
            str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A) : (self.WARP_SCATTER_WIDTH_A) ,
            str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B) : (self.WARP_SCATTER_WIDTH_B) ,
            str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A) : (self.THREAD_SCATTER_WIDTH_A) ,
            str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B) : (self.THREAD_SCATTER_WIDTH_B) ,
            str(ConfigKeywords.KEY_LOCAL_SPLIT_U) : (self.LOCAL_SPLIT_U) ,
            str(ConfigKeywords.KEY_BLOCK_MAPPING) : (self.BLOCK_MAPPING) ,
            str(ConfigKeywords.KEY_GLOB_STORE_WIDTH) : (self.GLOB_STORE_WIDTH ) ,
            str(ConfigKeywords.KEY_UNROLL_NUM) : (self.UNROLL_NUM) ,
            str(ConfigKeywords.KEY_REG_PREFETCH) : (self.REG_PREFETCH) ,
            str(ConfigKeywords.KEY_SHARED_PREFETCH) : (self.SHARED_PREFETCH) ,
            str(ConfigKeywords.KEY_LOAD_CONTINUOUS) : (self.LOAD_CONTINUOUS) ,
            str(ConfigKeywords.KEY_REDUCE_C_CONTINUOUS) : (self.REDUCE_C_CONTINUOUS) ,
        }
        return obj
    
    def assignWithDict(self, config : Dict) :
        kw = ConfigKeywords    
        self.M , self.N, self.K , self.batch = config[kw.KEY_M],config[kw.KEY_N],config[kw.KEY_K],config[kw.KEY_BATCH]
        self.dtA, self.dtB, self.dtC = EnumKernelDType(config[kw.KEY_DTYPE_A]), EnumKernelDType(config[kw.KEY_DTYPE_B]),EnumKernelDType(config[kw.KEY_DTYPE_C])
        self.BLOCK_SIZE_M = config[kw.KEY_BLOCK_SIZE_M]
        self.BLOCK_SIZE_N = config[kw.KEY_BLOCK_SIZE_N]
        self.BLOCK_SIZE_K = config[kw.KEY_BLOCK_SIZE_K]
        self.THREAD_SIZE_M = config[kw.KEY_THREAD_SIZE_M]
        self.THREAD_SIZE_N = config[kw.KEY_THREAD_SIZE_N]
        self.WARP_SIZE = config[kw.KEY_WARP_SIZE]
        self.BLOCK_LAYOUT_M = config[kw.KEY_BLOCK_LAYOUT_M]
        self.BLOCK_LAYOUT_N = config[kw.KEY_BLOCK_LAYOUT_N]
        self.WARP_LAYOUT_M = config[kw.KEY_WARP_LAYOUT_M]
        self.WARP_LAYOUT_N = config[kw.KEY_WARP_LAYOUT_N]
        self.isATranspose = config[kw.KEY_IS_A_TRANSPOSE]
        self.GLOB_LOAD_WIDTH_A = config[kw.KEY_GLOB_LOAD_WIDTH_A]
        self.GLOB_LOAD_WIDTH_B = config[kw.KEY_GLOB_LOAD_WIDTH_B]
        self.WARP_SCATTER_WIDTH_A = config[kw.KEY_WARP_SCATTER_WIDTH_A]
        self.WARP_SCATTER_WIDTH_B = config[kw.KEY_WARP_SCATTER_WIDTH_B]
        self.THREAD_SCATTER_WIDTH_A = config[kw.KEY_THREAD_SCATTER_WIDTH_A]
        self.THREAD_SCATTER_WIDTH_B = config[kw.KEY_THREAD_SCATTER_WIDTH_B]
        self.LOCAL_SPLIT_U = config[kw.KEY_LOCAL_SPLIT_U]
        self.BLOCK_MAPPING = config[kw.KEY_BLOCK_MAPPING]
        self.GLOB_STORE_WIDTH = config[kw.KEY_GLOB_STORE_WIDTH]
        self.UNROLL_NUM = config[kw.KEY_UNROLL_NUM]
        self.REG_PREFETCH = config[kw.KEY_REG_PREFETCH]
        self.SHARED_PREFETCH = config[kw.KEY_SHARED_PREFETCH]
        self.LOAD_CONTINUOUS = config[kw.KEY_LOAD_CONTINUOUS]
        self.REDUCE_C_CONTINUOUS = config[kw.KEY_REDUCE_C_CONTINUOUS]
    
    # def assignWithEncoder(self, cfgstr : int, tse : TuningSpaceEncoder) : 
    #     config = tse.decode(cfgstr)
    #     self.assignWithDict(config)
    
    def assignWithJson(self, jsonObj) : 
        self.BLOCK_SIZE_M = jsonObj[ConfigKeywords.KEY_BLOCK_SIZE_M] 
        self.BLOCK_SIZE_N = jsonObj[ConfigKeywords.KEY_BLOCK_SIZE_N] 
        self.BLOCK_SIZE_K = jsonObj[ConfigKeywords.KEY_BLOCK_SIZE_K] 
        self.THREAD_SIZE_M = jsonObj[ConfigKeywords.KEY_THREAD_SIZE_M] 
        self.THREAD_SIZE_N = jsonObj[ConfigKeywords.KEY_THREAD_SIZE_N] 
        self.WARP_SIZE = jsonObj[ConfigKeywords.KEY_WARP_SIZE] 
        self.BLOCK_LAYOUT_M = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_M] 
        self.BLOCK_LAYOUT_N = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_N] 
        self.WARP_LAYOUT_M = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_M] 
        self.WARP_LAYOUT_N = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_N] 
        self.dtA=  int(jsonObj[ConfigKeywords.KEY_DTYPE_A])
        self.dtB = int(jsonObj[ConfigKeywords.KEY_DTYPE_B])
        self.dtC = int(jsonObj[ConfigKeywords.KEY_DTYPE_C])
        self.M  = jsonObj[ConfigKeywords.KEY_M]
        self.N  = jsonObj[ConfigKeywords.KEY_N]
        self.K  = jsonObj[ConfigKeywords.KEY_K]
        self.batch  = jsonObj[ConfigKeywords.KEY_BATCH]
        self.isATranspose  = jsonObj[ConfigKeywords.KEY_IS_A_TRANSPOSE] > 0 
        self.GLOB_LOAD_WIDTH_A  = jsonObj[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A]
        self.GLOB_LOAD_WIDTH_B  = jsonObj[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B]
        self.WARP_SCATTER_WIDTH_A  = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A]
        self.WARP_SCATTER_WIDTH_B  = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B]
        self.THREAD_SCATTER_WIDTH_A  = jsonObj[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A]
        self.THREAD_SCATTER_WIDTH_B  = jsonObj[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B]
        self.LOCAL_SPLIT_U  = jsonObj[ConfigKeywords.KEY_LOCAL_SPLIT_U]
        self.BLOCK_MAPPING  = jsonObj[ConfigKeywords.KEY_BLOCK_MAPPING]
        self.GLOB_STORE_WIDTH   = jsonObj[ConfigKeywords.KEY_GLOB_STORE_WIDTH]
        self.UNROLL_NUM  = jsonObj[ConfigKeywords.KEY_UNROLL_NUM]
        self.REG_PREFETCH  = jsonObj[ConfigKeywords.KEY_REG_PREFETCH]
        self.SHARED_PREFETCH  = jsonObj[ConfigKeywords.KEY_SHARED_PREFETCH]
        self.LOAD_CONTINUOUS  = jsonObj[ConfigKeywords.KEY_LOAD_CONTINUOUS]
        self.REDUCE_C_CONTINUOUS  = jsonObj[ConfigKeywords.KEY_REDUCE_C_CONTINUOUS]

    
    def check(self) :
        # problem size check
        assert self.M % self.BLOCK_SIZE_M == 0 
        assert self.N % self.BLOCK_SIZE_N == 0 
        assert self.K % self.BLOCK_SIZE_K == 0 
        assert self.batch >= 1 
        # warp-block validation check
        assert self.BLOCK_SIZE_M % self.THREAD_SIZE_M == 0
        assert self.BLOCK_SIZE_N % self.THREAD_SIZE_N == 0
        assert (self.BLOCK_LAYOUT_M * self.WARP_LAYOUT_M) == (self.BLOCK_SIZE_M / self.THREAD_SIZE_M)
        assert (self.BLOCK_LAYOUT_N * self.WARP_LAYOUT_N) == (self.BLOCK_SIZE_N / self.THREAD_SIZE_N)
        assert self.WARP_LAYOUT_N * self.WARP_LAYOUT_M == self.WARP_SIZE
        # shm size check
        assert 2*(self.BLOCK_SIZE_M + self.BLOCK_SIZE_N) * self.BLOCK_SIZE_K <= 65536
        
        print("===== config check ok!")
    
    def dtype(self,index:str)->EnumKernelDType :
        if index=='A':
            return self.dtA
        if index=='B':
            return self.dtB
        if index=='C':
            return self.dtC
        
    def getGridDims(self) -> List[int]: ...
    def getBlockDims(self) -> List[int]: ...
    def getShmBytes(self) -> int : ...

    def __str__(self):
        return str(self.jsonfy())

# 算子生成逻辑
class MatmulOp(OpInterface) :
    def __init__(self):
        super().__init__()
        self.BaseArgs = MatmulBaseArgs()
        self.CompileKernelMatmul = None
        self.SetPlatform = None

    def GetBaselineInputTensor(self, devId : int) -> List[torch.Tensor] : 
        if self.InputTensors_Baseline is None :
            [batch, m,n,k, dtypeInt] = self.BaseArgs.intValues 
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            if batch > 1 :
                a = torch.rand((batch, m,k),dtype=ety, device=f"cuda:{devId}" )
                b = torch.rand((batch, k,n),dtype=ety, device=f"cuda:{devId}" )
            else:
                a = torch.rand((m,k),dtype=ety, device=f"cuda:{devId}" )
                b = torch.rand((k,n),dtype=ety, device=f"cuda:{devId}" )
            self.InputTensors_Baseline = [a,b]
        return self.InputTensors_Baseline
            
    def GetBenchmarkInputTensor(self,devId : int) -> List[torch.Tensor] : 
        if self.InputTensors_Benchmark is None :
            [a,b] = self.GetBaselineInputTensor(devId)
            [batch, m,n,k, dtypeInt] = self.BaseArgs.intValues 
            print(f"self.BaseArgs.intValues = {self.BaseArgs.intValues}" )
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            if batch > 1 :
                aa = a.transpose(1,2).contiguous()
                c = torch.empty((batch,m,n), dtype=ety, device=f"cuda:{devId}")
            else :
                aa = a.transpose(0,1).contiguous()
                c = torch.empty((m,n), dtype=ety, device=f"cuda:{devId}")
            self.InputTensors_Benchmark = [aa,b,c]
        return self.InputTensors_Benchmark
    
    # def GetBenchmarkOutputTensor(self,devId : int) -> List[torch.Tensor] : 
    #     return self.GetBenchmarkInputTensor(devId)[-1]
    
    # def GetBaselineOutputTensor(self) -> List[torch.Tensor] : 
    #     return self.OutputTensor_Baseline
    
    def InitLibInterface(self) :
        if self.CompileKernelMatmul is None or self.SetPlatform is None :
            import importlib.util
            print(f"PathManager.kcg_compiler_path() = {PathManager.kcg_compiler_path()}",flush=True)
            spec = importlib.util.spec_from_file_location("KCGCompiler", PathManager.kcg_compiler_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernelMatmul = mod.compile_kernel_matmul
            self.SetPlatform = mod.set_platform

        # attn_spec = importlib.util.spec_from_file_location("attention", PathManager.kcg_compiler_attention_path())
        # attn_mod = importlib.util.module_from_spec(attn_spec)
        # attn_spec.loader.exec_module(attn_mod)
        # self.__compile_kernel_FA = attn_mod.compile_attn
    
    def Compile(self, deviceId:int, backendtype : EnumBackendType, arch : str, info : CompileNeededInfo) -> Tuple[List,KernelConfigs,CompiledKernel] :
        Print = print
        _backend = 0
        if backendtype.value == EnumBackendType.CUDA.value :
            _backend = 1
        elif backendtype.value == EnumBackendType.HIP.value :
            _backend = 2
        else:
            assert False, f'invalid backendtype {backendtype}, Ty is {type(backendtype)}'
        print("compiling matmul",flush=True)
        print("ta=",*info.tsArgs,flush=True)
        print("ba=",info.baseArgs,flush=True)
        self.InitLibInterface()
        self.SetPlatform(_backend,arch)
        # Print("===== call compileKernel(kpm)[0] ========")
        res = self.CompileKernelMatmul( *info.tsArgs)
        hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,shmBytes = res[0]
        print(f"blockdims = {blockDimX,blockDimY,blockDimZ}")
        print(f"griddims = {gridDimX,gridDimY,gridDimZ}")
        Print("========= hsacoPath = ",hsacoPath)
        Print("========= kernelName = ",kernelName)
        print(f"==== backend is {backendtype}")
        print(f"==== shmBytes is {shmBytes}")
        dt = info.torchDataType
        inConfig = KernelConfigs(hsacoPath, kernelName, [ dt,dt,dt ], backendtype)
        inConfig.m_gridDims = [gridDimX,gridDimY,gridDimZ]
        inConfig.m_blockDims = [blockDimX,blockDimY,blockDimZ]
        inConfig.operatorKind = EnumOperator.Matmul
        inConfig.shmBytes = shmBytes
        packedKernel = self.GetCompiledKernel(inConfig,deviceId)
        return (info.baseArgs, inConfig, packedKernel)  # 
  
    def GetCompiledKernel(self, info : KernelConfigs, deviceId : int) -> CompiledKernel :
        signature = self.GetSignature(info.dtypes)
        return CompiledKernel(
            info.backend,
            info.binaryPath,
            info.kernelFuncName,
            info.sharedMem(),
            signature,
            info.gridDims(),
            info.blockDims(),
            deviceId
        )
    
    def GetSignature(self, dtypes : List[torch.dtype]) -> dict :
        # signature只和输入的dtype有关，尺寸无关
        dtypeA = dtypes[0]
        a = torch.rand(100, 100, device='cpu', dtype=dtypeA)
        b = torch.rand(100, 100, device='cpu', dtype=dtypeA)
        c = torch.empty(100, 100, device='cpu', dtype=dtypeA)
        # get function signature
        outSignature = _matmul(a, b, c)
        return outSignature
    
    def SetTuningArgs(self, tuningArgs : List) :
        self.TuningArgs.assignWithList(*tuningArgs)

    def InitBaseArgs(self, args : List[int]) :
        batch, m, n, k , dtypeInt = args
        self.BaseArgs.intValues = [batch, m,n,k, dtypeInt]
        ety = EnumKernelDType(dtypeInt)
        self.TuningArgs = MatmulTuningArgs(batch,m,n,k,ety)  
    
    def Test_warmup(self, packedKernel : CompiledKernel, warmupCount : int, devId : int) :
        [a0,b0] = self.GetBaselineInputTensor(devId)
        [a,b,c] = self.GetBenchmarkInputTensor(devId)
        torchMM = torch.matmul
        if self.TuningArgs.batch > 1:
            assert len(a0.shape) == 3, f"shape not match : len(a0.shape)={len(a0.shape)}, TuningArgs.batch={self.TuningArgs.batch}" 
            torchMM = torch.bmm
        for i in range(0,warmupCount) : 
            torchMM(a0,b0)
            packedKernel.run(a,b,c)

    def Test_baseline(self, devId : int) -> Tuple[torch.Tensor,float]:
        [matrixA, matrixB] = self.GetBaselineInputTensor(devId)
        torchMM = torch.matmul
        if len(matrixA.shape) > 2 :
            torchMM = torch.bmm
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        self.OutputTensor_Baseline = torchMM(matrixA, matrixB)
        ev_end.record()
        torch.cuda.synchronize()
        eps = ev_start.elapsed_time(ev_end)
        return (self.OutputTensor_Baseline, eps)
    
    def Test_benchmark(self, packedKernel : CompiledKernel,benchmarkCount : int , devId : int) -> Tuple[torch.Tensor,float] : 
        assert self.InputTensors_Benchmark  is not None, "error benchmark"
        eps = []
        for i in range(benchmarkCount):
            a,b,c = self.GetBenchmarkInputTensor(devId)
            st = torch.cuda.Event(enable_timing=True)
            et = torch.cuda.Event(enable_timing=True)
            st.record()
            packedKernel.run(a,b,c)
            et.record()
            torch.cuda.synchronize()
            elapsed_time = st.elapsed_time(et)
            eps.append(elapsed_time)
        t = np.median(eps)
        return (c,t)
    
    def InitInputTensorsWithDatalist(self,  devId) -> None:
        assert self.BaseArgs is not None
        assert self.TuningArgs is not None
        assert isinstance(self.TuningArgs, MatmulTuningArgs)
        assert isinstance(self.BaseArgs , MatmulBaseArgs)
        # init baseline inputs
        matA = None; matB = None
        if self.InputTensors_Baseline is None :
            batch, m,n,k , dtypeInt = self.BaseArgs.getIntDatalist()
            datatype = ToTorchType(EnumKernelDType(dtypeInt))
            if batch > 1:
                matA = torch.randn(batch,m,k, dtype= datatype, device=f'cuda:{devId}')
                matB = torch.randn(batch,k,n, dtype= datatype, device=f'cuda:{devId}')
            else:
                matA = torch.randn(m,k, dtype= datatype, device=f'cuda:{devId}')
                matB = torch.randn(k,n, dtype= datatype, device=f'cuda:{devId}')
            self.InputTensors_Baseline = [matA ,matB]
        else:
            matA, matB = self.InputTensors_Baseline
        # init benchmark inputs
        if self.InputTensors_Benchmark is None:
            aUse = None
            if self.TuningArgs.isATranspose :
                d0,d1 = 0,1
                if len(matA.shape) == 3 :
                    d0,d1 = 1,2
                atrans = torch.transpose(matA,d0,d1).contiguous()  # 转置会令底层存储不连续，导致失败。必须使其连续
                assert(matA.is_contiguous())
                assert(matB.is_contiguous())
                assert(atrans.is_contiguous())
                aUse = atrans
            else:
                aUse = matA
            self.InputTensors_Benchmark = [aUse,matB]

    
    def InitBaselineOutputTensor(self,  devId : int) -> None :
        # batch,m,n,k = self.BaseArgs.intValues[0:4]
        if self.OutputTensor_Baseline is None :
            b,m,n,k,dtypeInt = self.BaseArgs.getIntDatalist()
            dt = ToTorchType(EnumKernelDType(dtypeInt))
            if b > 1:
                ret = torch.empty(b,m,n,dtype=dt, device=f'cuda:{devId}')
            else:
                ret = torch.empty(m,n,dtype=dt, device=f'cuda:{devId}')
            self.OutputTensor_Baseline = ret

    
    # def GetBenchmarkOutputTensor(self,  devId : int) -> torch.Tensor :
    #     b,m,n,k,dtypeInt = self.BaseArgs.getIntDatalist()
    #     dt = ToTorchType(EnumKernelDType(dtypeInt))
    #     if b > 1:
    #         ret = torch.empty(b,m,n,dtype=dt, device=f'cuda:{devId}')
    #     else:
    #         ret = torch.empty(m,n,dtype=dt, device=f'cuda:{devId}')
    #     return ret

    
    
class TuningSpaceChecker_Matmul :
    @staticmethod
    def check_shm_size(config : Dict) :
        # 计算约束条件
        dtypeC = config[ ConfigKeywords.KEY_DTYPE_C]
        dtypeBytes = sizeof(get_dtype_from_int(dtypeC))
        value1 = (config[ ConfigKeywords.KEY_BLOCK_SIZE_M] + config[ ConfigKeywords.KEY_BLOCK_SIZE_N]) * config[ ConfigKeywords.KEY_BLOCK_SIZE_K] * dtypeBytes * config[ ConfigKeywords.KEY_LOCAL_SPLIT_U]
        value2 = config[ ConfigKeywords.KEY_BLOCK_SIZE_M] * config[ ConfigKeywords.KEY_BLOCK_SIZE_N] * dtypeBytes * config[ ConfigKeywords.KEY_LOCAL_SPLIT_U]

        if max(value1, value2) < 16384 :
            # 展开 KEY_DTYPE 到 KEY_DTYPE_A, KEY_DTYPE_B, KEY_DTYPE_C
            dtype_value = config.pop( ConfigKeywords.KEY_DTYPE_C)
            config[ ConfigKeywords.KEY_DTYPE_A] = dtype_value
            config[ ConfigKeywords.KEY_DTYPE_B] = dtype_value
            config[ ConfigKeywords.KEY_DTYPE_C] = dtype_value
            return True
        return False
    
    @staticmethod
    def check_warp(config : Dict) -> bool :
        wlm = config[ConfigKeywords.KEY_WARP_LAYOUT_M]
        wln = config[ConfigKeywords.KEY_WARP_LAYOUT_N]
        warpsz = config[ConfigKeywords.KEY_WARP_SIZE]
        if wlm * wln == warpsz :
            return True
        return False
    
    @staticmethod
    def check_size(config : Dict) -> bool :
        bm = config[ConfigKeywords.KEY_BLOCK_SIZE_M]
        bn = config[ConfigKeywords.KEY_BLOCK_SIZE_N]
        bk = config[ConfigKeywords.KEY_BLOCK_SIZE_K]
        tm = config[ConfigKeywords.KEY_THREAD_SIZE_M]
        tn = config[ConfigKeywords.KEY_THREAD_SIZE_N]
        m = config[ConfigKeywords.KEY_M]
        n = config[ConfigKeywords.KEY_N]
        k = config[ConfigKeywords.KEY_K]
        
        blm = config[ConfigKeywords.KEY_BLOCK_LAYOUT_M]
        bln = config[ConfigKeywords.KEY_BLOCK_LAYOUT_N]
        
        wlm = config[ConfigKeywords.KEY_WARP_LAYOUT_M]
        wln = config[ConfigKeywords.KEY_WARP_LAYOUT_N]
        
        if m % bm != 0 :
            return False
        if n % bn != 0 :
            return False
        if k % bk != 0 :
            return False
        
        blockDim_m = blm * wlm
        blockDim_n = bln * wln
        if blockDim_m * tm != bm or blockDim_n * tn != bn :
            return False
        
        wswa = config[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A]
        wswb = config[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B]
        tswa = config[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A]
        tswb = config[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B]
        if wswa < tswa or wswb < tswb :
            return False
        if wswa % tswa != 0 or wswb % tswb != 0 :
            return False
        return True
    
    @staticmethod
    def check_glw(config : Dict) -> bool :
        glwa = config[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A]
        glwb = config[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B]
        bm = config[ConfigKeywords.KEY_BLOCK_SIZE_M]
        bn = config[ConfigKeywords.KEY_BLOCK_SIZE_N]
        bk = config[ConfigKeywords.KEY_BLOCK_SIZE_K]
        tm = config[ConfigKeywords.KEY_THREAD_SIZE_M]
        tn = config[ConfigKeywords.KEY_THREAD_SIZE_N]
        nThreads = bm / tm * bn / tn
        if bm * bk / nThreads < 1 :
            return False
        if bn * bk / nThreads < 1 :
            return False
        return True
    