import numpy as np
import torch
from kcg.Kernel import *


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
        self.BLOCK_LAYOUT_Y : int = 4
        self.BLOCK_LAYOUT_X : int = 1
        self.WARP_LAYOUT_Y : int = 16
        self.WARP_LAYOUT_X : int = 4
        self.dtA : EnumKernelDType = enumDType
        self.dtB : EnumKernelDType = enumDType
        self.dtC : EnumKernelDType = enumDType
        self.M : int = m
        self.N : int = n
        self.K : int = k
        self.batch = batch
        self.isATranspose : int = 1
        self.GLOB_LOAD_WIDTH_A : int = 0
        self.GLOB_LOAD_WIDTH_B : int = 0
        self.BLOCK_SCATTER_WIDTH_M : int = 0
        self.BLOCK_SCATTER_WIDTH_N : int = 0
        self.WARP_SCATTER_WIDTH_M : int = 0
        self.WARP_SCATTER_WIDTH_N : int = 0
        self.LOCAL_SPLIT_U : int = 0
        self.BLOCK_MAPPING : int = 0
        self.GLOB_STORE_WIDTH : int = 0
        self.UNROLL_NUM : int = 1
        self.REG_PREFETCH : int = 0
        self.SHARED_PREFETCH : int = 0
        self.LOAD_CONTINUOUS : int = 0
        self.STORE_CONTINUOUS : int = 0
    
    def getCompileNeededInfo(self) -> CompileNeededInfo :
        kernelName = self.generateKernelName()
        configDict = {
            kernelName : self.jsonfy()
        }
        ret = CompileNeededInfo()
        ret.kernelName = kernelName
        ret.baseArgs = [self.batch, self.M, self.N, self.K, int(self.dtA)]
        ret.tsArgs = [[self.batch, self.M, self.N, self.K] , configDict  ]
        ret.torchDataType = ToTorchType(self.dtA)
        gridDim = self.M / self.BLOCK_SIZE_M * self.N / self.BLOCK_SIZE_N
        blockDim = (self.BLOCK_SIZE_M / self.THREAD_SIZE_M) * ( self.BLOCK_SIZE_N / self.THREAD_SIZE_N )
        shmBytes = (self.BLOCK_SIZE_M + self.BLOCK_SIZE_N) * self.BLOCK_SIZE_K
        if self.SHARED_PREFETCH > 0 :
            shmBytes *= 2
        if self.LOCAL_SPLIT_U > 1 :
            blockDim *= self.LOCAL_SPLIT_U
        shm_reduce = self.BLOCK_SIZE_M * self.BLOCK_SIZE_N * self.LOCAL_SPLIT_U
        if shm_reduce > shmBytes :
            shmBytes = shm_reduce
        ret.blockDims = [int(blockDim),1,1]
        ret.gridDims = [int(gridDim)]
        if int(gridDim) >= 65535 :
            return None
        if int(blockDim) >= 65535 :
            return None
        if len(self.batch) > 0 :
            ret.gridDims += self.batch  # 处理方式： 将batch维度加到griddim的y,z上. 即batch数组的维度不超过2
        assert len(ret.gridDims) <= 3
        while len(ret.gridDims) < 3:
            ret.gridDims.append(1)   # 不够三维的部分用1 补全
        ret.shmBytes = int(shmBytes * sizeof(self.dtA))
        return ret
    
    def assignWithList(self, *args):
        self.BLOCK_SIZE_M = args[0]
        self.BLOCK_SIZE_N = args[1]
        self.BLOCK_SIZE_K = args[2]
        self.THREAD_SIZE_M = args[3]
        self.THREAD_SIZE_N = args[4]
        self.GLOB_LOAD_WIDTH_A  = args[5]
        self.GLOB_LOAD_WIDTH_B  = args[6]
        self.BLOCK_LAYOUT_Y = args[7]
        self.BLOCK_LAYOUT_X = args[8]
        self.WARP_LAYOUT_Y = args[9]
        self.WARP_LAYOUT_X = args[10]
        self.BLOCK_SCATTER_WIDTH_M  = args[11]
        self.BLOCK_SCATTER_WIDTH_N  = args[12]
        self.WARP_SCATTER_WIDTH_M  = args[13]
        self.WARP_SCATTER_WIDTH_N  = args[14]
        self.SHARED_PREFETCH  = args[15]
        self.REG_PREFETCH  = args[16]
        self.LOAD_CONTINUOUS  = args[17]
        self.LOCAL_SPLIT_U  = args[18]
        self.GLOB_STORE_WIDTH   = args[19]
        self.STORE_CONTINUOUS  = args[20]
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
            str(ConfigKeywords.KEY_BLOCK_LAYOUT_Y) : (self.BLOCK_LAYOUT_Y),
            str(ConfigKeywords.KEY_BLOCK_LAYOUT_X) : (self.BLOCK_LAYOUT_X),
            str(ConfigKeywords.KEY_WARP_LAYOUT_Y) : (self.WARP_LAYOUT_Y),
            str(ConfigKeywords.KEY_WARP_LAYOUT_X) : (self.WARP_LAYOUT_X),
            str(ConfigKeywords.KEY_DTYPE_A) : int(self.dtA),
            str(ConfigKeywords.KEY_DTYPE_B) : int(self.dtB), 
            str(ConfigKeywords.KEY_DTYPE_C) : int(self.dtC), 
            str(ConfigKeywords.KEY_M) : (self.M) ,
            str(ConfigKeywords.KEY_N) : (self.N) ,
            str(ConfigKeywords.KEY_K) : (self.K) ,
            str(ConfigKeywords.KEY_BATCH) : 1 ,  # in tsArgs , we don't need batch info
            str(ConfigKeywords.KEY_IS_A_TRANSPOSE) : (self.isATranspose) ,
            str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A) : (self.GLOB_LOAD_WIDTH_A) ,
            str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B) : (self.GLOB_LOAD_WIDTH_B) ,
            str(ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_M) : (self.BLOCK_SCATTER_WIDTH_M) ,
            str(ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_N) : (self.BLOCK_SCATTER_WIDTH_N) ,
            str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_M) : (self.WARP_SCATTER_WIDTH_M) ,
            str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_N) : (self.WARP_SCATTER_WIDTH_N) ,
            str(ConfigKeywords.KEY_LOCAL_SPLIT_U) : (self.LOCAL_SPLIT_U) ,
            str(ConfigKeywords.KEY_BLOCK_MAPPING) : (self.BLOCK_MAPPING) ,
            str(ConfigKeywords.KEY_GLOB_STORE_WIDTH) : (self.GLOB_STORE_WIDTH ) ,
            str(ConfigKeywords.KEY_UNROLL_NUM) : (self.UNROLL_NUM) ,
            str(ConfigKeywords.KEY_REG_PREFETCH) : (self.REG_PREFETCH) ,
            str(ConfigKeywords.KEY_SHARED_PREFETCH) : (self.SHARED_PREFETCH) ,
            str(ConfigKeywords.KEY_LOAD_CONTINUOUS) : (self.LOAD_CONTINUOUS) ,
            str(ConfigKeywords.KEY_STORE_CONTINUOUS) : (self.STORE_CONTINUOUS) ,
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
        self.BLOCK_LAYOUT_Y = config[kw.KEY_BLOCK_LAYOUT_Y]
        self.BLOCK_LAYOUT_X = config[kw.KEY_BLOCK_LAYOUT_X]
        self.WARP_LAYOUT_Y = config[kw.KEY_WARP_LAYOUT_Y]
        self.WARP_LAYOUT_X = config[kw.KEY_WARP_LAYOUT_X]
        self.isATranspose = config[kw.KEY_IS_A_TRANSPOSE]
        self.GLOB_LOAD_WIDTH_A = config[kw.KEY_GLOB_LOAD_WIDTH_A]
        self.GLOB_LOAD_WIDTH_B = config[kw.KEY_GLOB_LOAD_WIDTH_B]
        self.BLOCK_SCATTER_WIDTH_M = config[kw.KEY_BLOCK_SCATTER_WIDTH_M]
        self.BLOCK_SCATTER_WIDTH_N = config[kw.KEY_BLOCK_SCATTER_WIDTH_N]
        self.WARP_SCATTER_WIDTH_M = config[kw.KEY_WARP_SCATTER_WIDTH_M]
        self.WARP_SCATTER_WIDTH_N = config[kw.KEY_WARP_SCATTER_WIDTH_N]
        self.LOCAL_SPLIT_U = config[kw.KEY_LOCAL_SPLIT_U]
        self.BLOCK_MAPPING = config[kw.KEY_BLOCK_MAPPING]
        self.GLOB_STORE_WIDTH = config[kw.KEY_GLOB_STORE_WIDTH]
        self.UNROLL_NUM = config[kw.KEY_UNROLL_NUM]
        self.REG_PREFETCH = config[kw.KEY_REG_PREFETCH]
        self.SHARED_PREFETCH = config[kw.KEY_SHARED_PREFETCH]
        self.LOAD_CONTINUOUS = config[kw.KEY_LOAD_CONTINUOUS]
        self.STORE_CONTINUOUS = config[kw.KEY_STORE_CONTINUOUS]
    
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
        self.BLOCK_LAYOUT_Y = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_Y] 
        self.BLOCK_LAYOUT_X = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_X] 
        self.WARP_LAYOUT_Y = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_Y] 
        self.WARP_LAYOUT_X = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_X] 
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
        self.BLOCK_SCATTER_WIDTH_M  = jsonObj[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_M]
        self.BLOCK_SCATTER_WIDTH_N  = jsonObj[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_N]
        self.WARP_SCATTER_WIDTH_M  = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_M]
        self.WARP_SCATTER_WIDTH_N  = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_N]
        self.LOCAL_SPLIT_U  = jsonObj[ConfigKeywords.KEY_LOCAL_SPLIT_U]
        self.BLOCK_MAPPING  = jsonObj[ConfigKeywords.KEY_BLOCK_MAPPING]
        self.GLOB_STORE_WIDTH   = jsonObj[ConfigKeywords.KEY_GLOB_STORE_WIDTH]
        self.UNROLL_NUM  = jsonObj[ConfigKeywords.KEY_UNROLL_NUM]
        self.REG_PREFETCH  = jsonObj[ConfigKeywords.KEY_REG_PREFETCH]
        self.SHARED_PREFETCH  = jsonObj[ConfigKeywords.KEY_SHARED_PREFETCH]
        self.LOAD_CONTINUOUS  = jsonObj[ConfigKeywords.KEY_LOAD_CONTINUOUS]
        self.STORE_CONTINUOUS  = jsonObj[ConfigKeywords.KEY_STORE_CONTINUOUS]

    
    def check(self) :
        # problem size check
        assert self.M % self.BLOCK_SIZE_M == 0 
        assert self.N % self.BLOCK_SIZE_N == 0 
        assert self.K % self.BLOCK_SIZE_K == 0 
        # assert self.batch >= 1 
        # warp-block validation check
        assert self.BLOCK_SIZE_M % self.THREAD_SIZE_M == 0
        assert self.BLOCK_SIZE_N % self.THREAD_SIZE_N == 0
        assert (self.BLOCK_LAYOUT_Y * self.WARP_LAYOUT_Y) == (self.BLOCK_SIZE_M / self.THREAD_SIZE_M)
        assert (self.BLOCK_LAYOUT_X * self.WARP_LAYOUT_X) == (self.BLOCK_SIZE_N / self.THREAD_SIZE_N)
        assert self.WARP_LAYOUT_X * self.WARP_LAYOUT_Y == self.WARP_SIZE
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
    
    def generateKernelName(self) -> str :
        ret = "kcg_MM_"
        ret += 'b'
        for e in self.batch :
            ret += f"{e}_" 
        ret += f"M{ self.M }" 
        ret += f"N{ self.N }" 
        ret += f"K{ self.K }" 
        ret += f"isAT{ self.isATranspose }" 
        ret += f"W{ self.WARP_SIZE }_" 
        ret += f"BM{ self.BLOCK_SIZE_M }" 
        ret += f"BN{ self.BLOCK_SIZE_N }" 
        ret += f"BK{ self.BLOCK_SIZE_K }" 
        ret += f"TM{ self.THREAD_SIZE_M }" 
        ret += f"TN{ self.THREAD_SIZE_N }" 
        ret += f"BLY{ self.BLOCK_LAYOUT_Y }" 
        ret += f"BLX{ self.BLOCK_LAYOUT_X }" 
        ret += f"WLY{ self.WARP_LAYOUT_Y }" 
        ret += f"WLX{ self.WARP_LAYOUT_X }" 
        ret += f"GLWA{ self.GLOB_LOAD_WIDTH_A }" 
        ret += f"GLWB{ self.GLOB_LOAD_WIDTH_B }" 
        ret += f"BSWM{ self.BLOCK_SCATTER_WIDTH_M }" 
        ret += f"BSWN{ self.BLOCK_SCATTER_WIDTH_N }" 
        ret += f"WSWM{ self.WARP_SCATTER_WIDTH_M }" 
        ret += f"WSWN{ self.WARP_SCATTER_WIDTH_N }" 
        ret += f"LSU{ self.LOCAL_SPLIT_U }" 
        ret += f"Map{ self.BLOCK_MAPPING }" 
        ret += f"GSW{ self.GLOB_STORE_WIDTH }" 
        ret += f"UN{ self.UNROLL_NUM }" 
        ret += f"RP{ self.REG_PREFETCH }" 
        ret += f"SP{ self.SHARED_PREFETCH }" 
        ret += f"LC{ self.LOAD_CONTINUOUS }" 
        ret += f"RC{ self.STORE_CONTINUOUS }" 
        return ret
    
    def assignWithKernelName(self,name : str) -> bool :
        st = 0
        items = name.split('_')
        cfgstr = items[-1]
        basestr = items[-2]
        batches = items[2:-2]
        print('items = ',items)
        if len(batches) > 0 :
            self.batch = []
            for _b in batches :
                if _b.startswith('b') :
                    _b = _b[1:]
                self.batch.append(int(_b))
        else:
            self.batch = []
            
        i_m = basestr.find('M')
        i_n = basestr.find('N')
        i_k = basestr.find('K')
        i_isAT = basestr.find('isAT')
        i_w = basestr.find('W')
        
        self.M = int(basestr[i_m+1:i_n])
        self.N = int(basestr[i_n+1:i_k])
        self.K = int(basestr[i_k+1:i_isAT])
        self.isATranspose = int(basestr[i_isAT + 4:i_w]) > 0
        self.WARP_SIZE = int(basestr[i_w+1:])
        
        i_BM = cfgstr.find('BM') 
        i_BN = cfgstr.find('BN') 
        i_BK = cfgstr.find('BK') 
        i_TM = cfgstr.find('TM') 
        i_TN = cfgstr.find('TN') 
        i_BLY = cfgstr.find('BLY') 
        i_BLX = cfgstr.find('BLX') 
        i_WLY = cfgstr.find('WLY') 
        i_WLX = cfgstr.find('WLX') 
        i_GLWA = cfgstr.find('GLWA') 
        i_GLWB = cfgstr.find('GLWB') 
        i_BSWM = cfgstr.find('BSWM') 
        i_BSWN = cfgstr.find('BSWN') 
        i_WSWM = cfgstr.find('WSWM') 
        i_WSWN = cfgstr.find('WSWN') 
        i_LSU = cfgstr.find('LSU') 
        i_Map = cfgstr.find('Map') 
        i_GSW = cfgstr.find('GSW') 
        i_UN = cfgstr.find('UN') 
        i_RP = cfgstr.find('RP') 
        i_SP = cfgstr.find('SP') 
        i_LC = cfgstr.find('LC') 
        i_RC = cfgstr.find('RC') 
        
        self.BLOCK_SIZE_M = int(cfgstr[i_BM + 2 : i_BN]) 
        self.BLOCK_SIZE_N = int(cfgstr[i_BN + 2 : i_BK]) 
        self.BLOCK_SIZE_K = int(cfgstr[i_BK + 2 : i_TM]) 
        self.THREAD_SIZE_M = int(cfgstr[i_TM + 2 : i_TN]) 
        self.THREAD_SIZE_N = int(cfgstr[i_TN + 2 : i_BLY]) 
        self.BLOCK_LAYOUT_Y = int(cfgstr[i_BLY + 3 : i_BLX]) 
        self.BLOCK_LAYOUT_X = int(cfgstr[i_BLX + 3 : i_WLY]) 
        self.WARP_LAYOUT_Y = int(cfgstr[i_WLY + 3 : i_WLX]) 
        self.WARP_LAYOUT_X = int(cfgstr[i_WLX + 3 : i_GLWA]) 
        self.GLOB_LOAD_WIDTH_A = int(cfgstr[i_GLWA + 4 : i_GLWB]) 
        self.GLOB_LOAD_WIDTH_B = int(cfgstr[i_GLWB + 4 : i_BSWM]) 
        self.BLOCK_SCATTER_WIDTH_M = int(cfgstr[i_BSWM + 4 : i_BSWN]) 
        self.BLOCK_SCATTER_WIDTH_N = int(cfgstr[i_BSWN + 4 : i_WSWM]) 
        self.WARP_SCATTER_WIDTH_M = int(cfgstr[i_WSWM + 4 : i_WSWN]) 
        self.WARP_SCATTER_WIDTH_N = int(cfgstr[i_WSWN + 4 : i_LSU]) 
        self.LOCAL_SPLIT_U = int(cfgstr[i_LSU + 3 : i_Map]) 
        self.BLOCK_MAPPING = int(cfgstr[i_Map + 3 : i_GSW]) 
        self.GLOB_STORE_WIDTH = int(cfgstr[i_GSW + 3 : i_UN]) 
        self.UNROLL_NUM = int(cfgstr[i_UN + 2 : i_RP]) 
        self.REG_PREFETCH = int(cfgstr[i_RP + 2 : i_SP]) 
        self.SHARED_PREFETCH = int(cfgstr[i_SP + 2 : i_LC]) 
        self.LOAD_CONTINUOUS = int(cfgstr[i_LC + 2 : i_RC]) 
        self.STORE_CONTINUOUS = int(cfgstr[i_RC + 2 : ]) 
        return True
        
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
            assert isinstance(batch, List)
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            shapeA = batch + [m,k]
            shapeB = batch + [k,n]
            a = torch.rand(shapeA,dtype=ety, device=f"cuda:{devId}" )
            b = torch.rand(shapeB,dtype=ety, device=f"cuda:{devId}" )
            self.InputTensors_Baseline = [a,b]
        return self.InputTensors_Baseline
            
    def GetBenchmarkInputTensor(self,devId : int) -> List[torch.Tensor] : 
        if self.InputTensors_Benchmark is None :
            [a,b] = self.GetBaselineInputTensor(devId)
            [batch, m,n,k, dtypeInt] = self.BaseArgs.intValues 
            print(f"self.BaseArgs.intValues = {self.BaseArgs.intValues}" )
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            aa = a.transpose(-1,-2).contiguous()
            shapeC = batch + [m,n]
            c = torch.empty(shapeC, dtype=ety, device=f"cuda:{devId}")
            # else :
            #     aa = a.transpose(0,1).contiguous()
            #     c = torch.empty((m,n), dtype=ety, device=f"cuda:{devId}")
            self.InputTensors_Benchmark = [aa,b,c]
        return self.InputTensors_Benchmark
    
    # def GetBenchmarkOutputTensor(self,devId : int) -> List[torch.Tensor] : 
    #     return self.GetBenchmarkInputTensor(devId)[-1]
    
    # def GetBaselineOutputTensor(self) -> List[torch.Tensor] : 
    #     return self.OutputTensor_Baseline
    
    def InitLibInterface(self) :
        if self.CompileKernelMatmul is None or self.SetPlatform is None :
            import importlib.util
            print(f"libdeepgen = {PathManager.kcg_lib_deepgen_path()}",flush=True)
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernelMatmul = mod.compile_mm
            self.SetPlatform = mod.set_platform
            self.SetKernelName = mod.set_kernel_name

        # attn_spec = importlib.util.spec_from_file_location("attention", PathManager.kcg_compiler_attention_path())
        # attn_mod = importlib.util.module_from_spec(attn_spec)
        # attn_spec.loader.exec_module(attn_mod)
        # self.__compile_kernel_FA = attn_mod.compile_attn
    
    def Compile(self, deviceId:int, backendtype : EnumBackendType, arch : str, info : CompileNeededInfo, opt : CompileOption = None) -> Tuple[List,KernelConfigs,CompiledKernel] :
        Print = print
        _backend = 0
        if backendtype.value == EnumBackendType.CUDA.value :
            _backend = 1
        elif backendtype.value == EnumBackendType.HIP.value :
            _backend = 2
        elif backendtype.value == EnumBackendType.MLU.value :  # hanwuji mlu
            _backend = 3
        elif backendtype.value == EnumBackendType.NPU.value :  # huawei npu
            _backend = 4
        else:
            assert False, f'invalid backendtype {backendtype}, Ty is {type(backendtype)}'
        print("compiling matmul",flush=True)
        print("ta=",*info.tsArgs,flush=True)
        print("ba=",info.baseArgs,flush=True)
        Print("===== call InitLibInterface ========",flush=True)
        self.InitLibInterface()
        Print(f"===== call SetPlatform ========, arch = {arch}",flush=True)
        self.SetPlatform(_backend,arch)
        Print("===== call SetKernelName ========",flush=True)
        self.SetKernelName(info.kernelName)
        Print("===== call CompileKernelMatmul ========",flush=True)
        shape, cfg = info.tsArgs
        # batch,m,n,k = shape
        if len(shape[0]) > 0 :
            shape = shape[0] + shape[1:]
        else:
            shape = shape[1:]
        print(f"shape = {shape}, cfg = {cfg}",flush=True)
        def is_power_of_two(num : int) :
            return (num & (num-1)) == 0
        hsacoPath = self.CompileKernelMatmul( shape,cfg)
        # hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,shmBytes = res[0]
        kernelName = info.kernelName
        gridDimX,gridDimY,gridDimZ = info.gridDims
        blockDimX,blockDimY,blockDimZ = info.blockDims
        shmBytes = info.shmBytes
        
        # print(f"blockdims = {blockDimX,blockDimY,blockDimZ}")
        # print(f"griddims = {gridDimX,gridDimY,gridDimZ}")
        # Print("========= hsacoPath = ",hsacoPath)
        Print("========= kernelName = ",kernelName)
        if is_power_of_two(shape[-1]) and is_power_of_two(shape[-2]) and is_power_of_two(shape[-3]) :
            ...
        else:
            hsacoPath = None
        # print(f"==== backend is {backendtype}")
        # print(f"==== shmBytes is {shmBytes}")
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
        if len(self.TuningArgs.batch) == 1 :
            # assert len(a0.shape) == 3, f"shape not match : len(a0.shape)={len(a0.shape)}, TuningArgs.batch={self.TuningArgs.batch}" 
            torchMM = torch.bmm
        for i in range(0,warmupCount) : 
            torchMM(a0,b0)
            packedKernel.run(a,b,c)

    def Test_baseline(self, devId : int) -> Tuple[torch.Tensor,float]:
        [matrixA, matrixB] = self.GetBaselineInputTensor(devId)
        torchMM = torch.matmul
        if len(matrixA.shape) == 3 :
            torchMM = torch.bmm
        
        epsList = []
        for i in range(5) :
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            self.OutputTensor_Baseline = torchMM(matrixA, matrixB)
            ev_end.record()
            torch.cuda.synchronize()
            eps = ev_start.elapsed_time(ev_end)
            epsList.append(eps)
            
        return (self.OutputTensor_Baseline, np.median(epsList))
    
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
            shapeA = batch + [m,k]
            shapeB = batch + [k,n]
            matA = torch.randn(shapeA, dtype= datatype, device=f'cuda:{devId}')
            matB = torch.randn(shapeB, dtype= datatype, device=f'cuda:{devId}')
            self.InputTensors_Baseline = [matA ,matB]
        else:
            matA, matB = self.InputTensors_Baseline
        # init benchmark inputs
        if self.InputTensors_Benchmark is None:
            aUse = None
            if self.TuningArgs.isATranspose :
                atrans = torch.transpose(matA,-1,-2).contiguous()  # 转置会令底层存储不连续，导致失败。必须使其连续
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
            shapeC = b + [m,n]
            ret = torch.empty(shapeC,dtype=dt, device=f'cuda:{devId}')
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
        wlm = config[ConfigKeywords.KEY_WARP_LAYOUT_Y]
        wln = config[ConfigKeywords.KEY_WARP_LAYOUT_X]
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
        
        blm = config[ConfigKeywords.KEY_BLOCK_LAYOUT_Y]
        bln = config[ConfigKeywords.KEY_BLOCK_LAYOUT_X]
        
        wlm = config[ConfigKeywords.KEY_WARP_LAYOUT_Y]
        wln = config[ConfigKeywords.KEY_WARP_LAYOUT_X]
        
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
        
        wswa = config[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_M]
        wswb = config[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_N]
        tswa = config[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_M]
        tswb = config[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_N]
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
    