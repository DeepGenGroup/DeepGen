import torch
# from kcg.KCGCompiler import KCGCompiler
from kcg.CompiledKernel import *
from kcg.Kernel import *
from kcg.Utils import *


@kcg_kernel
def _attention_kernel( a_ptr, b_ptr, c_ptr, out_ptr) : ...

# Call hook. 在这里带入实参并调用

def _attention(a : torch.Tensor, b : torch.Tensor, c : torch.Tensor, d : torch.Tensor):
    # Check constraints.
    dimsizeA = len(a.shape) 
    dimsizeB = len(b.shape)
    dimsizeC = len(c.shape)
    dimsizeD = len(d.shape)
    assert dimsizeA == dimsizeB == dimsizeC == dimsizeD, "ABC must with same dim size"
    if dimsizeA==3:
        assert a.shape[1] == b.shape[0], "AB have Incompatible shape"
    if dimsizeA==4:
        assert a.shape[0] == b.shape[0] == c.shape[0], "ABC must have same batch"
    
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert c.is_contiguous(), "Matrix B must be contiguous"
    
    return _attention_kernel(a, b, c, d)


# public interface:
def getAttentionSignature(dtypeA: torch.dtype, dtypeB : torch.dtype, dtypeC : torch.dtype, dtypeD : torch.dtype) -> dict:
    # signature只和输入的dtype有关，尺寸无关
    a = torch.randn((1024, 1024), device='cpu', dtype=dtypeA)
    b = torch.randn((1024, 1024), device='cpu', dtype=dtypeB)
    c = torch.randn((1024, 1024), device='cpu', dtype=dtypeC)
    # Allocates output.
    M, K = a.shape
    K, N = b.shape
    N, P = c.shape
    d = torch.empty((M, P), device='cpu', dtype=dtypeD)
    # get function signature
    outSignature = _attention(a, b, c, d)
    # print(f"[D] mm signature = {outSignature}, type =  {type(outSignature.values())}",)
    return outSignature


class KernelArgAttention :
    def __init__(self,typeA : EnumKernelDType,typeB : EnumKernelDType,typeC : EnumKernelDType, typeD : EnumKernelDType):
        self.__dataType_A : EnumKernelDType = typeA
        self.__dataType_B : EnumKernelDType = typeB
        self.__dataType_C : EnumKernelDType = typeC
        self.__dataType_D : EnumKernelDType = typeD
        
        self.Br = 0
        self.Bc = 0
        self.Hd = 0
        self.Slice1 = 0
        self.Slice2 = 0
        self.PTr = 0
        self.PTc = 0
        self.OTr = 0
        self.OTc = 0
        self.GLOB_LOAD_WIDTH_Q = 0
        self.GLOB_LOAD_WIDTH_K = 0
        self.GLOB_LOAD_WIDTH_V = 0
        self.BLOCK_LAYOUT_P_Y = 0
        self.BLOCK_LAYOUT_P_X = 0
        self.WARP_LAYOUT_P_Y = 0
        self.WARP_LAYOUT_P_X = 0
        self.BLOCK_SCATTER_WIDTH_Q = 0
        self.BLOCK_SCATTER_WIDTH_K = 0
        self.WARP_SCATTER_WIDTH_Q = 0
        self.WARP_SCATTER_WIDTH_K = 0
        self.BLOCK_LAYOUT_O_Y = 0
        self.BLOCK_LAYOUT_O_X = 0
        self.WARP_LAYOUT_O_Y = 0
        self.WARP_LAYOUT_O_X = 0
        self.BLOCK_SCATTER_WIDTH_P = 0
        self.BLOCK_SCATTER_WIDTH_V = 0
        self.WARP_SCATTER_WIDTH_P = 0
        self.WARP_SCATTER_WIDTH_V = 0
        self.UNROLL_NUM = 0
        self.WARP_SIZE = 0
        self.LOAD_CONTINUOUS_P = 0
        self.LOAD_CONTINUOUS_O = 0
        self.SHARED_PREFETCH_P = 0
        self.REG_PREFETCH_P = 0
        self.REG_PREFETCH_O = 0
    
    def setArgs(self, *args):
        self.Br = args[0], self.Bc = args[1], self.Hd = args[2], self.Slice1 = args[3], self.Slice2 = args[4], 
        self.PTr = args[5], self.PTc = args[6], self.OTr = args[7], self.OTc = args[8],
        # global to shared
        self.GLOB_LOAD_WIDTH_Q = args[9], self.GLOB_LOAD_WIDTH_K = args[10], self.GLOB_LOAD_WIDTH_V = args[11],
        # P = Q * K
        self.BLOCK_LAYOUT_P_Y = args[12], self.BLOCK_LAYOUT_P_X = args[13], self.WARP_LAYOUT_P_Y = args[14], self.WARP_LAYOUT_P_X = args[15],
        self.BLOCK_SCATTER_WIDTH_Q = args[16], self.BLOCK_SCATTER_WIDTH_K = args[17], self.WARP_SCATTER_WIDTH_Q = args[18], self.WARP_SCATTER_WIDTH_K = args[19],
        # O = P * V
        self.BLOCK_LAYOUT_O_Y = args[20], self.BLOCK_LAYOUT_O_X = args[21], self.WARP_LAYOUT_O_Y = args[22], self.WARP_LAYOUT_O_X = args[23], 
        self.BLOCK_SCATTER_WIDTH_P = args[24], self.BLOCK_SCATTER_WIDTH_V = args[25], self.WARP_SCATTER_WIDTH_P = args[26], self.WARP_SCATTER_WIDTH_V = args[27],

        self.UNROLL_NUM = args[28], self.WARP_SIZE = args[29], 
        self.LOAD_CONTINUOUS_P = args[30], self.LOAD_CONTINUOUS_O = args[31], 
        # prefecth
        self.SHARED_PREFETCH_P = args[32], self.REG_PREFETCH_P = args[33], self.REG_PREFETCH_O = args[34],

    def jsonfy(self) : 
        obj = {
           str(ConfigKeywords.KEY_Br)  : self.Br,
           str(ConfigKeywords.KEY_Bc)  : self.Bc,
           str(ConfigKeywords.KEY_Hd)  : self.Hd,
           str(ConfigKeywords.KEY_Slice1)  : self.Slice1,
           str(ConfigKeywords.KEY_Slice2)  : self.Slice2,
           str(ConfigKeywords.KEY_PTr)  : self.PTr,
           str(ConfigKeywords.KEY_PTc)  : self.PTc,
           str(ConfigKeywords.KEY_OTr)  : self.OTr,
           str(ConfigKeywords.KEY_OTc)  : self.OTc,
           str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_Q)  : self.GLOB_LOAD_WIDTH_Q,
           str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_K)  : self.GLOB_LOAD_WIDTH_K,
           str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_V)  : self.GLOB_LOAD_WIDTH_V,
           str(ConfigKeywords.KEY_BLOCK_LAYOUT_P_Y)  : self.BLOCK_LAYOUT_P_Y,
           str(ConfigKeywords.KEY_BLOCK_LAYOUT_P_X)  : self.BLOCK_LAYOUT_P_X,
           str(ConfigKeywords.KEY_WARP_LAYOUT_P_Y)  : self.WARP_LAYOUT_P_Y,
           str(ConfigKeywords.KEY_WARP_LAYOUT_P_X)  : self.WARP_LAYOUT_P_X,
           str(ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_Q)  : self.BLOCK_SCATTER_WIDTH_Q,
           str(ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_K)  : self.BLOCK_SCATTER_WIDTH_K,
           str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_Q)  : self.WARP_SCATTER_WIDTH_Q,
           str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_K)  : self.WARP_SCATTER_WIDTH_K,
           str(ConfigKeywords.KEY_BLOCK_LAYOUT_O_Y)  : self.BLOCK_LAYOUT_O_Y,
           str(ConfigKeywords.KEY_BLOCK_LAYOUT_O_X)  : self.BLOCK_LAYOUT_O_X,
           str(ConfigKeywords.KEY_WARP_LAYOUT_O_Y)  : self.WARP_LAYOUT_O_Y,
           str(ConfigKeywords.KEY_WARP_LAYOUT_O_X)  : self.WARP_LAYOUT_O_X,
           str(ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_P)  : self.BLOCK_SCATTER_WIDTH_P,
           str(ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_V)  : self.BLOCK_SCATTER_WIDTH_V,
           str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_P)  : self.WARP_SCATTER_WIDTH_P,
           str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_V)  : self.WARP_SCATTER_WIDTH_V,
           str(ConfigKeywords.KEY_UNROLL_NUM)  : self.UNROLL_NUM,
           str(ConfigKeywords.KEY_WARP_SIZE)  : self.WARP_SIZE,
           str(ConfigKeywords.KEY_LOAD_CONTINUOUS_P)  : self.LOAD_CONTINUOUS_P,
           str(ConfigKeywords.KEY_LOAD_CONTINUOUS_O)  : self.LOAD_CONTINUOUS_O,
           str(ConfigKeywords.KEY_SHARED_PREFETCH_P)  : self.SHARED_PREFETCH_P,
           str(ConfigKeywords.KEY_REG_PREFETCH_P)  : self.REG_PREFETCH_P,
           str(ConfigKeywords.KEY_REG_PREFETCH_O)  : self.REG_PREFETCH_O,
        }
        return obj
    
    def assignWithJson(self, jsonObj) : 
        self.Br = jsonObj[ConfigKeywords.KEY_Br]
        self.Bc = jsonObj[ConfigKeywords.KEY_Bc]
        self.Hd = jsonObj[ConfigKeywords.KEY_Hd]
        self.Slice1 = jsonObj[ConfigKeywords.KEY_Slice1]
        self.Slice2 = jsonObj[ConfigKeywords.KEY_Slice2]
        self.PTr = jsonObj[ConfigKeywords.KEY_PTr]
        self.PTc = jsonObj[ConfigKeywords.KEY_PTc]
        self.OTr = jsonObj[ConfigKeywords.KEY_OTr]
        self.OTc = jsonObj[ConfigKeywords.KEY_OTc]
        self.GLOB_LOAD_WIDTH_Q = jsonObj[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_Q]
        self.GLOB_LOAD_WIDTH_K = jsonObj[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_K]
        self.GLOB_LOAD_WIDTH_V = jsonObj[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_V]
        self.BLOCK_LAYOUT_P_Y = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_P_Y]
        self.BLOCK_LAYOUT_P_X = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_P_X]
        self.WARP_LAYOUT_P_Y = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_P_Y]
        self.WARP_LAYOUT_P_X = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_P_X]
        self.BLOCK_SCATTER_WIDTH_Q = jsonObj[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_Q]
        self.BLOCK_SCATTER_WIDTH_K = jsonObj[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_K]
        self.WARP_SCATTER_WIDTH_Q = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_Q]
        self.WARP_SCATTER_WIDTH_K = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_K]
        self.BLOCK_LAYOUT_O_Y = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_O_Y]
        self.BLOCK_LAYOUT_O_X = jsonObj[ConfigKeywords.KEY_BLOCK_LAYOUT_O_X]
        self.WARP_LAYOUT_O_Y = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_O_Y]
        self.WARP_LAYOUT_O_X = jsonObj[ConfigKeywords.KEY_WARP_LAYOUT_O_X]
        self.BLOCK_SCATTER_WIDTH_P = jsonObj[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_P]
        self.BLOCK_SCATTER_WIDTH_V = jsonObj[ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_V]
        self.WARP_SCATTER_WIDTH_P = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_P]
        self.WARP_SCATTER_WIDTH_V = jsonObj[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_V]
        self.UNROLL_NUM = jsonObj[ConfigKeywords.KEY_UNROLL_NUM]
        self.WARP_SIZE = jsonObj[ConfigKeywords.KEY_WARP_SIZE]
        self.LOAD_CONTINUOUS_P = jsonObj[ConfigKeywords.KEY_LOAD_CONTINUOUS_P]
        self.LOAD_CONTINUOUS_O = jsonObj[ConfigKeywords.KEY_LOAD_CONTINUOUS_O]
        self.SHARED_PREFETCH_P = jsonObj[ConfigKeywords.KEY_SHARED_PREFETCH_P]
        self.REG_PREFETCH_P = jsonObj[ConfigKeywords.KEY_REG_PREFETCH_P]
        self.REG_PREFETCH_O = jsonObj[ConfigKeywords.KEY_REG_PREFETCH_O]
    
    def check(self) :
        pass
        # problem size check
        # assert self.M % self.BLOCK_SIZE_M == 0 
        # assert self.N % self.BLOCK_SIZE_N == 0 
        # assert self.K % self.BLOCK_SIZE_K == 0 
        # assert self.batch >= 1 
        # warp-block validation check
        # assert self.BLOCK_SIZE_M % self.THREAD_SIZE_M == 0
        # assert self.BLOCK_SIZE_N % self.THREAD_SIZE_N == 0
        # assert (self.BLOCK_LAYOUT_M * self.WARP_LAYOUT_M) == (self.BLOCK_SIZE_M / self.THREAD_SIZE_M)
        # assert (self.BLOCK_LAYOUT_N * self.WARP_LAYOUT_N) == (self.BLOCK_SIZE_N / self.THREAD_SIZE_N)
        # assert self.WARP_LAYOUT_N * self.WARP_LAYOUT_M == self.WARP_SIZE
        # # shm size check
        # assert 2*(self.BLOCK_SIZE_M + self.BLOCK_SIZE_N) * self.BLOCK_SIZE_K <= 65536
        
        print("===== config check ok!")
    
    def dtype(self,index:str)->EnumKernelDType :
        if index=='A':
            return self.__dataType_A
        if index=='B':
            return self.__dataType_B
        if index=='C':
            return self.__dataType_C
    
    def dtypeTorch(self,index:str)->torch.dtype:
        if index=='A':
            return ToTorchType(self.__dataType_A)
        if index=='B':
            return ToTorchType(self.__dataType_B)
        if index=='C':
            return ToTorchType(self.__dataType_C)
    
    def __str__(self):
        retstr = '{\n'
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_M)}\" :  {str(self.BLOCK_SIZE_M)} , \n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_N)}\"  :  {str(self.BLOCK_SIZE_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_K)}\"  :  {str(self.BLOCK_SIZE_K)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SIZE_M)}\"  :  {str(self.THREAD_SIZE_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SIZE_N)}\"  :  {str(self.THREAD_SIZE_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SIZE)}\"  :  {str(self.WARP_SIZE)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_LAYOUT_M)}\"  :  {str(self.BLOCK_LAYOUT_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_LAYOUT_N)}\"  :  {str(self.BLOCK_LAYOUT_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_LAYOUT_M)}\"  :  {str(self.WARP_LAYOUT_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_LAYOUT_N)}\"  :  {str(self.WARP_LAYOUT_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_A)}\"  :  {self.__dataType_A} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_B)}\"  :  {self.__dataType_B} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_C)}\"  :  {self.__dataType_C} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_M)}\"  :  {str(self.M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_N)}\"  :  {str(self.N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_K)}\"  :  {str(self.K)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BATCH)}\"  :  {str(self.batch)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_IS_A_TRANSPOSE)}\"  :  {str(self.isATranspose)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A)}\"  :  {str(self.GLOB_LOAD_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B)}\"  :  {str(self.GLOB_LOAD_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A)}\"  :  {str(self.WARP_SCATTER_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B)}\"  :  {str(self.WARP_SCATTER_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A)}\"  :  {str(self.THREAD_SCATTER_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B)}\"  :  {str(self.THREAD_SCATTER_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_LOCAL_SPLIT_U)}\"  :  {str(self.LOCAL_SPLIT_U)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_MAPPING)}\"  :  {str(self.BLOCK_MAPPING)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_STORE_WIDTH)}\"  :  {str(self.GLOB_STORE_WIDTH )} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_UNROLL_NUM)}\"  :  {str(self.UNROLL_NUM)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_REG_PREFETCH)}\"  :  {str(self.REG_PREFETCH)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_SHARED_PREFETCH)}\"  :  {str(self.SHARED_PREFETCH)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_LOAD_CONTINUOUS)}\"  :  {str(self.LOAD_CONTINUOUS)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_REDUCE_C_CONTINUOUS)}\"  :  {str(self.REDUCE_C_CONTINUOUS)} \n"
        retstr += '}'
        return retstr




class AttentionBaseArgs(OpBaseArgs) :
    def __init__(self):
        # [M*K]*[K*N] -> softmax([M*N]) -> [M*N] * [N*P] = [M*P]
        # inputs : [M*K],[K*N],[N*P]
        self.operatorKind = EnumOperator.Attention
        self.argDict = {
            "batch" : 0,
            "head_num" : 0,
            "seq_len" : 0,
            "head_dim" : 0,
            "kind" : self.operatorKind,
            "dtype" : 0
        }
        # memref<1x32x128x2048xf32, 1>, %arg1: memref<1x32x128x2048xf32, 1>, %arg2: memref<1x32x2048x128xf32, 1>, %arg3: memref<1x32x2048x128xf32, 1>
    def getTorchDType(self) -> torch.dtype:
        dtInt = self.values[-1]
        return ToTorchType(EnumKernelDType(dtInt))        
    
    def getArgList(self) -> List :
        return self.values[0:4]
    
    def parseFromTemplateDict(self,templateDict : Dict):
        batch = templateDict[ConfigKeywords.KEY_Br][0]
        m = templateDict[ConfigKeywords.KEY_Bc][0]
        n = templateDict[ConfigKeywords.KEY_Hd][0]
        dtype : int = templateDict[ConfigKeywords.KEY_DTYPE_C][0]
        self.values = [batch,m,n,k,dtype]
        
    def parseFromJsonfile(self,path : str):
        import json
        obj = None
        with open(path) as f :
            obj = json.load(f)
        self.values = [obj['b'],obj['m'],obj['n'],obj['k'], obj['dtype']]
        # self.operatorKind = obj['kind']
        
    def dumpToJson(self,path : str):
        import json
        self.argDict["kind"] = self.operatorKind
        self.argDict["b"] = self.values[0]
        self.argDict["m"] = self.values[1]
        self.argDict["n"] = self.values[2]
        self.argDict["k"] = self.values[3]
        self.argDict["dtype"] = self.values[4]
        with open(path,'w') as f:
            json.dump(self.argDict,f)
        


class AttentionTuningArgs(TuningArgsInterface) :
    def __init__(self,m = 0,n = 0,k = 0,batch = 1,typeA : EnumKernelDType = EnumKernelDType.float32,typeB : EnumKernelDType = EnumKernelDType.float32, typeC : EnumKernelDType = EnumKernelDType.float32):
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
        self.__dataType_A : EnumKernelDType = typeA
        self.__dataType_B : EnumKernelDType = typeB
        self.__dataType_C : EnumKernelDType = typeC
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
    
    def setArgs(self, *args):
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
            str(ConfigKeywords.KEY_DTYPE_A) : int(self.__dataType_A),
            str(ConfigKeywords.KEY_DTYPE_B) : int(self.__dataType_B), 
            str(ConfigKeywords.KEY_DTYPE_C) : int(self.__dataType_C), 
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
    
    def assignWithEncoder(self, cfgstr : int, tse : TuningSpaceEncoder) : 
        kw = ConfigKeywords    
        config = tse.decode(cfgstr)
        self.M , self.N, self.K , self.batch = config[kw.KEY_M],config[kw.KEY_N],config[kw.KEY_K],config[kw.KEY_BATCH]
        self.__dataType_A, self.__dataType_B, self.__dataType_C = EnumKernelDType(config[kw.KEY_DTYPE_A]), EnumKernelDType(config[kw.KEY_DTYPE_B]),EnumKernelDType(config[kw.KEY_DTYPE_C])
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
        self.__dataType_A=  int(jsonObj[ConfigKeywords.KEY_DTYPE_A])
        self.__dataType_B = int(jsonObj[ConfigKeywords.KEY_DTYPE_B])
        self.__dataType_C = int(jsonObj[ConfigKeywords.KEY_DTYPE_C])
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
            return self.__dataType_A
        if index=='B':
            return self.__dataType_B
        if index=='C':
            return self.__dataType_C
    
    def dtypeTorch(self,index:str)->torch.dtype:
        if index=='A':
            return ToTorchType(self.__dataType_A)
        if index=='B':
            return ToTorchType(self.__dataType_B)
        if index=='C':
            return ToTorchType(self.__dataType_C)
    
    def __str__(self):
        retstr = '{\n'
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_M)}\" :  {str(self.BLOCK_SIZE_M)} , \n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_N)}\"  :  {str(self.BLOCK_SIZE_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_SIZE_K)}\"  :  {str(self.BLOCK_SIZE_K)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SIZE_M)}\"  :  {str(self.THREAD_SIZE_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SIZE_N)}\"  :  {str(self.THREAD_SIZE_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SIZE)}\"  :  {str(self.WARP_SIZE)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_LAYOUT_M)}\"  :  {str(self.BLOCK_LAYOUT_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_LAYOUT_N)}\"  :  {str(self.BLOCK_LAYOUT_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_LAYOUT_M)}\"  :  {str(self.WARP_LAYOUT_M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_LAYOUT_N)}\"  :  {str(self.WARP_LAYOUT_N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_A)}\"  :  {self.__dataType_A} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_B)}\"  :  {self.__dataType_B} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_DTYPE_C)}\"  :  {self.__dataType_C} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_M)}\"  :  {str(self.M)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_N)}\"  :  {str(self.N)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_K)}\"  :  {str(self.K)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BATCH)}\"  :  {str(self.batch)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_IS_A_TRANSPOSE)}\"  :  {str(self.isATranspose)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A)}\"  :  {str(self.GLOB_LOAD_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B)}\"  :  {str(self.GLOB_LOAD_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A)}\"  :  {str(self.WARP_SCATTER_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B)}\"  :  {str(self.WARP_SCATTER_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A)}\"  :  {str(self.THREAD_SCATTER_WIDTH_A)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B)}\"  :  {str(self.THREAD_SCATTER_WIDTH_B)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_LOCAL_SPLIT_U)}\"  :  {str(self.LOCAL_SPLIT_U)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_BLOCK_MAPPING)}\"  :  {str(self.BLOCK_MAPPING)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_GLOB_STORE_WIDTH)}\"  :  {str(self.GLOB_STORE_WIDTH )} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_UNROLL_NUM)}\"  :  {str(self.UNROLL_NUM)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_REG_PREFETCH)}\"  :  {str(self.REG_PREFETCH)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_SHARED_PREFETCH)}\"  :  {str(self.SHARED_PREFETCH)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_LOAD_CONTINUOUS)}\"  :  {str(self.LOAD_CONTINUOUS)} ,\n"
        retstr += f" \"{str(ConfigKeywords.KEY_REDUCE_C_CONTINUOUS)}\"  :  {str(self.REDUCE_C_CONTINUOUS)} \n"
        retstr += '}'
        return retstr


class MatmulOp(OpInterface) :
    def __init__(self):
        super().__init__()
        self.TuningArgs = MatmulTuningArgs()
        self.BaseArgs = MatmulBaseArgs()
    
    @staticmethod
    def Compile(tuningArgs : TuningArgsInterface, deviceId:int, backendtype : EnumBackendType, arch : str) -> Tuple[TuningArgsInterface,KernelConfigs,CompiledKernel] :
        Print = print
        # compile kernel
        # Print("===== KCGCompiler ctor ========")
        assert isinstance(tuningArgs, MatmulTuningArgs)
        kernelCompiler = KCGCompiler()
        _backend = 0
        if backendtype.value == EnumBackendType.CUDA.value :
            _backend = 1
        elif backendtype.value == EnumBackendType.HIP.value :
            _backend = 2
        else:
            assert False, f'invalid backendtype {backendtype}, Ty is {type(backendtype)}'
        kernelCompiler.set_platform(_backend,arch)
        # Print("===== call compileKernel(kpm)[0] ========")
        hsacoPath,kernelName,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,shmBytes = kernelCompiler.compileKernel(tuningArgs)[0] 
        # print(f"blockdims = {blockDimX,blockDimY,blockDimZ}")
        # print(f"griddims = {gridDimX,gridDimY,gridDimZ}")
        # Print("========= hsacoPath = ",hsacoPath)
        # Print("========= kernelName = ",kernelName)
        # print(f"==== backend is {backendtype}")
        inConfig = KernelConfigs(hsacoPath,kernelName, [tuningArgs.dtypeTorch('A'),tuningArgs.dtypeTorch('B'),tuningArgs.dtypeTorch('C')],backendtype)
        inConfig.m_gridDims = [gridDimX,gridDimY,gridDimZ]
        inConfig.m_blockDims = [blockDimX,blockDimY,blockDimZ]
        inConfig.operatorKind = EnumOperator.Matmul
        inConfig.shmBytes = shmBytes
        packedKernel = CompiledKernelFactory.getKernel(inConfig, deviceId)
        return (tuningArgs,inConfig,packedKernel)  # 
  
    
    @staticmethod
    def GetCompiledKernel(info : KernelConfigs, deviceId : int) -> CompiledKernel :
        # signature = getMatmulSignature(info.kernelParam.dtypeTorch('A'),info.kernelParam.dtypeTorch('B'),info.kernelParam.dtypeTorch('C'))
        signature = MatmulOp.GetSignature(info.dtypes)
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
    
    @staticmethod
    def GetSignature(dtypes : List[torch.dtype]) -> dict :
            # signature只和输入的dtype有关，尺寸无关
        dtypeA = dtypes[0]
        dtypeB = dtypes[1]
        dtypeC = dtypes[2]
        a = torch.randn((1024, 1024), device='cpu', dtype=dtypeA)
        b = torch.randn((1024, 1024), device='cpu', dtype=dtypeB)
        # Allocates output.
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device='cpu', dtype=dtypeC)
        # get function signature
        outSignature = _matmul(a, b, c)
        # print(f"[D] mm signature = {outSignature}, type =  {type(outSignature.values())}",)
        return outSignature
    
    @staticmethod
    def SetTuningArgs(tuningArgs : List) :
        MatmulOp.TuningArgs.setArgs(*tuningArgs)
    
    
    @staticmethod
    def InitBaseArgs(args : List) :
        m,n,k,batch , dtypeInt = args
        MatmulOp.BaseArgs.values = [m,n,k,batch,dtypeInt]
        ety = EnumKernelDType(dtypeInt)
        MatmulOp.TuningArgs = MatmulTuningArgs(m,n,k,batch,ety,ety,ety)  
    
    @staticmethod
    def Test_warmup(inputTensors : List[torch.Tensor], outputTensor : torch.Tensor , packedKernel : CompiledKernel, warmupCount : int) :
        a,b = MatmulOp.GetSelfArglistTensors(inputTensors)
        a0,b0 = inputTensors
        # warmup
        torchMM = torch.matmul
        if MatmulOp.TuningArgs.batch > 1:
            torchMM = torch.bmm
        for i in range(0,warmupCount) : 
            torchMM(a0,b0)
            packedKernel.run(a,b,outputTensor)

    
    @staticmethod
    def Test_baseline(inputs : List[torch.Tensor]) -> Tuple[torch.Tensor,float]:
        matrixA = inputs[0]
        matrixB = inputs[1]
        torchMM = torch.matmul
        if len(matrixA.shape) > 2 :
            torchMM = torch.bmm
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        resultTensor = torchMM(matrixA, matrixB)
        ev_end.record()
        torch.cuda.synchronize()
        eps = ev_start.elapsed_time(ev_end)
        return (resultTensor, eps)
    
    @staticmethod
    def Test_benchmark(argList : List[torch.Tensor], packedKernel : CompiledKernel, 
                  start_event : torch.cuda.Event, end_event : torch.cuda.Event) -> Tuple[torch.Tensor,float]: 
        a,b,c = argList
        start_event.record()
        packedKernel.run(a,b,c)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        return (c,elapsed_time)
    
    @staticmethod
    def GetSelfArglistTensors(inputTensors : List[torch.Tensor]) -> List[torch.Tensor] :
        d0,d1 = 0,1
        matA,matB = inputTensors
        if len(matA.shape) == 3 :
            d0,d1 = 1,2
        atrans = torch.transpose(matA,d0,d1).contiguous()  # 转置会令底层存储不连续，导致失败。必须使其连续
        assert(matA.is_contiguous())
        assert(matB.is_contiguous())
        assert(atrans.is_contiguous())
        assert(isinstance(MatmulOp.TuningArgs, MatmulTuningArgs))
        if MatmulOp.TuningArgs.batch > 1:
            b, M, K = matA.shape
            b, K, N = matB.shape
        res = []
        aUse = None
        
        if MatmulOp.TuningArgs.isATranspose :
            aUse = atrans
        else:
            aUse = matA
        return [aUse, matB]
        # # 计算torch的eps
        # if self.torch_eps <= 0 or self.matD is None:
        #     self._init_torch_eps()
        
        # # benchmark
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # for i in range(0,benchmarkCount) : 
        #     self.matC,eps = self._inner_test_kcg(aUse, self.matB, self.matC, packedKernel, start_event, end_event)
        #     res.append(eps)
        # print("c=",self.matC)
    
    @staticmethod
    def InitInputTensors(inConfig:KernelConfigs, devId : int) -> List[torch.Tensor] :
        m,n,k,batch = MatmulOp.BaseArgs[0:4]
        if batch > 1:
            matA = torch.randn(batch,m,k,dtype=inConfig.dtypes[0], device=f'cuda:{devId}')
            matB = torch.randn(batch,k,n,dtype=inConfig.dtypes[1], device=f'cuda:{devId}')
        else:
            matA = torch.randn(m,k,dtype=inConfig.dtypes[0], device=f'cuda:{devId}')
            matB = torch.randn(k,n,dtype=inConfig.dtypes[1], device=f'cuda:{devId}')
        return [matA,matB]
    
    @staticmethod
    def InitBaselineOutputTensor(inConfig:KernelConfigs, devId : int) -> torch.Tensor :
        m,n,k,batch = MatmulOp.BaseArgs[0:4]
        if batch > 1:
            ret = torch.empty(batch,m,n,dtype=inConfig.dtypes[0], device=f'cuda:{devId}')
        else:
            ret = torch.empty(m,n,dtype=inConfig.dtypes[0], device=f'cuda:{devId}')
        return ret
        
    
    @staticmethod
    def InitInputTensorsWithDatalist( dataList, datatype, devId) -> List[torch.Tensor]:
        batch, m,n,k = dataList
        if batch > 1:
            matA = torch.randn(batch,m,k, dtype= datatype, device=f'cuda:{devId}')
            matB = torch.randn(batch,k,n, dtype= datatype, device=f'cuda:{devId}')
        else:
            matA = torch.randn(m,k, dtype= datatype, device=f'cuda:{devId}')
            matB = torch.randn(k,n, dtype= datatype, device=f'cuda:{devId}')
        return [matA,matB]
    