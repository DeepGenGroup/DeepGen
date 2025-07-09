import importlib
import torch
import torch.nn.functional as F
from kcg.Kernel import *
from kcg.HIPCompiler import *


@kcg_kernel
def _attention_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr
):
    'DUMP CODES'
    pass
 
# Call hook. 在这里带入实参并调用
def _attention(a : torch.Tensor, b : torch.Tensor, c : torch.Tensor, out : torch.Tensor):
    dimsizeA = len(a.shape) 
    dimsizeB = len(b.shape)
    dimsizeC = len(c.shape)
    dimsizeD = len(out.shape)
    assert dimsizeA == dimsizeB == dimsizeC == dimsizeD, "ABC must with same dim size"
    return _attention_kernel(
        a, b, c, out
    )

# 基础参数
class AttentionBaseArgs(OpBaseArgs) :
    def __init__(self):
        super().__init__()
        self.operatorKind = EnumOperator.Attention
        self.argDict = {
            "kind" : self.operatorKind,
            "shape" : [0,0,0,0],
            "dtype" : 0
        }
        # self.intValues : [shape : List[int] , dtypeInt]
    
    def getEnumDType(self):
        return EnumKernelDType(self.intValues[1])
    
    def getIntDatalist(self) -> List[int] :
        # get shape of attention
        return self.intValues[0]
    
        
    def parseFromJsonfile(self,path : str):
        import json
        obj = None
        with open(path) as f :
            obj = json.load(f)
        self.intValues = [obj['shape'] , obj['dtype']]
        # print(f"[ attention ] shape,dtype = {self.intValues}")
        # self.operatorKind = obj['kind']
        
    def dumpToJson(self,path : str):
        import json
        self.argDict["kind"] = self.operatorKind
        self.argDict["shape"] = self.intValues[0]  #  batch, head_num, seq_len, head_dim = shape
        self.argDict["dtype"] = self.intValues[1]
        with open(path,'w') as f:
            json.dump(self.argDict,f)
        

# 调优参数
class AttentionTuningArgs(TuningArgsInterface) :
    def __init__(self, enumDType : EnumKernelDType = EnumKernelDType.float32):
        super().__init__()
        self.basearg = AttentionBaseArgs()
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
        
    def assignWithList(self, *args):
        self.Br= args[0]
        self.Bc= args[1]
        self.Hd= args[2]
        self.Slice1= args[3]
        self.Slice2= args[4]
        self.PTr= args[5]
        self.PTc= args[6]
        self.OTr= args[7]
        self.OTc= args[8]
        self.GLOB_LOAD_WIDTH_Q= args[9]
        self.GLOB_LOAD_WIDTH_K= args[10]
        self.GLOB_LOAD_WIDTH_V= args[11]
        self.BLOCK_LAYOUT_P_Y= args[12]
        self.BLOCK_LAYOUT_P_X= args[13]
        self.WARP_LAYOUT_P_Y= args[14]
        self.WARP_LAYOUT_P_X= args[15]
        self.BLOCK_SCATTER_WIDTH_Q= args[16]
        self.BLOCK_SCATTER_WIDTH_K= args[17]
        self.WARP_SCATTER_WIDTH_Q= args[18]
        self.WARP_SCATTER_WIDTH_K= args[19]
        self.BLOCK_LAYOUT_O_Y= args[20]
        self.BLOCK_LAYOUT_O_X= args[21]
        self.WARP_LAYOUT_O_Y= args[22]
        self.WARP_LAYOUT_O_X= args[23]
        self.BLOCK_SCATTER_WIDTH_P= args[24]
        self.BLOCK_SCATTER_WIDTH_V= args[25]
        self.WARP_SCATTER_WIDTH_P= args[26]
        self.WARP_SCATTER_WIDTH_V= args[27]
        self.UNROLL_NUM= args[28]
        self.WARP_SIZE= args[29]
        self.LOAD_CONTINUOUS_P= args[30]
        self.LOAD_CONTINUOUS_O= args[31]
        self.SHARED_PREFETCH_P= args[32]
        self.REG_PREFETCH_P= args[33]
        self.REG_PREFETCH_O= args[34]
        
        # gridSize = [int(shape[2]/cfg[1]), shape[1], shape[0]]  # bx, by, bz
        # blockSize = [cfg[-1][0]]  # tx
        # sharedSize = cfg[-1][1]  # shared memroy size
    def getCompileNeededInfo(self) -> CompileNeededInfo : 
        ...
    
    def jsonfy(self) : 
        
        obj = {
            ConfigKeywords.KEY_Br : self.Br,
            ConfigKeywords.KEY_Bc : self.Bc,
            ConfigKeywords.KEY_Hd : self.Hd,
            ConfigKeywords.KEY_Slice1 : self.Slice1,
            ConfigKeywords.KEY_Slice2 : self.Slice2,
            ConfigKeywords.KEY_PTr : self.PTr,
            ConfigKeywords.KEY_PTc : self.PTc,
            ConfigKeywords.KEY_OTr : self.OTr,
            ConfigKeywords.KEY_OTc : self.OTc,
            ConfigKeywords.KEY_GLOB_LOAD_WIDTH_Q : self.GLOB_LOAD_WIDTH_Q,
            ConfigKeywords.KEY_GLOB_LOAD_WIDTH_K : self.GLOB_LOAD_WIDTH_K,
            ConfigKeywords.KEY_GLOB_LOAD_WIDTH_V : self.GLOB_LOAD_WIDTH_V,
            ConfigKeywords.KEY_BLOCK_LAYOUT_P_Y : self.BLOCK_LAYOUT_P_Y,
            ConfigKeywords.KEY_BLOCK_LAYOUT_P_X : self.BLOCK_LAYOUT_P_X,
            ConfigKeywords.KEY_WARP_LAYOUT_P_Y : self.WARP_LAYOUT_P_Y,
            ConfigKeywords.KEY_WARP_LAYOUT_P_X : self.WARP_LAYOUT_P_X,
            ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_Q : self.BLOCK_SCATTER_WIDTH_Q,
            ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_K : self.BLOCK_SCATTER_WIDTH_K,
            ConfigKeywords.KEY_WARP_SCATTER_WIDTH_Q : self.WARP_SCATTER_WIDTH_Q,
            ConfigKeywords.KEY_WARP_SCATTER_WIDTH_K : self.WARP_SCATTER_WIDTH_K,
            ConfigKeywords.KEY_BLOCK_LAYOUT_O_Y : self.BLOCK_LAYOUT_O_Y,
            ConfigKeywords.KEY_BLOCK_LAYOUT_O_X : self.BLOCK_LAYOUT_O_X,
            ConfigKeywords.KEY_WARP_LAYOUT_O_Y : self.WARP_LAYOUT_O_Y,
            ConfigKeywords.KEY_WARP_LAYOUT_O_X : self.WARP_LAYOUT_O_X,
            ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_P : self.BLOCK_SCATTER_WIDTH_P,
            ConfigKeywords.KEY_BLOCK_SCATTER_WIDTH_V : self.BLOCK_SCATTER_WIDTH_V,
            ConfigKeywords.KEY_WARP_SCATTER_WIDTH_P : self.WARP_SCATTER_WIDTH_P,
            ConfigKeywords.KEY_WARP_SCATTER_WIDTH_V : self.WARP_SCATTER_WIDTH_V,
            ConfigKeywords.KEY_UNROLL_NUM : self.UNROLL_NUM,
            ConfigKeywords.KEY_WARP_SIZE : self.WARP_SIZE,
            ConfigKeywords.KEY_LOAD_CONTINUOUS_P : self.LOAD_CONTINUOUS_P,
            ConfigKeywords.KEY_LOAD_CONTINUOUS_O : self.LOAD_CONTINUOUS_O,
            ConfigKeywords.KEY_SHARED_PREFETCH_P : self.SHARED_PREFETCH_P,
            ConfigKeywords.KEY_REG_PREFETCH_P : self.REG_PREFETCH_P,
            ConfigKeywords.KEY_REG_PREFETCH_O : self.REG_PREFETCH_O,
            
            ConfigKeywords.KEY_BLOCK_DIM_X : self.blockDimX ,
            ConfigKeywords.KEY_BLOCK_DIM_Y : self.blockDimY ,
            ConfigKeywords.KEY_BLOCK_DIM_Z : self.blockDimZ ,
            ConfigKeywords.KEY_SHM_BYTES : self.shmBytes ,
            ConfigKeywords.KEY_GRID_DIM_X : self.gridDimX ,
            ConfigKeywords.KEY_GRID_DIM_Y : self.gridDimY ,
            ConfigKeywords.KEY_GRID_DIM_Z : self.gridDimZ 
        }
        return obj
    
    def assignWithDict(self, config : Dict) :
        kw = ConfigKeywords    
        self.Br = config[kw.KEY_Br]
        self.Bc = config[kw.KEY_Bc]
        self.Hd = config[kw.KEY_Hd]
        self.Slice1 = config[kw.KEY_Slice1]
        self.Slice2 = config[kw.KEY_Slice2]
        self.PTr = config[kw.KEY_PTr]
        self.PTc = config[kw.KEY_PTc]
        self.OTr = config[kw.KEY_OTr]
        self.OTc = config[kw.KEY_OTc]
        self.GLOB_LOAD_WIDTH_Q = config[kw.KEY_GLOB_LOAD_WIDTH_Q]
        self.GLOB_LOAD_WIDTH_K = config[kw.KEY_GLOB_LOAD_WIDTH_K]
        self.GLOB_LOAD_WIDTH_V = config[kw.KEY_GLOB_LOAD_WIDTH_V]
        self.BLOCK_LAYOUT_P_Y = config[kw.KEY_BLOCK_LAYOUT_P_Y]
        self.BLOCK_LAYOUT_P_X = config[kw.KEY_BLOCK_LAYOUT_P_X]
        self.WARP_LAYOUT_P_Y = config[kw.KEY_WARP_LAYOUT_P_Y]
        self.WARP_LAYOUT_P_X = config[kw.KEY_WARP_LAYOUT_P_X]
        self.BLOCK_SCATTER_WIDTH_Q = config[kw.KEY_BLOCK_SCATTER_WIDTH_Q]
        self.BLOCK_SCATTER_WIDTH_K = config[kw.KEY_BLOCK_SCATTER_WIDTH_K]
        self.WARP_SCATTER_WIDTH_Q = config[kw.KEY_WARP_SCATTER_WIDTH_Q]
        self.WARP_SCATTER_WIDTH_K = config[kw.KEY_WARP_SCATTER_WIDTH_K]
        self.BLOCK_LAYOUT_O_Y = config[kw.KEY_BLOCK_LAYOUT_O_Y]
        self.BLOCK_LAYOUT_O_X = config[kw.KEY_BLOCK_LAYOUT_O_X]
        self.WARP_LAYOUT_O_Y = config[kw.KEY_WARP_LAYOUT_O_Y]
        self.WARP_LAYOUT_O_X = config[kw.KEY_WARP_LAYOUT_O_X]
        self.BLOCK_SCATTER_WIDTH_P = config[kw.KEY_BLOCK_SCATTER_WIDTH_P]
        self.BLOCK_SCATTER_WIDTH_V = config[kw.KEY_BLOCK_SCATTER_WIDTH_V]
        self.WARP_SCATTER_WIDTH_P = config[kw.KEY_WARP_SCATTER_WIDTH_P]
        self.WARP_SCATTER_WIDTH_V = config[kw.KEY_WARP_SCATTER_WIDTH_V]
        self.UNROLL_NUM = config[kw.KEY_UNROLL_NUM]
        self.WARP_SIZE = config[kw.KEY_WARP_SIZE]
        self.LOAD_CONTINUOUS_P = config[kw.KEY_LOAD_CONTINUOUS_P]
        self.LOAD_CONTINUOUS_O = config[kw.KEY_LOAD_CONTINUOUS_O]
        self.SHARED_PREFETCH_P = config[kw.KEY_SHARED_PREFETCH_P]
        self.REG_PREFETCH_P = config[kw.KEY_REG_PREFETCH_P]
        self.REG_PREFETCH_O = config[kw.KEY_REG_PREFETCH_O]
        
        self.blockDimX = config[kw.KEY_BLOCK_DIM_X]
        self.blockDimY = config[kw.KEY_BLOCK_DIM_Y]
        self.blockDimZ = config[kw.KEY_BLOCK_DIM_Z]
        self.shmBytes = config[kw.KEY_SHM_BYTES]
        self.gridDimX = config[kw.KEY_GRID_DIM_X]
        self.gridDimY = config[kw.KEY_GRID_DIM_Y]
        self.gridDimZ = config[kw.KEY_GRID_DIM_Z]
    
    # def assignWithEncoder(self, cfgstr, tse : TuningSpaceEncoder):
    #     config = tse.decode(cfgstr)
    #     self.assignWithDict(config)
    def assignWithKernelName(self,name : str) -> bool :
        # kcg_Attention_Br32Bc64Hd128_Sa16Sb8PTr4PTc4OTr4OTc8GLWQ4GLWK4GLWV4BLPY2BLPX1WLPY4WLPX16BSWQ4BSWK2WSWQ4WSWK2BLOY2BLOX1WLOY4WLOX16BSWP4BSWV2WSWP4WSWV1Un16W64LCP1LCO1SPP0RPP0RPO0
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
      
    def generateKernelName(self) -> str : 
        ret = "kcg_Attention_"
        ret += f"{self.basearg.argDict['shape']}".replace(' ','').replace(',','_')[1:-1]
        ret += f"_Br{ self.Br }"
        ret += f"Bc{ self.Bc }"
        ret += f"Hd{ self.Hd }"
        ret += f"_Sa{ self.Slice1 }"
        ret += f"Sb{ self.Slice2 }"
        ret += f"PTr{ self.PTr }"
        ret += f"PTc{ self.PTc }"
        ret += f"OTr{ self.OTr }"
        ret += f"OTc{ self.OTc }"
        ret += f"GLWQ{ self.GLOB_LOAD_WIDTH_Q }"
        ret += f"GLWK{ self.GLOB_LOAD_WIDTH_K }"
        ret += f"GLWV{ self.GLOB_LOAD_WIDTH_V }"
        ret += f"BLPY{ self.BLOCK_LAYOUT_P_Y }"
        ret += f"BLPX{ self.BLOCK_LAYOUT_P_X }"
        ret += f"WLPY{ self.WARP_LAYOUT_P_Y }"
        ret += f"WLPX{ self.WARP_LAYOUT_P_X }"
        ret += f"BSWQ{ self.BLOCK_SCATTER_WIDTH_Q }"
        ret += f"BSWK{ self.BLOCK_SCATTER_WIDTH_K }"
        ret += f"WSWQ{ self.WARP_SCATTER_WIDTH_Q }"
        ret += f"WSWK{ self.WARP_SCATTER_WIDTH_K }"
        ret += f"BLOY{ self.BLOCK_LAYOUT_O_Y }"
        ret += f"BLOX{ self.BLOCK_LAYOUT_O_X }"
        ret += f"WLOY{ self.WARP_LAYOUT_O_Y }"
        ret += f"WLOX{ self.WARP_LAYOUT_O_X }"
        ret += f"BSWP{ self.BLOCK_SCATTER_WIDTH_P }"
        ret += f"BSWV{ self.BLOCK_SCATTER_WIDTH_V }"
        ret += f"WSWP{ self.WARP_SCATTER_WIDTH_P }"
        ret += f"WSWV{ self.WARP_SCATTER_WIDTH_V }"
        ret += f"Un{ self.UNROLL_NUM }"
        ret += f"W{ self.WARP_SIZE }"
        ret += f"LCP{ self.LOAD_CONTINUOUS_P }"
        ret += f"LCO{ self.LOAD_CONTINUOUS_O }"
        ret += f"SPP{ self.SHARED_PREFETCH_P }"
        ret += f"RPP{ self.REG_PREFETCH_P }"
        ret += f"RPO{ self.REG_PREFETCH_O }"
        return ret
    
    def assignWithJson(self, jsonObj) : 
        kw = ConfigKeywords    
        self.Br = jsonObj[kw.KEY_Br]
        self.Bc = jsonObj[kw.KEY_Bc]
        self.Hd = jsonObj[kw.KEY_Hd]
        self.Slice1 = jsonObj[kw.KEY_Slice1]
        self.Slice2 = jsonObj[kw.KEY_Slice2]
        self.PTr = jsonObj[kw.KEY_PTr]
        self.PTc = jsonObj[kw.KEY_PTc]
        self.OTr = jsonObj[kw.KEY_OTr]
        self.OTc = jsonObj[kw.KEY_OTc]
        self.GLOB_LOAD_WIDTH_Q = jsonObj[kw.KEY_GLOB_LOAD_WIDTH_Q]
        self.GLOB_LOAD_WIDTH_K = jsonObj[kw.KEY_GLOB_LOAD_WIDTH_K]
        self.GLOB_LOAD_WIDTH_V = jsonObj[kw.KEY_GLOB_LOAD_WIDTH_V]
        self.BLOCK_LAYOUT_P_Y = jsonObj[kw.KEY_BLOCK_LAYOUT_P_Y]
        self.BLOCK_LAYOUT_P_X = jsonObj[kw.KEY_BLOCK_LAYOUT_P_X]
        self.WARP_LAYOUT_P_Y = jsonObj[kw.KEY_WARP_LAYOUT_P_Y]
        self.WARP_LAYOUT_P_X = jsonObj[kw.KEY_WARP_LAYOUT_P_X]
        self.BLOCK_SCATTER_WIDTH_Q = jsonObj[kw.KEY_BLOCK_SCATTER_WIDTH_Q]
        self.BLOCK_SCATTER_WIDTH_K = jsonObj[kw.KEY_BLOCK_SCATTER_WIDTH_K]
        self.WARP_SCATTER_WIDTH_Q = jsonObj[kw.KEY_WARP_SCATTER_WIDTH_Q]
        self.WARP_SCATTER_WIDTH_K = jsonObj[kw.KEY_WARP_SCATTER_WIDTH_K]
        self.BLOCK_LAYOUT_O_Y = jsonObj[kw.KEY_BLOCK_LAYOUT_O_Y]
        self.BLOCK_LAYOUT_O_X = jsonObj[kw.KEY_BLOCK_LAYOUT_O_X]
        self.WARP_LAYOUT_O_Y = jsonObj[kw.KEY_WARP_LAYOUT_O_Y]
        self.WARP_LAYOUT_O_X = jsonObj[kw.KEY_WARP_LAYOUT_O_X]
        self.BLOCK_SCATTER_WIDTH_P = jsonObj[kw.KEY_BLOCK_SCATTER_WIDTH_P]
        self.BLOCK_SCATTER_WIDTH_V = jsonObj[kw.KEY_BLOCK_SCATTER_WIDTH_V]
        self.WARP_SCATTER_WIDTH_P = jsonObj[kw.KEY_WARP_SCATTER_WIDTH_P]
        self.WARP_SCATTER_WIDTH_V = jsonObj[kw.KEY_WARP_SCATTER_WIDTH_V]
        self.UNROLL_NUM = jsonObj[kw.KEY_UNROLL_NUM]
        self.WARP_SIZE = jsonObj[kw.KEY_WARP_SIZE]
        self.LOAD_CONTINUOUS_P = jsonObj[kw.KEY_LOAD_CONTINUOUS_P]
        self.LOAD_CONTINUOUS_O = jsonObj[kw.KEY_LOAD_CONTINUOUS_O]
        self.SHARED_PREFETCH_P = jsonObj[kw.KEY_SHARED_PREFETCH_P]
        self.REG_PREFETCH_P = jsonObj[kw.KEY_REG_PREFETCH_P]
        self.REG_PREFETCH_O = jsonObj[kw.KEY_REG_PREFETCH_O]

        self.blockDimX = jsonObj[kw.KEY_BLOCK_DIM_X]
        self.blockDimY = jsonObj[kw.KEY_BLOCK_DIM_Y]
        self.blockDimZ = jsonObj[kw.KEY_BLOCK_DIM_Z]
        self.shmBytes = jsonObj[kw.KEY_SHM_BYTES]
        self.gridDimX = jsonObj[kw.KEY_GRID_DIM_X]
        self.gridDimY = jsonObj[kw.KEY_GRID_DIM_Y]
        self.gridDimZ = jsonObj[kw.KEY_GRID_DIM_Z]
        
    def check(self) :
        print("===== config check ok!")

    def __str__(self):
        return str(self.jsonfy())

# 算子生成逻辑
class AttentionOp(OpInterface) :
    def __init__(self):
        super().__init__()
        self.TuningArgs = AttentionTuningArgs()
        self.BaseArgs = AttentionBaseArgs()
        self.CompileKernel = None
        self.SetPlatform = None
        self.fastCompile = True

    def GetBaselineInputTensor(self, devId : int) -> List[torch.Tensor] : 
        if self.InputTensors_Baseline is None :
            # [shape : List[int] , dtypeInt]
            [shapeList, dtypeInt] = self.BaseArgs.intValues 

            assert len(shapeList)==4, f"shapeList= {shapeList}"
            [bs,hn,sl,hd ] = shapeList
            # print(f"GetBaselineInputTensor : shape = {b0,b1,m,n}")
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            # print("ety =", ety)
            q = torch.rand((bs,hn,sl,hd ),dtype=ety, device=f"cuda:{devId}" )  # matmul(softmax(matmul(mn, nm)) , mn) = mn
            k = torch.rand((bs,hn,hd,sl ),dtype=ety, device=f"cuda:{devId}" )
            v = torch.rand((bs,hn,sl,hd ),dtype=ety, device=f"cuda:{devId}" )
            self.InputTensors_Baseline = [q,k,v]
        return self.InputTensors_Baseline

    def GetBenchmarkInputTensor(self,devId : int) -> List[torch.Tensor] : 
        if self.InputTensors_Benchmark is None :
            [q,k,v] = self.GetBaselineInputTensor(devId)
            shapeList = self.BaseArgs.intValues[0] 
            dtypeInt = self.BaseArgs.intValues[1] 
            assert len(shapeList)==4
            [ b0, b1, m, n] = shapeList
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            qq = q.transpose(-1,-2).contiguous() 
            kk = k.transpose(-1,-2).contiguous()
            d = torch.empty((b0, b1,m,n), dtype=ety, device=f"cuda:{devId}")
            self.InputTensors_Benchmark = [qq,kk,v,d]
        return self.InputTensors_Benchmark

    
    def InitLibInterface(self) :
        if self.CompileKernel is None or self.SetPlatform is None :
            print(f"libdeepgen = {PathManager.kcg_lib_deepgen_path()}",flush=True)
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernel = mod.compile_attn
            self.SetKernelName = mod.set_kernel_name
            self.SetPlatform = mod.set_platform

    def Compile(self, deviceId:int, backendtype : EnumBackendType, arch : str, info : CompileNeededInfo, opt : CompileOption = None ) -> Tuple[List,KernelConfigs,CompiledKernel] :
        Print = print
        # compile kernel
        # Print("===== KCGCompiler ctor ========")
        assert isinstance(self.TuningArgs, AttentionTuningArgs)
        _backend = 0
        if backendtype.value == EnumBackendType.CUDA.value :
            _backend = 1
        elif backendtype.value == EnumBackendType.HIP.value :
            _backend = 2
        else:
            assert False, f'invalid backendtype {backendtype}, Ty is {type(backendtype)}'

        self.InitLibInterface()
        self.SetPlatform(_backend,arch)
        print(f"set arch : {arch}")
        # Print("===== call compileKernel(kpm)[0] ========")
        dataTypeInt = ToEnumIntDType(info.torchDataType)
        self.InitBaseArgs([info.baseArgs, dataTypeInt])

        shape, config = info.tsArgs
        assert info.kernelName is not None
        kernelName = info.kernelName
        self.SetKernelName( kernelName )

        # if is_hip():
        #     # hip compile
        #     hsacopath = f"{PathManager.default_dump_dir()}/hs_{info.kernelName}.hsaco" 
        #     fastCompile = True
        #     if opt is not None :
        #         fastCompile = opt.fastCompile
        #     res = HIPCompiler().build(Kernel.Attention, shape, config[info.kernelName], hsacopath, info.kernelName, fastCompile)
        # else:
            # compile using llvm
        res = self.CompileKernel(shape , config)
##########
        hsacoPath = res
        blockDimX, blockDimY ,blockDimZ = info.blockDims
        gridDimX, gridDimY, gridDimZ = info.gridDims
        # kernelName = 'attention1'
        shmBytes = info.shmBytes
        # print(f"blockdims = {blockDimX,blockDimY,blockDimZ}")
        # print(f"griddims = {gridDimX,gridDimY,gridDimZ}")
        Print("========= hsacoPath = ",hsacoPath)
        Print("========= kernelName = ",info.kernelName)
        # print(f"==== backend is {backendtype}")
        # print(f"==== shmBytes is {shmBytes}")
        dt = self.BaseArgs.getTorchDType()
        inConfig = KernelConfigs(hsacoPath,info.kernelName, [dt,dt,dt,dt], backendtype)
        inConfig.m_gridDims = [gridDimX,gridDimY,gridDimZ]
        inConfig.m_blockDims = [blockDimX,blockDimY,blockDimZ]
        inConfig.operatorKind = EnumOperator.Attention
        inConfig.shmBytes = shmBytes
        # batch(几个句子), seqLen（句子长度）, (hiddenDim(一个单词编码以后的向量长度) -> headnum * headDim),   
        packedKernel = self.GetCompiledKernel(inConfig,deviceId)
        return ([info.baseArgs, dataTypeInt], inConfig, packedKernel)  # 
  
    def GetCompiledKernel(self, info : KernelConfigs, deviceId : int) -> CompiledKernel :
        signature = self.GetSignature(info.dtypes)
        print(f"GetCompiledKernel attop : funame = {info.kernelFuncName}")
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

        a = torch.randn((1,3,100,100), device='cpu', dtype=dtypeA)
        b = torch.randn((1,3,100,100), device='cpu', dtype=dtypeA)
        c = torch.randn((1,3,100,100), device='cpu', dtype=dtypeA)
        d = torch.empty((1,3,100,100), device='cpu', dtype=dtypeA)
        # get function signature
        outSignature = _attention(a, b, c, d)
        return outSignature
    
    def SetTuningArgs(self, tuningArgs : List) :
        self.TuningArgs.assignWithList(*tuningArgs)

    def InitBaseArgs(self, args : List) :
        # print("InitBaseArgs=", args)
        shape, dtypeInt = args
        self.BaseArgs.intValues = [shape, dtypeInt]
        ety = EnumKernelDType(dtypeInt)
        self.TuningArgs = AttentionTuningArgs(ety)  
    
    def Test_warmup(self, packedKernel : CompiledKernel, warmupCount : int, devId : int) :
        [q,k,v] = self.GetBaselineInputTensor(devId)
        [qq,kk,vv,out] = self.GetBenchmarkInputTensor(devId)
        for i in range(0,warmupCount) : 
            # F.scaled_dot_product_attention(q, k, v)
            packedKernel.run(qq,kk,vv,out)
            # print("out=",out)
        return
    
    def Test_baseline(self, devId : int) -> Tuple[torch.Tensor,float]:
        [q,k,v] = self.GetBaselineInputTensor(devId)
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        d = q.shape[1] * q.shape[3]
        ev_start.record()
        p = torch.matmul(q, k) / math.sqrt(d)
        s = F.softmax(p, dim=-1)
        self.OutputTensor_Baseline = torch.matmul(s,v)
        ev_end.record()
        torch.cuda.synchronize()
        eps = ev_start.elapsed_time(ev_end)
        return (self.OutputTensor_Baseline, eps)
    
    def Test_benchmark(self, packedKernel : CompiledKernel, benchmarkCount : int, devId : int) -> Tuple[torch.Tensor,float] : 
        a,b,c,d = self.GetBenchmarkInputTensor(devId)
        # print("a.shape = ",a.shape)
        # print("b.shape = ",b.shape)
        # print("c.shape = ",c.shape)
        # print("d.shape = ",d.shape)
        # a = torch.rand((1, 32, 128,2048),dtype=torch.float32, device=f"cuda:{devId}")
        # b = torch.rand((1, 32, 128,2048),dtype=torch.float32, device=f"cuda:{devId}")
        # c = torch.rand((1, 32, 2048, 128),dtype=torch.float32, device=f"cuda:{devId}")
        # d = torch.rand((1, 32, 2048, 128),dtype=torch.float32, device=f"cuda:{devId}")
        # assert self.InputTensors_Benchmark  is not None, "error benchmark"
        st = torch.cuda.Event(enable_timing=True)
        et = torch.cuda.Event(enable_timing=True)
        st.record()
        packedKernel.run(a,b,c,d)
        et.record()
        torch.cuda.synchronize()
        # print(d)
        elapsed_time = st.elapsed_time(et)
        return (d,elapsed_time)
    
    def InitInputTensorsWithDatalist(self,  devId) -> None:
        assert self.BaseArgs is not None
        assert self.TuningArgs is not None
        assert isinstance(self.TuningArgs, AttentionTuningArgs)
        assert isinstance(self.BaseArgs , AttentionBaseArgs)
        self.GetBaselineInputTensor(devId)
        self.GetBenchmarkInputTensor(devId)

    
    def InitBaselineOutputTensor(self,  devId : int) -> None :
        if self.OutputTensor_Baseline is None :
            b,bb,m,n = self.BaseArgs.getIntDatalist()
            dt = self.BaseArgs.getTorchDType()
            self.OutputTensor_Baseline = torch.empty(m,n,dtype=dt, device=f'cuda:{devId}')

