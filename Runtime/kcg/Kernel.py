# 存放 Kernel 相关的类
from dataclasses import dataclass
import inspect
from functools import cached_property
import ast
import functools
import hashlib
import os
import textwrap
from collections import defaultdict, namedtuple
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from abc import ABC, abstractmethod
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Loader import *

# from kcg.CompiledKernel import CompiledKernel


# Kernel runtime config data
class KernelConfigs:
    def __init__(self,binary_path:str,
                 kernel_func_name:str, 
                 dtypes : List[torch.dtype],  # 用于初始化 TuningArg  
                 backend : EnumBackendType):
        self.operatorKind = EnumOperator.Invalid
        self.binaryPath = binary_path
        self.kernelFuncName = kernel_func_name
        self.dtypes = dtypes
        self.m_gridDims = [1,1,1]
        self.m_blockDims = [1,1,1]
        self.backend = backend
        self.shmBytes = 0
        
    def gridDims(self):  # 行优先矩阵，行方向为x方向，尺寸为n
        return self.m_gridDims
    
    def blockDims(self):
        return self.m_blockDims
        
    def sharedMem(self):
        return self.shmBytes


class KernelParam :
    def __init__(self, index:int, param:inspect.Parameter):
        self.m_index = index
        self.m_param = param

    @cached_property
    def name(self):
        return self.m_param.name

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @property
    def default(self):
        return self.m_param.default

    @property
    def has_default(self):
        return self.m_param.default != inspect.Parameter.empty


class KernelArg:
    """Represents an argument to a @jit'ed function.

    An argument is a parameter plus a value.
    """

    def __init__(self, value, param):
        self.value = value
        self.param = param

    @property
    def name(self):
        return self.param.name

    def signature_key(self):
        annotation = self.param.annotation
        if "Tensor" in annotation:
            return self.value.dtype
        elif annotation == "bool":
            return "i1"
        elif annotation == "float":
            return "fp32"
        else:
            return KernelFunction._key_of(self.value)

    def specialization_key(self):
        assert not self.param.do_not_specialize

        try:
            return (self.value.data_ptr() % KernelFunction.divisibility == 0, )
        except AttributeError: 
            ...

        if isinstance(self.value, int):
            # bool is a subclass of int, so we don't check explicitly above.
            return (
                self.value % KernelFunction.divisibility == 0,
                self.value % KernelFunction.divisibility_8 == 0,
                self.value == 1,
            )
        return (False, )


class KernelFunction :
    divisibility = 16
    divisibility_8 = 8
    @staticmethod
    def _key_of(arg):
        if hasattr(arg, "dtype"):
            return arg.dtype
        elif isinstance(arg, bool):
            return "i1"
        elif isinstance(arg, int):
            if -(2**31) <= arg and arg <= 2**31 - 1:
                return "i32"
            elif 2**63 <= arg and arg <= 2**64 - 1:
                return "u64"
            else:
                return "i64"
        elif isinstance(arg, float):
            return "fp32"
        elif arg is None:
            return None
        else:
            raise TypeError(f"Unsupported type {type(arg)} for {arg}")
    
    
    @staticmethod
    def _type_of(key):
        # `None` is nullptr.  Implicitly convert to *i8.
        if key is None:
            return "*i8"
        dtype_str = str(key).split(".")[-1]
        tys = {
            "bool": "i1",
            "float8e4nv": "fp8e4nv",
            "float8_e4m3fn": "fp8e4nv",
            "float8e4b8": "fp8e4b8",
            "float8_e4m3fnuz": "fp8e4b8",
            "float8e5": "fp8e5",
            "float8_e5m2": "fp8e5",
            "float8e5b16": "fp8e5b16",
            "float8_e5m2fnuz": "fp8e5b16",
            "float8e4b15": "fp8e4b15",
            "float8e4b15x4": "fp8e4b15x4",
            "float8_e4m3fn": "fp8e4nv",
            "float8_e5m2": "fp8e5",
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "fp32",
            "float64": "fp64",
            "int8": "i8",
            "int16": "i16",
            "int32": "i32",
            "int64": "i64",
            "uint8": "u8",
            "uint16": "u16",
            "uint32": "u32",
            "uint64": "u64",
        }
        # reinterpret can create triton type
        for v in list(tys.values()):
            tys[v] = v
        return key if isinstance(key, str) else f"*{tys[dtype_str]}"


    def __init__(self,fn, version=None, do_not_specialize=None, debug=None, noinline=None):
        self.fn = fn
        self.module = fn.__module__
        self.version = version
        self.signature = inspect.signature(fn)
        self.do_not_specialize = do_not_specialize

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            self.params.append(KernelParam(i, param))

        # function source code (without decorators)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def"):]
        # cache of just-in-time compiled kernels
        self.cache = defaultdict(dict)
        self.hash = None
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = True if os.environ.get("KCG_DEBUG", "0") == "1" else debug
        self.noinline = noinline

        # tma info
        # self.tensormaps_info = TMAInfos()

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        # self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # re-use docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
    
    def _getSignature(self,*args, **kwargs):
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        assert len(bound_args.arguments) == len(self.params)
        args = [KernelArg(arg_value, param) for (_, arg_value), param in zip(bound_args.arguments.items(), self.params)]
        kernelSignature = {
                arg.param.m_index: self._type_of(self._key_of(arg.value))
                for arg in args
            }
        return kernelSignature
    
    # 获取kernel函数的signature表示（dict）
    def __call__(self, *args, **kwargs):
        return self._getSignature(*args,**kwargs)
    
    
    
# decortator @kcg_kernel
def kcg_kernel(
    fn: None,
    *,
    version=None,
    do_not_specialize: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) :
    def decorator(fn) -> KernelFunction:
        assert callable(fn)
        return KernelFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                debug=debug,
                noinline=noinline,
            )
    if fn is not None:
        return decorator(fn)
    else:
        return decorator


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
        self.backend = backend
        self.kernelBinaryPath = kernelBinaryPath
        self.kernelName = kernelName
        self.kernel_signature = kernel_signature
        self.gridDims = gridDims
        self.blockDims = blockDims
        self.device = device
        self.shmSize = shmSize
    
    def buildLoaderAndLauncher(self) :
        if self.backend.value == EnumBackendType.HIP.value :
            # print(f"[D] gridDims={gridDims} , blockDims={blockDims}, device ={device}")
            self.m_loader = HIPLoaderST()
            self.m_launcher = HIPLauncher(self.kernelBinaryPath,self.kernelName,self.shmSize,self.signature,self.gridDims,self.blockDims,self.device)
        elif self.backend.value == EnumBackendType.CUDA.value :
            # print(f"[D] gridDims={gridDims} , blockDims={blockDims}, device ={device}")
            self.m_loader = CudaLoaderST()
            self.m_launcher = CUDALauncher(self.kernelBinaryPath,self.kernelName,self.shmSize,self.signature,self.gridDims,self.blockDims,self.device)
        else:
            assert False, f"Invalid backend value {self.backend.value}"
        
    def deleteBinary(self):
        if self.m_launcher is not None :
            if os.path.exists(self.m_launcher.m_kernelLib.m_filePath) :
                os.remove(self.m_launcher.m_kernelLib.m_filePath)
            # print(f"deleted {self.m_launcher.m_kernelLib.m_filePath}")

    def setDevice(self,devId : int) :
        if self.m_launcher is None:
            self.buildLoaderAndLauncher()
        self.m_launcher.m_kernelLib.m_device = devId
    
    def run(self,*args):
        if self.m_launcher is None :
            self.buildLoaderAndLauncher()
        self.m_launcher.launchKernel(*args)
    

##################################### 算子统一接口定义 ############################################

# 算子基本参数（用于描述问题的基本属性（调优过程中保持不动的部分，如问题规模MNK，算子类型，dtype等））
class OpBaseArgs(ABC) :
    def __init__(self):
        self.operatorKind = EnumOperator.Invalid
        self.intValues = []  # [问题规模+数据类型] , arglist 中的参数为int类型 （int,torch.dtype）
        self.argDict = {  # 参数字典。
            "kind" : self.operatorKind,
            "dtype" : 0
        }
        
    @abstractmethod
    # 问题规模的参数列表
    def getIntDatalist(self) -> List[int] : ...
    
    def getTorchDType(self) -> torch.dtype:
        return ToTorchType(self.getEnumDType())    
    
    @abstractmethod
    def getEnumDType(self) -> EnumKernelDType : ...
  
    @abstractmethod
    # 从json反序列化
    def parseFromJsonfile(self,path : str): ...

    @abstractmethod
    # 序列化
    def dumpToJson(self,path : str): ...
        
class TuningArgsInterface(ABC) :
    def __init__(self):
        super().__init__()
        # 可以根据不同Op的实现，选择是否使用以下三个基本参数，以便生成TuneSpace时直接计算得到不同配置的基础参数
        self.blockDimX = 0
        self.blockDimY = 0
        self.blockDimZ = 0
        self.gridDimX = 0
        self.gridDimY = 0
        self.gridDimZ = 0
        self.shmBytes = 0
        
    @abstractmethod
    def assignWithList(self, *args) :  ...
    @abstractmethod
    def jsonfy(self) :  ...
    @abstractmethod
    def assignWithJson(self, jsonObj) :  ...
    # @abstractmethod
    # def assignWithEncoder(self, cfgstr : int, tse : TuningSpaceEncoder) :  ...
    @abstractmethod
    def assignWithDict(self, config : Dict) : ...
    @abstractmethod
    def check(self) :  ...
    @abstractmethod
    def generateKernelName(self) -> str : ...
    @abstractmethod
    def __str__(self): ...

# 算子接口
class OpInterface(ABC) :
    def __init__(self):
        super().__init__()
        self.KernelName : str = None
        self.BaseArgs : OpBaseArgs = None   # 基本参数，不能改变，定义了问题形状等（如M,N,K,batch）。为问题的基本属性
        self.InputTensors_Baseline : List[torch.Tensor] = None
        self.InputTensors_Benchmark : List[torch.Tensor] = None
        self.OutputTensor_Baseline : torch.Tensor = None

    def GetKernelName(self) -> str : 
        if self.KernelName is None:
            self.KernelName = self.TuningArgs.generateKernelName()
        return self.KernelName
    
    @abstractmethod
    ### [basearg, kernelRunINfo, packedKernel]
    def Compile(self, deviceId:int, backendtype : EnumBackendType, arch : str, info : CompileNeededInfo) -> Tuple[List, KernelConfigs, CompiledKernel] : ...
    @abstractmethod
    def GetBaselineInputTensor(self, devId : int) -> List[torch.Tensor] : ...
    @abstractmethod
    def GetBenchmarkInputTensor(self,devId : int) -> List[torch.Tensor] : ...

    @abstractmethod
    # Init BaseArgs, build TuningArgs object
    def InitBaseArgs(self, args : List[int]) :  ...
    
    @abstractmethod
    # how to build CompiledKernel object
    def GetCompiledKernel(self, info : KernelConfigs, deviceId : int) -> CompiledKernel : ...
    
    @abstractmethod
    # how to get signature
    def GetSignature(self, dtypes : List[torch.dtype]) -> dict : ...
    
    @abstractmethod
    def SetTuningArgs(self, tuningArgs : List) : ...    
    
    @abstractmethod
    def Test_warmup(self, packedKernel : CompiledKernel, warmupCount : int, devId : int) : ...
    
    @abstractmethod
    def Test_baseline(self, devId : int) -> Tuple[torch.Tensor,float]: ...

    @abstractmethod
    def Test_benchmark(self, packedKernel : CompiledKernel, benchmarkCount : int , devId : int) -> Tuple[torch.Tensor,float] :  ...
    
    @abstractmethod
    # initialize input tensors for Test_baseline with given dataList & datatype & devId
    def InitInputTensorsWithDatalist( self, devId ) -> None : ...
    
    @abstractmethod
    def InitBaselineOutputTensor(self,  devId : int) -> None : ...
    
