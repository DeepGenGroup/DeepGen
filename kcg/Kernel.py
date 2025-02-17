# 存放 Kernel 相关的类
from dataclasses import dataclass
import inspect
from kcg.Utils import *
from functools import cached_property
import ast
import functools
import hashlib
import os
import textwrap
from collections import defaultdict, namedtuple
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload


class KernelParam :
    def __init__(self, index:int, param:inspect.Parameter):
        self.m_index = index
        self.m_param = param

    @cached_property
    def name(self):
        return self.m_param.name

    # @cached_property
    # def annotation(self):
    #     if not self.m_param.annotation or self.m_param.annotation == inspect.Parameter.empty:
    #         return ""
    #     return _normalize_ty(self.m_param.annotation)

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @property
    def default(self):
        return self.m_param.default

    @property
    def has_default(self):
        return self.m_param.default != inspect.Parameter.empty


# kernel运行时信息
class KernelRuntimeInfo :
    def __init__(self,module,func,regs,spills):
        self.m_module = module
        self.m_function = func
        self.m_nRegs = regs
        self.m_nSpills = spills


class KernelLibFile :
    def __init__(self,
        filePath : str,  # hsaco文件路径
        backendType : EnumBackendType,   # 后端类型（CUDA | HIP）
        kernelFuncName ,  # 核函数名字
        sharedMemSize,   # shm大小
        signature,   # kernel signature
        gridDims : list,
        blockDims : list,
        device=DeviceInfo.get_current_device()): # device号
        
        self.m_filePath = filePath
        self.m_backendType : EnumBackendType = backendType
        self.m_kernelInfo : KernelRuntimeInfo = None  # loader解析得到的地址等信息
        self.m_signature = signature  
        self.m_kernelFuncName = kernelFuncName
        self.m_shmSize = sharedMemSize
        self.m_device = device
        self.m_gridDims = gridDims
        self.m_blockDims = blockDims
    
    def __hash__(self) -> int:
        return calculate_file_hash(self.m_filePath) 
    
    @functools.lru_cache
    def hash(self)->int :
        return calculate_file_hash(self.m_filePath) 

    def signature_str(self) -> str :
        ret = ''
        for v in self.m_signature.values() :
            ret += str(v)
        return ret
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
            pass

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
        self.debug = True if os.environ.get("TRITON_DEBUG", "0") == "1" else debug
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
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

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

