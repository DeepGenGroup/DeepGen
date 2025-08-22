# 公共函数和基本类

from abc import ABC, abstractmethod
import hashlib
from enum import Enum, IntEnum
import contextlib
import functools
import io
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import sysconfig
from typing import Any, Generator, List, Optional,Type
import setuptools
import torch
from typing import List,Tuple,Dict
from datetime import datetime
import traceback
from kcg.TorchNamespace import *

class CacheManager(ABC):
    def __init__(self, key):
        pass
    @abstractmethod
    def get_file(self, filename) -> Optional[str]:
        pass

    @abstractmethod
    def has_file(self, filename) -> bool:
        pass

    @abstractmethod
    def put(self, data, filename, binary=True) -> str:
        pass

    @abstractmethod
    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        pass

    @abstractmethod
    def put_group(self, filename: str, group: Dict[str, str]):
        pass


class FileCacheManager(CacheManager):
    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        PathManager.init()
        if dump:
            self.cache_dir = PathManager.default_dump_dir()
            # self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = PathManager.default_override_dir()
            # self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = PathManager.default_cache_dir()
            if self.cache_dir:
                # self.cache_dir = os.path.join(self.cache_dir, str(self.key))
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")
    
    def _make_path(self, filename) -> str:
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename) -> bool:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename) -> Optional[str]:
        if self.has_file(filename):
            return self._make_path(filename)
        else:
            return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        grp_filename = f"__grp__{filename}"
        if not self.has_file(grp_filename):
            return None
        grp_filepath = self._make_path(grp_filename)
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get("child_paths", None)
        # Invalid group data.
        if child_paths is None:
            return None
        result = {}
        for c in child_paths:
            p = self._make_path(c)
            if os.path.exists(p):
                result[c] = p
        return result

    # Note a group of pushed files as being part of a group
    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        grp_contents = json.dumps({"child_paths": sorted(list(group.keys()))})
        grp_filename = f"__grp__{filename}"
        return self.put(grp_contents, grp_filename, binary=False)

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = random.randint(0, 1000000)
        # we use the PID incase a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use tempfile to be robust against program interruptions
        temp_path = f"{filepath}.tmp.pid_{pid}_{rnd_id}"
        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        return filepath



# TODO: is_hip shouldn't be here
def is_hip():
    import torch
    return torch.version.hip is not None


def serialize_to_file(pkl_path, obj) :
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(obj, file=f)  # 进行序列化
    except Exception as e:
        print('[E] generatePklError : ',e)
        
def deserialize_from_file(pkl_path) :
    with open(pkl_path, 'rb') as f:
        try:
            temp = pickle.load(f)  # 进行反序列化
        except EOFError as e:
            print("EOF Error!")
            return []
        return temp

def delete_files_in_directory(directory):
    # 确保目录存在
    if os.path.exists(directory) and os.path.isdir(directory):
        # 遍历目录中的所有文件和子目录
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            # 如果是文件，删除它
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)
        print(f"Deleted files in {directory}")
    else:
        print(f"The directory {directory} does not exist.")


@functools.lru_cache()
def libcuda_dirs():
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    dirs.append("/home/xushilong/anaconda3/lib/stubs")
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the file.'
    else:
        msg += 'Please make sure GPU is setup and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def rocm_path_dir():
    default_path = os.path.join(os.path.dirname(__file__), "..", "third_party", "hip")
    # Check if include files have been populated locally.  If so, then we are 
    # most likely in a whl installation and he rest of our libraries should be here
    if (os.path.exists(default_path+"/include/hip/hip_runtime.h")):
        return default_path
    else:
        return os.getenv("HIP_PATH", default="/opt/rocm")


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


@functools.lru_cache()
def cuda_include_dir():
    ret = PathManager.cuda_install_dir() + "/include"
    return ret


def build(name, src, srcdir):
    p = get_platform_type()
    lib_dir = '/usr/lib'
    include_dir = '/usr/include'
    if p == 'dcu' :
        if is_hip():
            lib_dir = os.path.join(rocm_path_dir(), "lib")
            include_dir = os.path.join(rocm_path_dir(), "include")
        else:
            lib_dir = libcuda_dirs()
            include_dir = cuda_include_dir()
    elif p == 'npu' or p == 'mlu' :
        ...
    else :
        ...
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    if p=='dcu':
        if is_hip():
            ret = subprocess.check_call([
                cc, src, f"-I{include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC",
                f"-L{lib_dir}", "-lamdhip64", f"-Wl,-rpath,{lib_dir}", "-o", so
            ])
        else:
            cc_cmd = [
                cc, src, "-O3", f"-I{include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-lcuda",
                "-o", so
            ]
            cc_cmd += [f"-L{dir}" for dir in lib_dir]
            ret = subprocess.check_call(cc_cmd)
    elif p == 'mlu' or p == 'npu':
        ret = 0
    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = lib_dir
    include_dirs = [srcdir, include_dir]
    libraries = ['cuda']
    # extra arguments
    extra_link_args = []
    # create extension module
    ext = setuptools.Extension(
        name=name,
        language='c',
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ['-O3'],
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    with quiet():
        setuptools.setup(**args)
    return so



class EnumBackendType(IntEnum):
    CUDA = 1
    HIP = 2  # DCU
    MLU = 3  # hanwuji
    NPU = 4  # huawei
    INVALID = 9
    def __str__(self):
        return f'{self.name}'

class EnumRunMode(IntEnum):
    # 本地只作为Perftester运行kernel的benchmark。编译&调优空间生成&文件传输由其他host承担
    AsRemotePerftester = 3
    # 只在本地进行编译，将 benchmark任务所需文件推送到远程
    CallRemotePerftester = 5
    # 只在本地生产调优空间，不进行编译以及benchmark
    GetTuneSpace_Local_Only = 4
    
    def __str__(self):
        return f'{self.name}'
    
class EnumOperator:
    Invalid = "INVALID"
    Matmul = "MATMUL"
    Convolution = "CONV"
    Poll = "POLL"
    Attention = "ATTENTION"
    def __str__(self):
        return f'{self.name}'

class EnumKernelDType(IntEnum):
    float8 = 1
    float16 = 2
    float32 = 4
    float64 = 8
    float128 = 16
    int8 = 31
    int16 = 32
    int32 = 34
    int64 = 38
    def __str__(self):
        return f'{self.name}'

def ToTorchType (t : EnumKernelDType) -> torch.dtype:
    if t == EnumKernelDType.float32.value :
        return torch.float32
    if t == EnumKernelDType.float64.value :
        return torch.float64
    if t == EnumKernelDType.float16.value :
        return torch.float16

def ToEnumIntDType (t : torch.dtype) -> EnumKernelDType:
    if t is torch.float32 :
        return EnumKernelDType.float32
    if t is torch.float64 :
        return EnumKernelDType.float64
    if t is torch.float16 :
        return EnumKernelDType.float16

def sizeof(t : EnumKernelDType) : # bytes
    assert(t is not None)
    return int(t) % 30

def printTime() :
    now = datetime.now()
    formattime = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--------[ {formattime} ]------------")

def get_kernel_name(src: str, pattern: str) -> str:
    '''
    Get kernel name from ptx/amdgcn code.
    This Kernel name is required when launching the kernel.
    '''
    # There is a name mangling in PTX codegen, so the original kernel names in Triton IR are not available in PTX/cubin.
    assert src
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith(pattern):
            return line.split()[-1]
    
def calculate_file_hash(file_path ,algorithm='md5',hash_len=10) -> int:
    # 以二进制只读模式打开文件
    ret = ""
    try:
        with open(file_path, 'rb') as file:
            # 选择哈希算法
            if algorithm == 'md5':
                hasher = hashlib.md5()
            elif algorithm == 'sha1':
                hasher = hashlib.sha1()
            elif algorithm == 'sha256':
                hasher = hashlib.sha256()
            else:
                raise ValueError("Unsupported algorithm. Please choose from 'md5', 'sha1', or 'sha256'.")

            # 逐块更新哈希值
            for chunk in iter(lambda: file.read(4096), b''):
                hasher.update(chunk)

            # 返回计算得到的哈希值
            ret = hasher.hexdigest()
            return int(ret[:hash_len],16)
    except Exception as e :
        print('file_hash calcuate skipped ')


class DeviceInfo :
    __gpuInfoLibPath = None
    __get_gpu_info = None
    __get_device_count = None
    @staticmethod
    def get_cuda_stream(idx=None):
        if idx is None:
            idx = DeviceInfo.get_current_device()
        try:
            # print(f"[D]--------- DeviceInfo.get_current_device() is {idx}")
            from torch._C import _cuda_getCurrentRawStream
            return _cuda_getCurrentRawStream(idx)
        except ImportError:
            import torch
            return torch_ns.current_stream(idx).cuda_stream

    @staticmethod
    def get_current_device():
        import torch
        return torch_ns.current_device()

    @staticmethod
    def set_current_device(idx):
        import torch
        torch_ns.set_device(idx)

    @staticmethod
    def set_visible_devices(devids : List):
        import torch
        import os
        envname = 'CUDA_VISIBLE_DEVICES'
        if is_hip() :
            envname = 'HIP_VISIBLE_DEVICES'
        # if DeviceInfo.get_visible_devices() is None:
        expr = ''
        for id in devids:
            expr += str(id) + ','
        os.environ[envname] = expr[0:-1]
        print(f"==== set {envname}={os.environ[envname]}  =====",flush=True)

    
    @staticmethod
    def get_visible_devices():
        import os
        envname = 'CUDA_VISIBLE_DEVICES'
        if is_hip() :
            envname = 'HIP_VISIBLE_DEVICES'
        return os.environ.get(envname) 
    
    @staticmethod
    def get_device_capability(idx):
        import torch
        return torch_ns.get_device_capability(idx)
    
    @staticmethod
    def get_warp_size():
        if is_hip():
            return 64
        else:
            return 32
    
    @staticmethod
    def init_cuda(_devId : List) :
        DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
        print("init_cuda devid=",_devId)
        DeviceInfo.set_visible_devices(_devId)
        DeviceInfo.set_current_device(_devId[0])  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
        if not torch_ns.is_available() :
            torch_ns.init()
            torch_ns.empty_cache()
    
    @staticmethod
    def __init_gpu_info() :
        if DeviceInfo.__gpuInfoLibPath is None :
            if is_hip() :
                src = Path(PathManager.gpuinfo_c_path_hip()).read_text()
                fname = "gpuinfo_hip.so"
            else:
                src = Path(PathManager.gpuinfo_c_path_cuda()).read_text()
                fname = "gpuinfo_cuda.so"
            tmpdir = PathManager.default_cache_dir()
            src_path = tmpdir + "/main_gpuinfo.c"
            with open(src_path, "w") as f:
                f.write(src)
            DeviceInfo.__gpuInfoLibPath = build(fname, src_path, tmpdir)
        if DeviceInfo.__get_gpu_info is None or DeviceInfo.__get_device_count is None :
            import importlib.util
            spec = importlib.util.spec_from_file_location("gpu_info", DeviceInfo.__gpuInfoLibPath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            DeviceInfo.__get_gpu_info = getattr(mod, "get_gpu_info")
            DeviceInfo.__get_device_count = getattr(mod, "get_device_count")
    
    @staticmethod
    def get_gpu_info() :
        DeviceInfo.__init_gpu_info()
        return DeviceInfo.__get_gpu_info()
    
    @staticmethod
    def get_device_count() :
        DeviceInfo.__init_gpu_info()
        return DeviceInfo.__get_device_count()
    
# 路径管理器。存放了各种路径设置
class PathManager :
    _s_cuda_install_dir = ""
    _s_path_obj = None
    @staticmethod
    def project_dir()->str:
        return Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    
    @staticmethod
    def cuda_install_dir()->str:
        if PathManager._s_path_obj is None :
            userPathConfigfile = str(PathManager.project_dir()) + "/thirdPartyPath.json"
            assert os.path.exists(userPathConfigfile)
            with open(userPathConfigfile) as f:
                import json
                PathManager._s_path_obj = json.load(f)
        if len(PathManager._s_path_obj['cuda_install_dir']) > 1:
            return PathManager._s_path_obj['cuda_install_dir']
        if len(PathManager._s_cuda_install_dir) <= 0 :
            import subprocess
            result = subprocess.run(['which','nvcc'], capture_output=True, text=True)
            ret = result.stdout.strip()
            index = ret.find("/bin/nvcc")
            if index < 0 :
                assert False, "Cannot found nvcc. PLease set PATH env first!"
            PathManager._s_cuda_install_dir = ret[0:index]
        return PathManager._s_cuda_install_dir
    
    @staticmethod
    def third_party_dir()->str:
        return f"{PathManager.project_dir()}/third_party"
    
    @staticmethod
    def pikle_dir() ->str :
        return str(PathManager.project_dir())+'/_pkls'
    
    @staticmethod
    def tmp_dir() ->str :
        return str(PathManager.project_dir())+'/_tmp'
    
    @staticmethod
    def default_cache_dir()->str:
        # return os.path.join(Path.home(), ".kcg", "cache")
        return str(PathManager.project_dir()) + '/_cache'

    @staticmethod
    def default_override_dir()->str:
        # return os.path.join(Path.home(), ".kcg", "override")
        return str(PathManager.project_dir()) + '/_override'

    @staticmethod
    def default_dump_dir()->str:
        # return os.path.join(Path.home(), ".kcg", "dump")
        return str(PathManager.project_dir()) + '/_dump'

    @staticmethod
    def loader_c_path_hip()->str:
        return os.path.join(PathManager.project_dir(),"Runtime/kcg/loaderCCode/hip.c")
    
    @staticmethod
    def loader_c_path_cuda()->str:
        return os.path.join(PathManager.project_dir(),"Runtime/kcg/loaderCCode/cuda.c")
    
    @staticmethod
    def gpuinfo_c_path_hip()->str:
        return os.path.join(PathManager.project_dir(),"Runtime/kcg/gpuInfoCode/HipGPUInfo.cc")
    
    @staticmethod
    def gpuinfo_c_path_cuda()->str:
        return os.path.join(PathManager.project_dir(),"Runtime/kcg/gpuInfoCode/CudaGPUInfo.cc")
    
    @staticmethod
    def kcg_compiler_path()->str:
        return os.path.join(PathManager.project_dir(),"bin/libkcg_compiler.so")
        # return PathManager.__project_dir() + "/bin/libkcg_compiler.so"
    
    @staticmethod
    def kcg_lib_deepgen_path()->str:
        return os.path.join(PathManager.project_dir(),"bin/libdeepgen.so")
        # return PathManager.__project_dir() + "/bin/libkcg_compiler.so"
    
    @staticmethod
    def init(clearPkl = False, clearDump = False, clearOverride = False, clearCache = False, clearTmp = False) :
        # print("PathManager initializing ... ",flush=True)
        os.makedirs(PathManager.pikle_dir(),exist_ok=True)
        os.makedirs(PathManager.default_cache_dir(),exist_ok=True)
        os.makedirs(PathManager.default_override_dir(),exist_ok=True)
        os.makedirs(PathManager.default_dump_dir(),exist_ok=True)
        os.makedirs(PathManager.tmp_dir(),exist_ok=True)
        try:
            if clearPkl :
                delete_files_in_directory(PathManager.pikle_dir())
            if clearCache :
                delete_files_in_directory(PathManager.default_cache_dir())
            if clearOverride :
                delete_files_in_directory(PathManager.default_override_dir())
            if clearDump :
                delete_files_in_directory(PathManager.default_dump_dir())
            if clearTmp :
                delete_files_in_directory(PathManager.tmp_dir())
        except OSError as e :
            ...
        userPathConfigfile = str(PathManager.project_dir()) + "/thirdPartyPath.json"
        assert os.path.exists(userPathConfigfile)
        with open(userPathConfigfile) as f:
            import json
            PathManager._s_path_obj = json.load(f)

class CompileNeededInfo :
    def __init__(self):
        self.baseArgs : List = []  # 问题定义（ 基础不变量，各个算子自定义.如对于matmul，其为 mnk ）
        self.tsArgs : List = []
        self.torchDataType : torch.dtype = None
        self.blockDims : List[int] = None # optional. If needed, we can assign and use
        self.gridDims : List[int] = None # optional. If needed, we can assign and use
        self.shmBytes : int = None # optional. If needed, we can assign and use
        self.kernelName : str = None #  optional. If need, we can assign and use

class CompileOption :
    def __init__(self):
        self.toolchain = "llvm" # "other"
        self.fastCompile = False  # 是否关闭优化以加速编译

#  关键字
class ConfigKeywords :
    # common
    KEY_BLOCK_DIM_X =   "blockDim.x"
    KEY_BLOCK_DIM_Y =   "blockDim.y"
    KEY_BLOCK_DIM_Z =   "blockDim.z"
    KEY_GRID_DIM_X =   "gridDim.x"
    KEY_GRID_DIM_Y =   "gridDim.y"
    KEY_GRID_DIM_Z =   "gridDim.z"
    KEY_SHM_BYTES =   "shmBytes"
    # gemm
    KEY_BLOCK_SIZE_M =         "BLOCK_SIZE_M"
    KEY_BLOCK_SIZE_N =         "BLOCK_SIZE_N"
    KEY_BLOCK_SIZE_K =         "BLOCK_SIZE_K"
    KEY_THREAD_SIZE_M =        "THREAD_SIZE_M"
    KEY_THREAD_SIZE_N =        "THREAD_SIZE_N"
    KEY_WARP_SIZE =            "WARP_SIZE"
    KEY_BLOCK_LAYOUT_Y =       "BLOCK_LAYOUT_Y"
    KEY_BLOCK_LAYOUT_X =       "BLOCK_LAYOUT_X"
    KEY_WARP_LAYOUT_Y =        "WARP_LAYOUT_Y"
    KEY_WARP_LAYOUT_X =        "WARP_LAYOUT_X"
    KEY_DTYPE_A =              "DATATYPE_A"
    KEY_DTYPE_B =              "DATATYPE_B"
    KEY_DTYPE_C =              "DATATYPE_C"
    KEY_M =                    "M_SIZE"
    KEY_N =                    "N_SIZE"
    KEY_K =                    "K_SIZE"
    KEY_BATCH =                "BATCH_SIZE"
    KEY_IS_A_TRANSPOSE =       "IS_ATRANS"
    KEY_GLOB_LOAD_WIDTH_A =     "GLOB_LOAD_WIDTH_A"
    KEY_GLOB_LOAD_WIDTH_B =     "GLOB_LOAD_WIDTH_B"
    KEY_BLOCK_SCATTER_WIDTH_M =    "BLOCK_SCATTER_WIDTH_M"
    KEY_BLOCK_SCATTER_WIDTH_N =    "BLOCK_SCATTER_WIDTH_N"
    KEY_WARP_SCATTER_WIDTH_M =    "WARP_SCATTER_WIDTH_M"
    KEY_WARP_SCATTER_WIDTH_N =    "WARP_SCATTER_WIDTH_N"
    KEY_LOCAL_SPLIT_U =     "LOCAL_SPLIT_U"
    KEY_BLOCK_MAPPING =     "BLOCK_MAPPING"
    KEY_GLOB_STORE_WIDTH =    "GLOB_STORE_WIDTH"
    KEY_UNROLL_NUM =            "UNROLL_NUM"
    KEY_REG_PREFETCH =          "REG_PREFETCH"
    KEY_SHARED_PREFETCH =       "SHARED_PREFETCH"
    KEY_LOAD_CONTINUOUS =       "LOAD_CONTINUOUS"
    KEY_STORE_CONTINUOUS =   "STORE_CONTINUOUS"
    # attention
    KEY_Br = "Br"
    KEY_Bc = "Bc"
    KEY_Hd = "Hd"
    KEY_Slice1 = "Slice1"
    KEY_Slice2 = "Slice2"
    KEY_PTr = "PTr"
    KEY_PTc = "PTc"
    KEY_OTr = "OTr"
    KEY_OTc = "OTc"
    KEY_GLOB_LOAD_WIDTH_Q = "GLOB_LOAD_WIDTH_Q"
    KEY_GLOB_LOAD_WIDTH_K = "GLOB_LOAD_WIDTH_K"
    KEY_GLOB_LOAD_WIDTH_V = "GLOB_LOAD_WIDTH_V"
    KEY_BLOCK_LAYOUT_P_Y = "BLOCK_LAYOUT_P_Y"
    KEY_BLOCK_LAYOUT_P_X = "BLOCK_LAYOUT_P_X"
    KEY_WARP_LAYOUT_P_Y = "WARP_LAYOUT_P_Y"
    KEY_WARP_LAYOUT_P_X = "WARP_LAYOUT_P_X"
    KEY_BLOCK_SCATTER_WIDTH_Q = "BLOCK_SCATTER_WIDTH_Q"
    KEY_BLOCK_SCATTER_WIDTH_K = "BLOCK_SCATTER_WIDTH_K"
    KEY_WARP_SCATTER_WIDTH_Q = "WARP_SCATTER_WIDTH_Q"
    KEY_WARP_SCATTER_WIDTH_K = "WARP_SCATTER_WIDTH_K"
    KEY_BLOCK_LAYOUT_O_Y = "BLOCK_LAYOUT_O_Y"
    KEY_BLOCK_LAYOUT_O_X = "BLOCK_LAYOUT_O_X"
    KEY_WARP_LAYOUT_O_Y = "WARP_LAYOUT_O_Y"
    KEY_WARP_LAYOUT_O_X = "WARP_LAYOUT_O_X"
    KEY_BLOCK_SCATTER_WIDTH_P = "BLOCK_SCATTER_WIDTH_P"
    KEY_BLOCK_SCATTER_WIDTH_V = "BLOCK_SCATTER_WIDTH_V"
    KEY_WARP_SCATTER_WIDTH_P = "WARP_SCATTER_WIDTH_P"
    KEY_WARP_SCATTER_WIDTH_V = "WARP_SCATTER_WIDTH_V"
    KEY_LOAD_CONTINUOUS_P = "LOAD_CONTINUOUS_P"
    KEY_LOAD_CONTINUOUS_O = "LOAD_CONTINUOUS_O"
    KEY_SHARED_PREFETCH_P = "SHARED_PREFETCH_P"
    KEY_REG_PREFETCH_P = "REG_PREFETCH_P"
    KEY_REG_PREFETCH_O = "REG_PREFETCH_O"
    
def get_dtype_from_int(dtype : int) :
    if dtype == int( EnumKernelDType.float8) :
        return EnumKernelDType.float8
    if dtype == int( EnumKernelDType.float16) :
        return EnumKernelDType.float16
    if dtype == int( EnumKernelDType.float32) :
        return EnumKernelDType.float32
    if dtype == int( EnumKernelDType.float64) :
        return EnumKernelDType.float64
    if dtype == int( EnumKernelDType.float128) :
        return EnumKernelDType.float128
    if dtype == int( EnumKernelDType.int8) :
        return EnumKernelDType.int8
    if dtype == int( EnumKernelDType.int16) :
        return EnumKernelDType.int16
    if dtype == int( EnumKernelDType.int32) :
        return EnumKernelDType.int32
    if dtype == int( EnumKernelDType.int64) :
        return EnumKernelDType.int64
    return None



# kernel运行时的二进制层面信息
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
        signature : dict,   # kernel signature
        gridDims : list,
        blockDims : list,
        device= 0): # device号
        
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
        if self.m_filePath is None :
            return 0
        return calculate_file_hash(self.m_filePath) 

    def signature_str(self) -> str :
        ret = ''
        for v in self.m_signature.values() :
            ret += str(v)
        return ret

TsGeneratorType = Generator[CompileNeededInfo, Any, None]         


def compare_with_error(tensor1, tensor2, abs_error=1e-2, rel_error=1e-2):
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor1) + 1e-5)  # 避免除以零的情况

    # 比较绝对误差和相对误差
    error_mask = (abs_diff > abs_error) & (rel_diff > rel_error)
    diff_elements = torch.sum(error_mask).item()
    max_error = torch.max(torch.abs(tensor1 - tensor2))
    return diff_elements, max_error