# 公共函数和基本类

import hashlib
from enum import Enum, IntEnum
import contextlib
import functools
import io
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import sysconfig
from typing import List,Type
import setuptools
import torch
from typing import List,Tuple,Dict
from datetime import datetime

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
    try:
        with open(pkl_path, 'rb') as f:
            try:
                temp = pickle.load(f)  # 进行反序列化
            except EOFError as e:
                print("EOF Error!")
                return []
            return temp
    except Exception as e:
        print("[dsError]",e)
    return None
    
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
        return os.getenv("ROCM_PATH", default="/opt/rocm")


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
    if is_hip():
        hip_lib_dir = os.path.join(rocm_path_dir(), "lib")
        hip_include_dir = os.path.join(rocm_path_dir(), "include")
    else:
        cuda_lib_dirs = libcuda_dirs()
        cu_include_dir = cuda_include_dir()
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

    if is_hip():
        ret = subprocess.check_call([
            cc, src, f"-I{hip_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC",
            f"-L{hip_lib_dir}", "-lamdhip64", f"-Wl,-rpath,{hip_lib_dir}", "-o", so
        ])
    else:
        cc_cmd = [
            cc, src, "-O3", f"-I{cu_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-lcuda",
            "-o", so
        ]
        cc_cmd += [f"-L{dir}" for dir in cuda_lib_dirs]
        ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = cuda_lib_dirs
    include_dirs = [srcdir, cu_include_dir]
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
    HIP = 2
    INVALID = 3
    def __str__(self):
        return f'{self.name}'

class EnumRunMode(IntEnum):
    # 在本机执行生成调优空间、编译kernel以及benchmark
    GetTuneSpace_Compile_Benchmark_Local = 1  
    # 本地只作为Perftester运行kernel的benchmark。编译&调优空间生成&文件传输由其他host承担
    AsRemotePerftester = 2
    # 只在本地生产调优空间，不进行编译以及benchmark
    GetTuneSpace_Local_Only = 3
    # 只在本地进行编译，将 benchmark任务所需文件推送到远程
    CallRemotePerftester = 4
    def __str__(self):
        return f'{self.name}'
    
class EnumOperator:
    Invalid = "INVALID"
    Matmul = "MATMUL"
    Convolution = "CONV"
    Poll = "POLL"
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
    if t.value == EnumKernelDType.float32.value :
        return torch.float32
    if t.value == EnumKernelDType.float64.value :
        return torch.float64
    if t.value == EnumKernelDType.float16.value :
        return torch.float16

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



class DeviceInfo :
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
            return torch.cuda.current_stream(idx).cuda_stream

    @staticmethod
    def get_current_device():
        import torch
        return torch.cuda.current_device()

    @staticmethod
    def set_current_device(idx):
        import torch
        torch.cuda.set_device(idx)

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
        return torch.cuda.get_device_capability(idx)
    
    @staticmethod
    def get_warp_size():
        if is_hip():
            return 64
        else:
            return 32

# 路径管理器。存放了各种路径设置
class PathManager :
    _s_cuda_install_dir = ""
    _s_path_obj = None
    @staticmethod
    def project_dir()->str:
        return Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    
    @staticmethod
    def cuda_install_dir()->str:
        assert PathManager._s_path_obj is not None
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
    def cluster_run_dir() ->str :
        return str(PathManager.project_dir())+'/_cluster_run'
    
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
    def kcg_compiler_path()->str:
        return os.path.join(PathManager.project_dir(),"bin/libkcg_compiler.so")
        # return PathManager.__project_dir() + "/bin/libkcg_compiler.so"
    
    @staticmethod
    def init(clearPkl = False, clearDump = False, clearOverride = False, clearCache = False, clearTmp = False) :
        # print("PathManager initializing ... ",flush=True)
        os.makedirs(PathManager.pikle_dir(),exist_ok=True)
        os.makedirs(PathManager.default_cache_dir(),exist_ok=True)
        os.makedirs(PathManager.default_override_dir(),exist_ok=True)
        os.makedirs(PathManager.default_dump_dir(),exist_ok=True)
        os.makedirs(PathManager.tmp_dir(),exist_ok=True)
        os.makedirs(PathManager.cluster_run_dir(),exist_ok=True)
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
        userPathConfigfile = str(PathManager.project_dir()) + "/thirdPartyPath.json"
        assert os.path.exists(userPathConfigfile)
        with open(userPathConfigfile) as f:
            import json
            PathManager._s_path_obj = json.load(f)
        
#  关键字
class ConfigKeywords :
    KEY_BLOCK_SIZE_M =         "BLOCK_SIZE_M"
    KEY_BLOCK_SIZE_N =         "BLOCK_SIZE_N"
    KEY_BLOCK_SIZE_K =         "BLOCK_SIZE_K"
    KEY_THREAD_SIZE_M =        "THREAD_SIZE_M"
    KEY_THREAD_SIZE_N =        "THREAD_SIZE_N"
    KEY_WARP_SIZE =            "WARP_SIZE"
    KEY_BLOCK_LAYOUT_M =       "BLOCK_LAYOUT_M"
    KEY_BLOCK_LAYOUT_N =       "BLOCK_LAYOUT_N"
    KEY_WARP_LAYOUT_M =        "WARP_LAYOUT_M"
    KEY_WARP_LAYOUT_N =        "WARP_LAYOUT_N"
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
    KEY_WARP_SCATTER_WIDTH_A =    "WARP_SCATTER_WIDTH_A"
    KEY_WARP_SCATTER_WIDTH_B =    "WARP_SCATTER_WIDTH_B"
    KEY_THREAD_SCATTER_WIDTH_A =    "THREAD_SCATTER_WIDTH_A"
    KEY_THREAD_SCATTER_WIDTH_B =    "THREAD_SCATTER_WIDTH_B"
    KEY_LOCAL_SPLIT_U =     "LOCAL_SPLIT_U"
    KEY_BLOCK_MAPPING =     "BLOCK_MAPPING"
    KEY_GLOB_STORE_WIDTH =    "GLOB_STORE_WIDTH"
    KEY_UNROLL_NUM =            "UNROLL_NUM"
    KEY_REG_PREFETCH =          "REG_PREFETCH"
    KEY_SHARED_PREFETCH =       "SHARED_PREFETCH"
    KEY_LOAD_CONTINUOUS =       "LOAD_CONTINUOUS"
    KEY_REDUCE_C_CONTINUOUS =   "REDUCE_C_CONTINUOUS"
    
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

class OperatorBaseArgs :
    def __init__(self):
        self.operatorKind = EnumOperator.Invalid
        self.argList = []  # arglist 中的参数为int类型 （int,torch.dtype）
        self._innerDict = {
            "kind" : self.operatorKind,
            "b" : 0,
            "m" : 0,
            "n" : 0,
            "k" : 0,
            "dtype" : 0
        }
    def parseFromJsonfile(self,path : str):
        import json
        obj = None
        with open(path) as f :
            obj = json.load(f)
        self.argList = [obj['b'],obj['m'],obj['n'],obj['k'],obj['dtype']]
        self.operatorKind = obj['kind']
        
    def dumpToJson(self,path : str):
        import json
        self._innerDict["kind"] = self.operatorKind
        self._innerDict["b"] = self.argList[0]
        self._innerDict["m"] = self.argList[1]
        self._innerDict["n"] = self.argList[2]
        self._innerDict["k"] = self.argList[3]
        self._innerDict["dtype"] = self.argList[4]
        with open(path,'w') as f:
            json.dump(self._innerDict,f)
        
    
class GEMMBaseArgs(OperatorBaseArgs) :
    def __init__(self):
        super().__init__()
        self.operatorKind = EnumOperator.Matmul
    def getDetailedInfo(self) :
        batch = int(self.argList[1])
        m = int(self.argList[2])
        n = int(self.argList[3])
        k = int(self.argList[4])
        dtype = ToTorchType(EnumKernelDType(int(self.argList[5])))
        return (batch,m,n,k,dtype)

if __name__ == '__main__' :
    # PathManager.init()
    pass