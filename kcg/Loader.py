# CUDA & HIP Loader和Launcher的定义
import abc
import hashlib
import os
import tempfile
from pathlib import Path

from kcg.Utils import build
from kcg.Cache import *
from kcg.Kernel import *



class CudaLoaderST(object):
    # singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaLoaderST, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # compile loader_cuda.so and cache it
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "loader", "cuda.c")).read_text()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = FileCacheManager(key)
        fname = "loader_cuda.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build("loader_cuda", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util

        spec = importlib.util.spec_from_file_location("loader_cuda", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # function binding
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.CUtensorMapDataType = mod.CUtensorMapDataType
        self.CUtensorMapInterleave = mod.CUtensorMapInterleave
        self.CUtensorMapSwizzle = mod.CUtensorMapSwizzle
        self.CUtensorMapL2promotion = mod.CUtensorMapL2promotion
        self.CUtensorMapFloatOOBfill = mod.CUtensorMapFloatOOBfill
        self.cuTensorMapEncodeTiled = mod.cuTensorMapEncodeTiled
        self.cuMemAlloc = mod.cuMemAlloc
        self.cuMemcpyHtoD = mod.cuMemcpyHtoD
        self.cuMemFree = mod.cuMemFree
    # 加载二进制文件，返回其信息
    def loadBinary(self, kernelFile : KernelLibFile) -> KernelRuntimeInfo :
        binaryPath = kernelFile.m_filePath
        name = kernelFile.m_kernelFuncName
        shared = int(kernelFile.m_shmSize)
        device = int(kernelFile.m_device)
        mod,func, n_regs, n_spills = self.load_binary(name,binaryPath,shared,device)
        kernelFile.m_kernelInfo = KernelRuntimeInfo(mod,func,n_regs,n_spills)
        return kernelFile.m_kernelInfo
    
class HIPLoaderST(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPLoaderST, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        src = Path(PathManager.loader_c_path_hip()).read_text()
        self.key = calculate_file_hash(file_path=PathManager.loader_c_path_hip())
        self.cache = FileCacheManager(self.key)
        self.fname = "loader_hip.so"
        self.cache_path = self.cache.get_file(self.fname)
        self.load_binary = None
        # print('cache_path=',self.cache_path)
        if self.cache_path is None:
            tmpdir = PathManager.default_cache_dir()
            # with tempfile.TemporaryDirectory() as tmpdir:
            src_path = tmpdir + "/main.c"
            with open(src_path, "w") as f:
                f.write(src)
            so = build("loader_hip", src_path, tmpdir)
            with open(so, "rb") as f:
                self.cache_path = self.cache.put(f.read(), self.fname, binary=True)
        
    def loadBinary(self, kernelFile : KernelLibFile) -> KernelRuntimeInfo :
        if self.load_binary is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("loader_hip", self.cache_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.load_binary = mod.load_binary
            self.get_device_properties = mod.get_device_properties
        if kernelFile.m_kernelInfo is None:
            binaryPath = kernelFile.m_filePath
            name = kernelFile.m_kernelFuncName
            shared = int(kernelFile.m_shmSize)
            device = int(kernelFile.m_device)
            mod,func, n_regs, n_spills = self.load_binary(name,binaryPath,shared,device)
            info = KernelRuntimeInfo(mod,func,n_regs,n_spills)
            kernelFile.m_kernelInfo = info
            
        return kernelFile.m_kernelInfo

