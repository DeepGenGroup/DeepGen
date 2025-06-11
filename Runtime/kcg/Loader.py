# CUDA & HIP Loader和Launcher的定义
import abc
import hashlib
import os
import tempfile
from pathlib import Path

from kcg.Utils import *
from kcg.Cache import *
# from kcg.Kernel import KernelLibFile


class CudaLoaderST(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaLoaderST, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        src = Path(PathManager.loader_c_path_cuda()).read_text()
        self.key = calculate_file_hash(file_path=PathManager.loader_c_path_cuda())
        self.cache = FileCacheManager(self.key)
        self.fname = "loader_cuda.so"
        self.cache_path = self.cache.get_file(self.fname)
        self.load_binary = None
        self.unload_binary = None
        # print('cache_path=',self.cache_path)
        if self.cache_path is None:
            tmpdir = PathManager.default_cache_dir()
            # with tempfile.TemporaryDirectory() as tmpdir:
            src_path = tmpdir + "/main_loader_cuda.c"
            with open(src_path, "w") as f:
                f.write(src)
            so = build("loader_cuda", src_path, tmpdir)
            with open(so, "rb") as f:
                self.cache_path = self.cache.put(f.read(), self.fname, binary=True)
        
    def loadBinary(self, kernelFile : KernelLibFile) -> KernelRuntimeInfo :
        if self.load_binary is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("loader_cuda", self.cache_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.load_binary = mod.load_binary
            self.unload_binary = mod.unload_binary
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
    
    def unloadBinary(self, kernelFile : KernelLibFile) :
        if kernelFile.m_kernelInfo is not None and kernelFile.m_kernelInfo.m_module is not None :
            self.unload_binary(kernelFile.m_kernelInfo.m_module)

    
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
            src_path = tmpdir + "/main_loader_hip.c"
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
            print(f"[loader] name,binaryPath,shared,device = {name,binaryPath,shared,device}")
            mod,func, n_regs, n_spills = self.load_binary(name,binaryPath,shared,device)
            info = KernelRuntimeInfo(mod,func,n_regs,n_spills)
            kernelFile.m_kernelInfo = info
            
        return kernelFile.m_kernelInfo

