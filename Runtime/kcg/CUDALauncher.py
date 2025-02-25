import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Tuple
from kcg.Cache import *

# from kcg.common.backend import BaseBackend, register_backend, compute_core_version_key
# from kcg.Utils import generate_cu_signature

# from kcg.Launcher.make_launcher import get_cache_manager, make_so_cache_key
from kcg.Cache import *
from kcg.Utils import *
from kcg.Kernel import *
from kcg.Loader import CudaLoaderST
import importlib.util
# ----- stub --------


def make_so_cache_key(version_hash, signature, constants, ids, **kwargs):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}-{constants}-{ids}"
    for kw in kwargs:
        key = f"{key}-{kwargs.get(kw)}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def make_stub(kernelLibFile : KernelLibFile) -> str :
    so_cache_key = str(kernelLibFile.hash())
    so_cache_manager = FileCacheManager(so_cache_key)
    # so_name = f"{so_cache_key + kernelLibFile.m_kernelFuncName}.so"
    so_name = f"LaunCuda_{kernelLibFile.signature_str()}.so"
    # retrieve stub from cache if it exists
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        # with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = PathManager().default_cache_dir()
        src = generate_launcher_cuda(kernelLibFile)
        src_path = os.path.join(tmpdir, "stub_main_cuda.c")
        with open(src_path, "w") as f:
            for line in src:
                f.write(line)  # generate stub code
        so = build(so_name, src_path, tmpdir)
        with open(so, "rb") as f:
            return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path


# ----- source code generation --------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t" if is_hip() else "CUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def generate_launcher_cuda(kernelLib : KernelLibFile):
    # start_desc = len(signature)
    # warp_size = DeviceInfo.get_warp_size()
    kernelSignature : dict = kernelLib.m_signature
    # gridDims = kernelLib.m_gridDims
    # blockDims = kernelLib.m_blockDims
    print(type(kernelSignature))
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{index}" for index,ty in kernelSignature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    format = "iiiiiiiiiKKOOO" + ''.join([format_of(_extracted_type(ty)) for i,ty in kernelSignature.items()])

    # generate glue code
    # folded_without_constexprs = [c for c in ids['ids_of_folded_args'] if c not in ids['ids_of_const_exprs']]
    params = [i for i,ty in kernelSignature.items()]

    src = f"""
#include \"cuda.h\"
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "DeepGen Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {{
  // Open the shared library
  void* handle = dlopen("libcuda.so", RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so");
    return NULL;
  }}
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so");
    return NULL;
  }}
  return cuLaunchKernelExHandle;
}}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
    if (num_ctas == 1 || clusterDimX ==0 || clusterDimY == 0 || clusterDimZ == 0) {{
      CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
    }} else {{
      CUlaunchAttribute launchAttr[2];
      launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttr[0].value.clusterDim.x = clusterDimX;
      launchAttr[0].value.clusterDim.y = clusterDimY;
      launchAttr[0].value.clusterDim.z = clusterDimZ;
      launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttr[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      CUlaunchConfig config;
      config.gridDimX = gridX * clusterDimX;
      config.gridDimY = gridY * clusterDimY;
      config.gridDimZ = gridZ * clusterDimZ;
      config.blockDimX = 32 * num_warps;
      config.blockDimY = 1;
      config.blockDimZ = 1;
      config.sharedMemBytes = shared_memory;
      config.hStream = stream;
      config.attrs = launchAttr;
      config.numAttrs = 2;
      static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
      if (cuLaunchKernelExHandle == NULL) {{
        cuLaunchKernelExHandle = getLaunchKernelExHandle();
      }}
      CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int num_ctas;
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in kernelSignature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &num_ctas, &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel{', ' + ', '.join(f"&_arg{i}" for i, ty in kernelSignature.items()) if len(kernelSignature) > 0 else ''})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in kernelSignature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in kernelSignature.items()) if len(kernelSignature) > 0 else ''});
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__kcg_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___kcg_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class CUDALauncher :
    def __init__(self, kernelBinaryPath,kernelFuncName,shmSize,signature:dict,gridDims:list,blockDims:list,device=DeviceInfo.get_current_device()):
        self.m_cWrapper = None
        self.m_kernelLib = KernelLibFile(kernelBinaryPath,EnumBackendType.CUDA,kernelFuncName,shmSize,signature,gridDims,blockDims,device)
        self.m_launcherLibPath = None  # launcher.so 的路径
        
    def __hash__(self):
        return "launchCUDA_"+self.m_kernelLib.hash()
    
    def __loadKernel(self):
        loader = CudaLoaderST()
        loader.loadBinary(self.m_kernelLib)
    
    def _getWrapper(self) -> Callable:
        if self.m_launcherLibPath is None :
            if self.m_kernelLib.m_kernelInfo is None : 
                self.__loadKernel()
            # compile launcher.so
            self.m_launcherLibPath = make_stub(self.m_kernelLib)
        if self.m_cWrapper is None :
			# import launcher.so as module
            spec = importlib.util.spec_from_file_location("__kcg_launcher", self.m_launcherLibPath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.m_cWrapper = getattr(mod, "launch")
        return self.m_cWrapper

    def launchKernel(self,*args):
        wrapper = self._getWrapper()
        devid = self.m_kernelLib.m_device
        stream = DeviceInfo.get_cuda_stream(devid)
        if wrapper is None:
            raise Exception("kcg: _getWrapper failed")
        gridDims = self.m_kernelLib.m_gridDims
        blockDims = self.m_kernelLib.m_blockDims
        clusterDims = [0,0,0]  # Grid > Cluster > CTA(=Block=WorkGroup) > Wavefront(=Warp) > workitem(=thread) 
        enterHookFunc = None
        exitHookFunc = None
        numCTAs = gridDims[0]*gridDims[1]*gridDims[2]
        # print(f"[Runtime] gridDims = {gridDims}, blockdims={blockDims} ")
        numWarps = int(blockDims[0]*blockDims[1]*blockDims[2] / 32)
        # print(f'numwarps={numWarps}')
        if numWarps < 1 :
          numWarps = 1
        wrapper(gridDims[0],gridDims[1],gridDims[2], numWarps,
                numCTAs,
                clusterDims[0],clusterDims[1],clusterDims[2],
                self.m_kernelLib.m_shmSize,
                stream,
                self.m_kernelLib.m_kernelInfo.m_function, 
                enterHookFunc,
                exitHookFunc,
                self,*args )

        if wrapper is None :
            print("[D] error cwrapper")
            pass
        else:
            print("[D] success cwrapper")
            pass