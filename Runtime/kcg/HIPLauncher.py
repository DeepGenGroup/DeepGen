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
from kcg.Loader import HIPLoaderST
import importlib.util
# HIP_BACKEND_MODE = False

# if HIP_BACKEND_MODE:
#     from ..._C.librocm_backend_for_triton import triton as _triton
# else:
#     from ..._C.libtriton import triton as _triton


def make_stub(kernelLibFile : KernelLibFile) -> str :
    so_cache_key = str(kernelLibFile.hash())
    so_cache_manager = FileCacheManager(so_cache_key)
    # so_name = f"{so_cache_key + kernelLibFile.m_kernelFuncName}.so"
    so_name = f"LaunHip_{kernelLibFile.signature_str()}.so"
    # retrieve stub from cache if it exists
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        # with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = PathManager().default_cache_dir()
        src = generate_launcher_hip(kernelLibFile)
        src_path = os.path.join(tmpdir, "stub_main.c")
        with open(src_path, "w") as f:
            for line in src:
                f.write(line)  # generate stub code
        with open('/home/xushilong/CodeGenDemo/tempstub.c', "w") as f:
            for line in src:
                f.write(line)  # generate stub code
        so = build(so_name, src_path, tmpdir)
        with open(so, "rb") as f:
            return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t"
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

def generate_launcher_hip(kernelLib : KernelLibFile):
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

    format = "iiiiiiiiiiiKKOOO" + ''.join([format_of(_extracted_type(ty)) for i,ty in kernelSignature.items()])

    # generate glue code
    params = [i for i,ty in kernelSignature.items()]
    src = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <Python.h>
#include <stdbool.h>
#include <dlfcn.h>

static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "KCG Error [HIP]: ";
      const char* str = hipGetErrorString(code);
      char err[1024] = {{0}};
      snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  // printf("_launch hip kernel\\n");
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
      HIP_CHECK(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, shared_memory, stream, params, 0));
    }}
  }}

typedef struct _DevicePtrInfo {{
    hipDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
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
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    hipError_t status = hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == hipErrorInvalidValue) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
   // printf("launch\\n");
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int blockX, blockY, blockZ; 
  int num_ctas;
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in kernelSignature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &blockX,&blockY,&blockZ, &num_ctas, &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel{', ' + ', '.join(f"&_arg{i}" for i, ty in kernelSignature.items()) if len(kernelSignature) > 0 else ''})) {{
    return NULL;
  }}
  // printf("hiplauncher parsed : \\n");
  // printf("- gridX : %d \\n",gridX);
  // printf("- gridY : %d \\n",gridY);
  // printf("- gridZ : %d \\n",gridZ);
  // printf("- blockX : %d \\n",blockX);
  // printf("- blockY : %d \\n",blockY);
  // printf("- blockZ : %d \\n",blockZ);
  // printf("- num_ctas : %d \\n",num_ctas);
  // printf("- clusterDimX : %d \\n",clusterDimX);
  // printf("- clusterDimY : %d \\n",clusterDimY);
  // printf("- clusterDimZ : %d \\n",clusterDimZ);
  // printf("- shared_memory : %d \\n",shared_memory);
  if (launch_enter_hook != Py_None) {{
    PyObject_CallObject(launch_enter_hook, args);
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in kernelSignature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  // printf("- call _launch function \\n");
  _launch(gridX, gridY, gridZ, blockX, blockY, blockZ, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in kernelSignature.items()) if len(kernelSignature) > 0 else ''});
  Py_END_ALLOW_THREADS;

  if (launch_exit_hook != Py_None) {{
    PyObject_CallObject(launch_exit_hook, args);
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


class HIPLauncher :
    def __init__(self, kernelBinaryPath,kernelFuncName,shmSize,signature:dict,gridDims:list,blockDims:list,device=DeviceInfo.get_current_device()):
        self.m_cWrapper = None
        self.m_kernelLib = KernelLibFile(kernelBinaryPath,EnumBackendType.HIP,kernelFuncName,shmSize,signature,gridDims,blockDims,device)
        self.m_launcherLibPath = None  # launcher.so 的路径
        
    def __hash__(self):
        return "launch_"+self.m_kernelLib.hash()
    
    def __loadKernel(self):
        loader = HIPLoaderST()
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

    def launchKernel(self,devId:int,*args):
        wrapper = self._getWrapper()
        stream = DeviceInfo.get_cuda_stream(devId)
        if wrapper is None:
            raise Exception("kcg: _getWrapper failed")
        gridDims = self.m_kernelLib.m_gridDims
        blockDims = self.m_kernelLib.m_blockDims
        clusterDims = [1,1,1]  # Grid > Cluster > CTA(=Block=WorkGroup) > Wavefront(=Warp) > workitem(=thread) 
        enterHookFunc = None
        exitHookFunc = None
        numCTAs = gridDims[0]*gridDims[1]*gridDims[2]
        # print(f"[Runtime] gridDims = {gridDims}, blockdims={blockDims} ")
        wrapper(gridDims[0],gridDims[1],gridDims[2],blockDims[0],blockDims[1],blockDims[2],
                # m.num_ctas,
                numCTAs,
                clusterDims[0],clusterDims[1],clusterDims[2],
                self.m_kernelLib.m_shmSize,
                stream,
                self.m_kernelLib.m_kernelInfo.m_function, 
                enterHookFunc,
                exitHookFunc,
                self,*args )

        if wrapper is None :
            # print("[D] error cwrapper")
            pass
        else:
            # print("[D] success cwrapper")
            pass