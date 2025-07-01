import os
import re
import sysconfig
import torch
from deepgen.utils import loadLibModule, compileModuleFromSrc
from deepgen.HIPCompiler import Kernel, HIPCompiler

dirname = os.path.dirname(os.path.realpath(__file__))
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

def makeHostSrc(kernel_name, arg_num, grid, block, smem, arc_path, target="rocm"):
  ptr_type = "CUdeviceptr" if target == "cuda" else "hipDeviceptr_t"
  src = f"""
{"#define __HIP_PLATFORM_AMD__" if target == "rocm" else ""}
{"#include <cuda.h>" if target == "cuda" else ""}
#include <{"cuda" if target == "cuda" else "hip/hip"}_runtime.h>
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <dlfcn.h>

void _launch({", ".join(f"{ptr_type} arg{i}" for i in range(arg_num))}) {{
  void* args[] = {{ {", ".join(f"&arg{i}" for i in range(arg_num))} }};
  {"CUmodule" if target == "cuda" else "hipModule_t"} module;
  {"CUfunction" if target == "cuda" else "hipFunction_t"} kernel_fn;
  {"cu" if target == "cuda" else "hip"}ModuleLoad(&module, "{arc_path}");
  {"cu" if target == "cuda" else "hip"}ModuleGetFunction(&kernel_fn, module, "{kernel_name}");
  {"cu" if target == "cuda" else "hipModule"}LaunchKernel(
    kernel_fn,
    {grid[0]}, {grid[1]}, {grid[2]},
    {block[0]}, {block[1]}, {block[2]},
    {smem}, nullptr, args, nullptr
  );
}}

typedef struct _DevicePtrInfo {{
    {ptr_type} dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = ({ptr_type})PyLong_AsUnsignedLongLong(obj);
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
    ptr_info.dev_ptr = ({ptr_type})PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      Py_DECREF(ret);
      return ptr_info;
    }}

    {"CUpointer_attribute attributes[] = {{CU_POINTER_ATTRIBUTE_DEVICE_POINTER}}; CUresult status; void* data;" if target == "cuda" else "hipError_t status; uint64_t dev_ptr;"}
    status = {"hip" if target == "rocm" else "cu"}PointerGetAttribute(&{"dev_ptr, HIP" if target == "rocm" else "data, CU"}_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == {"hipErrorInvalidValue" if target == "rocm" else "CUDA_SUCCESS"}) {{
        PyErr_Format(PyExc_ValueError, "Pointer argument (at %d) cannot be accessed from Deepgen (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    {"ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;" if target == "rocm" else ""}
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  {"ptr_info.valid = false;" if target == "cuda" else ""}
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  PyObject {", ".join(f"*arg{i}" for i in range(arg_num))};
  if (!PyArg_ParseTuple(args, "{"".join("O" for i in range(arg_num))}", {", ".join(f"&arg{i}" for i in range(arg_num))})) {{
      return NULL;
  }}
  {" ".join(f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" for i in range(arg_num))}
  _launch({", ".join(f"ptr_info{i}.dev_ptr" for i in range(arg_num))});
  return Py_None;
}}

static PyMethodDef launchMethods[] = {{
    {{"launch", launch, METH_VARARGS, "Launch kernel"}}, 
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef deepgenmodule = {{
  PyModuleDef_HEAD_INIT,
  "launch",
  NULL,
  -1,
  launchMethods
}};

PyMODINIT_FUNC PyInit_launch(void) {{
  PyObject *m = PyModule_Create(&deepgenmodule);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, launchMethods);
  return m;
}}
"""
  # print(src)
  return src

class Runtime:
  def __init__(self, target="rocm", arch="gfx906"):
    lib_name = "deepgen"
    self.target = target
    deep_gen_so = os.path.join(dirname, "../../bin", f"lib{lib_name}.so")
    mod = loadLibModule(lib_name, deep_gen_so)
    match = re.findall(r'(?:gfx|sm_)(\d+)', arch)
    mod.set_platform(target, match[0])
    self.compile_mm = mod.compile_mm
    self.compile_attn = mod.compile_attn
  
  def compile(self, kernel: str, cfg: dict):
    if kernel == "matmul":  # compile_mm 应该接受 cfg["type"]
      kernel_dir = self.compile_mm(cfg["shape"], cfg["config"])
    elif kernel == "attention":
      if self.target == "rocm":
        hipcc = HIPCompiler()
        kernel_dir = hipcc.build(Kernel.Attention, cfg["shape"], cfg["config"]["attention"])
      else:
        kernel_dir = self.compile_attn(cfg["shape"], cfg["config"])
    host_src = makeHostSrc(kernel, len(cfg["type"]), cfg["grid"], cfg["block"], cfg["smem"], kernel_dir, self.target)
    # print(host_src)
    mod = compileModuleFromSrc("launch", host_src, self.target)
    return mod.launch