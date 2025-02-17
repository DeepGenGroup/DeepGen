
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <Python.h>
#include <stdbool.h>
#include <dlfcn.h>

static inline void gpuAssert(hipError_t code, const char *file, int line)
{
   if (code != HIP_SUCCESS)
   {
      const char* prefix = "KCG Error [HIP]: ";
      const char* str = hipGetErrorString(code);
      char err[1024] = {0};
      snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }
}

#define HIP_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static void _launch(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function, hipDeviceptr_t arg0, hipDeviceptr_t arg1, hipDeviceptr_t arg2) {
  // printf("_launch hip kernel\n");
  void *params[] = { &arg0, &arg1, &arg2 };
  if (gridX*gridY*gridZ > 0) {
      HIP_CHECK(hipModuleLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, shared_memory, stream, params, 0));
    }
  }

typedef struct _DevicePtrInfo {
    hipDeviceptr_t dev_ptr;
    bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    hipError_t status = hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == hipErrorInvalidValue) {
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }
    ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}

static PyObject* launch(PyObject* self, PyObject* args) {
   // printf("launch\n");
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
  PyObject* _arg0;  PyObject* _arg1;  PyObject* _arg2; 
  if(!PyArg_ParseTuple(args, "iiiiiiiiiiiKKOOOOOO", &gridX, &gridY, &gridZ, &blockX,&blockY,&blockZ, &num_ctas, &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, &_arg0, &_arg1, &_arg2)) {
    return NULL;
  }
  // printf("hiplauncher parsed : \n");
  // printf("- gridX : %d \n",gridX);
  // printf("- gridY : %d \n",gridY);
  // printf("- gridZ : %d \n",gridZ);
  // printf("- blockX : %d \n",blockX);
  // printf("- blockY : %d \n",blockY);
  // printf("- blockZ : %d \n",blockZ);
  // printf("- num_ctas : %d \n",num_ctas);
  // printf("- clusterDimX : %d \n",clusterDimX);
  // printf("- clusterDimY : %d \n",clusterDimY);
  // printf("- clusterDimZ : %d \n",clusterDimZ);
  // printf("- shared_memory : %d \n",shared_memory);
  if (launch_enter_hook != Py_None) {
    PyObject_CallObject(launch_enter_hook, args);
  }


  // raise exception asap
  DevicePtrInfo ptr_info0 = getPointer(_arg0, 0); if (!ptr_info0.valid) return NULL;; DevicePtrInfo ptr_info1 = getPointer(_arg1, 1); if (!ptr_info1.valid) return NULL;; DevicePtrInfo ptr_info2 = getPointer(_arg2, 2); if (!ptr_info2.valid) return NULL;;
  Py_BEGIN_ALLOW_THREADS;
  // printf("- call _launch function \n");
  _launch(gridX, gridY, gridZ, blockX, blockY, blockZ, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function, ptr_info0.dev_ptr, ptr_info1.dev_ptr, ptr_info2.dev_ptr);
  Py_END_ALLOW_THREADS;

  if (launch_exit_hook != Py_None) {
    PyObject_CallObject(launch_exit_hook, args);
  }

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__kcg_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___kcg_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
