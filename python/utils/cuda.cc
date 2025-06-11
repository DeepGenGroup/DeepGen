#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CUDA_CHECK(cmd) { cudaError_t error = cmd; if (error != cudaSuccess) { \
    PyErr_Format(PyExc_RuntimeError, "CUDA error %d: %s", error, cudaGetErrorString(error)); \
    return NULL; }}

// GPU 信息结构体
typedef struct {
    PyObject_HEAD
    char name[256];
    int compute_units;
    size_t shared_mem_per_block;
    int regs_per_block;
    int warp_size;
    char arch[32];
    int max_thread_per_block;
    int clock_rate_khz;
    int mem_clock_rate_khz;
    int mem_bus_width;
    size_t l2_cache_size;
    size_t global_mem;
} GPUInfoObject;

// GPUInfo 类型方法
static PyObject* GPUInfo_get_name(GPUInfoObject* self, void* closure) {
    return PyUnicode_FromString(self->name);
}
static PyObject* GPUInfo_get_compute_units(GPUInfoObject* self, void* closure) {
    return PyLong_FromLong(self->compute_units);
}
static PyObject* GPUInfo_get_shared_mem(GPUInfoObject* self, void* closure) {
    return PyLong_FromSize_t(self->shared_mem_per_block);
}
static PyObject* GPUInfo_get_regs(GPUInfoObject* self, void* closure) {
    return PyLong_FromLong(self->regs_per_block);
}
static PyObject* GPUInfo_get_warp_size(GPUInfoObject* self, void* closure) {
    return PyLong_FromLong(self->warp_size);
}
static PyObject* GPUInfo_get_arch(GPUInfoObject* self, void* closure) {
    return PyUnicode_FromString(self->arch);
}
static PyObject* GPUInfo_get_max_thread(GPUInfoObject* self, void* closure) {
    return PyLong_FromLong(self->max_thread_per_block);
}
static PyObject* GPUInfo_get_clock_rate(GPUInfoObject* self, void* closure) {
    return PyLong_FromLong(self->clock_rate_khz);
}
static PyObject* GPUInfo_get_mem_clock_rate(GPUInfoObject* self, void* closure) {
    return PyLong_FromLong(self->mem_clock_rate_khz);
}
static PyObject* GPUInfo_get_mem_bus_width(GPUInfoObject* self, void* closure) {
    return PyLong_FromLong(self->mem_bus_width);
}
static PyObject* GPUInfo_get_l2_cache(GPUInfoObject* self, void* closure) {
    return PyLong_FromSize_t(self->l2_cache_size);
}
static PyObject* GPUInfo_get_global_mem(GPUInfoObject* self, void* closure) {
    return PyLong_FromSize_t(self->global_mem);
}

// GPUInfo 属性定义
static PyGetSetDef GPUInfo_getset[] = {
    {"name", (getter)GPUInfo_get_name, NULL, "GPU name", NULL},
    {"compute_units", (getter)GPUInfo_get_compute_units, NULL, "Number of compute units (SMs)", NULL},
    {"shared_mem_per_block", (getter)GPUInfo_get_shared_mem, NULL, "Shared memory per block in bytes", NULL},
    {"regs_per_block", (getter)GPUInfo_get_regs, NULL, "Number of registers per block", NULL},
    {"warp_size", (getter)GPUInfo_get_warp_size, NULL, "Warp size", NULL},
    {"arch", (getter)GPUInfo_get_arch, NULL, "GPU architecture", NULL},
    {"max_thread_per_block", (getter)GPUInfo_get_max_thread, NULL, "Max threads per block", NULL},
    {"clock_rate_khz", (getter)GPUInfo_get_clock_rate, NULL, "Clock rate in kHz", NULL},
    {"mem_clock_rate_khz", (getter)GPUInfo_get_mem_clock_rate, NULL, "Memory clock rate in kHz", NULL},
    {"mem_bus_width", (getter)GPUInfo_get_mem_bus_width, NULL, "Memory bus width in bits", NULL},
    {"l2_cache_size", (getter)GPUInfo_get_l2_cache, NULL, "L2 Cache size in bytes", NULL},
    {"global_mem", (getter)GPUInfo_get_global_mem, NULL, "Total global memory size in bytes", NULL},
    {NULL}  /* Sentinel */
};

// GPUInfo 类型定义
static PyTypeObject GPUInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gpu_info.GPUInfo",
    .tp_basicsize = sizeof(GPUInfoObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "GPU information object",
    .tp_getset = GPUInfo_getset,
};

// 获取 GPU 信息的函数
static PyObject* get_gpu_info(PyObject* self, PyObject* args) {
    int device_id = 0;
    if (!PyArg_ParseTuple(args, "|i", &device_id)) {
        return NULL;
    }
    
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    
    // 创建 GPUInfo 对象
    GPUInfoObject* info = (GPUInfoObject*)GPUInfoType.tp_alloc(&GPUInfoType, 0);
    if (!info) {
        return NULL;
    }
    
    // 填充数据
    strncpy(info->name, props.name, sizeof(info->name));
    info->compute_units = props.multiProcessorCount;
    info->shared_mem_per_block = props.sharedMemPerBlock;
    info->regs_per_block = props.regsPerBlock;
    info->warp_size = props.warpSize;
    snprintf(info->arch, sizeof(info->arch), "sm_%d%d", props.major, props.minor);
    info->global_mem = props.totalGlobalMem;
    info->max_thread_per_block = props.maxThreadsPerBlock;
    info->clock_rate_khz = props.clockRate;
    info->mem_clock_rate_khz = props.memoryClockRate;
    info->mem_bus_width = props.memoryBusWidth;
    info->l2_cache_size = props.l2CacheSize;
    info->global_mem = props.totalGlobalMem;

    return (PyObject*)info;
}

// 获取设备数量的函数
static PyObject* get_device_count(PyObject* self, PyObject* args) {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return PyLong_FromLong(count);
}

// 模块方法定义
static PyMethodDef module_methods[] = {
    {"get_gpu_info", get_gpu_info, METH_VARARGS, "Get GPU information"},
    {"get_device_count", get_device_count, METH_NOARGS, "Get number of CUDA devices"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

// 模块定义
static struct PyModuleDef gpu_info_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "gpu_info",
    .m_doc = "Module for querying NVIDIA GPU information",
    .m_size = -1,
    .m_methods = module_methods,
};

// 模块初始化函数
PyMODINIT_FUNC PyInit_gpu_info(void) {
    PyObject* module;
    
    // 初始化 GPUInfo 类型
    if (PyType_Ready(&GPUInfoType) < 0) {
        return NULL;
    }

    // 创建模块
    module = PyModule_Create(&gpu_info_module);
    if (!module) {
        return NULL;
    }

    // 添加 GPUInfo 类型到模块
    Py_INCREF(&GPUInfoType);
    if (PyModule_AddObject(module, "GPUInfo", (PyObject*)&GPUInfoType) < 0) {
        Py_DECREF(&GPUInfoType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}