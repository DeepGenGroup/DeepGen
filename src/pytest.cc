#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Python.h"
#include <exception>

using namespace KernelCodeGen;

using TuneConfig = std::map<std::string, std::map<std::string, int64_t>>;
KernelCodeGen::Target __GlobalTarget = KernelCodeGen::Target::CUDA;
std::string __GlobalPlatDesc = "";
std::string __GlobalKernelName = "attention1";

static PyObject* packResultsToPythonObject(std::vector<KernelInfo>& kernels){
  PyObject *retArr;
  retArr = PyTuple_New(kernels.size());
    // 填充元组
  for (int i=0;i<kernels.size();++i) {
    // 假设元素数组元素是以字符串和整数对的方式存在
    // 这里你需要将每一对 (ss, i...) 插入
    const auto& kernel = kernels[i];
    std::vector<int> gridDims = {1,1,1};
    std::vector<int> blockDims = {1,1,1};
    for(int i=0;i<kernel.m_gridDims.size();++i){
      gridDims[i] = kernel.m_gridDims[i];
    }
    for(int i=0;i<kernel.m_blockDims.size();++i){
      blockDims[i] = kernel.m_blockDims[i];
    }
    PyObject* item = Py_BuildValue("(ssiiiiiii)",
      kernel.m_binaryPath.c_str(),
      kernel.m_kernelName.c_str(),
      gridDims[0],gridDims[1],gridDims[2],
      blockDims[0],blockDims[1],blockDims[2],
      kernel.m_shmBytes
    );
    if (item == NULL) {
      Py_DECREF(retArr);
      return NULL;  // 如果构建某个元素失败，释放资源并返回NULL
    }
    PyTuple_SetItem(retArr, i, item);  // 将每个元素插入元组
  }
  return retArr;
  // std::cout << "[pymod] ======== compile_kernel_matmul return " << std::endl;
}


std::string compile_attn(std::vector<int64_t> shape, const TuneConfig& config) {
    // auto attn = config.at("attention1");
  std::map<std::string, std::map<std::string, int64_t>> tileConfig;
  try{
    auto attn = config.at(__GlobalKernelName);
    tileConfig = {
      {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                  {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}}, 
      {"softmax1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                  {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
      {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")}, 
                  {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
    };
  }
  catch(std::exception & e){
    std::cout << "[Deepgen Fatal] invalid config or not set kernel name!" << std::endl;
    std::abort();
  }

  KernelCodeGenerator generator(__GlobalTarget, __GlobalPlatDesc);
  mlir::ModuleOp module = generator.createModule();
  std::vector<KernelData> kds;
  std::vector<FuseKernelData> fkds;
  KernelInfo info;
// ======  kernel  ======
  KernelData kd1, kd2, kd3;
  // matmul1
  kd1.name = "matmul1";
  kd1.type = "Matmul";
  kd1.argNames = {"A1", "B1", "C1"};
  kd1.shapes = {{shape[0], shape[1], shape[3], shape[2]}, 
                {shape[0], shape[1], shape[3], shape[2]}, 
                {shape[0], shape[1], shape[2], shape[2]}};
  kd1.dtypes = {"float32", "float32", "float32"};
  kd1.isTrans = {true, false};
  kd1.outputArgNum = 1;
  kds.push_back(kd1);
  //Softmax1
  kd2.name = "softmax1";
  kd2.type = "Softmax";
  kd2.argNames = {"C1", "C2"};
  kd2.shapes = {{shape[0], shape[1], shape[2], shape[2]}, 
                {shape[0], shape[1], shape[2], shape[2]}};
  kd2.dtypes = {"float32", "float32"};
  kd2.isTrans = {false};
  kd2.outputArgNum = 1;
  kds.push_back(kd2);
  // matmul2
  kd3.name = "matmul2";
  kd3.type = "Matmul";
  kd3.argNames = {"C2", "B2", "C3"};
  kd3.shapes = {{shape[0], shape[1], shape[2], shape[2]}, shape, shape};
  kd3.dtypes = {"float32", "float32", "float32"};
  kd3.isTrans = {false, false};
  kd3.outputArgNum = 1;
  kds.push_back(kd3);

  // ======  fuse kernel  ======
  std::cout << "[lib] ===========3" << std::endl;
  const std::string& kernelName = __GlobalKernelName;
  std::cout << "[lib]kernelName = " << kernelName << std::endl;
  FuseKernelData fkd = {
    kernelName,
    "FlashAttn",
    {"matmul1", "softmax1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd3.shapes[1], kd3.shapes[2]},
    {kd1.shapes[2]},
    {"float32", "float32", "float32", "float32"},
    {"float32"},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {{"matmul2", {2}}}}, 
    {{{"matmul1", {2}}, {"softmax1", {0, 1}} , {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd3.isTrans[1]},
    {"y"},
    1
  };
  fkds.push_back(fkd);

  // create kernels
  auto noSupKernels = generator.createKernels(module, kds);
  // fusing
  auto result = generator.fusing(module, fkds);
  // mpping
  result = generator.mapping(module, tileConfig);
  // optimize
  generator.optimize(module, config);
  // llvm::outs() << "=========== after optimize ===========\n"; llvm::outs().flush();module->dump();
  // lowering
  generator.lowering(module);
  // translate
  auto path = generator.translate(module);
  // std::cout << "[lib] ===========4" << std::endl;
  return path;
}


std::string compile_mm(std::vector<int64_t> shape, const TuneConfig& config) {
    // auto attn = config.at("attention1");
  std::map<std::string, std::map<std::string, int64_t>> tileConfig;
  try{
    auto mm = config.at(__GlobalKernelName);
    tileConfig = {
      {__GlobalKernelName , {{"BLOCK_SIZE_Y", mm.at(KEY_BLOCK_SIZE_M)}, {"THREAD_SIZE_Y", mm.at(KEY_THREAD_SIZE_M)}, 
                  {"BLOCK_SIZE_X", mm.at(KEY_BLOCK_SIZE_N)}, {"THREAD_SIZE_X", mm.at(KEY_THREAD_SIZE_N)}}}
    };
  }
  catch(std::exception& e){
    std::cout << "[Deepgen Fatal] invalid config or not set kernel name!" << std::endl;
    std::abort();
  }

  KernelCodeGenerator generator(__GlobalTarget, __GlobalPlatDesc);
  // std::cout << "[lib] ============ 1" << std::endl;
  mlir::ModuleOp module = generator.createModule();
  std::vector<KernelData> kds;
  // std::vector<FuseKernelData> fkds;
  KernelInfo info;
// ======  kernel  ======
  KernelData kd1;
  // matmul1
  // kd1.name = "matmul1";
  kd1.name = __GlobalKernelName;
  kd1.type = "Matmul";
  kd1.argNames = {"A1", "B1", "C1"};
  
  int LEN = shape.size();  // [.. , ..] , M , N , K
  assert(LEN >= 3);
  int64_t mVal; int64_t nVal;int64_t kVal;
  mVal = shape[LEN-3]; nVal = shape[LEN-2]; kVal = shape[LEN-1];
  std::vector<int64_t> shapeA;
  std::vector<int64_t> shapeB;
  std::vector<int64_t> shapeC;
  // std::cout << "[lib] ============ 2" << std::endl;
  for(int i=0;i<LEN-3;++i){
    shapeA.push_back(shape[i]);
    shapeB.push_back(shape[i]);
    shapeC.push_back(shape[i]);
  }
  shapeA.push_back(kVal);shapeA.push_back(mVal);
  shapeB.push_back(kVal);shapeB.push_back(nVal);
  shapeC.push_back(mVal);shapeC.push_back(nVal);

  kd1.shapes = {shapeA, shapeB, shapeC};
  kd1.dtypes = {"float32", "float32", "float32"};
  kd1.isTrans = {true, false};
  kd1.outputArgNum = 1;
  kds.push_back(kd1);

  // std::cout << "[lib] ============ 3" << std::endl;
  // create kernels
  auto noSupKernels = generator.createKernels(module, kds);
  // std::cout << "[lib] ============ 4" << std::endl;
  // mpping
  auto result = generator.mapping(module, tileConfig);
  // std::cout << "[lib] ============ 5" << std::endl;
  // optimize
  generator.optimize(module, config);
  // std::cout << "[lib] ============ 6" << std::endl;
  // llvm::outs() << "=========== after optimize ===========\n"; llvm::outs().flush();module->dump();
  // lowering
  generator.lowering(module);
  // translate
  auto path = generator.translate(module);
  // std::cout << "[lib] ============ 7" << std::endl;
  // std::cout << "[lib] ===========4" << std::endl;
  return path;
}


static bool py_list_to_vector(PyObject* py_list, std::vector<int64_t>& vec) {
  // list to vector
  if (!PyList_Check(py_list)) {
    PyErr_SetString(PyExc_TypeError, "Expected a list");
    return false;
  }
  Py_ssize_t size = PyList_Size(py_list);
  vec.resize(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "List items must be integers");
      return false;
    }
    vec[i] = PyLong_AsLongLong(item);
  }
  return true;
}


static bool py_dict_to_config(PyObject* py_dict, TuneConfig& config) {
  // dict to config
  if (!PyDict_Check(py_dict)) {
    PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
    return false;
  }
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(py_dict, &pos, &key, &value)) {
    if (!PyUnicode_Check(key)) {
      PyErr_SetString(PyExc_TypeError, "Dictionary keys must be strings");
      return false;
    }
    if (!PyDict_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Expected nested dictionary");
      return false;
    }
    const char* outer_key = PyUnicode_AsUTF8(key);
    std::map<std::string, int64_t> inner_map;
    PyObject *inner_key, *inner_value;
    Py_ssize_t inner_pos = 0;
    while (PyDict_Next(value, &inner_pos, &inner_key, &inner_value)) {
      if (!PyUnicode_Check(inner_key)) {
        PyErr_SetString(PyExc_TypeError, "Nested dictionary keys must be strings");
        return false;
      }
      if (!PyLong_Check(inner_value)) {
        PyErr_SetString(PyExc_TypeError, "Nested dictionary values must be integers");
        return false;
      }
      const char* key_str = PyUnicode_AsUTF8(inner_key);
      int64_t val = PyLong_AsLongLong(inner_value);
      inner_map[key_str] = val;
    }
      config[outer_key] = inner_map;
    }
  return true;
}


static PyObject* py_compile_attn(PyObject* self, PyObject* args) {
  // bind compile_attn func
  PyObject* py_shape;
  PyObject* py_config;
  if (!PyArg_ParseTuple(args, "OO", &py_shape, &py_config)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) {
    return NULL;
  }
  if (!py_dict_to_config(py_config, config)) {
    return NULL;
  }
  std::string result = compile_attn(shape, config);
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_mm(PyObject* self, PyObject* args) {
  // bind compile_attn func
  PyObject* py_shape;
  PyObject* py_config;
  if (!PyArg_ParseTuple(args, "OO", &py_shape, &py_config)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) {
    return NULL;
  }
  if (!py_dict_to_config(py_config, config)) {
    return NULL;
  }
  std::string result = compile_mm(shape, config);
  return PyUnicode_FromString(result.c_str());
}



static PyObject* set_platform(PyObject* self, PyObject* args) {
  int index = 0;
  char* platInfo = NULL;
  if(PyArg_ParseTuple(args, "is", &index, &platInfo)){
    if(index == 2){
      __GlobalTarget = KernelCodeGen::Target::ROCm;
    }
    else if(index == 1){
      __GlobalTarget = KernelCodeGen::Target::CUDA;
    }
    else{
      std::cout << "DeepGen Error : Invalid Platform id " << index << std::endl;
      std::abort();
    }
    __GlobalPlatDesc = std::string(platInfo);
    if(__GlobalPlatDesc.size() <= 0){
      std::cout << "DeepGen Error : Invalid Arch info " << __GlobalPlatDesc << std::endl;
      std::abort();
    }
  }
  return Py_None;
}


static PyObject* set_kernel_name(PyObject* self, PyObject* args) {
  char* kernelName = NULL;
  if(PyArg_ParseTuple(args, "s", &kernelName)){
    __GlobalKernelName = std::string(kernelName);
    if(__GlobalKernelName.size() <= 0){
      std::cout << "DeepGen Error : Invalid KernelName " << __GlobalKernelName << std::endl;
      std::abort();
    }
    else{
      std::cout << "[lib] setKernelNAme : "<< __GlobalKernelName << std::endl;
    }
  }
  return Py_None;
}

// 方法定义
static PyMethodDef DeepgenMethods[] = {
    {"compile_attn", py_compile_attn, METH_VARARGS, "Compile attention with given shape and config"},
    {"compile_mm", py_compile_mm, METH_VARARGS, "Compile matmul with given shape and config"},
    {"set_kernel_name", set_kernel_name, METH_VARARGS, "Compile attention with given shape and config"},
    {"set_platform", set_platform, METH_VARARGS, "Compile attention with given shape and config"},
    {NULL, NULL, 0, NULL}
};

// 模块定义
static struct PyModuleDef deepgenmodule = {
  PyModuleDef_HEAD_INIT,
  "deepgen",
  NULL,
  -1,
  DeepgenMethods
};

// 模块初始化
PyMODINIT_FUNC PyInit_deepgen(void) {
  return PyModule_Create(&deepgenmodule);
}