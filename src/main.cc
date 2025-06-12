#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Python.h"

using namespace KernelCodeGen;
using TuneConfig = std::map<std::string, std::map<std::string, int64_t>>;
using TileConfig = std::map<std::string, std::map<std::string, int64_t>>;

KernelCodeGen::KernelCodeGenerator generator;

std::string compile_kernel(TuneConfig tuneCfg, TileConfig tileCfg, std::vector<KernelData> kds, std::vector<FuseKernelData> fkds={}) {
  // compile func
  mlir::ModuleOp module = generator.createModule();
  auto noSupKernels = generator.createKernels(module, kds);  // create kernels
  auto result = generator.fusing(module, fkds);  // fusing
  result = generator.mapping(module, tileCfg);  // mpping
  generator.optimize(module, tuneCfg);  // optimize
  // generator.lowering(module);  // lowering
  generator.transform(module);
  generator.lowering_(module);  // lowering
  auto path = generator.translate(module);  // translate
  // std::cout << "[lib] ===========4" << std::endl;
  return path;
}

std::string matmul(std::vector<int64_t> shape, const TuneConfig& config) {
  // matmul compile func
  auto mm = config.at("matmul");
  TileConfig tileConfig  = {
    {"matmul", {{"BLOCK_SIZE_Y", mm.at(KEY_BLOCK_SIZE_M)}, {"THREAD_SIZE_Y", mm.at(KEY_THREAD_SIZE_M)}, 
                {"BLOCK_SIZE_X", mm.at(KEY_BLOCK_SIZE_N)}, {"THREAD_SIZE_X", mm.at(KEY_THREAD_SIZE_N)}}}
  };
  // create new shapes
  int len = shape.size(), bl = shape.size()-3;
  int64_t m = shape[len-3], n = shape[len-2], k = shape[len-1];  // m, n, k
  std::vector<int64_t> b(shape.begin(), shape.begin()+bl);  // batch
  std::vector<int64_t> sha{k, m}, shb{k, n}, shc{m, n};
  for (int i=b.size()-1; i>=0; i--) {  // add batch
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
  }
  // create kernel info
  KernelData kd = {
    "matmul", "Matmul", {sha, shb, shc}, {"float32", "float32", "float32"}, {true, false}, 1
  };
  std::vector<KernelData> kds{kd};
  // compile kernel
  return compile_kernel(config, tileConfig, kds);
}

std::string attention(std::vector<int64_t> shape, const TuneConfig& config) {
  // attn compile func
  // shape: {batch, head_num, seq_len, head_dim}
  auto attn = config.at("attention");
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}}, 
    {"softmax1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")}, 
                {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
  };
  // create new shapes
  int len = shape.size(), bl = shape.size()-2;
  int64_t sl = shape[len-2], hd = shape[len-1];
  std::vector<int64_t> b(shape.begin(), shape.begin()+bl);
  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh1{sl, sl}, sh2{sl, sl};
  std::vector<int64_t> sha1{sl, sl}, shb1{sl, hd}, shc1{sl, hd};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh1.insert(sh1.begin(), b[i]); sh2.insert(sh2.begin(), b[i]); 
    sha1.insert(sha1.begin(), b[i]); shb1.insert(shb1.begin(), b[i]); shc1.insert(shc1.begin(), b[i]);
  }
  // kernel info
  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {"float32", "float32", "float32"}, {true, false}, 1
  };
  KernelData kd2 = {
    "softmax1", "Softmax", {sh1, sh2}, {"float32", "float32"}, {false}, 1
  };
  KernelData kd3 = {
    "matmul2", "Matmul", {sha1, shb1, shc1}, {"float32", "float32", "float32"}, {false, false}, 1
  };
  // kernel fusing
  FuseKernelData fkd = {
    "attention", "FlashAttn", {"matmul1", "softmax1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd3.shapes[1], kd3.shapes[2]}, {kd1.shapes[2]},
    {"float32", "float32", "float32", "float32"}, {"float32"},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {{"matmul2", {2}}}},
    {{{"matmul1", {2}}, {"softmax1", {0, 1}} , {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd3.isTrans[1]}, {"y"}, 1
  };
  std::vector<KernelData> kds{kd1, kd2, kd3};
  std::vector<FuseKernelData> fkds{fkd};
  // compile kernel
  return compile_kernel(config, tileConfig, kds, fkds);
}

// bind python module
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
  std::string result = attention(shape, config);
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
  std::string result = matmul(shape, config);
  return PyUnicode_FromString(result.c_str());
}

static PyObject* set_platform(PyObject* self, PyObject* args) {
  char* target = NULL;
  char* platInfo = NULL;
  if(PyArg_ParseTuple(args, "ss", &target, &platInfo)){
    if(std::string(target) == "cuda"){
      generator.setPaltform(Target::CUDA, std::string(platInfo));
    }else if (std::string(target) == "rocm") {
      generator.setPaltform(Target::ROCm, std::string(platInfo));
    } else{
      std::cout << "DeepGen Error : Invalid Platform id " << target << std::endl;
      std::abort();
    }
  }
  return Py_None;
}

static PyMethodDef DeepgenMethods[] = {
    {"compile_attn", py_compile_attn, METH_VARARGS, "Compile attention with given shape and config"},
    {"compile_mm", py_compile_mm, METH_VARARGS, "Compile matmul with given shape and config"},
    {"set_platform", set_platform, METH_VARARGS, "Compile attention with given shape and config"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef deepgenmodule = {
  PyModuleDef_HEAD_INIT,
  "deepgen",
  NULL,
  -1,
  DeepgenMethods
};

PyMODINIT_FUNC PyInit_deepgen(void) {
  return PyModule_Create(&deepgenmodule);
}