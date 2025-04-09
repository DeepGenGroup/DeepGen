#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "KernelCodeGen.h"
#include "Common/Utils.h"
#include "Common/ThreadPool.h"

#include <Python.h>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

using namespace KernelCodeGen;

PYBIND11_MODULE(deepGen, m) {

  py::enum_<Target>(m, "Target")
    .value("CUDA", Target::CUDA)
    .value("ROCM", Target::ROCM)
    .export_values();

  py::class_<KernelCodeGenerator>(m, "KernelCodeGenerator")
  .def(py::init<Target, const std::string&>())
  .def("createModule", &KernelCodeGenerator::createModule)
  .def("createKernels", &KernelCodeGenerator::createKernels)
  .def("fusing", &KernelCodeGenerator::fusing)
  .def("splitModule", &KernelCodeGenerator::splitModule)
  .def("mapping", &KernelCodeGenerator::mapping)
  .def("optimize", &KernelCodeGenerator::optimize)
  .def("lowering", &KernelCodeGenerator::lowering)
  .def("translate", &KernelCodeGenerator::translate)
}