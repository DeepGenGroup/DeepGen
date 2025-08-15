#include "ffi.h"

PYBIND11_MODULE(deepgengraph_ffi, m) {
  m.doc() = "TODO";
  mlir::deepgengraph::init_ffi_ir(m.def_submodule("ir"));
  mlir::deepgengraph::init_ffi_passes(m.def_submodule("passes"));
}