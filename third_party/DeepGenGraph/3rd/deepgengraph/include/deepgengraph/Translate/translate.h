#ifndef DEEPGENGRAPH_TRANSLATE_H
#define DEEPGENGRAPH_TRANSLATE_H

#include <string>

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"

namespace mlir::deepgengraph {

::mlir::LogicalResult module_to_py_impl(::mlir::ModuleOp, ::mlir::raw_ostream &, bool benchmark = true);
::mlir::LogicalResult kernel_to_py_impl(::mlir::deepgengraph::KernelOp, ::mlir::raw_ostream &, bool import = true,
                                        bool benchmark = true);

} // namespace mlir::deepgengraph

#endif // DEEPGENGRAPH_TRANSLATE_H