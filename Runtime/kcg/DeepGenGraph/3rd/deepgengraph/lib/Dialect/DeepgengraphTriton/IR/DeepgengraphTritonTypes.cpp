#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"

using namespace mlir;
using namespace mlir::deepgengraph;
using namespace mlir::deepgengraph::triton;

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.cpp.inc"

void DeepgengraphTritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.cpp.inc"
      >();
}
