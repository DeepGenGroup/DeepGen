#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphTypes.h"

using namespace mlir;
using namespace mlir::deepgengraph;

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphTypes.cpp.inc"
