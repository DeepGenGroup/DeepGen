#ifndef DEEPGENGRAPH_TYPES_H_
#define DEEPGENGRAPH_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphTypes.h.inc"

#endif // DEEPGENGRAPH_TYPES_H_