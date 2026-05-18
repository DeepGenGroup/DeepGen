#ifndef FRISK_TYPES_H_
#define FRISK_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Frisk/IR/FriskTypes.h.inc"

#endif // FRISK_TYPES_H_
