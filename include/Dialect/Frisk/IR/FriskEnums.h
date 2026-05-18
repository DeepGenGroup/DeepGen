#ifndef FRISK_ENUMS_H
#define FRISK_ENUMS_H

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ENUM_CLASSES
#include "Dialect/Frisk/IR/FriskEnums.h.inc"
#undef GET_ENUM_CLASSES

#endif // FRISK_ENUMS_H
