#ifndef FRISK_ATTRIBUTES_H
#define FRISK_ATTRIBUTES_H

#include <string>

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"

#include "Dialect/Frisk/IR/FriskEnums.h"

namespace mlir::frisk {
class FriskDialect;
} // namespace mlir::frisk

#define GET_ATTRDEF_CLASSES
#include "Dialect/Frisk/IR/FriskAttributes.h.inc"

namespace mlir::frisk {
// Emit a compact description of the layout/fragment that mirrors tilelang's
// LayoutNode::DebugOutput()/FragmentNode::DebugOutput() helpers.
void printLayoutDebug(LayoutAttr layout, llvm::raw_ostream &os);

// Convenience wrapper that returns the debug string instead of writing it to
// an ostream.
std::string layoutDebugString(LayoutAttr layout);
} // namespace mlir::frisk

#endif // FRISK_ATTRIBUTES_H
