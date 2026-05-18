#ifndef FRISK_UTILS_LAYOUT_UTILS_H
#define FRISK_UTILS_LAYOUT_UTILS_H

#include <optional>
#include <cstdint>

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"

namespace mlir::frisk {

/// Estimate how many distinct values an affine map may produce from the
/// dimension at `placeholderPos` when that dimension ranges over
/// `[0, placeholderExtent)`. The analysis mirrors tilelang's
/// `CondenseReplicateVar` by collecting the used iterator splits of the target
/// dimension (via its `mod`/`floordiv` usage) and returning the product of the
/// participating split extents. A return value of 1 means the dimension does
/// not affect the map. Returns `std::nullopt` when the map references the
/// placeholder through unsupported affine constructs or when overflow occurs.
std::optional<int64_t>
computeUsedExtentForDim(AffineMapAttr mapAttr, unsigned placeholderPos,
                        int64_t placeholderExtent);

} // namespace mlir::frisk

#endif // FRISK_UTILS_LAYOUT_UTILS_H
