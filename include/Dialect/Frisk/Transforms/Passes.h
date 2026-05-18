#ifndef FRISK_TRANSFORMS_PASSES_H
#define FRISK_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::frisk {

#define GEN_PASS_DECL
#include "Dialect/Frisk/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDeepgenGraphSimplifyPass();
std::unique_ptr<Pass> createConvertKernelOpToFriskPass();
std::unique_ptr<Pass> createConvertMemOpPass();
std::unique_ptr<Pass> createConvertCalcOpPass();
std::unique_ptr<Pass> createDeepgenGraphMemspaceAnalyzePass();
std::unique_ptr<Pass> createConvertScfForOpPass();
std::unique_ptr<Pass> createOverlapPass();

#define GEN_PASS_REGISTRATION
#include "Dialect/Frisk/Transforms/Passes.h.inc"

} // namespace mlir::frisk

#endif // FRISK_TRANSFORMS_PASSES_H
