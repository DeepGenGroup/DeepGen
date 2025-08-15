#ifndef DEEPGENGRAPHTRITON_TRANSFORMS_PASSES_H_
#define DEEPGENGRAPHTRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::deepgengraph::triton {

std::unique_ptr<Pass> createSqueezeBlockPass();

// std::unique_ptr<Pass> createBlockingPass();
std::unique_ptr<Pass> createUserReplicatePass();

#define GEN_PASS_DECL
#include "deepgengraph/Dialect/DeepgengraphTriton/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Dialect/DeepgengraphTriton/Transforms/Passes.h.inc"

} // namespace mlir::deepgengraph::triton

#endif