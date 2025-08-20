#ifndef DEEPGENGRAPH_CONVERSION_DEEPGENGRAPHTODEEPGENGRAPHTRITON_DEEPGENGRAPHTODEEPGENGRAPHTRITON_PASS_H
#define DEEPGENGRAPH_CONVERSION_DEEPGENGRAPHTODEEPGENGRAPHTRITON_DEEPGENGRAPHTODEEPGENGRAPHTRITON_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::deepgengraph {

std::unique_ptr<mlir::Pass> createConvertDeepgengraphToDeepgengraphTritonPass();

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Conversion/DeepgengraphToDeepgengraphTriton/Passes.h.inc"

} // namespace mlir::deepgengraph

#endif