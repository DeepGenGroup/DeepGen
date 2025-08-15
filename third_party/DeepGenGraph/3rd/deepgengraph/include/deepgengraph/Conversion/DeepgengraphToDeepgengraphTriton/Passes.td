#ifndef DEEPGENGRAPHTODEEPGENGRAPHTRITON_CONVERSION_PASSES
#define DEEPGENGRAPHTODEEPGENGRAPHTRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDeepgengraphToDeepgengraphTriton: Pass<"convert-deepgengraph-to-deepgengraphtriton", "::mlir::deepgengraph::KernelOp"> {
  let summary = "convert deepgengraph to deepgengraph triton";
  let description = [{}];

  let constructor = "mlir::deepgengraph::createConvertDeepgengraphToDeepgengraphTritonPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect"
  ];
}



#endif // DEEPGENGRAPHTODEEPGENGRAPHTRITON_CONVERSION_PASSES