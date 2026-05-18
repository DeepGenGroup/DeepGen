#ifndef FRISK_PASSES
#define FRISK_PASSES

include "mlir/Pass/PassBase.td"

def DeepgengraphSimplify: Pass<"deepgengraph-simplify", "::mlir::deepgengraph::KernelOp"> {
  let summary = "简化 Deepgengraph IR表示";
  let description = [{
  inline device_kenrel_op
  将 return arg 放进参数列表，并修改函数签名
  之后修改相关 empty_ptr_op为 ptrOp
  }];

  let constructor = "mlir::frisk::createDeepgenGraphSimplifyPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

def KernelOpToFrisk: Pass<"kernelop-convert-to-frisk", "::mlir::ModuleOp"> {
  let summary = "Deepgengraph kernelop convert to Frisk";
  let description = [{
  }];

  let constructor = "mlir::frisk::createConvertKernelOpToFriskPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::frisk::FriskDialect",
    "gpu::GPUDialect"
  ];
}

def MemOpToFrisk: Pass<"memop-convert-to-frisk", "frisk::KernelOp"> {
  let summary = "Deepgengraph memop convert to Frisk";
  let description = [{
  }];

  let constructor = "mlir::frisk::createConvertMemOpPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::frisk::FriskDialect",
    "gpu::GPUDialect"
  ];
}

def CalculateOpToFrisk: Pass<"calculateop-convert-to-frisk", "frisk::KernelOp"> {
  let summary = "Deepgengraph calculate op convert to frisk.blockOp";
  let description = [{
  }];

  let constructor = "mlir::frisk::createConvertCalcOpPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::frisk::FriskDialect",
    "gpu::GPUDialect"
  ];
}

def friskOverlap: Pass<"frisk-overlap", "::mlir::frisk::KernelOp"> {
  let summary = "frisk overlap ";
  let description = [{
    overlap
  }];

  let constructor = "mlir::frisk::createOverlapPass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect",
  ];
}

#endif // FRISK_PASSES
