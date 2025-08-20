#ifndef DEEPGENGRAPHTRITON_DIALECT
#define DEEPGENGRAPHTRITON_DIALECT

include "mlir/IR/OpBase.td"


def DeepgengraphTriton_Dialect : Dialect {
  let name = "deepgengraph_triton";

  let cppNamespace = "::mlir::deepgengraph::triton";

  let summary = "DeepgengraphTriton Dialact in MLIR";

  let description = [{
    DeepgengraphTriton Dialect.
  }];

  let dependentDialects = [
    "::mlir::deepgengraph::DeepgengraphDialect",

    // "::mlir::triton::TritonDialect",
    // "::mlir::triton::gpu::TritonGPUDialect",
    // "::mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
    // "::mlir::triton::nvgpu::NVGPUDialect",

    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  // need explicitly set
  let useDefaultTypePrinterParser = 1;
}

#endif // DEEPGENGRAPH_DIALECT