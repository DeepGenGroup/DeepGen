#ifndef DEEPGENGRAPH_DIALECT
#define DEEPGENGRAPH_DIALECT

include "mlir/IR/OpBase.td"


def Deepgengraph_Dialect : Dialect {
  let name = "deepgengraph";

  let cppNamespace = "::mlir::deepgengraph";

  let summary = "Deepgengraph Dialact in MLIR";

  let description = [{
    Deepgengraph Dialect.
  }];

  let dependentDialects = [
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

  let useDefaultAttributePrinterParser = 1;
}

#endif // DEEPGENGRAPH_DIALECT