#ifndef DEEPGENGRAPHTRITON_PASSES
#define DEEPGENGRAPHTRITON_PASSES

include "mlir/Pass/PassBase.td"

def DeepgengraphTritonSqueezeBlock: Pass<"deepgengraphtriton-squeeze-block", "::mlir::deepgengraph::KernelOp"> {
  let summary = ""; 
  let description = [{}];
  let constructor = "mlir::deepgengraph::triton::createSqueezeBlockPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect",
    "mlir::scf::SCFDialect"
  ];
}

def DeepgengraphTritonUserReplicate: Pass<"deepgengraphtriton-user-replicate", "mlir::ModuleOp"> {
  let summary = "deepgengraph triton replicate for only one use in device_kernel";
  let description = [{}];

  let constructor = "mlir::deepgengraph::triton::createUserReplicatePass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::deepgengraph::triton::DeepgengraphTritonDialect"
  ];
}


#endif // DEEPGENGRAPHTRITON_PASSES
