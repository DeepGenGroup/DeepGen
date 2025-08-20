#ifndef DEEPGENGRAPH_PASSES
#define DEEPGENGRAPH_PASSES

include "mlir/Pass/PassBase.td"

def DeepgengraphSimplify: InterfacePass<"deepgengraph-simplify", "FunctionOpInterface"> {
  let summary = "Deepgengraph simplify";
  let description = [{
    remove redundent op
  }];

  let constructor = "mlir::deepgengraph::createSimplifyPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
  ];
}

def DeepgengraphReplaceExpAndLog: Pass<"deepgengraph-replace-exp-and-log", "::mlir::deepgengraph::KernelOp"> {
  let summary = "";
  let description = [{
    replace exp and log by base 2
  }];
  let constructor = "mlir::deepgengraph::createReplaceExpAndLogPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}


// FIXME: this pass should be applied on module_op or function op interface?
def DeepgengraphLowerComplexReduce: InterfacePass<"deepgengraph-lower-complex-reduce", "FunctionOpInterface"> {
  let summary = "";
  let description = [{}];
  let constructor = "mlir::deepgengraph::createLowerComplexReducePass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}


def DeepgengraphEraseTypeInKernel: Pass<"deepgengraph-erase-type-in-kernel", "::mlir::deepgengraph::KernelOp"> {
  let summary = "";
  let description = [{}];  
  let constructor = "mlir::deepgengraph::createEraseTypeInKernelPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphBroadcastTransform: Pass<"deepgengraph-broadcast-transform", "mlir::ModuleOp"> {
  let summary = "";  
  let description = [{}];
  let constructor = "mlir::deepgengraph::createBroadcastTransformPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphEquivalentTransform: Pass<"deepgengraph-equivalent-transform", "::mlir::deepgengraph::KernelOp"> {
  let summary = "";  
  let description = [{}];
  let constructor = "mlir::deepgengraph::createEquivalentTransformPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphMulScalarHoisting: Pass<"deepgengraph-mul-scalar-hoisting", "mlir::ModuleOp"> {
  let summary = "deepgengraph mul scalar hoisting";
  let description = [{
  }];

  let constructor = "mlir::deepgengraph::createMulScalarHoistingPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
  ];
}

def DeepgengraphPermuteHoisting: Pass<"deepgengraph-permute-hoisting", "mlir::ModuleOp"> {
  let summary = "deepgengraph permute hoisting";
  let description = [{
  }];

  let constructor = "mlir::deepgengraph::createPermuteHoistingPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];

}

def DeepgengraphAnnotateParallelism: Pass<"deepgengraph-annotate-parallelism", "::mlir::deepgengraph::KernelOp"> {
  let summary = "deepgengraph annotate parallelism";
  let description = [{}];

  let constructor = "mlir::deepgengraph::createAnnotateParallelismPass()";

  let options = [
    Option<"verbose", "verbose", "bool", /*default=*/"false", "verbose output">,
  ];

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

// FIXME: or function op interface?
def DeepgengraphToMask: Pass<"deepgengraph-to-mask", "::mlir::deepgengraph::KernelOp"> {
  let summary = "deepgengraph other op to mask op";
  let description = [{}];

  let constructor = "mlir::deepgengraph::createToMaskPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphParallelize: Pass<"deepgengraph-parallelize", "::mlir::deepgengraph::KernelOp"> {
  let summary = "deepgengraph parallelize";
  let description = [{}];

  let constructor = "mlir::deepgengraph::createParallelizePass()";

  let options = [
    Option<"partition", "partition", "bool", /*default=*/"true", "whether to partition">,
  ];

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphTiling: Pass<"deepgengraph-tiling", "::mlir::deepgengraph::KernelOp"> {
  let summary = "deepgengraph tiling";
  let description = [{}];
  let constructor = "mlir::deepgengraph::createTilingPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphDynamicFor: Pass<"deepgengraph-dynamic-for", "::mlir::deepgengraph::KernelOp"> {
  let summary = "deepgengraph dynamic for";
  let description = [{}];
  let constructor = "mlir::deepgengraph::createDynamicForPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphRecoverTypeInKernel: Pass<"deepgengraph-recover-type-in-kernel", "::mlir::deepgengraph::KernelOp"> {
  let summary = "";
  let description = [{}];
  let constructor = "mlir::deepgengraph::createRecoverTypeInKernelPass()";
  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}

def DeepgengraphUserReplicate: Pass<"deepgengraph-user-replicate", "mlir::ModuleOp"> {
  let summary = "deepgengraph replicate for only one use in device_kernel";
  let description = [{}];

  let constructor = "mlir::deepgengraph::createUserReplicatePass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect"
  ];
}


#endif // DEEPGENGRAPH_PASSES