#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Common/ThreadPool.h"

using namespace KernelCodeGen;

int main(int argc, char* argv[]) {
  using namespace mlir;
  KernelCodeGenerator generator(Target::CUDA, "");
  mlir::MLIRContext testContext;
  testContext.loadDialect<
    mlir::affine::AffineDialect,
    func::FuncDialect,
    memref::MemRefDialect,
    scf::SCFDialect,
    mlir::vector::VectorDialect,
    mlir::cf::ControlFlowDialect,
    gpu::GPUDialect, 
    NVVM::NVVMDialect, 
    arith::ArithDialect, cf::ControlFlowDialect, LLVM::LLVMDialect, ROCDL::ROCDLDialect,
    mlir::math::MathDialect
  >();
  auto temp = mlir::parseSourceFile<ModuleOp>(argv[1], &testContext);
  auto module = *temp;
  // lowering
  generator.lowering(module);
  // translate
  auto path = generator.translate(module);
  llvm::outs() << "bin path: " << path << "\n";
  return 0;
}