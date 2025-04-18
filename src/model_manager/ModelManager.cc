#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "ModelManager/ModelManager.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace KernelCodeGen {

bool ModelManager::process(){
  return false;
}

bool ModelManager::importModelFromIR(const std::string& filepath){
  MLIRContext ctx;
  // 首先，注册需要的 dialect
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect, stablehlo::StablehloDialect >();
  // 读入文件
  auto src = parseSourceFile<ModuleOp>(filepath, &ctx);
  // 输出dialect，也可以输出到 llvm::errs(), llvm::dbgs()
  src->print(llvm::outs());
  // 简单的输出，在 debug 的时候常用
  src->dump();
  return true;
}

};

