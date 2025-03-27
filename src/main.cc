#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Common/ThreadPool.h"

using namespace KernelCodeGen;

int main(){
  int64_t bs = 1;
  int64_t hn = 16;
  int64_t sl = 4096;
  int64_t hd = 128;

  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  KernelCodeGenerator generator(Target::CUDA, "");

  std::vector<KernelData> kds;
  KernelData kd1, kd2, kd3;
  // matmul1
  kd1.name = "matmul1";
  kd1.type = "Matmul";
  kd1.argNames = {"A1", "B1", "C1"};
  kd1.shapes = {{bs, hn, hd, sl}, {bs, hn, hd, sl}, {bs, hn, sl, sl}};
  kd1.dtypes = {"float32", "float32", "float32"};
  kd1.isTrans = {true, false};
  kd1.outputArgNum = 1;
  kds.push_back(kd1);
  //Softmax1
  kd2.name = "softmax1";
  kd2.type = "Softmax";
  kd2.argNames = {"C1"};
  kd2.shapes = {{bs, hn, sl, sl}};
  kd2.dtypes = {"float32"};
  kd2.isTrans = {false};
  kd2.outputArgNum = 1;
  kds.push_back(kd2);
  // matmul2
  kd3.name = "matmul2";
  kd3.type = "Matmul";
  kd3.argNames = {"C1", "B2", "C2"};
  kd3.shapes = {{bs, hn, sl, sl}, {bs, hn, sl, hd}, {bs, hn, sl, hd}};
  kd3.dtypes = {"float32", "float32", "float32"};
  kd3.isTrans = {false, false};
  kd3.outputArgNum = 1;
  kds.push_back(kd3);

  // fuse kernel list
  std::vector<FuseKernelData> fkds;
  FuseKernelData fkd = {
    "attention1",
    "Attention",
    {"matmul1", "softmax1", "matmul2"},
    {{bs, hn, hd, sl}, {bs, hn, hd, sl}, {bs, hn, sl, hd}, {bs, hn, sl, hd}},
    {{bs, hn, sl, sl}},
    {"float32", "float32", "float32", "float32"},
    {"float32"},
    {{{"matmul1", 0}}, {{"matmul1", 1}}, {{"matmul2", 1}}, {{"matmul2", 2}}}, 
    {{{"matmul1", 2}, {"softmax1", 0}, {"matmul2", 0}}},
    1
  };
  fkds.push_back(fkd);

  auto noSupKernels = generator.createModel(module, kds);
  llvm::outs() << module << "\n";
  if (noSupKernels.size() != 0) {
    llvm::errs() << "UnSupport Kernel: ";
    for (auto nsk : noSupKernels) {
      llvm::errs() << nsk << ", ";
    }
    llvm::errs() << "\n";
  } else {
    auto result = generator.fusing(module, fkds);
    llvm::outs() << module << "\n";
    result = generator.mapping(module);
  }

  
  // matmul
  // std::vector<int64_t> dims1{bs, hn, sl, sl, hd};
  // std::vector<std::string> dtypes{"float32", "float32", "float32"};
  // generator.create<Operators::Matmul>(module, dims1, dtypes, "GEMM1", true);

  // softmax
  // std::vector<int64_t> dims2{bs, hn, sl, sl};
  // std::string dtype{"float32"};
  // generator.create<Operators::Softmax>(module, dims2, dtype, "Softmax1");

  llvm::outs() << module << "\n";  
  return 0;
}


