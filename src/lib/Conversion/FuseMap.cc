#include "Conversion/FuseMap.h"

namespace KernelCodeGen {

std::vector<mlir::func::FuncOp> getKernelFuncOps(mlir::ModuleOp mod, std::vector<std::string> kernelNames) {
  // 从 module 中获取需要进行融合的kernel 的funcop
  std::vector<mlir::func::FuncOp> fks;
  for (auto fk : kernelNames) {
    mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
      if (funcOp.getName().str() == fk) {
        fks.push_back(funcOp);
        return;
      }
    });
  }
  return fks;
}

std::tuple<mlir::func::FuncOp, std::vector<mlir::Value>, std::vector<mlir::Value>> 
  createFuseFuncAndMidMems(mlir::OpBuilder& builder, FuseKernelData fkd) {
  // 创建一个新的融合 funcop ，并且生成fuse kernel之间的中间变量
  // create new FuncOp
  std::vector<mlir::Type> newInputTypes;
  for (int i=0; i<fkd.newArgsShape.size(); i++) {
    auto mlirType = tools::getDType(builder, fkd.newArgsDtype[i]);
    auto shape = fkd.newArgsShape[i];
    auto memSpaceAttr = static_cast<int>(MemorySpace::global);
    auto type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape), mlirType, {}, memSpaceAttr);
    newInputTypes.push_back(type);
  }
  mlir::func::FuncOp newFuncOp = buildFunction(builder, fkd.fkName, fkd.type, newInputTypes, fkd.outputArgNum);
  auto newArgs_ = newFuncOp.getArguments();

  // create global var
  std::vector<mlir::Value> newVars, newArgs{newArgs_.begin(), newArgs_.end()};
  for (int i=0; i<fkd.newVarsShape.size(); i++) {
    auto mlirType = tools::getDType(builder, fkd.newVarsDtype[i]);
    auto shape = fkd.newVarsShape[i];
    auto memSpaceAttr = static_cast<int>(MemorySpace::global);
    auto type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape), mlirType, {}, memSpaceAttr);
    auto memOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), type);
    newVars.push_back(memOp);
  }
  return {newFuncOp, newArgs, newVars};
}

std::vector<std::vector<mlir::Value>> collectOldMems(std::vector<std::map<std::string, int64_t>> newMemsIndex, std::vector<mlir::func::FuncOp> fks) {
  // newMemsIndex 每一个item对应一个新的函数签名变量，新的变量需要替换掉旧的函数签名变量，采用新变量索引->func(旧索引)的方式获取旧的 memroy value
  std::vector<std::vector<mlir::Value>> memsVec;
  for (int i=0; i<newMemsIndex.size(); i++) {
    std::vector<mlir::Value> temp;
    auto itemMap = newMemsIndex[i];
    for (auto oldFuncOp : fks) {
      auto name = oldFuncOp.getName().str();
      if (itemMap.count(name)) {
        auto arg = oldFuncOp.getBody().getArgument(itemMap[name]);
        temp.push_back(arg);
      }
    }
    memsVec.push_back(temp);
  }
  return memsVec;
}

void fuseKernels(mlir::func::FuncOp funcOp, std::vector<mlir::func::FuncOp> fks, 
                 std::vector<mlir::Value> newArgs, std::vector<mlir::Value> newVars, 
                 std::vector<std::vector<mlir::Value>> oldArgs, std::vector<std::vector<mlir::Value>> oldVars) {
  // 将 old func 融合到 new func 中的操作
  // move funcOp body
  for (int i=fks.size()-1; i>=0; i--) {
    fks[i].getBody().front().back().erase();  // remove returnOp
    spliceHaveBlockOp(funcOp, fks[i], newVars.size());
  }
  // 替换val
  for (int i=0; i<newArgs.size(); i++) {
    auto newBuf = newArgs[i];
    for (auto oldBuf : oldArgs[i]) {
      replaceLoadAndStoreOpBuf(oldBuf, newBuf);
    }
  }
  for (int i=0; i<newVars.size(); i++) {
    auto newBuf = newVars[i];
    for (auto oldBuf : oldVars[i]) {
      replaceLoadAndStoreOpBuf(oldBuf, newBuf);
    }
  }
  // erase origin kernel
  for (auto it = fks.begin(); it != fks.end(); ) {
    auto fop = *it;
    fop.erase();
    it = fks.erase(it);
  }
}


}