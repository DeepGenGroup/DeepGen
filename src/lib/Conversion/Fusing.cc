#include "Conversion/Fusing.h"

namespace KernelCodeGen {

std::vector<mlir::func::FuncOp> getKernelFuncOps(mlir::ModuleOp mod, 
                                                 const std::vector<std::string>& kernelNames) {
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

std::vector<std::vector<mlir::affine::AffineForOp>> getBatchFors(const std::vector<mlir::func::FuncOp>& fks) {
  // 获取fuse kernel中的所有batch for
  std::vector<std::vector<mlir::affine::AffineForOp>> funcBatchs;
  for (auto fk : fks) {
    std::vector<mlir::affine::AffineForOp> batchs;
    fk.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      if (auto forDesc = forOp->getAttr(FORDESC)) {
        auto forAttr = mlir::dyn_cast<mlir::StringAttr>(forDesc);
        if (forAttr.getValue().str() == "batch") {
          batchs.push_back(forOp);
        }
      }
    });
    funcBatchs.push_back(batchs);
  }
  return funcBatchs;
}

std::tuple<mlir::func::FuncOp, std::vector<mlir::Value>, std::vector<mlir::Value>> createFuseFuncAndMidMems(mlir::OpBuilder& builder, 
                                                                                                            FuseKernelData fkd) {
  // 创建一个新的融合 funcop ，并且生成fuse kernel之间的中间变量
  // create new FuncOp
  std::vector<mlir::Type> newInputTypes;
  for (int i=0; i<fkd.funcArgShapes.size(); i++) {
    auto mlirType = tools::getDType(builder, fkd.funcArgDtypes[i]);
    auto shape = fkd.funcArgShapes[i];
    auto memSpaceAttr = static_cast<int>(MemorySpace::global);
    auto type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape), mlirType, {}, memSpaceAttr);
    newInputTypes.push_back(type);
  }
  mlir::func::FuncOp newFuncOp = buildFunction(builder, fkd.name, fkd.type, newInputTypes, fkd.paraDims, fkd.outputArgNum);
  auto funcArgs_ = newFuncOp.getArguments();
  // create global var
  std::vector<mlir::Value> midVars, funcArgs{funcArgs_.begin(), funcArgs_.end()};
  for (int i=0; i<fkd.midVarShapes.size(); i++) {
    auto mlirType = tools::getDType(builder, fkd.midVarDtypes[i]);
    auto shape = fkd.midVarShapes[i];
    auto memSpaceAttr = static_cast<int>(MemorySpace::global);
    auto type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape), mlirType, {}, memSpaceAttr);
    auto memOp = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), type);
    midVars.push_back(memOp);
  }
  return {newFuncOp, funcArgs, midVars};
}

std::vector<std::vector<mlir::Value>> collectOldMems(const std::vector<std::map<std::string, int64_t>>& newMemsIndex, 
                                                     const std::vector<mlir::func::FuncOp>& fks) {
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

void moveOperation(mlir::func::FuncOp funcOp, 
                   std::vector<mlir::func::FuncOp> fks, 
                   const std::vector<mlir::Value>& funcArgs, 
                   const std::vector<mlir::Value>& midVars, 
                   const std::vector<std::vector<mlir::Value>>& argToArgs, 
                   const std::vector<std::vector<mlir::Value>>& midToArgs) {
  // 将 old func 融合到 new func 中的操作
  // move funcOp body
  for (int i=fks.size()-1; i>=0; i--) {
    // no move returnOp of oldfunc
    spliceHaveBlockOp(funcOp, fks[i], /*insertPoint*/midVars.size(), 0, -2);
  }
  // 函数参数替换
  for (int i=0; i<funcArgs.size(); i++) {
    auto newBuf = funcArgs[i];
    for (auto oldBuf : argToArgs[i]) {
      auto users = getValueUsers(oldBuf);
      for (auto user : users) {
        replaceOpOperands(user, oldBuf, newBuf);
      }
    }
  }
  // 中间变量替换
  for (int i=0; i<midVars.size(); i++) {
    auto newBuf = midVars[i];
    for (auto oldBuf : midToArgs[i]) {
      auto users = getValueUsers(oldBuf);
      for (auto user : users) {
        replaceOpOperands(user, oldBuf, newBuf);
      }
    }
  }
  // erase origin kernel
  for (auto it=fks.rbegin(); it!=fks.rend(); ++it) {
    mlir::func::FuncOp oldFunc = *it;
    oldFunc.erase();
  }
}


std::vector<mlir::affine::AffineForOp> fuseBatchForOps(mlir::OpBuilder builder, 
                                                       std::vector<std::vector<mlir::affine::AffineForOp>> batchs) {
  // fuse all batch forOp
  llvm::SmallVector<int64_t> lbs, ubs, steps;
  for (auto batch : batchs[0]) {
    auto [lb, ub, step] = getLoopBoundAndStep(batch);
    lbs.push_back(lb);
    ubs.push_back(ub);
    steps.push_back(step);
  }
  // create new batch loops
  auto [newBatchs, newIvs] = createNestedLoops(builder, lbs, ubs, steps);
  for (int i=0; i<newBatchs.size(); i++) {
    copyAttr<mlir::StringAttr>(batchs[0][i], newBatchs[i], FORDESC);
    copyAttr<mlir::IntegerAttr>(batchs[0][i], newBatchs[i], BATCHNUM);
  }
  // move ops
  for (int i=batchs.size()-1; i>=0; i--) {
    std::vector<mlir::Value> oldIvs;
    for (auto b : batchs[i]) {
      oldIvs.push_back(b.getInductionVar());
    }
    // move op
    spliceHaveBlockOp(newBatchs.back(), batchs[i].back(), 0, 0, -2);
    // replace operands
    replaceOpsOperands(newBatchs.back(), oldIvs, newIvs);
    batchs[i][0].erase();
  }
  return newBatchs;
}

}