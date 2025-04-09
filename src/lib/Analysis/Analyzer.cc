#include "Analysis/Analyzer.h"


namespace KernelCodeGen {
namespace Analyzer {


int64_t getThreadPerBlock(mlir::affine::AffineParallelOp parallelLevel) {
  // 获取block的线程数量
  int64_t threadSum = 1;
  auto oldRanges = parallelLevel.getConstantRanges();
  for (auto i : *oldRanges) {
    threadSum *= i;
  }
  return threadSum;
}

std::vector<int64_t> getParallelNumber(mlir::affine::AffineParallelOp parallelLevel, int64_t& totalNumber) {
  auto dim = parallelLevel.getNumDims();
  totalNumber = 1;
  std::vector<int64_t> result;
  for (int i = 0; i < dim; i++) {
    auto map = parallelLevel.getUpperBoundMap(i);
    auto exprs = map.getResults();
    assert(exprs.size() == 1);
    auto constExpr = exprs[0].dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr);
    totalNumber *= constExpr.getValue();
    result.push_back(constExpr.getValue());
  }
  return result;
}

std::vector<mlir::affine::AffineForOp> collectFuncLoops(mlir::func::FuncOp funcOp) {
  std::vector<mlir::affine::AffineForOp> res;
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    res.push_back(forOp);
  });
  return res;
}

std::map<std::string, std::string> collectNameTypeMap(mlir::ModuleOp& module) {
  std::map<std::string, std::string> ntMap;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
    auto type = funcOp->getAttr(std::string("func.op.type"));
    auto typeAttr = type.dyn_cast<mlir::StringAttr>();
    ntMap[funcOp.getName().str()] = typeAttr.getValue().str();
  });
  return ntMap;
}

std::set<std::string> collectFuncTypes(mlir::ModuleOp& module) {
  std::set<std::string> result;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
    auto type = funcOp->getAttr(std::string("func.op.type"));
    auto typeAttr = type.dyn_cast<mlir::StringAttr>();
    result.insert(typeAttr.getValue().str()); 
  });
  return result;
}

int getThreadsPerCTA(mlir::ModuleOp module) {
  int threadNum = 1;
  for (auto &op : module.getBody()->getOperations()) {
    if (auto funcOp = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      if (!funcOp->hasAttr("func.op.type")) continue;
      auto blockDims = funcOp->getAttrOfType<mlir::DenseI32ArrayAttr>("func.block.dim");
      for (size_t i=0; i<blockDims.size(); i++) {
        threadNum *= blockDims[i];
      }
      return threadNum;
    }
  }
  return threadNum;
}


std::vector<mlir::Value> getParallelIdx(mlir::affine::AffineParallelOp parallelLevel) {
  // auto dim = parallelLevel.getNumDims();
  std::vector<mlir::Value> idxes;
  auto ivs = parallelLevel.getIVs();
  for (int i=ivs.size()-1; i>=0; i--) {
    idxes.push_back(ivs[i]);
  }
  return idxes;
}


mlir::affine::AffineForOp findRootLoop(mlir::Operation* op) {
  while (true) {
    auto parentOp = op->getParentOp();
    if (!parentOp) assert(false);
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(parentOp)) {
      return mlir::dyn_cast<mlir::affine::AffineForOp>(op);
    } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)){
      return mlir::dyn_cast<mlir::affine::AffineForOp>(op);
    } else if (auto parallel = mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
      return mlir::dyn_cast<mlir::affine::AffineForOp>(op);
    }
    op = mlir::dyn_cast<mlir::affine::AffineForOp>(parentOp);
    if (!op) {
      op = mlir::dyn_cast<mlir::affine::AffineIfOp>(parentOp);
    }
    if (!op) {
      assert(false);
    }
  }
}

mlir::Block* getClostScopeOp(mlir::Operation* op) {
  while (true) {
    auto parentOp = op->getParentOp();
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(parentOp)) {
      return module.getBody();
    } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)){
      return &(func.getBlocks().front());
    } else if (auto parallelOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
      return parallelOp.getBody();
    }
    op = parentOp;
  }
}

}

}