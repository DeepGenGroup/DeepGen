#include "Conversion/Mapping.h"

namespace KernelCodeGen {

std::vector<mlir::func::FuncOp> getKernels(mlir::ModuleOp mod) {
  // get all kernel in module
  std::vector<mlir::func::FuncOp> funOps;
  mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
    funOps.push_back(funcOp);
  });
  return funOps;
}

std::vector<std::string> getArrayStringAttr(mlir::Operation* op, std::string attrName) {
  // get array string attrs(parallelDims/IterVar)
  std::vector<std::string> arrayStrAttr;
  auto descArr = op->getAttr(attrName);
  auto descArrAttr = mlir::dyn_cast<mlir::ArrayAttr>(descArr);
  for (auto desc : descArrAttr) {
    auto descAttr = mlir::dyn_cast<mlir::StringAttr>(desc);
    auto descStr = descAttr.getValue().str();
    arrayStrAttr.push_back(descStr);
  }
  return arrayStrAttr;
}

void normalizeParaForOp(std::vector<mlir::affine::AffineForOp> &yloops, std::vector<std::map<std::string, int64_t>> &paraCfg) {
  // 规范化所有的并行循环，保证fory下就是forx
  std::vector<int> idxs;
  std::vector<mlir::affine::AffineForOp> newLoops;
  std::vector<std::map<std::string, int64_t>> newCfg;
  for (int i=0; i<yloops.size(); i++) {
    int index = 0;
    yloops[i].walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      if (auto desc = forOp->getAttr(FORDESC)){
        auto descAttr = mlir::dyn_cast<mlir::StringAttr>(desc);
        if (descAttr.getValue().str() == "x") {
          if (index > 0) {
            auto newY = decoupleNestedLoop({yloops[i]}, forOp);
            newCfg.push_back(paraCfg[i]);
            newLoops.push_back(newY[0]);
            idxs.push_back(i);
          }
          index++;
        }
      }
    });
  }
  // update yloops and paraCfg
  for (int i=0; i<idxs.size(); i++) {
    yloops.insert(yloops.begin() + idxs[idxs.size()-1-i], newLoops[newLoops.size()-1-i]);
    paraCfg.insert(paraCfg.begin() + idxs[idxs.size()-1-i], newCfg[newLoops.size()-1-i]);
  }
}

void blockForOpShiftDown(std::vector<mlir::affine::AffineForOp>& blockForOps) {
  // 将代表blockx的for循环下移到thread的parallel下
  for (auto blockForOp : blockForOps) {
    blockForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineParallelOp paraOp) {
      int index = 0;
      for (auto &op : paraOp.getBody()->getOperations()) {
        if (mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) index++;
      }
      spliceHaveBlockOp(blockForOp, paraOp, /*insertPos*/0, index, -2);
      paraOp->moveBefore(blockForOp);
      mlir::Operation* parentOp = blockForOp->getParentOp();
      auto idx = getOpIndex(parentOp, blockForOp);
      spliceHaveBlockOp(paraOp, parentOp, /*insertPos*/index, idx, idx+1);
    });
  }
}

mlir::affine::AffineParallelOp fuseParallelOp(mlir::OpBuilder builder, 
                                              std::vector<mlir::affine::AffineParallelOp> parallelOps) {
  // fuse parallel
  // collect old parallel data
  llvm::SmallVector<int64_t> ranges;
  auto oldRanges = parallelOps[0].getConstantRanges();
  for (int64_t oldRange : *oldRanges) {
    ranges.push_back(oldRange);
  }
  // create new parallel op
  auto newParallelOp = builder.create<mlir::affine::AffineParallelOp>(builder.getUnknownLoc(), 
                                                                      mlir::TypeRange(), 
                                                                      llvm::ArrayRef<mlir::arith::AtomicRMWKind>(), 
                                                                      llvm::ArrayRef<int64_t>(ranges));
  auto newIvs = newParallelOp.getIVs();
  std::vector<mlir::Value> newIvsVec(newIvs.begin(), newIvs.end());
  // move ops in parallel
  for (auto it = parallelOps.rbegin(); it != parallelOps.rend(); ++it) {
    mlir::affine::AffineParallelOp paraOp = *it;
    auto oldIvs = paraOp.getIVs();
    std::vector<mlir::Value> oldIvsVec(oldIvs.begin(), oldIvs.end());
    //replace operands
    replaceOpsOperands(paraOp, oldIvsVec, newIvsVec);
    // move
    spliceHaveBlockOp(newParallelOp, paraOp, /*insertPos*/0, 0, -2);
    // move apply to begin of parallelop 
    auto& bodyOps = newParallelOp.getBody()->getOperations();
    mlir::Operation* firstOp = &bodyOps.front();
    for (auto &op : bodyOps) {
      if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
        if (firstOp != applyOp)
          applyOp->moveBefore(firstOp);
      }
    }
    paraOp.erase();
  }
  return newParallelOp;
}

}