#include "Conversion/Mapping.h"

namespace KernelCodeGen {


void blockForOpShiftDown(std::vector<mlir::affine::AffineForOp>& blockForOps) {
  // 将代表blockx的for循环下移到thread的parallel下
  for (auto blockForOp : blockForOps) {
    blockForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineParallelOp paraOp) {
      spliceHaveBlockOp(blockForOp, paraOp, /*insertPos*/0, 0, -2);
      paraOp->moveBefore(blockForOp);
      mlir::Operation* parentOp = blockForOp->getParentOp();
      auto idx = getOpIndex(parentOp, blockForOp);
      spliceHaveBlockOp(paraOp, parentOp, /*insertPos*/0, idx, idx+1);
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
  copyAttr(parallelOps[0], newParallelOp, AttrGPUIndex);
  auto newIvs = newParallelOp.getIVs();
  std::vector<mlir::Value> newIvsVec(newIvs.begin(), newIvs.end());
  // move ops in parallel
  for (auto it = parallelOps.rbegin(); it != parallelOps.rend(); ++it) {
    mlir::affine::AffineParallelOp paraOp = *it, tParaOp;
    auto oldIvs = paraOp.getIVs();
    std::vector<mlir::Value> oldIvsVec(oldIvs.begin(), oldIvs.end());
    // move
    spliceHaveBlockOp(newParallelOp, paraOp, /*insertPos*/0, 0, -2);
    // replace operands
    for (int i=0; i<oldIvsVec.size(); i++) {
      oldIvsVec[i].replaceAllUsesWith(newIvsVec[i]);
    }
    paraOp.erase();
  }
  // if fuse blockidx
  auto gpuIdx = getStrAttr(newParallelOp, AttrGPUIndex);
  if (gpuIdx == BLOCKIDX) {    // parallelOp have blockIdx attr
    // collect old applyOp and affineMap of old applyOp
    std::vector<mlir::affine::AffineApplyOp> applyOps;
    std::map<std::string, mlir::AffineMap> applyMap;
    for (auto &op : newParallelOp.getBody()->getOperations()) {
      if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
        applyOps.push_back(applyOp);
        auto applyDesc = getStrAttr(applyOp, APPLYDESC);
        if (!applyDesc.empty()) {
          applyMap.emplace(applyDesc, applyOp.getAffineMap());
        }
      }
    }
    // create new applyop
    for (auto& [applyDesc, amap] : applyMap) {
      builder.setInsertionPointToStart(newParallelOp.getBody());
      auto op = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), amap, mlir::ValueRange({newIvs[0]}));
      op->setAttr(APPLYDESC, builder.getStringAttr(applyDesc));
      // replace old applyop
      for (auto it = applyOps.rbegin(); it != applyOps.rend(); ++it) {
        mlir::affine::AffineApplyOp apOp = *it;
        auto applyDesc_ = getStrAttr(apOp, APPLYDESC);
        if (applyDesc_ == applyDesc) {
          apOp.getResult().replaceAllUsesWith(op.getResult());
          apOp.erase();
        }
      }
    }
  }
  return newParallelOp;
}

void eraseSingleIterForOps(mlir::func::FuncOp funcOp) {
  // 遍历funcop中所有的forop，找到为循环次数为1的循环，将其删除，map也修改
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    auto [lb, ub, step] = getLoopBoundAndStep(forOp);
    if ((ub - lb) / step == 1) {
      auto attrval = getStrAttr(forOp, FORDESC);
      if((attrval == "k") || (attrval == "ttilex") || (attrval == "ttiley")){
        return;
      }
      eraseSingleIterForOp(forOp);
    }
  });
}

}