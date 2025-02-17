#include "Conversion/General/GeneralFuncs.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <map>
#include <cmath>

namespace KernelCodeGen {

mlir::OpBuilder getBuilder(mlir::Operation* op, Position pos) {
  // 按照位置和op创建builder
  switch (pos){
  case Position::after:
  {
    mlir::OpBuilder builder(op->getContext());
    builder.setInsertionPointAfter(op);
    return builder;
  }
  case Position::before:
  {
    mlir::OpBuilder builder(op);
    return builder;
  }
  case Position::begin:
  {
    return mlir::OpBuilder::atBlockBegin(&op->getRegion(0).front());
  }
  case Position::end:
  {
    return mlir::OpBuilder::atBlockEnd(&op->getRegion(0).front());
  }
  default:
    assert(false);
  }
}

std::tuple<int64_t, int64_t, int64_t> getLoopBoundAndStep(mlir::affine::AffineForOp loop) {
  // 获取forOp的上界、下界以及步长
  int64_t ub = loop.getConstantUpperBound();
  int64_t lb = loop.getConstantLowerBound();
  int64_t step = loop.getStep().getLimitedValue();
  return std::make_tuple(lb, ub, step);
}

void replaceAndErase(mlir::Operation* newOp, mlir::Operation* oldOp) {
  // 替换后面op使用到oldOp的值，且删除oldOp
  auto oldResult = oldOp->getResult(0);
  oldResult.replaceAllUsesWith(newOp->getResult(0));
  oldOp->erase();
}

void spliceHaveBlockOp(mlir::Operation* newOp, mlir::Operation* oldOp, int index) {
  // 将 oldOp 中的 ops 转到 newOp 中，index决定转移newOp的位置
    auto& newOpOperations = newOp->getRegion(0).front().getOperations();
    auto& oldOpOperations = oldOp->getRegion(0).front().getOperations();
    llvm::iplist<mlir::Operation>::iterator it = newOpOperations.begin();
    std::advance(it, index);
    newOpOperations.splice(it, oldOpOperations);
}

int getOpIndex(mlir::Operation* haveBlockOp, mlir::Operation* targetOp) {
  // 找到op在block中的index
  auto& ops = haveBlockOp->getRegion(0).front().getOperations();
  int index = -1;
  for (auto& op : ops) {
    index++;
    if (&op == targetOp) return index;
  }
  return -1;
}

std::set<mlir::Operation*> getValueUsers(mlir::Value var) {
  // 获取value的使用者
  std::set<mlir::Operation*> users;
  for (auto user: var.getUsers()) {
    users.insert(user);
  }
  return users;
}

mlir::AffineExpr getOrderExpr(mlir::OpBuilder builder, int dimCount) {
  // 获取一个有序的连续累加的affine表达式
  mlir::AffineExpr sumExpr = builder.getAffineConstantExpr(0);
  for (int i=0; i<dimCount; i++) {
    sumExpr = sumExpr + builder.getAffineDimExpr(i);
  }
  return sumExpr;
}

mlir::AffineExpr shiftAffineExprDim(mlir::MLIRContext* context, mlir::AffineExpr expr, int shift) {
  // d0 + d1 + d2  =>  shift==1  =>  d1 + d2 + d3
  if (auto dimExpr_ = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return mlir::getAffineDimExpr(dimExpr_.getPosition() + shift, context);
  } else if (auto binaryExpr_ = expr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = shiftAffineExprDim(context, binaryExpr_.getLHS(), shift);
    auto RHS = shiftAffineExprDim(context, binaryExpr_.getRHS(), shift);
    return mlir::getAffineBinaryOpExpr(binaryExpr_.getKind(), LHS, RHS);
  } else {
    // allowed dim, constant, binaryOp
    auto constExpr_ = expr.dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr_);
    return constExpr_;
  }
}

mlir::AffineMap addDimToMap(mlir::OpBuilder builder, mlir::AffineMap oldMap) {
  // d0 + d1 -> d0 + d1 + d2
  auto oldExprs = oldMap.getResults();
  mlir::SmallVector<mlir::AffineExpr> newExprs;
  for (int i=0; i<oldExprs.size(); i++) {
    if (i != oldExprs.size() - 1) {
      newExprs.push_back(oldExprs[i]);
    } else {
      auto dim = builder.getAffineDimExpr(oldMap.getNumDims());
      newExprs.push_back(oldExprs[i] + dim);
    }
  }
  return mlir::AffineMap::get(oldMap.getNumDims() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
}

mlir::AffineExpr getModifiedExpr(mlir::MLIRContext* context, mlir::AffineExpr inExpr, mlir::AffineExpr replaceExpr, int targetDim, int replaceNumberDims) {
  // d0 + d1 + d2  =>  target==1 & replace==[d1 + d2 + d3] =>  d0 + [d1 + d2 + d3] + d4
  if (auto dimExpr_ = inExpr.dyn_cast<mlir::AffineDimExpr>()) {
    if (dimExpr_.getPosition() == targetDim) {
      return replaceExpr;
    } else if (dimExpr_.getPosition() > targetDim) {
      return mlir::getAffineDimExpr(dimExpr_.getPosition() + replaceNumberDims - 1, context);
    } else {
      return dimExpr_;
    }
  } else if (auto binaryExpr_ = inExpr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = getModifiedExpr(context, binaryExpr_.getLHS(), replaceExpr, targetDim, replaceNumberDims);
    auto RHS = getModifiedExpr(context, binaryExpr_.getRHS(), replaceExpr, targetDim, replaceNumberDims);
    return mlir::getAffineBinaryOpExpr(binaryExpr_.getKind(), LHS, RHS);
  } else {
    // allowed dim, constant, binaryOp
    auto constExpr_ = inExpr.dyn_cast<mlir::AffineConstantExpr>();
    assert(constExpr_);
    return constExpr_;
  }
}

mlir::AffineMap getModifyedMap(mlir::OpBuilder builder, mlir::AffineMap oldMap, mlir::AffineExpr replaceExpr, int targetDim) {
  // [d0 + d1, d2, d1 + d2]  replaceExpr==d1 + 2   =>  [d0 + d1 + 2, d2, d1 + 2 + d2]
  llvm::SmallVector<mlir::AffineExpr> newExprs;
  for (auto oldEpr : oldMap.getResults()) {
    newExprs.push_back(getModifiedExpr(builder.getContext(), oldEpr, replaceExpr, targetDim, 1));
  }
  return mlir::AffineMap::get(oldMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
}

mlir::AffineMap mapDimToConstant(mlir::OpBuilder builder, mlir::AffineMap map, int targat, int constant) {
  // {d1, d0 + d1 + d2, d2} & target==1 & replace==0  => {0, d0 + 0 + d1, d2}
  mlir::MLIRContext* context = builder.getContext();
  auto oldExprs = map.getResults();
  mlir::SmallVector<mlir::AffineExpr> exprs;
  auto constantExpr = builder.getAffineConstantExpr(constant);
  for (auto expr : oldExprs) {
    auto expr_ = getModifiedExpr(context, expr, constantExpr, targat, 0);
    if (expr_.dyn_cast<mlir::AffineConstantExpr>() && expr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
      exprs.push_back(constantExpr);
    } else {
      exprs.push_back(expr_);
    }
  }
  return mlir::AffineMap::get(map.getNumDims()-1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), context);
}

mlir::AffineExpr shiftTargetAffineExprDim(mlir::OpBuilder builder, mlir::AffineExpr expr, int target, int shift) {
  // d0 + d1 + d2  target==1 & shift==1  => d0 + d2 + d3
  if (auto dimExpr = expr.dyn_cast<mlir::AffineDimExpr>()) {
    if (dimExpr.getPosition() >= target) {
      return mlir::getAffineDimExpr(dimExpr.getPosition() + shift, builder.getContext());
    } else {
      return dimExpr;
    }
  } else if (auto binaryExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = shiftTargetAffineExprDim(builder, binaryExpr.getLHS(), target, shift);
    auto RHS = shiftTargetAffineExprDim(builder, binaryExpr.getRHS(), target, shift);
    return mlir::getAffineBinaryOpExpr(binaryExpr.getKind(), LHS, RHS);
  } else {
    auto constExpr = expr.dyn_cast<mlir::AffineConstantExpr>();
    return constExpr;
  }
}

std::vector<int64_t> getOptVectorizeGroup(int64_t width) {
  // 计算最优的向量化组合根据width
  std::vector<int64_t> group;
  while (width != 0) {
    if (width - 4 >= 0) {
      width = width - 4;
      group.push_back(4);
    } else if (width - 2 >= 0) {
      width = width - 2;
      group.push_back(2);
    } else {
      width = width -1;
      group.push_back(1);
    }
  }
  return group;
}

mlir::affine::AffineForOp shiftBufferDatas(mlir::OpBuilder builder, mlir::Value src, mlir::Value dst, mlir::AffineMap srcMap, mlir::AffineMap dstMap, 
                                          llvm::SmallVector<mlir::Value> srcOperands, llvm::SmallVector<mlir::Value> dstOperands, 
                                          int64_t loadWidth, std:: vector<int> times) {
  // src -> dst  by  srcmap & dstmap
  auto srcNumDims = srcMap.getNumDims();
  auto dstType = dst.getType().dyn_cast<mlir::MemRefType>();
  mlir::Value ld;
  int nestedNum = 0;

  mlir::SmallVector<int64_t, 16> upperBounds(times.begin(), times.end());
  mlir::SmallVector<int64_t, 16> steps(times.size(), /*Value=*/1);
  mlir::SmallVector<int64_t, 16> lowerBounds(times.size(), /*Value=*/0);
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange ivs) {
      for (auto iv : ivs) {
        srcOperands.push_back(iv);
        dstOperands.push_back(iv);
        nestedNum++;
      }
      if (srcNumDims - srcOperands.size() == 1) {
        auto innerBody = [&](mlir::OpBuilder &b, mlir::Location nestedLoc, mlir::Value iv_inner, mlir::ValueRange iterArgs) {
          mlir::OpBuilder::InsertionGuard nestedGuard(b);
          srcOperands.push_back(iv_inner);
          dstOperands.push_back(iv_inner);
          nestedNum++;
          auto vectorType = mlir::VectorType::get(1, dstType.getElementType());
          ld = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorType, src, srcMap, srcOperands);
          b.create<mlir::affine::AffineVectorStoreOp>(b.getUnknownLoc(), ld, dst, dstMap, dstOperands);
          b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
        };
        b.create<mlir::affine::AffineForOp>(b.getUnknownLoc(), 0, loadWidth, 1, mlir::ValueRange({}), innerBody);
      } else {
        auto vectorType = mlir::VectorType::get(loadWidth, dstType.getElementType());
        ld = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorType, src, srcMap, srcOperands);
        b.create<mlir::affine::AffineVectorStoreOp>(b.getUnknownLoc(), ld, dst, dstMap, dstOperands);
      }
    }
  );
  // get mostouter affineforOp
  mlir::Operation* cur = ld.getDefiningOp();
  while (nestedNum != 0) {
    cur = cur->getParentOp();
    nestedNum--;
  }
  return mlir::dyn_cast<mlir::affine::AffineForOp>(cur);
}

int getLoopNestedNum(mlir::affine::AffineForOp forOp) {
  // 获取循环的嵌套次数
  int nestedNum = 0;
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp_) {
    nestedNum++;
  });
  return nestedNum;
}

std::vector<mlir::Value> collectNestedIvs(mlir::affine::AffineForOp forOp) {
  // 收集嵌套循环的iv
  std::vector<mlir::Value> ivs;
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp_) {
    ivs.push_back(forOp_.getInductionVar());
  });
  return ivs;
}

mlir::Value doubleBuffer(mlir::Value buffer) {
  // 在buffer下创建一个new buffer，size是两倍
  mlir::Operation* op = buffer.getDefiningOp();
  auto attr = mlir::dyn_cast<mlir::StringAttr>(op->getAttr(AttrBufDescription));
  auto bufDesc = attr.getValue().str();
  // 否则创建新的buf
  mlir::OpBuilder builder(op);
  mlir::SmallVector<int64_t> shape{2};
  auto bufType = buffer.getType().dyn_cast<mlir::MemRefType>();
  auto memSpace = static_cast<MemorySpace>(bufType.getMemorySpaceAsInt());
  for (auto s : bufType.getShape()) { shape.push_back(s); }
  mlir::Value newBuffer;
  if (memSpace == MemorySpace::shared) {
    auto alloc = createAllocOp<mlir::memref::AllocOp>(builder, shape, bufType.getElementType(), memSpace, KCG_ALIGNBYTE, bufDesc);
    newBuffer = alloc.getResult();
  } else {
    auto alloca = createAllocOp<mlir::memref::AllocaOp>(builder, shape, bufType.getElementType(), memSpace, KCG_ALIGNBYTE, bufDesc);
    newBuffer = alloca.getResult();
  }
  return newBuffer;
}

std::vector<std::vector<int64_t>> getNestedLoopData(mlir::affine::AffineForOp forOp) {
  // 获取嵌套循环的循环信息
  std::vector<int64_t> lowerBounds, upperBounds, steps;
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp fop) {
    auto loopData = getLoopBoundAndStep(fop);
    lowerBounds.push_back(std::get<0>(loopData));
    upperBounds.push_back(std::get<1>(loopData));
    steps.push_back(std::get<2>(loopData));
  });
  return {lowerBounds, upperBounds, steps};
}

std::pair<llvm::SmallVector<mlir::affine::AffineForOp>, llvm::SmallVector<mlir::Value>> 
createNestedLoops(mlir::OpBuilder builder, std::vector<std::vector<int64_t>> loopDatas) {
  // 根据loop的信息创建嵌套的loops
  llvm::SmallVector<mlir::Value> allIvs;
  llvm::SmallVector<mlir::affine::AffineForOp> mostLoops;
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    allIvs.push_back(iv);
    llvm::SmallVector<int64_t, 3> lowerBounds(loopDatas[0].begin()+1, loopDatas[0].end());
    llvm::SmallVector<int64_t, 3> upperBounds(loopDatas[1].begin()+1, loopDatas[1].end());
    llvm::SmallVector<int64_t, 3> steps(loopDatas[2].begin()+1, loopDatas[2].end());
    mlir::affine::buildAffineLoopNest(b, b.getUnknownLoc(), lowerBounds, upperBounds, steps,
      [&](mlir::OpBuilder &bb, mlir::Location loc, mlir::ValueRange ivs) {
        for (auto iv : ivs) { allIvs.push_back(iv); }
      });
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  auto outerLoop = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), loopDatas[0][0], 
                                          loopDatas[1][0], loopDatas[2][0], mlir::ValueRange({}), loopBody);
  mostLoops.push_back(outerLoop);
  mlir::affine::AffineForOp innerLoop;
  outerLoop.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp fop) {
    innerLoop = fop;
  });
  mostLoops.push_back(innerLoop);
  return std::make_pair(mostLoops, allIvs);
}

std::vector<mlir::affine::AffineForOp> createNewDataShiftForOp(mlir::OpBuilder builder, std::vector<mlir::affine::AffineForOp> forOps,  
                             std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps, mlir::Value mainIv, mlir::AffineExpr addExpr) {
  std::vector<mlir::affine::AffineForOp> newForOps;
  for (auto forOp : forOps) {
    auto loopDatas = getNestedLoopData(forOp);  // get nested loop datas
    auto allOps = collectInnerMostAllOps(forOp);  // collect all ops from most inner loop
    auto result = createNestedLoops(builder, loopDatas);  // create new nested loop
    mlir::affine::AffineForOp outerLoop = result.first[0], innerLoop = result.first[1];
    newForOps.push_back(outerLoop);
    llvm::SmallVector<mlir::Value> allIvs = result.second;
    mlir::OpBuilder b(innerLoop.getContext());
    b.setInsertionPointToStart(innerLoop.getBody());

    // create new loadOp 如果能在bufmap中找到这个loadOp的buf，则证明这个loadop的buf应该需要被替换
    // 先进行 loadOp 的创建
    mlir::Value newLoadOp;
    for (auto op : allOps) {
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
        auto result = getPerfetchMapDatas(b, loadOp, bufMaps, allIvs, mainIv, addExpr);
        newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), std::get<2>(result), std::get<1>(result), std::get<0>(result));
      } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
        auto result = getPerfetchMapDatas(b, vectorLoadOp, bufMaps, allIvs, mainIv, addExpr);
        newLoadOp = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorLoadOp.getVectorType(), 
                                                               std::get<2>(result), std::get<1>(result), std::get<0>(result));
      }
    }
    // storeOp 的创建
    for (auto op : allOps) {
      if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
        auto result = getPerfetchMapDatas(b, storeOp, bufMaps, allIvs, mainIv, addExpr);
        b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), newLoadOp, std::get<2>(result), std::get<1>(result), std::get<0>(result));
      } else if (auto vectorStoreOp = mlir::dyn_cast<mlir::affine::AffineVectorStoreOp>(op)) {
        auto result = getPerfetchMapDatas(b, vectorStoreOp, bufMaps, allIvs, mainIv, addExpr);
        b.create<mlir::affine::AffineVectorStoreOp>(b.getUnknownLoc(), newLoadOp, std::get<2>(result), std::get<1>(result), std::get<0>(result));
      }
    }
  }
  return newForOps;
}

void moveCalculateForOp(mlir::Operation* posOp, mlir::affine::AffineForOp &forOp, std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps, 
                        mlir::Value mainIv, mlir::AffineExpr addExpr) {
  // 移动计算的forOp
  forOp->moveAfter(posOp);
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    mlir::OpBuilder b(op);
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
      auto result = getCalculateMapDatas(b, loadOp, bufMaps, mainIv, addExpr);
      if(std::get<2>(result)) {
        auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), std::get<2>(result), std::get<1>(result), std::get<0>(result));
        replaceAndErase(newLoadOp, loadOp);
      }
    } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
      auto result = getCalculateMapDatas(b, vectorLoadOp, bufMaps, mainIv, addExpr);
      if (std::get<2>(result)) {
        auto newVectorLoadOp = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorLoadOp.getVectorType(), 
                                                                          std::get<2>(result), std::get<1>(result), std::get<0>(result));
        replaceAndErase(newVectorLoadOp, vectorLoadOp);
      }

    }
  });
}

mlir::affine::AffineForOp createRearCalculateForOp(mlir::OpBuilder builder, mlir::affine::AffineForOp calculateForOp, 
                                                   std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps) {
  // 寄存器预取会多出一个尾for
  mlir::IRMapping mapper;
  auto newBody = builder.clone(*calculateForOp, mapper);
  auto rearLoop = mlir::dyn_cast<mlir::affine::AffineForOp>(newBody);

  auto ops = collectInnerMostAllOps(rearLoop);
  for (auto op : ops) {
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
      auto buf = loadOp.getMemRef();
      mlir::OpBuilder b(loadOp);
      if (bufMaps.count(buf)) {
        llvm::SmallVector<mlir::AffineExpr> newExprs;
        auto map = loadOp.getAffineMap();
        newExprs.push_back(builder.getAffineConstantExpr(1));
        for (auto expr : map.getResults()) {
          newExprs.push_back(expr);
        }
        auto newMap = mlir::AffineMap::get(map.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
        auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), bufMaps[buf], newMap, loadOp.getMapOperands());
        replaceAndErase(newLoadOp, loadOp);
      }
    }
  }
  return rearLoop;
}

}