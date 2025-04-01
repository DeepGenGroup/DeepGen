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
  return {lb, ub, step};
}

mlir::Value createAllocOp(mlir::OpBuilder builder, llvm::SmallVector<int64_t> shape, mlir::Type dtype, MemorySpace space, int alignment, std::string bufDesc) {
  // 创建allocaOp
  mlir::Value allocVal;
  auto bufferType = mlir::MemRefType::get(shape, dtype, {}, static_cast<int>(space));
  if (space == MemorySpace::local) {
    auto reg = builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), bufferType);
    reg.setAlignment(alignment);
    reg->setAttr(AttrBufDescription, builder.getStringAttr(bufDesc));
    allocVal = reg.getResult();
  } else {
    auto sm = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), bufferType);
    sm.setAlignment(alignment);
    sm->setAttr(AttrBufDescription, builder.getStringAttr(bufDesc));
    allocVal = sm.getResult();
  }
  return allocVal;
}

std::pair<std::vector<mlir::affine::AffineForOp>, std::vector<mlir::Value>> createNestedLoops(mlir::OpBuilder builder, 
                                                                                              llvm::SmallVector<int64_t> lowerBounds, 
                                                                                              llvm::SmallVector<int64_t> upperBounds, 
                                                                                              llvm::SmallVector<int64_t> steps) {
  // 根据loop的信息创建嵌套的loops
  llvm::SmallVector<int64_t> outer{lowerBounds[0], upperBounds[0], steps[0]};
  lowerBounds.erase(lowerBounds.begin());
  upperBounds.erase(upperBounds.begin());
  steps.erase(steps.begin());
  // create for
  std::vector<mlir::Value> allIvs;
  std::vector<mlir::affine::AffineForOp> mostLoops;
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    allIvs.push_back(iv);
    mlir::affine::buildAffineLoopNest(b, b.getUnknownLoc(), lowerBounds, upperBounds, steps,
      [&](mlir::OpBuilder &bb, mlir::Location loc, mlir::ValueRange ivs) {
        for (auto iv : ivs) { allIvs.push_back(iv); }
      });
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  auto outerLoop = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), outer[0], outer[1], outer[2], 
                                                             mlir::ValueRange({}), loopBody);
  outerLoop.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp fop) {
    mostLoops.push_back(fop);
  });
  return {mostLoops, allIvs};
}

void replaceAndErase(mlir::Operation* newOp, mlir::Operation* oldOp) {
  // 替换后面op使用到oldOp的值，且删除oldOp
  auto oldResult = oldOp->getResult(0);
  oldResult.replaceAllUsesWith(newOp->getResult(0));
  oldOp->erase();
}

void spliceHaveBlockOp(mlir::Operation* newOp, mlir::Operation* oldOp, int insertPos, int startOpIndex, int endOpIndex) {
  // 将 oldOp 中的 ops 转到 newOp 中，index决定转移newOp的位置
  auto &oldBlock = oldOp->getRegion(0).front();
  auto &newBlock = newOp->getRegion(0).front();
  unsigned oldOpCount = oldBlock.getOperations().size();
  unsigned newOpCount = newBlock.getOperations().size();

  int startOpIdx = (startOpIndex >= 0) ? startOpIndex : oldOpCount + startOpIndex + 1;
  int endOpIdx = (endOpIndex >= 0) ? endOpIndex : oldOpCount + endOpIndex + 1;
  if (startOpIdx > oldOpCount || endOpIdx > oldOpCount || startOpIdx < 0 || endOpIdx < 0) {
    llvm::errs() << "index out of range of old block\n";
    return;
  }
  if (startOpIdx >= endOpIdx) return;

  int actInsertPos = (insertPos >= 0) ? insertPos : newOpCount + insertPos + 1;
  if (actInsertPos > newOpCount || actInsertPos < 0) {
    llvm::errs() << "actInsertPos: " << actInsertPos << " index out of range of new block\n";
    return;
  }

  auto& oldOps = oldBlock.getOperations();
  auto& newOps = newBlock.getOperations();
  auto opStart = oldOps.begin();
  std::advance(opStart, startOpIdx);
  auto opEnd = opStart;
  std::advance(opEnd, endOpIdx - startOpIdx);
  auto insertIt = newOps.begin();
  std::advance(insertIt, actInsertPos);
  newOps.splice(insertIt, oldOps, opStart, opEnd);
}

void replaceOpsOperands(mlir::Operation* parentOp, const std::vector<mlir::Value>& oldIvs, const std::vector<mlir::Value>& newIvs) {
  // 替换在parentOp内部的op的operands
  parentOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    auto operands = op->getOperands();
    std::vector<unsigned> indexs;
    for (auto iv : oldIvs) {
      for (unsigned i=0; i<operands.size(); i++) {
        if (iv == operands[i]) {
          indexs.push_back(i);
          break;
        }
      }
    }
    for (unsigned i=0; i<indexs.size(); i++) {
      op->setOperand(indexs[i], newIvs[i]);
    }
  });
}

void replaceOpOperands(mlir::Operation* op, mlir::Value oldOperand, mlir::Value newOperand) {
  // 将op中operand==oldOperand的operand替换newOperand
  auto operands = op->getOperands();
  unsigned index; 
  for (unsigned i=0; i<operands.size(); i++) {
    if (operands[i] == oldOperand) {
      index = i;
      break;
    }
  }
  op->setOperand(index, newOperand);
}

std::set<mlir::Operation*> getValueUsers(mlir::Value var) {
  // 获取value的使用者
  std::set<mlir::Operation*> users;
  for (auto user: var.getUsers()) {
    users.insert(user);
  }
  return users;
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

std::vector<mlir::affine::AffineForOp> decoupleNestedLoop(std::vector<mlir::affine::AffineForOp> upLoops, 
                                                          mlir::affine::AffineForOp lowLoop, 
                                                          bool carryDesc) {
  /*
  for (i to n){             for (i to n){
    ...                       ...
    for (j to m) {    =>    }
      ...                   for (i to n){
    }                         for (j to m) {
  }                             ...
                              }
                            }
  */
  // collect loops data
  std::vector<mlir::Value> oldIvs;
  llvm::SmallVector<int64_t> lowerBounds, upperBounds, steps;
  for (auto loop : upLoops) {
    oldIvs.push_back(loop.getInductionVar());
    auto [lb, ub, step] = getLoopBoundAndStep(loop);
    lowerBounds.push_back(lb);
    upperBounds.push_back(ub);
    steps.push_back(step);
  }
  // create new loops
  mlir::OpBuilder builder(upLoops[0]);
  auto [newLoops, newIvs] = createNestedLoops(builder, lowerBounds, upperBounds, steps);
  // set attr
  if (carryDesc) {
    for (int i=0; i<upLoops.size(); i++) {
      copyAttr<mlir::StringAttr>(upLoops[i], newLoops[i], FORDESC);
    }
  }
  // move ops
  int index = getOpIndex(upLoops[0], lowLoop);
  mlir::affine::AffineForOp innerLoop = newLoops.back();
  spliceHaveBlockOp(innerLoop, upLoops.back(), 0, 0, index);
  // modify ops(storeop) under init forop 
  replaceOpsOperands(innerLoop, oldIvs, newIvs);
  return newLoops;
}

void eraseForOpIterVar(mlir::affine::AffineForOp &forOp, llvm::SmallVector<mlir::Value> bufs, llvm::SmallVector<mlir::Value> ivs) {
  // 只是将含有迭代变量的forop转换成不含迭代遍历的for，迭代变量使用storeOp代替
  // 创建新的forop
  mlir::OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  llvm::SmallVector<mlir::Value> replaceValues;
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    for (auto buf : bufs) {
      auto loadOp = b.create<mlir::affine::AffineLoadOp>(loc, buf, ivs);
      replaceValues.push_back(loadOp.getResult());
    }
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  auto [lb, ub, step] = getLoopBoundAndStep(forOp);
  auto newLoop = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), lb, ub, step, mlir::ValueRange({}), loopBody);
  // set attr
  copyAttr<mlir::StringAttr>(forOp, newLoop, FORDESC);
  auto& oldYieldOp = forOp.getBody()->getOperations().back();

  // 将旧forop的body转移到新的forop中，且body中使用了迭代变量的使用 loadop 的结果代替
  spliceHaveBlockOp(newLoop, forOp, bufs.size(), 0, -2);
  forOp.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());
  for (int i=0; i<replaceValues.size(); i++) {
    forOp.getRegionIterArgs()[i].replaceAllUsesWith(replaceValues[i]);
    // 迭代过程使用storeOp代替
    builder.setInsertionPoint(&newLoop.getBody()->getOperations().back());
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), oldYieldOp.getOperand(i), bufs[i], ivs); 

    // 最后将forop之外使用了其迭代变量的地方全部替换成 loadOp 加载的数据
    auto iter = forOp.getResult(i);
    auto users = getValueUsers(iter);
    for (auto user : users) {
      builder.setInsertionPoint(user);
      auto loadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), bufs[i], ivs);
      replaceOpOperands(user, iter, loadOp.getResult());
    }
  }
  forOp.erase();
  forOp = newLoop;
}




mlir::AffineExpr getOrderExpr(mlir::OpBuilder builder, int dimCount) {
  // 获取一个有序的连续累加的affine表达式
  mlir::AffineExpr sumExpr = builder.getAffineConstantExpr(0);
  for (int i=0; i<dimCount; i++) {
    sumExpr = sumExpr + builder.getAffineDimExpr(i);
  }
  return sumExpr;
}

mlir::AffineExpr shiftAffineExprDim(mlir::OpBuilder builder, mlir::AffineExpr expr, int shift) {
  // d0 + d1 + d2  =>  shift==1  =>  d1 + d2 + d3
  if (auto dimExpr_ = expr.dyn_cast<mlir::AffineDimExpr>()) {
    return builder.getAffineDimExpr(dimExpr_.getPosition() + shift);
  } else if (auto binaryExpr_ = expr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = shiftAffineExprDim(builder, binaryExpr_.getLHS(), shift);
    auto RHS = shiftAffineExprDim(builder, binaryExpr_.getRHS(), shift);
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

mlir::AffineExpr getModifiedExpr(mlir::OpBuilder builder, mlir::AffineExpr inExpr, mlir::AffineExpr replaceExpr, int targetDim, int replaceNumberDims) {
  // d0 + d1 + d2  =>  target==1 & replace==[d1 + d2 + d3] =>  d0 + [d1 + d2 + d3] + d4
  if (auto dimExpr_ = inExpr.dyn_cast<mlir::AffineDimExpr>()) {
    if (dimExpr_.getPosition() == targetDim) {
      return replaceExpr;
    } else if (dimExpr_.getPosition() > targetDim) {
      return builder.getAffineDimExpr(dimExpr_.getPosition() + replaceNumberDims - 1);
    } else {
      return dimExpr_;
    }
  } else if (auto binaryExpr_ = inExpr.dyn_cast<mlir::AffineBinaryOpExpr>()){
    auto LHS = getModifiedExpr(builder, binaryExpr_.getLHS(), replaceExpr, targetDim, replaceNumberDims);
    auto RHS = getModifiedExpr(builder, binaryExpr_.getRHS(), replaceExpr, targetDim, replaceNumberDims);
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
    newExprs.push_back(getModifiedExpr(builder, oldEpr, replaceExpr, targetDim, 1));
  }
  return mlir::AffineMap::get(oldMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
}

mlir::AffineMap mapDimToConstant(mlir::OpBuilder builder, mlir::AffineMap map, int targat, int constant) {
  // {d1, d0 + d1 + d2, d2} & target==1 & replace==0  => {0, d0 + 0 + d1, d2}
  auto oldExprs = map.getResults();
  mlir::SmallVector<mlir::AffineExpr> exprs;
  auto constantExpr = builder.getAffineConstantExpr(constant);
  for (auto expr : oldExprs) {
    auto expr_ = getModifiedExpr(builder, expr, constantExpr, targat, 0);
    if (expr_.dyn_cast<mlir::AffineConstantExpr>() && expr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
      exprs.push_back(constantExpr);
    } else {
      exprs.push_back(expr_);
    }
  }
  return mlir::AffineMap::get(map.getNumDims()-1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
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
  mlir::Value newBuffer = createAllocOp(builder, shape, bufType.getElementType(), memSpace, KCG_ALIGNBYTE, bufDesc);
  return newBuffer;
}

std::tuple<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>> 
  getNestedLoopData(mlir::affine::AffineForOp forOp) {
  // 获取嵌套循环的循环信息，前提是完美循环
  llvm::SmallVector<int64_t> lowerBounds, upperBounds, steps;
  forOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp fop) {
    auto [lb, up, step] = getLoopBoundAndStep(fop);
    lowerBounds.push_back(lb);
    upperBounds.push_back(up);
    steps.push_back(step);
  });
  return {lowerBounds, upperBounds, steps};
}

std::vector<mlir::affine::AffineForOp> createNewDataShiftForOp(mlir::OpBuilder builder, std::vector<mlir::affine::AffineForOp> forOps,  
                             std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps, mlir::Value mainIv, mlir::AffineExpr addExpr) {
  std::vector<mlir::affine::AffineForOp> newForOps;
  for (auto forOp : forOps) {
    auto [lbs, ubs, steps] = getNestedLoopData(forOp);  // get nested loop datas
    auto allOps = collectInnerMostAllOps(forOp);  // collect all ops from most inner loop
    auto [loops, allIvs] = createNestedLoops(builder, lbs, ubs, steps);  // create new nested loop
    mlir::affine::AffineForOp outerLoop = loops[0], innerLoop = loops[1];
    newForOps.push_back(outerLoop);
    mlir::OpBuilder b(innerLoop.getContext());
    b.setInsertionPointToStart(innerLoop.getBody());

    // create new loadOp 如果能在bufmap中找到这个loadOp的buf，则证明这个loadop的buf应该需要被替换
    // 先进行 loadOp 的创建
    mlir::Value newLoadOp;
    for (auto op : allOps) {
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
        auto [operands, map, buf] = getPerfetchMapDatas(b, loadOp, bufMaps, allIvs, mainIv, addExpr);
        newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), buf, map, operands);
      } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
        auto [operands, map, buf] = getPerfetchMapDatas(b, vectorLoadOp, bufMaps, allIvs, mainIv, addExpr);
        newLoadOp = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorLoadOp.getVectorType(), buf, map, operands);
      }
    }
    // storeOp 的创建
    for (auto op : allOps) {
      if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
        auto [operands, map, buf] = getPerfetchMapDatas(b, storeOp, bufMaps, allIvs, mainIv, addExpr);
        b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), newLoadOp, buf, map, operands);
      } else if (auto vectorStoreOp = mlir::dyn_cast<mlir::affine::AffineVectorStoreOp>(op)) {
        auto [operands, map, buf] = getPerfetchMapDatas(b, vectorStoreOp, bufMaps, allIvs, mainIv, addExpr);
        b.create<mlir::affine::AffineVectorStoreOp>(b.getUnknownLoc(), newLoadOp, buf, map, operands);
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
      auto [operands, map, buf] = getCalculateMapDatas(b, loadOp, bufMaps, mainIv, addExpr);
      if(buf) {
        auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), buf, map, operands);
        replaceAndErase(newLoadOp, loadOp);
      }
    } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
      auto [operands, map, buf] = getCalculateMapDatas(b, vectorLoadOp, bufMaps, mainIv, addExpr);
      if (buf) {
        auto newVectorLoadOp = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorLoadOp.getVectorType(), buf, map, operands);
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

int32_t getNoBodyOpCount(mlir::Operation* op) {
  // 获取没有body的op数量
  int32_t count = 0;
  for (auto &op : op->getRegion(0).front().getOperations()) {
    if (auto forOp_ = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      auto [lb, ub, step] = getLoopBoundAndStep(forOp_);
      int32_t loopNum = (ub - lb) / step;
      count += getNoBodyOpCount(forOp_) * loopNum;
    } else if (auto ifOp = mlir::dyn_cast<mlir::affine::AffineIfOp>(op)) {
      count += getNoBodyOpCount(ifOp);
    } else if (!mlir::dyn_cast<mlir::affine::AffineYieldOp>(op)) {
      count++;
    }
  }
  return count;
}

}