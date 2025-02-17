#include "Conversion/General/Rewriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <map>
#include <cmath>

namespace KernelCodeGen {
namespace Rewriter {

mlir::Value _inner_alloc_buffer(mlir::OpBuilder &builder, mlir::MemRefType &type) {
  if (type.getMemorySpaceAsInt() == int(KernelCodeGen::MemorySpace::local)){
    return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), type)->getResult(0);
  }
  return builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), type);
}

void OpSetDesc(mlir::Operation* op, const std::string& attrValue){
  mlir::OpBuilder b(op->getContext());
  op->setAttr("kcg.desc",b.getStringAttr(attrValue));
}

std::vector<mlir::affine::AffineForOp> split(mlir::affine::AffineForOp forOp, uint64_t num_output, std::vector<int64_t>&& factors) {
  // 创建一个空的切开的嵌套循环
  std::sort(factors.begin(), factors.end(), std::greater<uint64_t>());
  auto loopBAS = getLoopBoundAndStep(forOp);
  factors.insert(factors.begin(), std::get<1>(loopBAS));
  factors.push_back(1);
  mlir::SmallVector<int64_t, 16> upperBounds(factors.begin(), --(factors.end()));
  mlir::SmallVector<int64_t, 16> steps(++(factors.begin()), factors.end());
  mlir::SmallVector<int64_t, 16> lowerBounds(num_output, /*Value=*/0);

  std::vector<mlir::Value> ivsVector;
  mlir::OpBuilder builder(forOp.getOperation());
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      //empty nested loops.
      for (auto iv : ivs) { ivsVector.push_back(iv); }
    }
  );

  // 将旧的forop内部的op转移到新的嵌套forOp中
  auto loops = collectInnerOps<mlir::affine::AffineForOp>(forOp->getPrevNode());
  // erase the yield op, as the forOp will bring the AffineYieldOp
  mlir::affine::AffineForOp innermostForOp = loops.back();
  innermostForOp.getBody()->back().erase();
  spliceHaveBlockOp(innermostForOp, forOp, -1);

  // 需要修改affineMap
  auto oldIv = forOp.getInductionVar();
  std::set<mlir::Operation*> users = getValueUsers(oldIv);
  mlir::AffineExpr sumExpr = getOrderExpr(builder, ivsVector.size());

  // 替换load/store/apply的map
  for (auto user : users) {
    mlir::OpBuilder builder(user);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    llvm::SmallVector<mlir::Value> operands;
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
      auto mem = loadOp.getMemref();
      int dimCount = replaceIndexWithExpr(oldIv, ivsVector, loadOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      auto newLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), mem, map, llvm::ArrayRef<mlir::Value>(operands));
      replaceAndErase(newLoadOp, loadOp);
    } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
      auto valueToStore = storeOp.getValue();
      auto mem = storeOp.getMemref();
      int dimCount = replaceIndexWithExpr(oldIv, ivsVector, storeOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), valueToStore, mem, map, llvm::ArrayRef<mlir::Value>(operands));
      storeOp.erase();
    } else if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(user)) {
      int dimCount = replaceIndexWithExpr(oldIv, ivsVector, applyOp, sumExpr, exprs, operands);
      mlir::AffineMap map = mlir::AffineMap::get(/*dimCount*/dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
      auto newApplyOp = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange(operands));
      replaceAndErase(newApplyOp, applyOp);
    } else {
      assert(false);
    }
  }

  forOp.erase();
  return loops;
}

mlir::Value bufferizeLoopCarryVar(mlir::affine::AffineForOp &carryVarLoop, std::vector<mlir::affine::AffineForOp> &loops, std::string bufDesc) {
  // 将迭代遍历变成buffer，loops为buffer提供索引值
  llvm::SmallVector<int64_t> bufferShape;
  llvm::SmallVector<mlir::Value> bufferAdrressOperand;
  for (auto loop : loops) {
    auto loopBAS = getLoopBoundAndStep(loop);
    bufferShape.push_back((std::get<1>(loopBAS) - std::get<0>(loopBAS)) / std::get<2>(loopBAS));
    bufferAdrressOperand.push_back(loop.getInductionVar());
  }
  
  auto builder = getBuilder(loops[0], Position::before);
  auto carryVar = carryVarLoop.getRegionIterArgs()[0];
  auto allocOp = createAllocOp<mlir::memref::AllocaOp>(builder, bufferShape, carryVar.getType(), MemorySpace::local, KCG_ALIGNBYTE, bufDesc);
  // step1: 将buffer初始化值
  auto initValue = carryVarLoop.getOperands().back();
  auto defineOp = initValue.getDefiningOp();
  builder.setInsertionPointAfter(defineOp);
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), initValue, allocOp.getResult(), bufferAdrressOperand);
  // step2: 替换含迭代变量的循环
  eraseForOpIterVar(builder, carryVarLoop, allocOp, bufferAdrressOperand);
  return allocOp.getResult();
}

void loopToParallelZ(mlir::affine::AffineForOp loop, mlir::affine::AffineParallelOp &parallelOp) {
  // loop add to parallel

  llvm::SmallVector<mlir::AffineMap> lbMaps;
  llvm::SmallVector<mlir::AffineMap> upMaps;
  llvm::SmallVector<mlir::Value> lbOperands;
  llvm::SmallVector<mlir::Value> upOperands;
  llvm::SmallVector<int64_t> steps;

  // loop operand
  lbMaps.push_back(loop.getLowerBoundMap());
  upMaps.push_back(loop.getUpperBoundMap());
  lbOperands.append(loop.getLowerBoundOperands().begin(), loop.getLowerBoundOperands().end());
  upOperands.append(loop.getUpperBoundOperands().begin(), loop.getUpperBoundOperands().end());
  steps.push_back(loop.getStep().getLimitedValue());

  // parallel operands
  auto ss = parallelOp.getSteps();
  for (unsigned i=0; i<parallelOp.getNumDims(); i++) {
    lbMaps.push_back(parallelOp.getLowerBoundMap(i));
    upMaps.push_back(parallelOp.getUpperBoundMap(i));
    steps.push_back(ss[i]);
  }
  lbOperands.append(parallelOp.getLowerBoundsOperands().begin(), parallelOp.getLowerBoundsOperands().end());
  upOperands.append(parallelOp.getUpperBoundsOperands().begin(), parallelOp.getUpperBoundsOperands().end());

  // create new parallelOp
  mlir::OpBuilder builder = getBuilder(parallelOp, Position::before);
  mlir::affine::AffineParallelOp newParallelOp = builder.create<mlir::affine::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<mlir::AffineMap>(lbMaps), lbOperands,
    llvm::ArrayRef<mlir::AffineMap>(upMaps), upOperands,
    llvm::ArrayRef<int64_t>(steps));
  
  // replace old parallelOp
  parallelOp.getBody()->back().erase();
  spliceHaveBlockOp(newParallelOp, parallelOp);
  auto oldIVs = parallelOp.getIVs();
  auto newIVs = newParallelOp.getIVs();
  for (int i=0; i<oldIVs.size(); i++) {
    oldIVs[i].replaceAllUsesWith(newIVs[i+1]);
  }
  parallelOp.erase();

  // erase loop
  auto parentOp = loop.getOperation()->getParentOp();
  int index = getOpIndex(parentOp, loop);
  loop.getBody()->back().erase();
  spliceHaveBlockOp(parentOp, loop, index);
  loop.getInductionVar().replaceAllUsesWith(newIVs[0]);
  loop.erase();

  parallelOp = newParallelOp;
}
 
// Swap two nested loops.
// if outer loop contains multiple Operations, clone the outer loop to maintain correctness.
void swap(mlir::affine::AffineForOp outer, mlir::affine::AffineForOp inner) {
  auto& ops = outer.getBody()->getOperations();
  auto opNumber = ops.size();
  int position = 0;
  mlir::Operation* innerOp = inner;
  for (auto& op : ops) {
    if (&op == innerOp) {
      break;
    }
    position += 1;
  }
  // must found.
  assert(position < opNumber);

  bool existOpBeforeLoop = position != 0;
  // considering the affine.yield
  bool existOpAfterLoop = position != opNumber - 2;

  if (existOpBeforeLoop) {
    mlir::OpBuilder b(outer->getBlock(), mlir::Block::iterator(outer));
    mlir::IRMapping mapper;
    // auto cloned = b.clone(*outer, mapper);
    b.clone(*outer, mapper);
    auto cloned = (--mlir::Block::iterator(outer));

    auto clonedFor = mlir::dyn_cast<mlir::affine::AffineForOp>(cloned);
    assert(clonedFor);
    auto& ops_ = clonedFor.getBody()->getOperations();
  
    int count = 0;
    auto iter = --(--(ops_.end()));
    int number = ops_.size();
    for (int i = 0; i < number - position - 1; i++) {
      ++count;
      // it seems that iter->erase() will cause segment false.
      (iter--)->erase();
    }
  }
  if (existOpAfterLoop) {
    mlir::OpBuilder b(outer->getBlock(), ++mlir::Block::iterator(outer));
    mlir::IRMapping mapper;
    auto cloned = b.clone(*outer, mapper);
    auto& ops_ = mlir::dyn_cast<mlir::affine::AffineForOp>(cloned).getBody()->getOperations();
    auto iter = ops_.end();
    int number = ops_.size();
    for (int i = 0; i < number - position; i++) --iter;
    for(int i = 0; i <= position; i++) {
      (iter--)->erase();
    }
  }
  // clear current outer loop
  if (existOpBeforeLoop || existOpAfterLoop) {
    auto iter = --(ops.end());
    int number = ops.size();
    for (int i = 0; i < number; i++) {
      if (i == number - 1 - position || i == 0) {
        --iter;
      } else {
        (iter--)->erase();
      }
    }

  }
  // int count = 0;
  // for (auto& op : ops) {
  //   if (count != position && count !=  opNumber-1) {
  //     op.erase();
  //   }
  //   count += 1;
  // }


  /// step1: move the body of inner to outer
  // erase the yield op
  inner.getBody()->back().erase();
  // this block contain the inner Op
  inner->getBlock()->getOperations().splice( // this block is belong to outer
    mlir::Block::iterator(inner),
    inner.getBody()->getOperations());

  /// step2: move inner before outer.
  inner->moveBefore(outer);

  /// step3: make the outer as the body of inner
  inner.getBody()->getOperations().splice(inner.getBody()->end(),
                  outer->getBlock()->getOperations(), mlir::Block::iterator(outer));//only the outer.

  mlir::OpBuilder builder(inner.getContext());
  builder.setInsertionPointToEnd(inner.getBody());
  builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
}

void reorder(const std::vector<mlir::affine::AffineForOp>& loops) {

  std::map<mlir::affine::AffineForOp, int, CompareLoop> loopPriority;
  int priority = loops.size();
  for (auto loop : loops) {
    loopPriority[loop] = priority--;
  }

  auto findFirstTargetLoop = [&](mlir::affine::AffineForOp root) {
    if (loopPriority.count(root) != 0) return root;
    mlir::affine::AffineForOp result;
    bool found = false;
    root.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      if ((!found) && loopPriority.count(forOp) != 0) {
        result = forOp;
        found = true;
      }
    });
    assert(found);
    return result;
  };

  auto containTargetLoop = [&](mlir::affine::AffineForOp root) {

    auto& ops = root.getBody()->getOperations();
    mlir::affine::AffineForOp sonLoop;
    bool result = false;

    for (auto& op : ops) {
      if (auto sonOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
        if (loopPriority.count(sonOp)) {
          result = true;
          sonLoop = sonOp;
          break;
        }
      }
    }
    return result ? sonLoop : nullptr;
  };

  bool swapped;

  mlir::affine::AffineForOp rootForOp = Analyzer::findRootLoop(loops[0]);

  auto parentLoop_ = findFirstTargetLoop(rootForOp);

  // bubble sort.
  do {
    swapped = false;
    mlir::affine::AffineForOp parentLoop = parentLoop_;
    while (auto sonLoop = containTargetLoop(parentLoop)) {
      if (loopPriority[parentLoop] < loopPriority[sonLoop]) {
        swap(parentLoop, sonLoop);
        swapped = true;
      } else {
        parentLoop = sonLoop;
      }
    }
  } while (swapped);
}

mlir::affine::AffineParallelOp parallel(const std::vector<mlir::affine::AffineForOp>& forOps) {
  // X, Y, Z
  assert(forOps.size() <= 3);
  llvm::SmallVector<mlir::AffineMap> lbMaps;
  llvm::SmallVector<mlir::AffineMap> upMaps;
  llvm::SmallVector<int64_t> steps;
  llvm::SmallVector<int64_t> applySteps;

  mlir::OpBuilder builder(forOps[0]);
  for (auto forOp : forOps) {
    int64_t step = forOp.getStep().getLimitedValue();
    int64_t lowerBound = forOp.getLowerBoundMap().getConstantResults()[0];
    int64_t upperBound = forOp.getUpperBoundMap().getConstantResults()[0] / step;
    lbMaps.push_back(mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(lowerBound)), builder.getContext()));
    upMaps.push_back(mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(upperBound)), builder.getContext()));
    steps.push_back(1);
    applySteps.push_back(step);
  }

  mlir::affine::AffineParallelOp parallelOp = builder.create<mlir::affine::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<mlir::AffineMap>(lbMaps), mlir::ValueRange({}),
    llvm::ArrayRef<mlir::AffineMap>(upMaps), mlir::ValueRange({}),
    llvm::ArrayRef<int64_t>(steps));

  // erase the yield op of innermost loop
  auto innermost = forOps.back();
  innermost.getBody()->back().erase();
  spliceHaveBlockOp(parallelOp, innermost);

  llvm::SmallVector<mlir::Value> applyResults;
  mlir::AffineExpr dim = builder.getAffineDimExpr(0);
  auto ivs = parallelOp.getIVs();
  builder.setInsertionPointToStart(parallelOp.getBody());
  for (int i=0; i<ivs.size(); i++) {
    auto map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim * applySteps[i]), builder.getContext());
    auto applyOp = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange({ivs[i]}));
    applyResults.push_back(applyOp.getResult());
  }

  int count = applyResults.size() - 1;
  for (auto iter = forOps.rbegin(); iter != forOps.rend(); ++iter) {
    auto forOp = *iter;
    forOp.getInductionVar().replaceAllUsesWith(applyResults[count--]);
    forOp.erase();
  }
  return parallelOp;
}

llvm::SmallVector<mlir::Value> parallelToOneDim(mlir::affine::AffineParallelOp &parallelOp, int* outUpperBound) {
  // 将parallelOp转成一维表示
  std::vector<int64_t> uppers;
  auto builder = getBuilder(parallelOp, Position::before);
  int64_t upperBound = 1;
  for (auto i : parallelOp.getUpperBoundsMap().getConstantResults()) {
    upperBound *= i;
    uppers.push_back(i);
  }
  if(outUpperBound != nullptr){
    *outUpperBound = upperBound;
  }
  // create new parallelOp
  auto lowerMap = mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(0)), builder.getContext());
  auto upperMap = mlir::AffineMap::get(0, 0, llvm::ArrayRef<mlir::AffineExpr>(builder.getAffineConstantExpr(upperBound)), builder.getContext());
  mlir::affine::AffineParallelOp newOp = builder.create<mlir::affine::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<mlir::AffineMap>(lowerMap), mlir::ValueRange({}),
    llvm::ArrayRef<mlir::AffineMap>(upperMap), mlir::ValueRange({}),
    llvm::ArrayRef<int64_t>({1}));

  // create new maps
  llvm::SmallVector<mlir::AffineMap> maps;
  builder.setInsertionPointToStart(newOp.getBody());
  mlir::AffineExpr tid = builder.getAffineDimExpr(0);
  int64_t front = 1;
  for (int i=1; i<uppers.size(); i++) {
    int64_t sum = 1;
    for (int j=i; j<uppers.size(); j++) { 
      sum *= uppers[j]; 
    }
    maps.push_back(mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(tid.floorDiv(sum)), builder.getContext()));
    tid = tid % sum;
    if (i == uppers.size() - 1) {
      maps.push_back(mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(tid), builder.getContext()));
    }
  }

  // create affineApplyOp
  llvm::SmallVector<mlir::Value> newIVs;
  for (auto map : maps) {
    auto axesApplyOp = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange(newOp.getIVs()));
    newIVs.push_back(axesApplyOp.getResult());
  }

  // move
  parallelOp.getBody()->back().erase();
  spliceHaveBlockOp(newOp, parallelOp, maps.size());
  auto oldIVs = parallelOp.getIVs();
  for (int i=0; i<oldIVs.size(); i++) {
    oldIVs[i].replaceAllUsesWith(newIVs[i]);
  }
  parallelOp.erase();
  parallelOp = newOp;

  return newIVs;
}

// dst is register.
mlir::affine::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                              std::vector<int64_t> widths, mlir::affine::AffineForOp compute_at, Position pos) {
  auto dimsNum = map.getNumDims();
  auto builder = getBuilder(compute_at, pos);
  auto dstType = dst.getType().dyn_cast<mlir::MemRefType>();
  int64_t totalWidth = dstType.getShape()[0];

  std::vector<int> times;
  mlir::AffineExpr expr = builder.getAffineConstantExpr(0);
  for (int i=0; i<widths.size(); i++) {
    auto dim = builder.getAffineDimExpr(i);
    expr = expr + dim * widths[i];
    times.push_back(totalWidth / widths[i]);
    totalWidth = widths[i];
  }
  mlir::AffineMap dstMap;
  if (dimsNum - (operands.size() + times.size()) == 1) {
    expr = expr + builder.getAffineDimExpr(widths.size());
    dstMap = mlir::AffineMap::get(/*dimCount*/widths.size() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  } else {
    dstMap = mlir::AffineMap::get(/*dimCount*/widths.size(), 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  }
  
  llvm::SmallVector<mlir::Value> dstOperands;
  auto load = shiftBufferDatas(builder, src, dst, map, dstMap, operands, dstOperands, widths.back(), times);
  return load;
}

// src is register
mlir::affine::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                                std::vector<int64_t> widths, mlir::affine::AffineForOp compute_at, Position pos) {
  auto dimsNum = map.getNumDims();
  auto builder = getBuilder(compute_at, pos);
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  int64_t totalWidth = srcType.getShape()[0];

  std::vector<int> times;
  mlir::AffineExpr expr = builder.getAffineConstantExpr(0);
  for (int i=0; i<widths.size(); i++) {
    auto dim = builder.getAffineDimExpr(i);
    expr = expr + dim * widths[i];
    times.push_back(totalWidth / widths[i]);
    totalWidth = widths[i];
  }
  mlir::AffineMap srcMap;
  if (dimsNum - (operands.size() + times.size()) == 1) {
    expr = expr + builder.getAffineDimExpr(widths.size());
    srcMap = mlir::AffineMap::get(/*dimCount*/widths.size() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  } else {
    srcMap = mlir::AffineMap::get(/*dimCount*/widths.size(), 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  }
  
  llvm::SmallVector<mlir::Value> srcOperands;
  auto store = shiftBufferDatas(builder, src, dst, srcMap, map, srcOperands, operands, widths.back(), times);
  return store;
}

// src is register
mlir::affine::AffineForOp write(mlir::OpBuilder& builder, mlir::Value src, mlir::Value dst, 
    mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width) {
  auto dimsNum = map.getNumDims();
  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  bool twoLoop = abs(dimsNum - operands.size()) == 2;
  auto srcMap = !twoLoop ? 
                mlir::AffineMap::get(/*dimCount*/1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width), builder.getContext()) :
                mlir::AffineMap::get(/*dimCount*/2, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * width + dim1), builder.getContext());
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  // registers is always 1 dim.
  auto storeTimes = srcType.getShape()[0] / width;
  auto storeBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv,
                      mlir::ValueRange iterArgs) {
    mlir::OpBuilder::InsertionGuard nestedGuard(builder);
    // loop iterator is the last operand.
    operands.push_back(iv);
    if (twoLoop) {
      auto innerBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv_inner,
                        mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        // loop iterator is the last operand.
        operands.push_back(iv_inner);
        auto vectorType = mlir::VectorType::get(1, srcType.getElementType());
        auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv, iv_inner}));
        auto st = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
        builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
      };
      auto storeInner = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 
          0, width, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), innerBody);
      builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
    } else { 
      auto vectorType = mlir::VectorType::get(width, srcType.getElementType());
      auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, src, srcMap, mlir::ValueRange({iv}));
      auto st = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), ld.getResult(), dst, map, operands);
      builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc());
    }
  };
  auto store = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 
     0, storeTimes, 1, /*iterArgs=lvm::None*/ mlir::ValueRange({}), storeBody);
  return store;
}

mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos) {
  auto builder = getBuilder(compute_at, pos);
  return builder.create<mlir::gpu::BarrierOp>(builder.getUnknownLoc());
}

mlir::gpu::BarrierOp barrier(mlir::OpBuilder builder) {
  return builder.create<mlir::gpu::BarrierOp>(builder.getUnknownLoc());
}

void cache_read(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands) {
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp load) {
    if (load.getMemref() != src) return;
    mlir::OpBuilder builder(load);
    auto newLoad = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), cached, map, operands);
    load.getResult().replaceAllUsesWith(newLoad.getResult());
    load.erase();
  });
}

void cache_write(mlir::affine::AffineForOp scope, mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands) {
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    if (store.getMemref() != src) return;
    mlir::OpBuilder builder(store);
    auto newStore = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), store.getValue(), cached, map, operands);
    store.erase();
  });
}

///TODO: two level vector.
std::vector<std::vector<mlir::affine::AffineForOp>> get_write(mlir::affine::AffineParallelOp parallelLevel, mlir::Value dst) {
  std::vector<std::vector<mlir::affine::AffineForOp>> results;
  std::vector<mlir::affine::AffineStoreOp> stores;
  parallelLevel.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    if (store.getMemref() != dst) return;
    stores.push_back(store);
  });
  for (auto store : stores) {
    std::vector<mlir::affine::AffineForOp> result;
    mlir::affine::AffineForOp parent;
    mlir::Operation* cur = store;
    while (parent = mlir::dyn_cast<mlir::affine::AffineForOp>(cur->getParentOp())) {
      result.push_back(parent);
      cur = parent;
    }
    std::reverse(result.begin(), result.end());
    results.push_back(result);
  }
  return results;
}

mlir::affine::AffineForOp vectorize(mlir::affine::AffineForOp readOrWrite, int64_t width) {
  int64_t step = readOrWrite.getStep().getLimitedValue();
  int64_t ub = readOrWrite.getConstantUpperBound();
  int64_t lb = readOrWrite.getConstantLowerBound();
  assert(step = 1 && lb == 0 && ub % width == 0);
  readOrWrite.setStep(width);
  readOrWrite.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp load) {
    mlir::OpBuilder builder(load);
    auto type = load.getMemRef().getType().dyn_cast<mlir::MemRefType>();
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorLoad = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, load.getMemRef(), load.getAffineMap(), load.getMapOperands());
    load.getResult().replaceAllUsesWith(vectorLoad.getResult());
    load.erase();
  });
  readOrWrite.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    mlir::OpBuilder builder(store);
     auto type = store.getMemRef().getType().dyn_cast<mlir::MemRefType>();
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorStore = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemRef(), store.getAffineMap(), store.getMapOperands());
    store.erase();
  });
  return readOrWrite;
}

std::pair<mlir::affine::AffineForOp, mlir::affine::AffineForOp> 
splitUReduce(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands,
               int localSplitU, int64_t globStoreWidth, mlir::affine::AffineForOp compute_at, Position pos) {
  // splitU!=1时，插入将多层结果进行累加求和的结构
  auto builder = getBuilder(compute_at, pos);
  auto dstType = dst.getType().dyn_cast<mlir::MemRefType>();
  int64_t regCTotalWidth = dstType.getShape()[0];   // 16
  int64_t globStoreTotalWidth = regCTotalWidth / localSplitU;  // 8
  int64_t globStoreNum = globStoreTotalWidth / globStoreWidth;  // 4

  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dstExpr = dim0 * globStoreWidth;
  auto reduceExpr = dim0 * globStoreWidth + dim1;
  auto dstMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dstExpr), builder.getContext());
  auto reduceMap = mlir::AffineMap::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(reduceExpr), builder.getContext());

  auto oneMap = mapDimToConstant(builder, map, /*target*/1, /*constant*/0);
  llvm::SmallVector<mlir::Value> oneLoopSrcOperands(operands.begin(), operands.end());
  auto oneLoopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    oneLoopSrcOperands.push_back(iv);
    llvm::SmallVector<mlir::Value> oneLoopDstOperands{iv};
    auto vectorType = mlir::VectorType::get(globStoreWidth, dstType.getElementType());
    auto ld = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorType, src, oneMap, oneLoopSrcOperands);
    b.create<mlir::affine::AffineVectorStoreOp>(b.getUnknownLoc(), ld, dst, dstMap, oneLoopDstOperands);
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  auto oneLoop = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 0, globStoreNum, 1, mlir::ValueRange({}), oneLoopBody);

  auto newMap = addDimToMap(builder, map);
  llvm::SmallVector<mlir::Value> twoLoopSrcOperands(operands.begin(), operands.end());
  mlir::SmallVector<int64_t, 2> upperBounds{localSplitU, globStoreNum, globStoreWidth};
  mlir::SmallVector<int64_t, 2> steps(3, /*Value=*/1);
  mlir::SmallVector<int64_t, 2> lowerBounds{1, 0, 0};
  mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange ivs) {
      for (auto iv : ivs) {
        twoLoopSrcOperands.push_back(iv);
      }
      llvm::SmallVector<mlir::Value> reduceOperands{ivs[1], ivs[2]};
      auto loadRegC = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), dst, reduceMap, reduceOperands);
      auto loadShC = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), src, newMap, twoLoopSrcOperands);
      auto addOp = b.create<mlir::arith::AddFOp>(b.getUnknownLoc(), loadRegC, loadShC);
      b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), addOp, dst, reduceMap, reduceOperands);
    });
  
  return std::make_pair(oneLoop, mlir::dyn_cast<mlir::affine::AffineForOp>(oneLoop->getNextNode()));
}

mlir::affine::AffineForOp splitUWrite(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                                      int localSplitU, int64_t globStoreWidth, mlir::affine::AffineForOp compute_at, Position pos) {
  // 将结果累加完成后，再将结果写回到C矩阵
  auto builder = getBuilder(compute_at, pos);
  auto dim0 = builder.getAffineDimExpr(0);
  auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
  int64_t regTotalWidth = srcType.getShape()[0];
  int globStoreNum = regTotalWidth / localSplitU / globStoreWidth;
  mlir::AffineMap srcMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * globStoreWidth), builder.getContext());
  llvm::SmallVector<mlir::Value> srcOperands;
  auto store = shiftBufferDatas(builder, src, dst, srcMap, map, srcOperands, operands, globStoreWidth, {globStoreNum});
  return store;
}

mlir::Value bufferCombine(std::vector<std::vector<mlir::Value>> buffers, std::string bufDesc) {
  // 将buffer合并到一个，“{{smA, smB}, {smC}}”，smA+smB的大小比较smC的大小，取最大的size创建一维的buffer
  // {smA, smB}与{smC}可复用
  std::vector<std::pair<mlir::Value, int64_t>> bufAndOffsets;
  int64_t maxBufSize = 0;
  for (auto buffer : buffers) {
    int64_t bufSize = 0;
    for (auto buf : buffer) {
      auto bufType = buf.getType().dyn_cast<mlir::MemRefType>();
      int64_t size = 1;
      for (auto shape : bufType.getShape()) { size *= shape; }
      bufSize += size;
      bufAndOffsets.push_back(std::make_pair(buf, bufSize - size));
    }
    if (maxBufSize < bufSize) { maxBufSize = bufSize; }
  }

  mlir::OpBuilder builder = getBuilder(buffers[0][0].getDefiningOp(), Position::before);
  auto bufType = buffers[0][0].getType().dyn_cast<mlir::MemRefType>();
  auto memSpace = static_cast<MemorySpace>(bufType.getMemorySpaceAsInt());
  auto elementType = bufType.getElementType();
  mlir::Value newBuffer;
  if (memSpace == MemorySpace::shared) {
    auto sharedAlloc = createAllocOp<mlir::memref::AllocOp>(builder, {maxBufSize}, elementType, memSpace, KCG_ALIGNBYTE, bufDesc);
    newBuffer = sharedAlloc.getResult();
  } else {
    auto localAlloca = createAllocOp<mlir::memref::AllocaOp>(builder, {maxBufSize}, elementType, memSpace, KCG_ALIGNBYTE, bufDesc);
    newBuffer = localAlloca.getResult();
  }

  for (auto bufAndOffset : bufAndOffsets) {
    auto users = getValueUsers(bufAndOffset.first);
    int64_t offset = bufAndOffset.second;
    for (auto user : users) {
      builder.setInsertionPointAfter(user);
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
        auto newMap = getOneDimMap(loadOp, offset);
        auto newLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), newBuffer, newMap, loadOp.getMapOperands());
        replaceAndErase(newLoadOp, loadOp);
      } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
        auto newMap = getOneDimMap(storeOp, offset);
        builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), storeOp.getValue(), newBuffer, newMap, storeOp.getMapOperands());
        storeOp.erase();
      } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(user)) {
        auto newMap = getOneDimMap(vectorLoadOp, offset);
        auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorLoadOp.getVectorType(), 
                                                                                newBuffer, newMap, vectorLoadOp.getMapOperands());
        replaceAndErase(newVectorLoadOp, vectorLoadOp);
      } else if (auto vectorStoreOp = mlir::dyn_cast<mlir::affine::AffineVectorStoreOp>(user)) {
        auto newMap = getOneDimMap(vectorStoreOp, offset);
        builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), vectorStoreOp.getValue(), 
                                                          newBuffer, newMap, vectorStoreOp.getMapOperands());
        vectorStoreOp.erase();
      }
    }
    mlir::Operation* defOp = bufAndOffset.first.getDefiningOp();
    defOp->erase();
  }
  return newBuffer;
}

void BlockMapping(mlir::affine::AffineParallelOp gridLevel, int64_t groupWidth, bool isCol) {
  // 重映射block的位置，提高L2 cache命中率
  std::vector<int64_t> uppers;
  for (auto i : gridLevel.getUpperBoundsMap().getConstantResults()) { uppers.push_back(i); }
  auto applyResults = parallelToOneDim(gridLevel);
  if (!groupWidth) return;

  int64_t groupHeight, otherWidth;
  if (isCol) {
    groupHeight = uppers[uppers.size()-1];
    otherWidth = uppers[uppers.size()-2];
  } else {
    groupHeight = uppers[uppers.size()-2];
    otherWidth = uppers[uppers.size()-1];
  }

  if (otherWidth % groupWidth != 0) return;  // 不可以整除

  auto ivs = gridLevel.getIVs();
  mlir::OpBuilder builder = getBuilder(gridLevel, Position::begin);
  mlir::AffineExpr dim = builder.getAffineDimExpr(0);
  int64_t groupNum = groupWidth * groupHeight;
  auto start = dim.floorDiv(groupNum) * groupWidth;
  auto exas0Map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(start + dim % groupWidth), builder.getContext());
  auto exas0 = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), exas0Map, mlir::ValueRange({ivs[0]}));
  auto exas1Map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>((dim % groupNum).floorDiv(groupWidth)), builder.getContext());
  auto exas1 = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), exas1Map, mlir::ValueRange({ivs[0]}));

  if (isCol) {
    applyResults[0].replaceAllUsesWith(exas0.getResult());
    applyResults[1].replaceAllUsesWith(exas1.getResult());
  } else {
    applyResults[0].replaceAllUsesWith(exas1.getResult());
    applyResults[1].replaceAllUsesWith(exas0.getResult());
  }
}

void unrollAttribute(mlir::ModuleOp module, int unrollNum) {
  // 添加unroll属性
  module.walk<mlir::WalkOrder::PostOrder>([&](mlir::affine::AffineForOp forOp) {
    auto loopBAS = getLoopBoundAndStep(forOp);
    int64_t loopNum = (std::get<1>(loopBAS) - std::get<0>(loopBAS)) / std::get<2>(loopBAS);
    if (loopNum >= unrollNum) return;
    mlir::OpBuilder builder(forOp->getContext());
    forOp->setAttr(std::string("affine.loop"), builder.getStringAttr("unroll"));
  });
}

std::pair<std::map<mlir::Value, mlir::Value, BufferCompare>, std::pair<std::vector<mlir::affine::AffineForOp>, std::vector<mlir::affine::AffineForOp>>>
sharedPrefetch(mlir::affine::AffineForOp &forOp, std::vector<mlir::affine::AffineForOp> &loadRegForOps, std::vector<mlir::affine::AffineForOp> &loadSharedForOps, 
               mlir::affine::AffineForOp &calculateForOp, std::vector<mlir::Value> buffers) {
  // double buffer save in map
  std::map<mlir::Value, mlir::Value, BufferCompare> doubleBufMaps;
  std::vector<mlir::affine::AffineForOp> newLoadSharedForOps, newLoadRegForOps, perfetchLoadSharedForOps, perfetchLoadRegForOps;
  for (auto buf : buffers) {
    doubleBufMaps.emplace(buf, doubleBuffer(buf));
  }

  // base datas
  mlir::OpBuilder builder = getBuilder(forOp, Position::before);
  auto forData = getLoopBoundAndStep(forOp);
  auto upperBound = std::get<1>(forData);
  auto step = std::get<2>(forData);
  auto k = builder.getAffineDimExpr(0);
  auto cst = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({upperBound - step - k}), llvm::ArrayRef<bool>({false}));

  // create new main forOp
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  auto mainForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), step, upperBound + step, step, mlir::ValueRange({}), loopBody);

  // create load reg ifOp
  mlir::Value mainIv = mainForOp.getInductionVar();
  builder.setInsertionPointToStart(mainForOp.getBody());
  auto ifOp0 = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{mainIv}, false);
  builder.setInsertionPointToStart(ifOp0.getThenBlock());
  newLoadRegForOps = createNewDataShiftForOp(builder, loadRegForOps, doubleBufMaps, mainIv);
  builder.setInsertionPointAfter(ifOp0);

  // move calculate forOp to here
  mlir::AffineExpr expr = (builder.getAffineDimExpr(0).floorDiv(step) - 1) % 2;
  moveCalculateForOp(ifOp0, calculateForOp, doubleBufMaps, mainIv, expr);
  
  // create load shared ifOp
  auto ifOp1 = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{mainIv}, false);
  builder.setInsertionPointToStart(ifOp1.getThenBlock());
  expr = builder.getAffineDimExpr(0).floorDiv(step) % 2;
  newLoadSharedForOps = createNewDataShiftForOp(builder, loadSharedForOps, doubleBufMaps, mainIv, expr);  // 涉及buoule buf
  barrier(builder);

  // 预取
  builder.setInsertionPoint(mainForOp);
  perfetchLoadRegForOps = createNewDataShiftForOp(builder, loadRegForOps, doubleBufMaps);
  perfetchLoadSharedForOps = createNewDataShiftForOp(builder, loadSharedForOps, doubleBufMaps);  // 涉及buoule buf 预取
  barrier(mainForOp, Position::before);

  // delete origin
  forOp.erase();
  forOp = mainForOp;
  loadRegForOps = newLoadRegForOps;
  loadSharedForOps = newLoadSharedForOps;
  for (auto oldBuf : buffers) {
    mlir::Operation* op = oldBuf.getDefiningOp();
    op->erase();
  }
  return std::make_pair(doubleBufMaps, std::make_pair(perfetchLoadRegForOps, perfetchLoadSharedForOps));
}

std::pair<std::map<mlir::Value, mlir::Value, BufferCompare>, std::pair<std::vector<mlir::affine::AffineForOp>, mlir::affine::AffineForOp>>
registersPrefetch(mlir::affine::AffineForOp &forOp, std::vector<mlir::affine::AffineForOp> &loadRegForOps, 
                       mlir::affine::AffineForOp &calculateForOp, std::vector<mlir::Value> buffers) {
  // registers double
  std::map<mlir::Value, mlir::Value, BufferCompare> doubleBufMaps;
  for (auto buf : buffers) {
    doubleBufMaps.emplace(buf, doubleBuffer(buf));
  }

  // base datas
  mlir::OpBuilder builder = getBuilder(forOp, Position::before);
  auto forData = getLoopBoundAndStep(forOp);
  auto upperBound = std::get<1>(forData);
  auto step = std::get<2>(forData);
  std::vector<mlir::affine::AffineForOp> newLoadRegForOps, perfetchLoadRegForOps;

  // rear outer forBK
  builder.setInsertionPointAfter(forOp);
  auto rearForOp = createRearCalculateForOp(builder, calculateForOp, doubleBufMaps);
  
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  builder.setInsertionPoint(forOp);
  auto mainForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 0, upperBound - step, step, mlir::ValueRange({}), loopBody);

  mlir::Value mainIv = mainForOp.getInductionVar();
  builder.setInsertionPointToStart(mainForOp.getBody());
  // for inner load
  mlir::AffineExpr expr = (builder.getAffineDimExpr(0).floorDiv(step) + 1) % 2;
  newLoadRegForOps = createNewDataShiftForOp(builder, loadRegForOps, doubleBufMaps, mainIv, expr);

  // move calculate
  expr = builder.getAffineDimExpr(0).floorDiv(step) % 2;
  moveCalculateForOp(newLoadRegForOps.back(), calculateForOp, doubleBufMaps, mainIv, expr);

  // perfetch
  builder.setInsertionPoint(mainForOp);
  perfetchLoadRegForOps = createNewDataShiftForOp(builder, loadRegForOps, doubleBufMaps);

  // delete origin forbk
  forOp.erase();
  forOp = mainForOp;
  loadRegForOps = newLoadRegForOps;
  for (auto oldBuf : buffers) {
    mlir::Operation* op = oldBuf.getDefiningOp();
    op->erase();
  }
  return std::make_pair(doubleBufMaps, std::make_pair(perfetchLoadRegForOps, rearForOp));
}

void doublePerfetchAdjust(std::vector<mlir::affine::AffineForOp> &shShPerfetchForOps, std::vector<mlir::affine::AffineForOp> &shRegPerfetchForOps, 
                          std::vector<mlir::affine::AffineForOp> &regPerfetchForOps, mlir::affine::AffineForOp &rearForOp, 
                          std::vector<mlir::Value> smBufs, std::vector<mlir::Value> regBufs) {
  // 使用两种预取后，需要进行的调整
  mlir::OpBuilder builder(rearForOp->getParentOp());
  // 1.寄存器预取提出最大循环，修改map的第一个expr为0
  for (auto forOp : regPerfetchForOps) {
    // get nested forOp data
    int nestedNum = 0;
    forOp.walk<mlir::WalkOrder::PostOrder>([&](mlir::affine::AffineForOp forOp){ nestedNum++; });
    mlir::IRMapping mapper;
    auto newBody = builder.clone(*forOp, mapper);
    auto regPerfetchLoop = mlir::dyn_cast<mlir::affine::AffineForOp>(newBody);
    auto ops = collectInnerMostAllOps(regPerfetchLoop);
    for (auto smBuf : smBufs) {
      for (auto op : ops) {
        mlir::OpBuilder b(op);
        if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
          auto buf = loadOp.getMemRef();
          if (smBuf.getDefiningOp() == buf.getDefiningOp()) {
            auto result = getRegPerfetchOuterAdjustDatas(b, loadOp, nestedNum);
            auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), buf, result.second, result.first);
            replaceAndErase(newLoadOp, loadOp);
          }
        } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
          auto buf = vectorLoadOp.getMemRef();
          if (smBuf.getDefiningOp() == buf.getDefiningOp()) {
            auto result = getRegPerfetchOuterAdjustDatas(b, vectorLoadOp, nestedNum);
            auto newVectorLoadOp = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorLoadOp.getVectorType(), buf, result.second, result.first);
            replaceAndErase(newVectorLoadOp, vectorLoadOp);
          }
        }
      }
    }
  }
  // 2. 调度最后的计算部分到第二个if的下方
  auto op = rearForOp->getParentOp();
  auto forKOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op);
  builder.setInsertionPoint(&(forKOp.getBody()->back()));
  rearForOp->remove();
  builder.insert(rearForOp);

  // 3. 在fork最后再加一个寄存器预取
  builder.setInsertionPoint(&(forKOp.getBody()->back()));
  auto step = std::get<2>(getLoopBoundAndStep(forKOp));
  auto forKIv = forKOp.getInductionVar();
  for (auto forOp : regPerfetchForOps) {
    forOp->remove();
    builder.insert(forOp);
    auto ops = collectInnerMostAllOps(forOp);
    for (auto smBuf : smBufs) {
      for (auto op : ops) {
        mlir::OpBuilder b(op);
        if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
          auto buf = loadOp.getMemRef();
          if (smBuf.getDefiningOp() == buf.getDefiningOp()) {
            auto newMap = getRegPerFetchInnerAdjustDatas(b, loadOp, forKIv, step);
            auto newLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buf, newMap, loadOp.getMapOperands());
            replaceAndErase(newLoadOp, loadOp);
          }
        } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
          auto buf = vectorLoadOp.getMemRef();
          if (smBuf.getDefiningOp() == buf.getDefiningOp()) {
            auto newMap = getRegPerFetchInnerAdjustDatas(b, vectorLoadOp, forKIv, step);
            auto newVectorLoadOp = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorLoadOp.getVectorType(), 
                                                                              buf, newMap, vectorLoadOp.getMapOperands());
            replaceAndErase(newVectorLoadOp, vectorLoadOp);
          }
        }
      }
    }
  }

  // 4. （可选）将shared的预取合并成为直接从glob到shared
  for (int i=0; i<shShPerfetchForOps.size(); i++) {
    auto globToRegOps = collectInnerMostAllOps(shRegPerfetchForOps[i]);
    auto regToSharedOps = collectInnerMostAllOps(shShPerfetchForOps[i]);
    if (globToRegOps.size() == regToSharedOps.size()) {  // 只有相等才是怎么取到reg的数据就怎么取到shared

      mlir::affine::AffineVectorLoadOp rtsLoadOp, newLoadOp;
      for (auto op : globToRegOps) {
        if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
          for (auto op_ : regToSharedOps) {
            if (auto vectorLoadOp_ = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op_)) {
              auto ivs = collectNestedIvs(shShPerfetchForOps[i]);
              // operands
              auto oldOperands = vectorLoadOp.getMapOperands();
              llvm::SmallVector<mlir::Value> newOperands(oldOperands.begin(), oldOperands.end());
              for (int i=0; i<ivs.size(); i++) {
                newOperands.erase(newOperands.end()-1);
              }
              for (auto iv : ivs) {
                newOperands.push_back(iv);
              }
              builder.setInsertionPointAfter(vectorLoadOp_);
              auto newVectorLoadOp = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorLoadOp.getVectorType(), 
                                                                                 vectorLoadOp.getMemRef(), vectorLoadOp.getAffineMap(), newOperands);
              replaceAndErase(newVectorLoadOp, vectorLoadOp_);
              break;
            }
          }
          break;
        }
      }
      shRegPerfetchForOps[i].erase();
    }
  }
}

} // rewriter

} // kernelcodegen