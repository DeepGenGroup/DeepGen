#include "Conversion/General/Rewriter.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <map>
#include <cmath>

namespace KernelCodeGen {
namespace Rewriter {


void OpSetDesc(mlir::Operation* op, const std::string& attrValue){
  mlir::OpBuilder b(op->getContext());
  op->setAttr("kcg.desc",b.getStringAttr(attrValue));
}

std::vector<mlir::affine::AffineForOp> split(mlir::affine::AffineForOp forOp, 
                                              const std::vector<int64_t>& tile, 
                                              const std::vector<std::string>& forDescs) {
  // 创建一个空的切开的嵌套循环
  std::vector<int64_t> ts(tile.begin(), tile.end());
  std::sort(ts.begin(), ts.end(), std::greater<uint64_t>());  // 降序
  auto [lb, ub, step] = getLoopBoundAndStep(forOp);
  ts.insert(ts.begin(), ub);
  ts.push_back(1);
  // create nest loops
  mlir::OpBuilder builder(forOp);
  llvm::SmallVector<int64_t> lbs(ts.size()-1, 0), ubs(ts.begin(), --(ts.end())), steps(++(ts.begin()), ts.end());
  auto [loops, ivsVector] = createNestedLoops(builder, lbs, ubs, steps);
  // loops add attr
  for (int i=0; i<forDescs.size(); i++) {
    loops[i]->setAttr(FORDESC, builder.getStringAttr(forDescs[i]));
  }
  // 将旧的forop内部的op转移到新的嵌套forOp中
  mlir::affine::AffineForOp innermostForOp = loops.back();
  innermostForOp.getBody()->back().erase();
  spliceHaveBlockOp(innermostForOp, forOp);
  // 需要修改affineMap
  auto oldIv = forOp.getInductionVar();
  std::set<mlir::Operation*> users = getValueUsers(oldIv);
  mlir::AffineExpr sumExpr = getOrderExpr(builder, ivsVector.size());
  // 替换load/store/apply的map
  for (auto user : users) {
    mlir::OpBuilder b(user);
    llvm::SmallVector<mlir::AffineExpr> exprs;
    llvm::SmallVector<mlir::Value> operands;
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
      auto [map, operands] = replaceIndexWithExpr(b, loadOp, sumExpr, oldIv, ivsVector);
      auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), loadOp.getMemref(), map, operands);
      replaceAndErase(newLoadOp, loadOp);
    } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
      auto [map, operands] = replaceIndexWithExpr(b, storeOp, sumExpr, oldIv, ivsVector);
      b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), storeOp.getValue(), storeOp.getMemref(), map, operands);
      storeOp.erase();
    }
  }
  forOp.erase();
  return loops;
}

llvm::SmallVector<mlir::Value> bufferizeLoopCarryVar(mlir::affine::AffineForOp &carryVarLoop, 
                                                      std::vector<mlir::affine::AffineForOp> &loops, 
                                                      MemorySpace ms,
                                                      const std::vector<std::string>& bufDescs) {
  // 将迭代遍历变成buffer，loops为buffer提供索引值
  // loops就是需要上移到循环，规定loops的第一个循环为上移的最外层循环
  std::vector<int64_t> bufferShape;
  llvm::SmallVector<mlir::Value> ivs, bufs;
  for (auto loop : loops) {
    auto [lb, ub, step] = getLoopBoundAndStep(loop);
    bufferShape.push_back(ub);
    ivs.push_back(loop.getInductionVar());
  }
  
  mlir::OpBuilder builder = getBuilder(loops[0]->getParentOp(), Position::begin);
  auto carryVars = carryVarLoop.getRegionIterArgs();
  for (int i=carryVars.size()-1; i>=0; i--) {
    auto desc = bufDescs[carryVars.size()-i-1];
    mlir::Value allocVal = createAllocOp(builder, bufferShape, carryVars[i].getType(), ms, KCG_ALIGNBYTE, desc);
    auto operands = carryVarLoop.getOperands();
    // step1: 将buffer初始化值
    auto cst = operands[operands.size()-i-1];
    mlir::OpBuilder b = getBuilder(cst.getDefiningOp(), Position::after);
    b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), cst, allocVal, ivs);
    bufs.push_back(allocVal);
  }
  // step2: 替换含迭代变量的循环
  eraseForOpIterVar(carryVarLoop, bufs, ivs);
  // step3: init buffer
  decoupleNestedLoop(builder, loops, carryVarLoop, /*uncopy*/false, "initBuf");
  return bufs;
}

void reorder(const std::vector<mlir::affine::AffineForOp>& loops) {
  // 重新排列for循环
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

mlir::affine::AffineParallelOp parallel(std::vector<mlir::affine::AffineForOp> forOps, 
                                        std::string GPUIndexDesc, bool useApply) {
  // X, Y, Z 并行化
  int64_t parallelSize = 1;
  llvm::SmallVector<int64_t> ranges, steps;
  for (auto forOp : forOps) {
    auto [lb, ub, step] = getLoopBoundAndStep(forOp);
    int64_t range = ub / step;
    ranges.push_back(range);
    steps.push_back(step);
    parallelSize *= range;
  }
  // create parallel op
  mlir::OpBuilder builder(forOps[0]);
  mlir::affine::AffineParallelOp parallelOp = builder.create<mlir::affine::AffineParallelOp>(
    builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
    llvm::ArrayRef<int64_t>({parallelSize}));
  // move inner for ops
  spliceHaveBlockOp(parallelOp, forOps.back(), 0, 0, -2);
  // replace old memop
  auto piv = parallelOp.getIVs()[0];
  auto dim = builder.getAffineDimExpr(0);
  for (int i=0; i<forOps.size(); i++) {
    int64_t stride = 1;
    for (int j=i+1; j<forOps.size(); j++) {
      stride *= ranges[j];
    }
    auto expr = dim.floorDiv(stride) * steps[i];
    auto fiv = forOps[i].getInductionVar();
    if (useApply) {
      builder.setInsertionPointToStart(parallelOp.getBody());
      auto map = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
      auto bIdx = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), map, mlir::ValueRange({piv}));
      auto forDesc = getStrAttr(forOps[i], FORDESC);
      if (!forDesc.empty()) {
        bIdx->setAttr(APPLYDESC, builder.getStringAttr(forDesc));
      }
      fiv.replaceAllUsesWith(bIdx.getResult());
    } else {
      auto users = getValueUsers(fiv);
      fuseOneDimExprToLSOp(users, expr, fiv, piv);
    }
    dim = dim % stride;
  }
  // erase for op
  for (auto iter = forOps.rbegin(); iter != forOps.rend(); ++iter) {
    auto forOp = *iter;
    forOp.erase();
  }
  if (!GPUIndexDesc.empty())
    parallelOp->setAttr(AttrGPUIndex, builder.getStringAttr(GPUIndexDesc));
  return parallelOp;
}

void addLoopsToParallel(std::vector<mlir::affine::AffineForOp> loops, 
                        std::vector<mlir::affine::AffineParallelOp> &parallelOps, 
                        bool fuse) {
  // 这个函数封装得太深，我自己也看不懂
  // 将loops添加到parallel到高维，loops {b1, b2} -> parallel {id, b2, b1}(unfuse) or parallel {newid}(fuse)
  // loops data
  llvm::SmallVector<int64_t> steps, forRanges;// loops steps / loops loop num
  for (int i=0; i<loops.size(); i++) {
    auto [lb, ub, step] = getLoopBoundAndStep(loops[i]);
    int64_t range = ub / step;
    forRanges.push_back(range);  // {b1, b2}
    steps.push_back(step);  // {b1, b2}
  }
  // all parallel should be add same loops
  for (int idx=parallelOps.size()-1; idx>=0; idx--) {
    llvm::SmallVector<int64_t> ranges(forRanges.rbegin(), forRanges.rend());  // parallel upperbound
    auto oldRanges = parallelOps[idx].getConstantRanges();  // old parallel upperbound (idx.x)
    int64_t rg = (*oldRanges)[0];  // upperBounds
    ranges.insert(ranges.begin(), rg);  // {id, b2, b1}
    llvm::SmallVector<int64_t> shapes(ranges.rbegin(), ranges.rend());  // {b1, b2, id}
    // new parallel data
    if (fuse) {
      int64_t paraNum = 1;
      for (int i=ranges.size()-1; i>=0; i--) {
        paraNum *= ranges[i];
      }
      ranges.clear();
      ranges.push_back(paraNum);
    }
    // create new parallel
    mlir::OpBuilder builder = getBuilder(parallelOps[idx], Position::before);
    mlir::affine::AffineParallelOp newParallelOp = builder.create<mlir::affine::AffineParallelOp>(
      builder.getUnknownLoc(), mlir::TypeRange(), llvm::ArrayRef<mlir::arith::AtomicRMWKind>(), llvm::ArrayRef<int64_t>(ranges));
    // old parallel attr
    copyAttr(parallelOps[idx], newParallelOp, AttrGPUIndex);
    // replace old parallelOp
    spliceHaveBlockOp(newParallelOp, parallelOps[idx], 0, 0, -2);
    
    auto newIvs = newParallelOp.getIVs();  // {id, b2, b1}  // {newid}
    mlir::AffineExpr dim = builder.getAffineDimExpr(0);
    for (int i=0; i<steps.size(); i++) {  // {b1, b2}
      auto fiv = loops[i].getInductionVar();
      auto users = getValueUsers(fiv, newParallelOp);
      if (fuse) {
        int64_t stride = 1;
        for (int j=i+1; j<shapes.size(); j++) {
          stride *= shapes[j];
        }
        mlir::AffineExpr expr = dim.floorDiv(stride) * steps[i];
        fuseOneDimExprToLSOp(users, expr, fiv, newIvs[0]);
        dim = dim % stride;
      } else {
        mlir::AffineExpr expr = dim * steps[i];
        fuseOneDimExprToLSOp(users, expr, fiv, newIvs[newIvs.size()-1-i]);
      }
      // operations in fuse forop are move out of forop
      auto parentOp = loops[i]->getParentOp();
      int index = getOpIndex(parentOp, loops[i]);
      spliceHaveBlockOp(parentOp, loops[i], index, 0, -2);
    }
    // replace parallel ivs
    auto oldIvs = parallelOps[idx].getIVs();
    if (fuse) {
      auto users = getValueUsers(oldIvs[0], newParallelOp);
      fuseOneDimExprToLSOp(users, dim, oldIvs[0], newIvs[0]);
    } else {
      oldIvs[0].replaceAllUsesWith(newIvs[0]);
    }    
    parallelOps[idx].erase();
    parallelOps[idx] = newParallelOp;
  }
  // erase forOp which fuse into parallel idx.x
  for (auto it = loops.rbegin(); it != loops.rend(); ++it) {
    mlir::affine::AffineForOp forOp = *it;
    forOp.erase();
  }
}


std::vector<mlir::Value> allocBuffers(const std::vector<std::vector<int64_t>>& shapes, 
                                      const std::vector<mlir::Type>& dtypes,
                                      MemorySpace ms, 
                                      const std::vector<std::string>& bufDescs, 
                                      mlir::Operation* op, 
                                      int alignment,
                                      Position pos) {
  // 创建buffer
  mlir::OpBuilder builder = getBuilder(op, pos);
  std::vector<mlir::Value> bufs;
  for (int i=0; i<shapes.size(); i++) {
    auto buf = createAllocOp(builder, shapes[i], dtypes[i], ms, alignment, bufDescs[i]);
    bufs.push_back(buf);
  }
  return bufs;
}

// dst is register.
mlir::affine::AffineForOp loadToRegisters(mlir::Value src, 
                                          mlir::Value dst, 
                                          mlir::AffineMap map, 
                                          llvm::SmallVector<mlir::Value> operands, 
                                          std::vector<int64_t> widths, 
                                          mlir::affine::AffineForOp compute_at, 
                                          Position pos, 
                                          const std::string& forDesc) {
  // 加载数据到寄存器
  auto dimsNum = map.getNumDims();
  auto builder = getBuilder(compute_at, pos);
  auto dstType = mlir::dyn_cast<mlir::MemRefType>(dst.getType());
  int64_t totalWidth = dstType.getShape()[0];

  // 构建reg中的访问map == dim0 * width0 + dim1 + width1 + ...
  std::vector<int> times;
  mlir::AffineExpr expr = builder.getAffineConstantExpr(0);
  for (int i=0; i<widths.size(); i++) {
    auto dim = builder.getAffineDimExpr(i);
    expr = expr + dim * widths[i];
    times.push_back(totalWidth / widths[i]);
    totalWidth = widths[i];
  }
  mlir::AffineMap dstMap;
  if (dimsNum - (operands.size() + times.size()) == 1) {  // 不能用vectorload取数据
    expr = expr + builder.getAffineDimExpr(widths.size());
    dstMap = mlir::AffineMap::get(/*dimCount*/widths.size() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  } else {
    dstMap = mlir::AffineMap::get(/*dimCount*/widths.size(), 0, llvm::ArrayRef<mlir::AffineExpr>(expr), builder.getContext());
  }
  
  llvm::SmallVector<mlir::Value> dstOperands;
  auto load = shiftBufferDatas(builder, src, dst, map, dstMap, operands, dstOperands, widths.back(), times, forDesc);
  return load;
}

// src is register
mlir::affine::AffineForOp loadFromRegisters(mlir::Value src, 
                                mlir::Value dst, 
                                mlir::AffineMap map, 
                                llvm::SmallVector<mlir::Value> operands, 
                                std::vector<int64_t> widths, 
                                mlir::affine::AffineForOp compute_at, 
                                Position pos, 
                                const std::string& forDesc) {
  // write store
  auto dimsNum = map.getNumDims();
  auto builder = getBuilder(compute_at, pos);
  auto srcType = mlir::dyn_cast<mlir::MemRefType>(src.getType());
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
  auto store = shiftBufferDatas(builder, src, dst, srcMap, map, srcOperands, operands, widths.back(), times, forDesc);
  return store;
}

mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos) {
  auto builder = getBuilder(compute_at, pos);
  return builder.create<mlir::gpu::BarrierOp>(builder.getUnknownLoc());
}

mlir::gpu::BarrierOp barrier(mlir::OpBuilder builder) {
  return builder.create<mlir::gpu::BarrierOp>(builder.getUnknownLoc());
}

void cache_read(mlir::affine::AffineForOp scope, 
                mlir::Value src, 
                mlir::Value cached, 
                mlir::AffineMap map, 
                llvm::SmallVector<mlir::Value> operands) {
  // reg read
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineLoadOp load) {
    if (load.getMemref() != src) return;
    mlir::OpBuilder builder(load);
    auto newLoad = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), cached, map, operands);
    load.getResult().replaceAllUsesWith(newLoad.getResult());
    load.erase();
  });
}

void cache_write(mlir::affine::AffineForOp scope, 
                 mlir::Value src, 
                 mlir::Value cached, 
                 mlir::AffineMap map, 
                 llvm::SmallVector<mlir::Value> operands) {
  // reg write
  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    if (store.getMemref() != src) return;
    mlir::OpBuilder builder(store);
    auto newStore = builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), store.getValue(), cached, map, operands);
    store.erase();
  });
}

void separateNoOpRelyForOp(std::vector<mlir::affine::AffineForOp> forOps) {
  // 按照依赖链的条数分离循环
  mlir::OpBuilder builder(forOps[0]);
  // get nest loop datas
  auto [lbs, ubs, steps, oldIvs, forDescs] = getNestedLoopDetailDatas(forOps[0]);
  std::vector<mlir::Value> lastIvs;
  // separate loop
  std::set<mlir::Operation*> oldLoadOps;
  auto chains = getOpRelyChains(forOps.back());
  for (auto chain : chains) {
    // create new chain
    builder.setInsertionPoint(forOps[0]);
    auto [newForOps, newIvs] = createNestedLoops(builder, lbs, ubs, steps, forDescs);
    auto cur = &newForOps.back().getBody()->getOperations().back();
    std::vector<mlir::affine::AffineLoadOp> tempOldLoadOps;
    // move ops
    for (auto op : chain) {
      op->moveBefore(cur);
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
        tempOldLoadOps.push_back(loadOp);
        oldLoadOps.insert(op);
      }
    }
    // create new loadop and replace old loadOp
    for (auto loadOp : tempOldLoadOps) {
      builder.setInsertionPoint(loadOp);
      auto buf = loadOp.getMemref();
      auto map = loadOp.getAffineMap();
      auto operands = loadOp.getMapOperands();
      auto cpLoadOp = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), buf, map, operands);
      std::vector<mlir::Value> oldLoadVals{loadOp.getResult()}, newLoadVals{cpLoadOp.getResult()};
      replaceOpsOperands(newForOps.back(), oldLoadVals, newLoadVals);
    }
    // replace new forIvs with old forIvs
    replaceOpsOperands(newForOps.back(), oldIvs, newIvs);
    replaceOpsOperands(newForOps.back(), lastIvs, newIvs);
    lastIvs = newIvs;
  }
  // erase old loadOp
  for (auto it=oldLoadOps.rbegin(); it!=oldLoadOps.rend(); ++it) {
    mlir::Operation* op = *it;
    op->erase();
  }
  // erase old forOp
  for (auto it=forOps.rbegin(); it!=forOps.rend(); ++it) {
    mlir::affine::AffineForOp forOp = *it;
    forOp.erase();
  }
}

std::vector<mlir::Value> createHierarchyInitBuf(mlir::affine::AffineForOp initForOp,
                                                const std::vector<int64_t>& newShape, 
                                                mlir::Operation* pos, 
                                                MemorySpace space) {
  // 根据提供的glob initbuf 的for创建sm/reg上的initbuf（也可以根据其他level的initbuf创建）
  std::vector<float> csts;
  std::vector<std::string> bufDescs;
  std::vector<mlir::Type> elemTypes;
  std::string initForDesc = getStrAttr(initForOp, FORDESC);
  // collect must datas
  auto innerForOp = getInnerMostOp<mlir::affine::AffineForOp>(initForOp);
  innerForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp storeOp) {
    // get constant value
    auto cstOp = mlir::dyn_cast<mlir::arith::ConstantOp>(storeOp.getValue().getDefiningOp());
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(cstOp.getValue())) {
      csts.push_back(floatAttr.getValueAsDouble());
    }
    // get type
    auto buf = storeOp.getMemRef();
    auto bufType = mlir::dyn_cast<mlir::MemRefType>(buf.getType());
    elemTypes.push_back(bufType.getElementType());
    // get buffer desc
    auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(buf.getDefiningOp());
    bufDescs.push_back(getStrAttr(allocOp, AttrBufDescription));
  });
  mlir::Value tid;
  int64_t thread_num;
  if (auto parallelOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(pos)) {
    tid = parallelOp.getIVs()[0];
    thread_num = (*(parallelOp.getConstantRanges()))[0];
  }
  // create new initbuf
  mlir::OpBuilder builder = getBuilder(pos, Position::begin);
  // create alloc sm buf
  std::vector<mlir::Value> newBufs;
  for (int i=0; i<elemTypes.size(); i++) {
    std::string tempDesc;
    if (space == MemorySpace::shared) {
      tempDesc = "sm" + bufDescs[i];
    } else {
      tempDesc = "reg" + bufDescs[i];
    }
    auto buf = createAllocOp(builder, newShape, elemTypes[i], space, KCG_ALIGNBYTE, tempDesc);
    newBufs.push_back(buf);
  }
  // get forop ub/lb/step datas
  llvm::SmallVector<int64_t> lbs, ubs, steps;
  for (auto dim : newShape) {
    lbs.push_back(0); ubs.push_back(dim); steps.push_back(1);
  }
  int64_t max = -1, maxIdx = -1;
  if (space == MemorySpace::shared) {
    for (int i=0; i<newShape.size(); i++) {
      if (max < newShape[i]) {
        maxIdx = i; max = newShape[i];
      }
    }
    steps[maxIdx] = thread_num;
  }
  // create init forop
  auto [newForOps, newIvs] = createNestedLoops(builder, lbs, ubs, steps);
  newForOps.front()->setAttr(FORDESC, builder.getStringAttr(initForDesc));
  builder.setInsertionPointToStart(newForOps.back().getBody());
  // create ifop
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<newIvs.size(); i++) {
    exprs.push_back(builder.getAffineDimExpr(i));
  }
  if (space == MemorySpace::shared) {
    mlir::AffineExpr expr = max - 1 - builder.getAffineDimExpr(0) - builder.getAffineDimExpr(1);
    auto cst = mlir::IntegerSet::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), llvm::ArrayRef<bool>({false}));
    auto ifOp = builder.create<mlir::affine::AffineIfOp>(builder.getUnknownLoc(), cst, mlir::ValueRange{tid, newIvs[maxIdx]}, false);
    builder.setInsertionPointToStart(ifOp.getThenBlock());
    exprs[maxIdx] = builder.getAffineDimExpr(maxIdx) + builder.getAffineDimExpr(newIvs.size());
    newIvs.push_back(tid);
  }
  //create cstop and store
  for (int i=0; i<csts.size(); i++) {
    auto cstAttr = builder.getFloatAttr(elemTypes[i], csts[i]);
    auto cstOp = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), cstAttr);
    auto map = mlir::AffineMap::get(newIvs.size(), 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
    builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), cstOp, newBufs[i], map, newIvs);
  }
  return newBufs;
}

///TODO: two level vector.
std::vector<std::vector<mlir::affine::AffineForOp>> get_write(mlir::affine::AffineParallelOp parallelLevel, 
                                                              mlir::Value dst) {
  // 
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
    auto type = mlir::dyn_cast<mlir::MemRefType>(load.getMemRef().getType());
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorLoad = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, load.getMemRef(), load.getAffineMap(), load.getMapOperands());
    load.getResult().replaceAllUsesWith(vectorLoad.getResult());
    load.erase();
  });
  readOrWrite.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineStoreOp store) {
    mlir::OpBuilder builder(store);
     auto type = mlir::dyn_cast<mlir::MemRefType>(store.getMemRef().getType());
    auto vectorType = mlir::VectorType::get(width, type.getElementType());
    auto vectorStore = builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), store.getValue(), store.getMemRef(), store.getAffineMap(), store.getMapOperands());
    store.erase();
  });
  return readOrWrite;
}

std::pair<mlir::affine::AffineForOp, 
mlir::affine::AffineForOp> splitUReduce(mlir::Value src, 
                                        mlir::Value dst, 
                                        mlir::AffineMap map, 
                                        llvm::SmallVector<mlir::Value> operands,
                                        int localSplitU, 
                                        int64_t globStoreWidth, 
                                        mlir::affine::AffineForOp compute_at, 
                                        Position pos) {
  // splitU!=1时，插入将多层结果进行累加求和的结构
  auto builder = getBuilder(compute_at, pos);
  auto dstType = mlir::dyn_cast<mlir::MemRefType>(dst.getType());
  int64_t regCTotalWidth = dstType.getShape()[0];   // 16
  int64_t globStoreTotalWidth = regCTotalWidth / localSplitU;  // 8
  int64_t globStoreNum = globStoreTotalWidth / globStoreWidth;  // 4

  auto dim0 = builder.getAffineDimExpr(0);
  auto dim1 = builder.getAffineDimExpr(1);
  auto dstExpr = dim0 * globStoreWidth;
  auto reduceExpr = dim0 * globStoreWidth + dim1;
  auto dstMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dstExpr), builder.getContext());
  auto reduceMap = mlir::AffineMap::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(reduceExpr), builder.getContext());

  auto cstExpr = builder.getAffineConstantExpr(0);
  auto oneMap = replaceExprInMap(builder, map, cstExpr, 1);
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

  auto newMap = addDimToLastExprInMap(builder, map);
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

mlir::affine::AffineForOp splitUWrite(mlir::Value src, 
                                      mlir::Value dst, 
                                      mlir::AffineMap map, 
                                      llvm::SmallVector<mlir::Value> operands, 
                                      int localSplitU, 
                                      int64_t globStoreWidth, 
                                      mlir::affine::AffineForOp compute_at, 
                                      Position pos, 
                                      const std::string& forDesc) {
  // 将结果累加完成后，再将结果写回到C矩阵
  auto builder = getBuilder(compute_at, pos);
  auto dim0 = builder.getAffineDimExpr(0);
  auto srcType = mlir::dyn_cast<mlir::MemRefType>(src.getType());
  int64_t regTotalWidth = srcType.getShape()[0];
  int globStoreNum = regTotalWidth / localSplitU / globStoreWidth;
  mlir::AffineMap srcMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(dim0 * globStoreWidth), builder.getContext());
  llvm::SmallVector<mlir::Value> srcOperands;
  auto store = shiftBufferDatas(builder, src, dst, srcMap, map, srcOperands, operands, globStoreWidth, {globStoreNum}, forDesc);
  return store;
}

mlir::Value bufferCombine(std::vector<mlir::Value> buf1, std::vector<mlir::Value> buf2, std::string bufDesc) {
  // 将buffer合并到一个，“buf1{smA, smB}, buf2{smC}”，smA+smB的大小比较smC的大小，取最大的size创建一维的buffer
  // get max size and offset
  std::vector<std::pair<mlir::Value, int64_t>> bufAndOffsets;
  int64_t buf1Size = 0, buf2Size = 0;
  for (auto buf : buf1) {
    auto bufType = mlir::dyn_cast<mlir::MemRefType>(buf.getType());
    int64_t size = 1;
    for (auto shape : bufType.getShape()) { size *= shape; }
    buf1Size += size;
    bufAndOffsets.push_back(std::make_pair(buf, buf1Size - size));
  }
  for (auto buf : buf2) {
    auto bufType = mlir::dyn_cast<mlir::MemRefType>(buf.getType());
    int64_t size = 1;
    for (auto shape : bufType.getShape()) { size *= shape; }
    buf2Size += size;
    bufAndOffsets.push_back(std::make_pair(buf, buf2Size - size));
  }
  int64_t maxBufSize = buf1Size > buf2Size ? buf1Size : buf2Size;
  // create new allocop
  mlir::OpBuilder b = getBuilder(buf1[0].getDefiningOp()->getParentOp(), Position::begin);
  auto bufType = mlir::dyn_cast<mlir::MemRefType>(buf1[0].getType());
  auto memSpace = static_cast<MemorySpace>(bufType.getMemorySpaceAsInt());
  auto elementType = bufType.getElementType();
  mlir::Value newBuffer = createAllocOp(b, {maxBufSize}, elementType, memSpace, KCG_ALIGNBYTE, bufDesc);

  for (auto bufAndOffset : bufAndOffsets) {
    auto users = getValueUsers(bufAndOffset.first);
    int64_t offset = bufAndOffset.second;
    for (auto user : users) {
      b.setInsertionPointAfter(user);
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
        auto newMap = getOneDimMap(loadOp, offset);
        auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), newBuffer, newMap, loadOp.getMapOperands());
        replaceAndErase(newLoadOp, loadOp);
      } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)) {
        auto newMap = getOneDimMap(storeOp, offset);
        b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), storeOp.getValue(), newBuffer, newMap, storeOp.getMapOperands());
        storeOp.erase();
      } else if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(user)) {
        auto newMap = getOneDimMap(vectorLoadOp, offset);
        auto newVectorLoadOp = b.create<mlir::affine::AffineVectorLoadOp>(b.getUnknownLoc(), vectorLoadOp.getVectorType(), 
                                                                                newBuffer, newMap, vectorLoadOp.getMapOperands());
        replaceAndErase(newVectorLoadOp, vectorLoadOp);
      } else if (auto vectorStoreOp = mlir::dyn_cast<mlir::affine::AffineVectorStoreOp>(user)) {
        auto newMap = getOneDimMap(vectorStoreOp, offset);
        b.create<mlir::affine::AffineVectorStoreOp>(b.getUnknownLoc(), vectorStoreOp.getValue(), 
                                                          newBuffer, newMap, vectorStoreOp.getMapOperands());
        vectorStoreOp.erase();
      }
    }
    mlir::Operation* defOp = bufAndOffset.first.getDefiningOp();
    defOp->erase();
  }
  return newBuffer;
}

std::array<mlir::Value, 2> blockMapping(mlir::affine::AffineParallelOp gridLevel, 
                                        const std::vector<int64_t>& blockTiles,
                                        const std::vector<int64_t>& gridShape, 
                                        int64_t groupM) {
  // 重映射block的位置，提高L2 cache命中率 
  // collect apply
  std::map<std::string, mlir::affine::AffineApplyOp> applyMap;
  for (auto &op : gridLevel.getBody()->getOperations()) {
    if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
      auto applyDesc = getStrAttr(applyOp, APPLYDESC);
      if (!applyDesc.empty()) {
        applyMap.emplace(applyDesc, applyOp);
      }
    }
  }
  auto ivs = gridLevel.getIVs();
  mlir::OpBuilder builder = getBuilder(gridLevel, Position::begin);
  // create expr and map
  mlir::AffineExpr bid = builder.getAffineDimExpr(0);
  mlir::AffineExpr block_mapping = builder.getAffineDimExpr(1);
  mlir::AffineExpr groupMExpr = builder.getAffineConstantExpr(groupM);
  int64_t groupNum = groupM * gridShape[1];
  auto start_y = bid.floorDiv(groupNum) * groupM;

  // create affine minop
  // mlir::SmallVector<mlir::AffineExpr> exprs{start_y, groupMExpr};
  // auto minMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  // auto minOp = builder.create<mlir::affine::AffineMinOp>(builder.getUnknownLoc(), minMap, mlir::ValueRange({ivs[0]}));
  // create new by and bx applyop
  // auto yExpr = (start_y + bid % block_mapping) * blockTiles[0];
  // auto xExpr = (bid % groupNum).floorDiv(block_mapping) * blockTiles[1];
  // auto yMap = mlir::AffineMap::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(yExpr), builder.getContext());
  // auto xMap = mlir::AffineMap::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(xExpr), builder.getContext());
  // auto by = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), yMap, mlir::ValueRange({ivs[0], minOp.getResult()}));
  // auto bx = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), xMap, mlir::ValueRange({ivs[0], minOp.getResult()}));

  auto yExpr = (start_y + bid % groupM) * blockTiles[0];
  auto xExpr = (bid % groupNum).floorDiv(groupM) * blockTiles[1];
  auto yMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(yExpr), builder.getContext());
  auto xMap = mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(xExpr), builder.getContext());
  auto by = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), yMap, mlir::ValueRange({ivs[0]}));
  auto bx = builder.create<mlir::affine::AffineApplyOp>(builder.getUnknownLoc(), xMap, mlir::ValueRange({ivs[0]}));
  by->setAttr(APPLYDESC, builder.getStringAttr("blocky"));
  bx->setAttr(APPLYDESC, builder.getStringAttr("blockx"));
  // replace old apply and remove old apply
  applyMap["blocky"].getResult().replaceAllUsesWith(by.getResult());
  applyMap["blocky"].erase();
  applyMap["blockx"].getResult().replaceAllUsesWith(bx.getResult());
  applyMap["blockx"].erase();
  return {by.getResult(), bx.getResult()};
}

void unrollAttribute(mlir::ModuleOp module, int32_t unrollNum) {
  // 添加unroll属性
  module.walk<mlir::WalkOrder::PostOrder>([&](mlir::affine::AffineForOp forOp) {
    int32_t noBodyOpNum = getNoBodyOpCount(forOp);
    auto [lb, ub, step] = getLoopBoundAndStep(forOp);
    int32_t loopNum = (ub - lb) / step;
    // llvm::outs() << "Count: " << noBodyOpNum << "\n";
    if (noBodyOpNum <= 128) {
      // if (loopNum > unrollNum) {
      //   if (loopNum % unrollNum == 0) {
          mlir::OpBuilder builder(forOp->getContext());
          forOp->setAttr(std::string("affine.loop"), builder.getStringAttr("unroll"));
          forOp->setAttr(std::string("affine.unroll.num"), builder.getI32IntegerAttr(unrollNum));
      //   }
      // } else {
        // mlir::OpBuilder builder(forOp->getContext());
        // forOp->setAttr(std::string("affine.loop"), builder.getStringAttr("unroll"));
        // forOp->setAttr(std::string("affine.unroll.num"), builder.getI32IntegerAttr(unrollNum));
    //   }
    }
  });
}

// ===================================== about double buffer ========================================

std::pair<std::map<mlir::Value, mlir::Value, BufferCompare>, 
std::pair<std::vector<mlir::affine::AffineForOp>, 
std::vector<mlir::affine::AffineForOp>>>
  sharedPrefetch(mlir::affine::AffineForOp &forOp, 
                 std::vector<mlir::affine::AffineForOp> &loadRegForOps, 
                 std::vector<mlir::affine::AffineForOp> &loadSharedForOps, 
                 mlir::affine::AffineForOp &calculateForOp, 
                 std::vector<mlir::Value> buffers) {
  // double buffer save in map
  std::map<mlir::Value, mlir::Value, BufferCompare> doubleBufMaps;
  std::vector<mlir::affine::AffineForOp> newLoadSharedForOps, newLoadRegForOps, perfetchLoadSharedForOps, perfetchLoadRegForOps;
  for (auto buf : buffers) {
    doubleBufMaps.emplace(buf, doubleBuffer(buf));
  }

  // base datas
  mlir::OpBuilder builder = getBuilder(forOp, Position::before);  // forK
  auto [lb, ub, step] = getLoopBoundAndStep(forOp);
  auto k = builder.getAffineDimExpr(0);
  auto cst = mlir::IntegerSet::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>({ub - step - k}), llvm::ArrayRef<bool>({false}));

  // create new main forOp
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  auto mainForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), step, ub + step, step, mlir::ValueRange({}), loopBody);

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

std::pair<std::map<mlir::Value, mlir::Value, BufferCompare>, 
std::pair<std::vector<mlir::affine::AffineForOp>, 
mlir::affine::AffineForOp>>
  registersPrefetch(mlir::affine::AffineForOp &forOp,
                    std::vector<mlir::affine::AffineForOp> &loadRegForOps, 
                    mlir::affine::AffineForOp &calculateForOp, 
                    std::vector<mlir::Value> buffers) {
  // registers double
  std::map<mlir::Value, mlir::Value, BufferCompare> doubleBufMaps;
  for (auto buf : buffers) {
    doubleBufMaps.emplace(buf, doubleBuffer(buf));
  }

  // base datas
  mlir::OpBuilder builder = getBuilder(forOp, Position::before);
  auto [lb, ub, step] = getLoopBoundAndStep(forOp);
  std::vector<mlir::affine::AffineForOp> newLoadRegForOps, perfetchLoadRegForOps;
  // rear outer forBK
  builder.setInsertionPointAfter(forOp);
  auto rearForOp = createRearCalculateForOp(builder, calculateForOp, doubleBufMaps);
  
  auto loopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterArgs) {
    b.create<mlir::affine::AffineYieldOp>(b.getUnknownLoc());
  };
  builder.setInsertionPoint(forOp);
  auto mainForOp = builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 0, ub - step, step, mlir::ValueRange({}), loopBody);

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

void doublePerfetchAdjust(std::vector<mlir::affine::AffineForOp> &shShPerfetchForOps, 
                          std::vector<mlir::affine::AffineForOp> &shRegPerfetchForOps, 
                          std::vector<mlir::affine::AffineForOp> &regPerfetchForOps, 
                          mlir::affine::AffineForOp &rearForOp, 
                          std::vector<mlir::Value> smBufs, 
                          std::vector<mlir::Value> regBufs) {
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

      for (auto op : globToRegOps) {
        if (auto vectorLoadOp = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
          for (auto op_ : regToSharedOps) {
            if (auto vectorLoadOp_ = mlir::dyn_cast<mlir::affine::AffineVectorLoadOp>(op_)) {
              auto ivs = collectNestedIvs(shShPerfetchForOps[i]);
              auto regivs = collectNestedIvs(shRegPerfetchForOps[i]);
              // operands
              if (ivs.size() > regivs.size()) {
                break;
              }
              auto oldOperands = vectorLoadOp.getMapOperands();
              llvm::SmallVector<mlir::Value> newOperands(oldOperands.begin(), oldOperands.end());
              for (int i=0; i<regivs.size(); i++) {
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