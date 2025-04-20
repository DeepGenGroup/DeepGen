#include "Conversion/Fusing.h"

namespace KernelCodeGen {

std::vector<std::vector<mlir::affine::AffineForOp>> getBatchFors(const std::vector<mlir::func::FuncOp>& fks) {
  // 获取fuse kernel中的所有batch for
  std::vector<std::vector<mlir::affine::AffineForOp>> funcBatchs;
  for (auto fk : fks) {
    std::vector<mlir::affine::AffineForOp> batchs;
    fk.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      auto result = getStrAttr(forOp, FORDESC);
      if (result == "batch") {
        batchs.push_back(forOp);
      }
    });
    funcBatchs.push_back(batchs);
  }
  return funcBatchs;
}

std::tuple<mlir::func::FuncOp, 
std::vector<mlir::Value>, 
std::vector<mlir::Value>> createFuseFuncAndMidMems(mlir::OpBuilder& builder, 
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
  mlir::func::FuncOp newFuncOp = buildFunction(builder, fkd.name, fkd.type, newInputTypes, fkd.isTranspose, fkd.paraDims, fkd.outputArgNum);
  auto funcArgs_ = newFuncOp.getArguments();
  // create global var
  std::vector<mlir::Value> midVars, funcArgs{funcArgs_.begin(), funcArgs_.end()};
  for (int i=0; i<fkd.midVarShapes.size(); i++) {
    auto mlirType = tools::getDType(builder, fkd.midVarDtypes[i]);
    auto memOp = createAllocOp(builder, fkd.midVarShapes[i], mlirType, MemorySpace::global, /*none*/0, "midBuf_"+std::to_string(i));
    midVars.push_back(memOp);
  }
  return {newFuncOp, funcArgs, midVars};
}

std::vector<std::vector<mlir::Value>> collectOldMems(const std::vector<std::map<std::string, std::vector<int64_t>>>& newMemsIndex, 
                                                     const std::vector<mlir::func::FuncOp>& fks) {
  // newMemsIndex 每一个item对应一个新的函数签名变量，新的变量需要替换掉旧的函数签名变量，采用新变量索引->func(旧索引)的方式获取旧的 memroy value
  std::vector<std::vector<mlir::Value>> memsVec;
  for (int i=0; i<newMemsIndex.size(); i++) {
    std::vector<mlir::Value> temp;
    auto itemMap = newMemsIndex[i];
    for (auto oldFuncOp : fks) {
      auto name = oldFuncOp.getName().str();
      if (itemMap.count(name)) {
        for (auto idx : itemMap[name]) {
          auto arg = oldFuncOp.getBody().getArgument(idx);
          temp.push_back(arg);         
        }
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

void normalizeParaForOp(std::vector<mlir::affine::AffineForOp>& yloops) {
  // 规范化所有的并行循环，保证fory下就是forx
  std::vector<int> idxs;
  std::vector<mlir::affine::AffineForOp> newLoops;
  for (int i=0; i<yloops.size(); i++) {
    int index = 0;
    yloops[i].walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      auto result = getStrAttr(forOp, FORDESC);
      if (result == "x") {
        if (index > 0) {
          mlir::OpBuilder b = getBuilder(yloops[i], Position::before);
          auto newY = decoupleNestedLoop(b, {yloops[i]}, forOp);
          newLoops.push_back(newY[0]);
          idxs.push_back(i);
        }
        index++;
      }
    });
  }
  // update yloops and paraCfg
  for (int i=0; i<idxs.size(); i++) {
    yloops.insert(yloops.begin() + idxs[idxs.size()-1-i], newLoops[newLoops.size()-1-i]);
  }
}

std::vector<mlir::affine::AffineForOp> collectIndexForOps(mlir::affine::AffineForOp parentForOp) {
  // 收集用来表示index的forop，forx/y/k
  std::vector<mlir::affine::AffineForOp> forOps;
    parentForOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
      auto forDesc = getStrAttr(forOp, FORDESC);
      if (forDesc == "y") forOps.push_back(forOp);
      else if (forDesc == "x") forOps.push_back(forOp);
      else if (forDesc == "k") forOps.push_back(forOp);
    });
    return forOps;
}

template<typename loadOrStoreOp>
std::vector<loadOrStoreOp> collectLoadOrStoreOps(mlir::affine::AffineForOp forOp) {
  // 收集loadop或者storeop
  std::vector<loadOrStoreOp> lsOps;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (auto lsOp = mlir::dyn_cast<loadOrStoreOp>(op)) {
      lsOps.push_back(lsOp);
    }
  }
  return lsOps;
}

std::vector<mlir::Operation*> collectOtherOps(mlir::affine::AffineForOp forOp) {
  // 收集除loadop、storeop之外的算子
  std::vector<mlir::Operation*> otherOps;
  for (auto &op : forOp.getBody()->getOperations()) {
    if (!mlir::dyn_cast<mlir::affine::AffineLoadOp>(op) && 
        !mlir::dyn_cast<mlir::affine::AffineStoreOp>(op) && 
        !mlir::dyn_cast<mlir::affine::AffineYieldOp>(op)) {
      otherOps.push_back(&op);
    }
  }
  return otherOps;
}

bool isReduce(mlir::affine::AffineForOp foryOp, 
              std::vector<mlir::affine::AffineLoadOp> loadOps, 
              std::vector<mlir::affine::AffineStoreOp> storeOps) {
  // 判断是否是reduce类型算子
  for (auto storeOp : storeOps) {
    auto storeBuf = storeOp.getMemRef();
    auto storeMap = storeOp.getAffineMap();
    auto storeMapOperands = storeOp.getMapOperands();
    for (auto loadOp : loadOps) {
      auto loadBuf = loadOp.getMemRef();
      auto loadMap = loadOp.getAffineMap();
      auto loadMapOperands = loadOp.getMapOperands();
      if (loadBuf == storeBuf && getOpIndex(storeOp->getParentOp(), storeOp) > getOpIndex(loadOp->getParentOp(), loadOp)) {  // buf/map是一样的
        if (loadMap == storeMap && loadMapOperands.size() == 1) {  // 
          mlir::Value iv = foryOp.getInductionVar();
          if (iv == storeMapOperands[0] && iv == loadMapOperands[0]) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void separate(std::vector<mlir::affine::AffineForOp> forOps, 
              std::vector<mlir::affine::AffineStoreOp> storeOps, 
              std::vector<mlir::Operation*> calculateOps) {
  // 分离foryx
  sortOps(calculateOps);
  for (auto calculateOp : calculateOps) {
    auto result = calculateOp->getResult(0);
    for (auto user : getValueUsers(result)) {
      auto it = std::find(calculateOps.begin(), calculateOps.end(), user);
      if (!mlir::dyn_cast<mlir::affine::AffineStoreOp>(user) && it != calculateOps.end()) {
        auto b = getBuilder(calculateOp, Position::after);
        auto buf = storeOps[0].getMemRef();
        auto map = storeOps[0].getAffineMap();
        auto mapOperands = storeOps[0].getMapOperands();
        b.create<mlir::affine::AffineStoreOp>(b.getUnknownLoc(), result, buf, map, mapOperands);
        auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(b.getUnknownLoc(), buf, map, mapOperands);
        replaceOpOperands(user, result, newLoadOp.getResult());
      }
    }
  }
  // move data
  llvm::SmallVector<int64_t> lbs, ubs, steps;
  std::vector<mlir::Value> oldIvs;
  for (auto loop : forOps) {
    auto [lb, ub, step] = getLoopBoundAndStep(loop);
    oldIvs.push_back(loop.getInductionVar());
    lbs.push_back(lb);
    ubs.push_back(ub);
    steps.push_back(step);
  }
  for (int i=0; i<calculateOps.size()-1; i++) {
    auto b = getBuilder(forOps[0], Position::before);
    auto [newLoops, newIvs] = createNestedLoops(b, lbs, ubs, steps);
    for (int i=0; i<newLoops.size(); i++) {
      copyAttrs(forOps[i], newLoops[i]);
    }
    int idx = getOpIndex(forOps.back(), calculateOps[i]);
    spliceHaveBlockOp(newLoops.back(), forOps.back(), 0, idx, idx+1);
    auto operands = calculateOps[i]->getOperands();
    for (auto operand : operands) {
      auto defOp = operand.getDefiningOp();
      defOp->moveBefore(calculateOps[i]);
    }
    auto result = calculateOps[i]->getResult(0);
    auto users = getValueUsers(result);
    (*users.begin())->moveAfter(calculateOps[i]);
    replaceOpsOperands(newLoops.back(), oldIvs, newIvs);
  }
}

void separateParaForOps(mlir::func::FuncOp funcOp) {
  // 这个函数将融合算子进行拆分，生成只有reduce、elem-wise、binary、matmul等层次的forx/y表示
  // 目前为了快速完成attention，先将这个地方写死，后续再进行细致的优化
  auto yLoops = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, std::string{"y"});
  for (auto yloop : yLoops) {
    auto forOps = collectIndexForOps(yloop);
    if (forOps.size() < 3) {  // 不是矩阵乘法
      auto loadOps = collectLoadOrStoreOps<mlir::affine::AffineLoadOp>(forOps.back());
      auto storeOps = collectLoadOrStoreOps<mlir::affine::AffineStoreOp>(forOps.back());
      auto calculateOps = collectOtherOps(forOps.back());
      if (!isReduce(yloop, loadOps, storeOps)) {   // 不是reduce （其实有些不是reduce的算子也还可以拆开，但是目前只考虑这么多）
        separate(forOps, storeOps, calculateOps);
      }
    }
  }
}

bool isValueVecEqual(mlir::ValueRange operands1, mlir::ValueRange operands2) {
  // 判断这个两个operands的value是否全部相等
  if (operands1.size() != operands2.size()) { return false; }
  for (int i=0; i<operands1.size(); i++) {
    if (operands1[i] != operands2[i]) { return false; }
  }
  return true;
}

bool isUseLoadOp(mlir::affine::AffineStoreOp fstoreOp) {
  // 检查整个func中，其位于这个parentOp(fstoreOp的parentOp)之后的位置还有没有使用loadop
  mlir::func::FuncOp funcOp;
  mlir::Operation *parentOp = fstoreOp->getParentOp();
  while(true) {
    if (funcOp = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)) {
      break;
    }
    parentOp = parentOp->getParentOp();
  }
  // 遍历func
  bool isfind = false, result = false;
  auto fsbuf = fstoreOp.getMemRef();
  auto fsmap = fstoreOp.getAffineMap();
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (mlir::dyn_cast<mlir::affine::AffineStoreOp>(op) && op == fstoreOp.getOperation()) {
      isfind = true;
      return;
    }
    if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
      if (isfind) {
        auto buf = loadOp.getMemRef();
        auto map = loadOp.getAffineMap();
        if (fsbuf == buf && fsmap == map) {
          result = true;
        }
      }
    }
  });
  return result;
}

void eliminate(std::vector<mlir::affine::AffineLoadOp> &floadOps, 
               std::vector<mlir::affine::AffineStoreOp> &fstoreOps, 
               std::vector<mlir::affine::AffineLoadOp> &bloadOps, 
               std::vector<mlir::affine::AffineStoreOp> &bstoreOps) {
  // eliminate
// eliminate redundant load
  for (int i=bloadOps.size()-1; i>=0; i--) {
    auto bloadOp = bloadOps[i];
    auto mapOperands = bloadOp.getMapOperands();
    auto map = bloadOp.getAffineMap();
    // 前面循环中的storeop与后循环中的loadop是否匹配上
    for (int j=fstoreOps.size()-1; j>=0; j--) {
      auto fstoreOp = fstoreOps[j];
      auto fmapOperands = fstoreOp.getMapOperands();
      auto fmap = fstoreOp.getAffineMap();
      if (fmap == map && isValueVecEqual(mapOperands, fmapOperands)) {
        bloadOp.getResult().replaceAllUsesWith(fstoreOp.getValue());
        bloadOp.erase();
        // 匹配上了就删除前面循环的loadop，同时检查后循环中是否有相同的storeop，有的话就删除前循环的storeop
        bool deled = false;
        for (int k=bstoreOps.size()-1; k>=0; k--) {
          auto bstoreOp = bstoreOps[k];
          auto bmapOperands = bstoreOp.getMapOperands();
          auto bmap = bstoreOp.getAffineMap();
          if (fmap == bmap && isValueVecEqual(fmapOperands, bmapOperands)) {
            fstoreOp.erase();
            fstoreOps.erase(fstoreOps.begin()+j);
            deled = true;
            break;
          }
        }
         // 后循环内部检查没有相同的storeop，再检查整个module在这个循环后面是否再次使用了loadop，使用了就不能删除storeop
        if (!deled && !isUseLoadOp(fstoreOp)) {
          fstoreOp.erase();
          fstoreOps.erase(fstoreOps.begin()+j);
        }
        // 
        bloadOps.erase(bloadOps.begin()+i);
        break;
      }
    }
  }
  // fload 替换 bload
  for (int i=bloadOps.size()-1; i>=0; i--) {
    auto bloadOp = bloadOps[i];
    auto mapOperands = bloadOp.getMapOperands();
    auto map = bloadOp.getAffineMap();
    for (int j=floadOps.size()-1; j>=0; j--) {
      auto floadOp = floadOps[i];
      auto fmapOperands = floadOp.getMapOperands();
      auto fmap = floadOp.getAffineMap();
      if (fmap == map && isValueVecEqual(mapOperands, fmapOperands)) {
        replaceAndErase(floadOp, bloadOp);
        bloadOps.erase(bloadOps.begin()+i);
        break;
      }
    }
  }
}

void fuse(mlir::affine::AffineForOp yloop1, mlir::affine::AffineForOp yloop2) {
  // 融合两个yfor
  auto fForOps = collectIndexForOps(yloop1);
  auto bForOps = collectIndexForOps(yloop2);
  // collect loadop or storeop
  auto floadOps = collectLoadOrStoreOps<mlir::affine::AffineLoadOp>(fForOps[1]);
  auto fstoreOps = collectLoadOrStoreOps<mlir::affine::AffineStoreOp>(fForOps[1]);
  auto bloadOps = collectLoadOrStoreOps<mlir::affine::AffineLoadOp>(bForOps[1]);
  auto bstoreOps = collectLoadOrStoreOps<mlir::affine::AffineStoreOp>(bForOps[1]);
  // fuse first forOp with second forOp
  std::pair<int, int> idsx{1, 0};
  std::vector<std::vector<mlir::affine::AffineForOp>> loops{{fForOps[0], fForOps[1]}, {bForOps[0], bForOps[1]}};
  auto newFors = fuseForOps(loops, idsx, Position::after);
  eliminate(floadOps, fstoreOps, bloadOps, bstoreOps);
}

void fuseParaForOps(mlir::func::FuncOp funcOp) {
  // fuse parallel forOp
  // 收集func中的fory循环，因为fory循环下必定是forx以及其他的for循环
  // ================ 目前只有attention的融合工作，所以先将attention使用手融合解决 =============
  auto yLoops = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, std::string{"y"});
  fuse(yLoops[0], yLoops[1]);  // matmul1 + front softmax(reduce)

  // 自我融合，我说fuse哪个就fuse哪个
  yLoops = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, std::string{"y"});
  fuse(yLoops[1], yLoops[2]);   // back softmax sub(binary) and exp(elem-wise) 

  // div move to matmul 2/3
  yLoops = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, std::string{"y"});
  auto fForOps = collectIndexForOps(yLoops[2]);
  auto bForOps = collectIndexForOps(yLoops[3]);  // matmul2
  // collect loadop or storeop
  auto floadOps = collectLoadOrStoreOps<mlir::affine::AffineLoadOp>(fForOps[1]);
  auto fstoreOps = collectLoadOrStoreOps<mlir::affine::AffineStoreOp>(fForOps[1]);
  auto bloadOps = collectLoadOrStoreOps<mlir::affine::AffineLoadOp>(bForOps[2]);
  auto bstoreOps = collectLoadOrStoreOps<mlir::affine::AffineStoreOp>(bForOps[2]);
  // move
  std::vector<mlir::Value> oldIvs{fForOps[0].getInductionVar(), fForOps[1].getInductionVar()};  // y, x
  std::vector<mlir::Value> newIvs{bForOps[0].getInductionVar(), bForOps[2].getInductionVar()};  // y, k
  spliceHaveBlockOp(bForOps.back(), fForOps.back(), 0, 0, -2);
  replaceOpsOperands(bForOps.back(), oldIvs, newIvs);
  eliminate(floadOps, fstoreOps, bloadOps, bstoreOps);
  yLoops[2].erase();
  // a*b0*c0 + a*b1*c1 + ... + a*bk*ck == a*(b0*c0 + b1*c1 + ... + bk*ck)
  std::vector<mlir::Value> matmul_ivs;
  for (auto loop : bForOps) {
    matmul_ivs.push_back(loop.getInductionVar());
  }
  bloadOps = collectLoadOrStoreOps<mlir::affine::AffineLoadOp>(bForOps[2]);
  for (auto bloadOp : bloadOps) {
    auto mapOperands = bloadOp.getMapOperands();
    int ivCount = 0;
    for (auto operand : mapOperands) {
      auto it = std::find(matmul_ivs.begin(), matmul_ivs.end(), operand);
      if (it != matmul_ivs.end()) ivCount++;
    }
    if (ivCount == 1) {  // 找到了iv为1的loadop
      auto result = bloadOp.getResult();
      auto users = getValueUsers(result);
      auto calculateOp = *users.begin();
      auto operands = calculateOp->getOperands();
      if (operands.size() == 2) {  // 计算的算子如果是binary算子
        mlir::affine::AffineLoadOp otherLoadOp;
        for (auto operand : operands) {
          if (operand != result) {
            otherLoadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(operand.getDefiningOp());
            break;
          }
        }
        if (otherLoadOp) {
          calculateOp->getResult(0).replaceAllUsesWith(otherLoadOp.getResult());
        }
        // move calculateOp and bloadOp to before storeop
        if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(bForOps[2]->getNextNode())) {
          bloadOp->moveBefore(storeOp);
          calculateOp->moveBefore(storeOp);
          replaceOpOperands(calculateOp, otherLoadOp.getResult(), storeOp.getValue());
          replaceOpOperands(storeOp, storeOp.getValue(), calculateOp->getResult(0));
        }
      }
    }
  }
}

}