#ifndef _General_Funcs_h_
#define _General_Funcs_h_

#include "Common/Utils.h"
#include "Analysis/Analyzer.h"
#include "mlir/Support/LLVM.h"
#include <vector>
#include <tuple>

namespace KernelCodeGen {

mlir::OpBuilder getBuilder(mlir::Operation* op, Position pos);

std::tuple<int64_t, int64_t, int64_t> getLoopBoundAndStep(mlir::affine::AffineForOp loop);

std::vector<mlir::func::FuncOp> getAllKernels(mlir::ModuleOp mod);

std::vector<mlir::func::FuncOp> getSpecifiedKernels(mlir::ModuleOp mod, 
                                                    const std::vector<std::string>& kernelNames);

void swap(mlir::affine::AffineForOp outer, mlir::affine::AffineForOp inner);

mlir::Value createAllocOp(mlir::OpBuilder builder, 
                          std::vector<int64_t> shape, 
                          mlir::Type dtype, 
                          MemorySpace space, 
                          int alignment, 
                          std::string bufDesc);

std::pair<std::vector<mlir::affine::AffineForOp>, 
std::vector<mlir::Value>> 
  createNestedLoops(mlir::OpBuilder builder, 
                    llvm::SmallVector<int64_t> lowerBounds, 
                    llvm::SmallVector<int64_t> upperBounds, 
                    llvm::SmallVector<int64_t> steps);

void replaceAndErase(mlir::Operation* newOp, mlir::Operation* oldOp);

void spliceHaveBlockOp(mlir::Operation* newOp, 
                       mlir::Operation* oldOp, 
                       int insertPos=0, 
                       int startOpIndex=0, 
                       int endOpIndex=-1);

template<typename AttrType>
void copyAttr(mlir::Operation* originOp, 
              mlir::Operation* newOp, 
              std::string attrName) {
  // 复制attr到新的op上
  if (auto desc = originOp->getAttr(attrName)) {
    auto descAttr = mlir::dyn_cast<AttrType>(desc);
    newOp->setAttr(attrName, descAttr);
  }
}

void replaceOpsOperands(mlir::Operation* parentOp, 
                        const std::vector<mlir::Value>& oldIvs, 
                        const std::vector<mlir::Value>& newIvs);

void replaceOpOperands(mlir::Operation* op, 
                      mlir::Value oldOperand, 
                      mlir::Value newOperand);

std::set<mlir::Operation*> getValueUsers(mlir::Value var, mlir::Operation* rangeOp=nullptr);

int getOpIndex(mlir::Operation* haveBlockOp, mlir::Operation* targetOp);

std::vector<mlir::affine::AffineForOp> decoupleNestedLoop(std::vector<mlir::affine::AffineForOp> upLoops, 
                                                          mlir::affine::AffineForOp lowLoop, 
                                                          bool carryDesc=true);

// =================================== AffineMap ====================================

mlir::AffineExpr getOrderExpr(mlir::OpBuilder builder, int dimCount);

bool isContainsDimInExpr(mlir::AffineExpr expr, unsigned dim);

int getGETargetExprNum(mlir::AffineExpr expr, unsigned target);

int getMaxDimInExpr(mlir::AffineExpr expr);

mlir::AffineExpr shiftExprDim(mlir::OpBuilder builder, 
                              mlir::AffineExpr expr, 
                              int shift);

mlir::AffineExpr shiftUpTargetExprDim(mlir::OpBuilder builder, 
                                    mlir::AffineExpr expr, 
                                    int target, 
                                    int shift);

mlir::AffineMap addDimToLastExprInMap(mlir::OpBuilder builder, mlir::AffineMap oldMap);

mlir::AffineExpr replaceExprInExpr(mlir::OpBuilder builder, 
                                 mlir::AffineExpr inExpr, 
                                 mlir::AffineExpr replaceExpr, 
                                 int targetDim, 
                                 int replaceNumberDims);

mlir::AffineMap replaceExprInMap_(mlir::OpBuilder builder, 
                               mlir::AffineMap oldMap, 
                               mlir::AffineExpr replaceExpr, 
                               int targetDim);

mlir::AffineMap replaceExprInMap(mlir::OpBuilder builder, 
                               mlir::AffineMap oldMap, 
                               mlir::AffineExpr replaceExpr, 
                               int targetDim);

// ==================================================================================================

template <typename AffineMemoryOp>
std::pair<mlir::AffineMap, llvm::SmallVector<mlir::Value>> getParaMapAndOperands(mlir::OpBuilder builder, 
                                                                                 AffineMemoryOp memOp, 
                                                                                 mlir::AffineExpr expr, 
                                                                                 mlir::Value fiv, 
                                                                                 mlir::Value piv) {
  // parallel pass需要将affineapply的映射放到affineload或者store内部
  auto oldOperands = memOp.getMapOperands();
  auto oldMap = memOp.getAffineMap();
  int pidx = -1, fidx = -1;
  for (int ii=0; ii<oldOperands.size(); ii++) {
    if (oldOperands[ii] == fiv) fidx = ii;
    if (oldOperands[ii] == piv) pidx = ii;
  }
  llvm::SmallVector<mlir::Value> newOperands(oldOperands.begin(), oldOperands.end());
  mlir::AffineExpr expr_;
  if (pidx < 0) {
    expr_ = shiftExprDim(builder, expr, fidx);
    newOperands[fidx] = piv;
    if (expr_.dyn_cast<mlir::AffineConstantExpr>()) {
      newOperands.erase(newOperands.begin()+fidx);
    }
  } else {
    // if (pidx > fidx) pidx--;  // 如果没有这一行代码的话 {d0, d1, d2} r[d2+2] t[1]  => {d0, [d2+2], d1}
    expr_ = shiftExprDim(builder, expr, pidx);
    newOperands.erase(newOperands.begin()+fidx);
  }
  mlir::AffineMap newMap = replaceExprInMap(builder, oldMap, expr_, fidx);
  return {newMap, newOperands};
}

template <typename AffineMemoryOp>
std::pair<mlir::AffineMap, llvm::SmallVector<mlir::Value>> replaceIndexWithExpr(mlir::OpBuilder builder,
                                                                                AffineMemoryOp memOp, 
                                                                                mlir::AffineExpr expr,
                                                                                mlir::Value oldIv, 
                                                                                std::vector<mlir::Value>& newIvs) {
  // d0 + d1 + d2 + [d3] + d4  => old:d3 & new: [d0 + d1 + d2]  =>  d0 + d1 + d2 + [d3 + d4 + d5] + d6
  auto oldOperands = memOp.getMapOperands();
  auto oldMap = memOp.getAffineMap();
  llvm::SmallVector<mlir::Value> newOperands(oldOperands.begin(), oldOperands.end());
  // 找到oldIv在Operands中的index，对应找到dx的x
  int target = -1;
  for (int i=0; i<oldOperands.size(); i++) {
    if (oldOperands[i] == oldIv) target = i;
  }
  expr = shiftExprDim(builder, expr, target);
  newOperands.erase(newOperands.begin()+target);
  for (int i=newIvs.size()-1; i>=0; i--) {
    newOperands.insert(newOperands.begin()+target, newIvs[i]);
  }
  mlir::AffineMap newMap = replaceExprInMap(builder, oldMap, expr, target);
  return {newMap, newOperands};
}

void fuseOneDimExprToLSOp(std::set<mlir::Operation*> users, 
                          mlir::AffineExpr expr, 
                          mlir::Value oldIv, 
                          mlir::Value newIv);

mlir::affine::AffineForOp shiftBufferDatas(mlir::OpBuilder builder, 
                                           mlir::Value src, mlir::Value dst, 
                                           mlir::AffineMap srcMap, 
                                           mlir::AffineMap dstMap, 
                                           llvm::SmallVector<mlir::Value> srcOperands, 
                                           llvm::SmallVector<mlir::Value> dstOperands, 
                                           int64_t loadWidth, 
                                           std:: vector<int> times);

template <typename AffineMemoryOp>
mlir::AffineMap getOneDimMap(AffineMemoryOp memOp, int64_t offset) {
  // [d0, d1, d2] -> offset + d0 * (shape[1] * shape[2]) + d1 * (shape[2]) + d2
  mlir::OpBuilder builder(memOp);
  auto exprs = memOp.getAffineMap().getResults();
  auto buf = memOp.getMemref();
  auto bufShape = buf.getType().getShape();

  mlir::AffineExpr oneDimExpr = builder.getAffineConstantExpr(offset);
  for (int i=0; i<exprs.size(); i++) {
    int64_t stride = 1;
    for (int j=i+1; j<exprs.size(); j++) { 
      stride *= bufShape[j];
    }
    oneDimExpr = oneDimExpr + exprs[i] * stride;
  }
  return mlir::AffineMap::get(memOp.getMapOperands().size(), 0, llvm::ArrayRef<mlir::AffineExpr>({oneDimExpr}), builder.getContext());
}

template <typename AffineMemoryOp>
int countOtherOperandNum(AffineMemoryOp loadOrStoreOp, int newLoopIvsSize) {
  // 检查 load 或者 store op 中operands除了新建的循环的变量和block或者thread的变量外，还有几个其他的变量
  int num = 0;
  auto operands = loadOrStoreOp.getMapOperands();
  for (int i=0; i<operands.size()-newLoopIvsSize; i++) {
    auto bkv = mlir::dyn_cast<mlir::BlockArgument>(operands[i]);
    mlir::Operation* op = bkv.getOwner()->getParentOp();
    if (auto parallelOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(op)) {
      continue;
    } else {
      num++;
    }
  }
  return num;
}

template <typename loadOrStoreOp>
std::tuple<std::vector<mlir::Value>, mlir::AffineMap, mlir::Value> /*operands, map, buf*/
  getPerfetchMapDatas(mlir::OpBuilder builder, 
                      loadOrStoreOp lSOp, 
                      std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps, 
                      std::vector<mlir::Value> newLoopIvs, 
                      mlir::Value mainIv, 
                      mlir::AffineExpr addExpr) {

  // 这个会对数据转移的load或者store op (也就是forOp中的load和store)进行分析，获取到load或者store op 的map，operands和buf
  // 大类分为两种形式：
  // 1. shared memory的double buffer (在两种forOp中完成，glob->reg; reg->shared)
  // (1) 将数据从glob取到temp的reg中：不是预取(glob map不需要修改，operands的forK变量修改，buf不改)；是预取(glob map设置forK为0，operands去除forK的变量，buf不变)；temp reg只需要修改operands即可
  // (2) 将数据从temp的reg中取到shared中：不是预取(shared map添加一个expr，operands添加forK变量，buf为double buf)；是预取(expr为0，operands不需要forK，buf为double buf)；temp reg同上
  // 2. registers的double buffer (在一种forOp中完成，shared->reg)
  // (1) 将数据从shared取到reg中：
  // new datas
  std::vector<mlir::Value> newOperands;
  mlir::AffineMap newMap;
  mlir::Value newBuf;

  // old datas
  auto buf = lSOp.getMemref();
  auto map = lSOp.getAffineMap();
  auto operands = lSOp.getMapOperands();

  if (bufMaps.count(buf)) {  // 如果lsop的buf在bufmaps中，则为需要进行double buf的load 或者 store op
    newBuf = bufMaps[buf];
    if (!mainIv) {  // 预取
      // operands
      for (auto i=0; i<operands.size() - newLoopIvs.size(); i++) {
        newOperands.push_back(operands[i]);
      }
      for (auto newLoopIv : newLoopIvs) {
        newOperands.push_back(newLoopIv);
      }
      // map
      llvm::SmallVector<mlir::AffineExpr> newExprs;
      for (auto oldExpr : map.getResults()) {
        newExprs.push_back(oldExpr);
      }
      newExprs.insert(newExprs.begin(), builder.getAffineConstantExpr(0));
      newMap = mlir::AffineMap::get(map.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
    } else {  // 非预取
      // operands
      int otherOperandNum = countOtherOperandNum(lSOp, newLoopIvs.size());
      // llvm::outs() << otherOperandNum << "\n";
      // otherOperandNum = 0;
      int otherIndex = operands.size() - newLoopIvs.size() - otherOperandNum;
      for (int i=0; i<otherIndex; i++) {
        newOperands.push_back(operands[i]);
      }
      newOperands.push_back(mainIv);
      for (int i=otherIndex; i<operands.size() - newLoopIvs.size(); i++) {
        newOperands.push_back(operands[i]);
      }
      for (auto newLoopIv : newLoopIvs) {
        newOperands.push_back(newLoopIv);
      }
      // map
      llvm::SmallVector<mlir::AffineExpr> newExprs;
      auto addExprDim = map.getNumDims() - newLoopIvs.size() - otherOperandNum;
      addExpr = shiftExprDim(builder, addExpr, addExprDim);
      newExprs.push_back(addExpr);
      for (auto oldExpr : map.getResults()) {
        auto newExpr = shiftUpTargetExprDim(builder, oldExpr, addExprDim, 1);
        newExprs.push_back(newExpr);
      }
      newMap = mlir::AffineMap::get(map.getNumDims() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
    }

  } else {  // 没有则有可能为 glob 的load 或者中转的 store
    newBuf = buf;
    if (operands.size() == newLoopIvs.size()) {  // 这个为中转的storeOp，中转operand只有nested loop 的ivs
      newMap = map;
      newOperands = newLoopIvs;
    } else {  // 这个是对 glob 的loadOp进行修改，若mainIv为nullptr则是预取的load，需要修改map
      int outerIvSize = operands.size() - newLoopIvs.size() - 1; 
      if (!mainIv) {  // 预取
        // operands
        for (int i=0; i<outerIvSize; i++) {
          newOperands.push_back(operands[i]);
        }
        for (auto iv : newLoopIvs) {
          newOperands.push_back(iv);
        }
        // map
        auto cstExpr = builder.getAffineConstantExpr(0);
        newMap = replaceExprInMap(builder, map, cstExpr, outerIvSize);
      } else {  // 非预取
        // operands
        for (int i=0; i<outerIvSize; i++) {
          newOperands.push_back(operands[i]);
        }
        newOperands.push_back(mainIv);
        for (auto iv : newLoopIvs) {
          newOperands.push_back(iv);
        }
        // check mainIv forOp
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(mainIv);
        mlir::Operation* op = blockArg.getOwner()->getParentOp();
        auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op);
        auto [lb, ub, step] = getLoopBoundAndStep(forOp);
        // map
        if (lb == 0) {
          mlir::AffineExpr expr = builder.getAffineDimExpr(outerIvSize) + builder.getAffineConstantExpr(step);
          newMap = replaceExprInMap(builder, map, expr, outerIvSize);
        } else {
          newMap = map;
        }
      }
    }
  }
  return std::make_tuple(newOperands, newMap, newBuf);
}

template <typename loadOp>
std::tuple<std::vector<mlir::Value>, mlir::AffineMap, mlir::Value> /*operands, map, buf*/
  getCalculateMapDatas(mlir::OpBuilder builder, 
                       loadOp lSOp, 
                       std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps,
                       mlir::Value mainIv, 
                       mlir::AffineExpr addExpr) {
  // 计算部分修改的全是load op，而且都是修改buf为double buf，且map多一个expr，operands多一个和ForK/BK相关的变量

  // old datas
  auto buf = lSOp.getMemref();
  auto map = lSOp.getAffineMap();
  auto operands = lSOp.getMapOperands();

  // new datas
  std::vector<mlir::Value> newOperands;
  mlir::AffineMap newMap;
  mlir::Value newBuf = nullptr;

  if (bufMaps.count(buf)) {
    newBuf = bufMaps[buf];

    // map
    int noParallelOperandNum = countOtherOperandNum(lSOp, 0);
    // llvm::outs() << noParallelOperandNum << "\n";
    // noParallelOperandNum = 0;
    llvm::SmallVector<mlir::AffineExpr> newExprs;
    auto addExprDim = map.getNumDims() - noParallelOperandNum;
    addExpr = shiftExprDim(builder, addExpr, addExprDim);
    newExprs.push_back(addExpr);
    for (auto oldExpr : map.getResults()) {
      auto newExpr = shiftUpTargetExprDim(builder, oldExpr, addExprDim, 1);
      newExprs.push_back(newExpr);
    }
    newMap = mlir::AffineMap::get(map.getNumDims()+1, 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());

    // operands
    for (int i=0; i<addExprDim; i++) {
      newOperands.push_back(operands[i]);
    }
    newOperands.push_back(mainIv);
    for (int i=addExprDim; i<map.getNumDims(); i++) {
      newOperands.push_back(operands[i]);
    }
  }
  return std::make_tuple(newOperands, newMap, newBuf);
}

template <typename loadOp>
std::pair<std::vector<mlir::Value>, mlir::AffineMap> /*operands, map*/ 
  getRegPerfetchOuterAdjustDatas(mlir::OpBuilder builder, 
                                 loadOp lSOp, 
                                 int nestedNum) {
  // 获取寄存器预取将其调整到forK外所需要的map operands buf
  auto cst0 = builder.getAffineConstantExpr(0);
  llvm::SmallVector<mlir::AffineExpr> newExprs{cst0};

  auto operands = lSOp.getMapOperands();
  int index = operands.size() - nestedNum - 1;
  auto map = lSOp.getAffineMap();
  for (int i=1; i<map.getNumResults(); i++) {
    auto newExpr = replaceExprInExpr(builder, map.getResult(i), cst0, index, 0);
    newExprs.push_back(newExpr);
  }
  auto newMap = mlir::AffineMap::get(map.getNumDims() - 1, 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());

  std::vector<mlir::Value> newOperands(operands.begin(), operands.end());
  newOperands.erase(newOperands.begin()+1);
  
  return std::make_pair(newOperands, newMap);
}

template <typename loadOp>/*map*/ 
mlir::AffineMap getRegPerFetchInnerAdjustDatas(mlir::OpBuilder builder, 
                                               loadOp lSOp, 
                                               mlir::Value outerForIv, 
                                               int step) {
  // 移动内部寄存器预取的for循环到forK的最后，提供map即可
  int dim = 0;
  auto oldOperands = lSOp.getMapOperands();
  for (auto oldOperand : oldOperands) {
    if (oldOperand == outerForIv) break;
    dim++;
  }
  llvm::SmallVector<mlir::AffineExpr> newExprs;
  auto map = lSOp.getAffineMap();
  for (int i=0; i<map.getNumResults(); i++) {
    if (i == 0) {
      auto newExpr = builder.getAffineDimExpr(dim).floorDiv(step) % 2;
      newExprs.push_back(newExpr);
    } else {
      newExprs.push_back(map.getResult(i));
    }
  }
  return mlir::AffineMap::get(map.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
}

int getLoopNestedNum(mlir::Operation* forOp); 

std::vector<mlir::Value> collectNestedIvs(mlir::affine::AffineForOp forOp);

void eraseForOpIterVar(mlir::affine::AffineForOp &forOp, 
                       llvm::SmallVector<mlir::Value> bufs, 
                       llvm::SmallVector<mlir::Value> ivs);

template <typename collectOp>
std::vector<collectOp> collectInnerOps(mlir::Operation* haveBlockOp) {
  // 收集含有block的op的下面指定的op
  std::vector<collectOp> ops;
  haveBlockOp->walk<mlir::WalkOrder::PreOrder>([&](collectOp op) {
    ops.push_back(op);
  });
  return ops;
}

template <typename haveBodyOp>
haveBodyOp getInnerMostOp(haveBodyOp imop) {
  // 找个最内层的那个op，这些算子都是含有block的op
  for (auto &op : imop.getBody()->getOperations()) {
    if (auto reOp = mlir::dyn_cast<haveBodyOp>(op)) {
      return getInnerMostOp(reOp);
    }
  }
  return imop;
}

template <typename haveBodyOp>
std::vector<mlir::Operation*> collectInnerMostAllOps(haveBodyOp haveBlockOp) {
  // 收集最里层的ops
  std::vector<mlir::Operation*> ops;
  auto bodyOp = getInnerMostOp(haveBlockOp);
  for (auto &op : bodyOp.getBody()->getOperations()) {
    ops.push_back(&op);
  }
  ops.pop_back();
  return ops;
}

mlir::Value doubleBuffer(mlir::Value buffer);

std::tuple<llvm::SmallVector<int64_t>, 
llvm::SmallVector<int64_t>, 
llvm::SmallVector<int64_t>> 
  getNestedLoopData(mlir::affine::AffineForOp forOp);

std::vector<mlir::affine::AffineForOp> createNewDataShiftForOp(mlir::OpBuilder builder, 
                                                               std::vector<mlir::affine::AffineForOp> forOps, 
                                                               std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps, 
                                                               mlir::Value mainIv=nullptr, 
                                                               mlir::AffineExpr 
                                                               addExpr=nullptr);

void moveCalculateForOp(mlir::Operation* posOp, 
                        mlir::affine::AffineForOp &forOp, 
                        std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps, 
                        mlir::Value mainIv, 
                        mlir::AffineExpr addExpr);

mlir::affine::AffineForOp createRearCalculateForOp(mlir::OpBuilder builder, 
                                                   mlir::affine::AffineForOp calculateForOp, 
                                                   std::map<mlir::Value, mlir::Value, BufferCompare> bufMaps);

int32_t getNoBodyOpCount(mlir::Operation* op);

}

#endif