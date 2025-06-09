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

bool isForOpArgsEqual(mlir::affine::AffineForOp forOp1, mlir::affine::AffineForOp forOp2);

std::vector<mlir::func::FuncOp> getAllKernels(mlir::ModuleOp mod);

std::vector<mlir::func::FuncOp> getSpecifiedKernels(mlir::ModuleOp mod, 
                                                    const std::vector<std::string>& kernelNames);

void swap(mlir::affine::AffineForOp outer, mlir::affine::AffineForOp inner);

mlir::Value createAllocOp(mlir::OpBuilder builder, 
                          const std::vector<int64_t>& shape, 
                          mlir::Type dtype, 
                          MemorySpace space, 
                          int alignment, 
                          std::string bufDesc);

std::pair<std::vector<mlir::affine::AffineForOp>, 
std::vector<mlir::Value>> 
  createNestedLoops(mlir::OpBuilder builder, 
                    llvm::SmallVector<int64_t> lowerBounds, 
                    llvm::SmallVector<int64_t> upperBounds, 
                    llvm::SmallVector<int64_t> steps, 
                    const std::vector<std::string>& forDescs={});

std::vector<mlir::affine::AffineForOp> fuseForOps(std::vector<std::vector<mlir::affine::AffineForOp>> forOps, 
                                                  std::pair<int, int> idxs={0, 0},
                                                  Position insertPos=Position::before,
                                                  const std::pair<std::string, std::string>& setAttr={});

void replaceAndErase(mlir::Operation* newOp, mlir::Operation* oldOp);

void spliceHaveBlockOp(mlir::Operation* newOp, 
                       mlir::Operation* oldOp, 
                       int insertPos=0, 
                       int startOpIndex=0, 
                       int endOpIndex=-1);

std::string getStrAttr(mlir::Operation* op, std::string attrName);

std::vector<std::string> getArrayStrAttr(mlir::Operation* op, 
                                         std::string attrName);

void copyAttr(mlir::Operation* originOp, 
              mlir::Operation* newOp, 
              const std::string& attrName);

void copyAttrs(mlir::Operation* originOp, 
               mlir::Operation* newOp, 
               const std::vector<std::string>& excludeAttrs={});

template<typename operation>
std::vector<operation> collectOpsInfuncOp(mlir::func::FuncOp funcOp, 
                                          const std::string& attrName, 
                                          const std::string& forDesc) {
  // 获取func中任意operation
  std::vector<operation> ops;
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](operation op) {
    if (auto desc = op->getAttr(attrName)) {
      auto descAttr = mlir::dyn_cast<mlir::StringAttr>(desc);
      auto descStr = descAttr.getValue().str();
      if (descStr == forDesc) {
        ops.push_back(op);
      }
    }
  });
  return ops;
}

void replaceOpsOperands(mlir::Operation* parentOp, 
                        const std::vector<mlir::Value>& oldIvs, 
                        const std::vector<mlir::Value>& newIvs);

void replaceOpOperands(mlir::Operation* op, 
                      mlir::Value oldOperand, 
                      mlir::Value newOperand);

void sortOps(std::vector<mlir::Operation*>& ops);

std::set<mlir::Operation*> getValueUsers(mlir::Value var, mlir::Operation* rangeOp=nullptr);

int getOpIndex(mlir::Operation* haveBlockOp, mlir::Operation* targetOp);

std::vector<mlir::affine::AffineForOp> decoupleNestedLoop(mlir::OpBuilder& builder,
                                                          std::vector<mlir::affine::AffineForOp> upLoops, 
                                                          mlir::affine::AffineForOp lowLoop, 
                                                          bool copyDesc=true, 
                                                          const std::string setDesc="");

bool isPrevOp(mlir::Operation* prevOp, mlir::Operation* backOp);

void eraseSingleIterForOp(mlir::affine::AffineForOp forOp);

std::vector<std::vector<mlir::Operation*>> getOpRelyChains(mlir::affine::AffineForOp forOp);

// ======================================== redece ================================================
using reduceFunc = llvm::function_ref<std::vector<mlir::Value>(mlir::OpBuilder &, std::vector<mlir::Value>, std::vector<mlir::Value>)>;

mlir::affine::AffineForOp warpReduce(mlir::OpBuilder &builder,
                                     int64_t ydim, 
                                     int64_t width, 
                                     const std::vector<mlir::Value>& bufs, 
                                     reduceFunc calculateFunc);

mlir::affine::AffineForOp blockReduce(mlir::OpBuilder &builder,
                                     int64_t ydim, 
                                     int64_t width, 
                                     mlir::Value tid,
                                     const std::vector<mlir::Value>& regBufs, 
                                     const std::vector<mlir::Value>& smBufs, 
                                     reduceFunc calculateFunc);

mlir::affine::AffineForOp warpBroadcast(mlir::OpBuilder &builder, 
                                        int64_t ydim, 
                                        int64_t width,
                                        const std::vector<mlir::Value>& bufs, 
                                        int64_t index);

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

mlir::AffineMap insertExprToMap(mlir::OpBuilder builder, 
                                mlir::AffineMap oldMap, 
                                mlir::AffineExpr expr, 
                                int index);

mlir::AffineExpr replaceExprInExpr(mlir::OpBuilder builder, 
                                 mlir::AffineExpr inExpr, 
                                 mlir::AffineExpr replaceExpr, 
                                 int targetDim, 
                                 int replaceNumberDims);

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
    if (mlir::dyn_cast<mlir::AffineConstantExpr>(expr_)) {
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
                                           std:: vector<int> times, 
                                           const std::string& forDesc);

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

// ======================================= double bffer =========================================

enum class ShiftType {
  GTRInner = 0,
  GTRPrefetch = 1,
  RTSInner = 2,
  RTSPrefetch = 3,
  STRInner = 4,
  STRPrefetch = 5,
};

template <typename loadOrStoreOp>
std::tuple<std::vector<mlir::Value>, mlir::AffineMap, mlir::Value> /*operands, map, buf*/
  getDataShiftMapInfos(mlir::OpBuilder builder, 
                      loadOrStoreOp lSOp, 
                      const std::map<mlir::Value, mlir::Value, ValueCompare>& bufMap,      // {sm1, sm2} / {reg1, reg2}
                      const std::map<mlir::Value, mlir::Value, ValueCompare>& allIvsMap,   // {oldiv0 : newiv0, oldiv1 : newiv1, ...}
                      const std::pair<mlir::Value, mlir::Value>& forKIv,
                      ShiftType sType,
                      mlir::AffineExpr addExpr) {
  /* 计算以下部分的map信息（数据移动）
   1. glob to reg (if/inner)
   2. reg to sm (if/inner)
   3. glob to reg (prefetch/outer)
   4. reg to sm (prefetch/outer)
   5. sm to reg (inner)
   6. sm to reg (prefetch/outer)
  */
  // memory access type
  bool isLoadOp = false;
  auto opName = lSOp.getOperationName();
  if (opName == "affine.load" || opName == "affine.vector_load") {
    isLoadOp = true;
  }
  // new datas / old datas
  mlir::Value newBuf, buf = lSOp.getMemref();
  mlir::AffineMap newMap, map = lSOp.getAffineMap();
  newBuf = buf; newMap = map;
  // 替换掉所有的shift for iv
  std::vector<mlir::Value> newOperands;
  for (auto operand : lSOp.getMapOperands()) {
    if (allIvsMap.count(operand)) {
      newOperands.push_back(allIvsMap.at(operand));
    } else {
      newOperands.push_back(operand);
    }
  }
  if ((sType == ShiftType::RTSInner || sType == ShiftType::STRInner) && !isLoadOp) {
    // (2/5/storeOp) 内部存储到double buf的infos
    newBuf = bufMap.at(buf);
    newOperands.push_back(forKIv.second);
    addExpr = shiftExprDim(builder, addExpr, map.getNumDims());
    newMap = insertExprToMap(builder, map, addExpr, /*index*/0);
  } else if ((sType == ShiftType::RTSPrefetch|| sType == ShiftType::STRPrefetch) && !isLoadOp) {
    // (4/6/storeOp) 预取存储到double buf的infos
    newBuf = bufMap.at(buf);
    newMap = insertExprToMap(builder, map, builder.getAffineConstantExpr(0), 0);
  } else if ((sType == ShiftType::GTRPrefetch || sType == ShiftType::STRPrefetch) && isLoadOp) {
    // (3/6/loadOp) 预取从非double buf中加载的infos
    int idx = -1;
    for (int i=0; i<newOperands.size(); i++) {
      if (newOperands[i] == forKIv.first) { idx = i; }
    }
    newOperands.erase(newOperands.begin() + idx);
    newMap = replaceExprInMap(builder, map, builder.getAffineConstantExpr(0), idx);
  } else if (sType == ShiftType::STRInner && isLoadOp) {
    // (5/loadOp) 寄存器预取，从非double buf中加载的infos <寄存器预取的内部forbk需要+1>
    int idx = -1;
    for (int i=0; i<newOperands.size(); i++) {
      if (newOperands[i] == forKIv.first) { 
        newOperands[i] = forKIv.second;
        idx = i; break;
      }
    }
    addExpr = shiftExprDim(builder, builder.getAffineDimExpr(0) + 1, idx);
    newMap = replaceExprInMap(builder, map, addExpr, idx);
  } else if (sType == ShiftType::GTRInner && isLoadOp) {
    // (1/loadOp) 内部从非double buf加载的infos
    for (int i=0; i<newOperands.size(); i++) {
      if (newOperands[i] == forKIv.first) { newOperands[i] = forKIv.second; }
    }
  }
  return std::make_tuple(newOperands, newMap, newBuf);
}

template <typename loadOp>
std::tuple<std::vector<mlir::Value>, mlir::AffineMap, mlir::Value> /*operands, map, buf*/
  getCalculateMapInfos(mlir::OpBuilder builder, 
                       loadOp lSOp, 
                       const std::map<mlir::Value, mlir::Value, ValueCompare>& bufMap,
                       mlir::Value forKIv, 
                       mlir::AffineExpr addExpr) {
  // 计算部分修改的全是load op，而且都是修改buf为double buf，且map多一个expr，operands多一个和ForK/BK相关的变量
  // old datas / new datas
  mlir::Value newBuf = bufMap.at(lSOp.getMemref());
  mlir::AffineMap newMap, map = lSOp.getAffineMap();
  // operands
  auto operands = lSOp.getMapOperands();
  std::vector<mlir::Value> newOperands(operands.begin(), operands.end());
  newOperands.push_back(forKIv);
  // map
  addExpr = shiftExprDim(builder, addExpr, map.getNumDims());
  newMap = insertExprToMap(builder, map, addExpr, /*index*/0);
  return std::make_tuple(newOperands, newMap, newBuf);
}


template <typename loadOp>
std::tuple<std::vector<mlir::Value>, mlir::AffineMap, mlir::Value> /*operands, map, buf*/ 
  getDoubleBufferAdjustInfos(mlir::OpBuilder builder, 
                             loadOp lSOp, 
                             int step=0) {
  // 获取寄存器预取将其调整到forK循环外部和fork内部的最后加一个预取
  // 外部：sm[0, 0, ...] / 尾部：sm[（iter/step)%2, 0, ...]
  auto map = lSOp.getAffineMap();
  auto operands = lSOp.getMapOperands();
  std::vector<mlir::Value> newOperands(operands.begin(), operands.end());
  mlir::AffineExpr newExpr;
  int dimCount = -1;
  if (step) {
    dimCount = map.getNumDims();
    newExpr = builder.getAffineDimExpr(map.getNumDims()-1).floorDiv(step) % 2;
  } else {
    dimCount = map.getNumDims() - 1;
    newExpr = builder.getAffineConstantExpr(0);
    newOperands.pop_back();
  }
  llvm::SmallVector<mlir::AffineExpr> newExprs{newExpr};
  auto oldExprs = map.getResults();
  for (int i=1; i<oldExprs.size(); i++) {
    newExprs.push_back(oldExprs[i]);
  }
  auto newMap = mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
  return std::make_tuple(newOperands, newMap, lSOp.getMemRef());
}

// ===========================================================================================

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

mlir::Value createDoubleBuffer(mlir::Value buffer);

std::tuple<llvm::SmallVector<int64_t>, 
llvm::SmallVector<int64_t>, 
llvm::SmallVector<int64_t>, 
std::vector<mlir::Value>> 
  getNestedLoopData(mlir::affine::AffineForOp forOp);

std::tuple<llvm::SmallVector<int64_t>, 
llvm::SmallVector<int64_t>, 
llvm::SmallVector<int64_t>,
std::vector<mlir::Value>,
std::vector<std::string>>
  getNestedLoopDetailDatas(mlir::affine::AffineForOp forOp);

// =========================================== about double buffer ============================================

std::vector<mlir::affine::AffineForOp> 
  createNewDataShift(mlir::OpBuilder builder, 
                          const std::vector<mlir::affine::AffineForOp>& forOps,  
                          const std::map<mlir::Value, mlir::Value, ValueCompare>& bufMap, 
                          const std::pair<mlir::Value, mlir::Value>& forKIv,
                          ShiftType sType,
                          mlir::AffineExpr addExpr=nullptr);

void moveCalculateForOp(mlir::Operation* posOp, 
                        mlir::affine::AffineForOp &forOp, 
                        const std::map<mlir::Value, mlir::Value, ValueCompare>& bufMap, 
                        mlir::Value forKIv,
                        mlir::AffineExpr addExpr);

mlir::affine::AffineForOp createRearCalculateForOp(mlir::OpBuilder builder, 
                                                   mlir::affine::AffineForOp calculateForOp, 
                                                   std::map<mlir::Value, mlir::Value, ValueCompare> bufMap);

// =============================================================================================================

int32_t getNoBodyOpCount(mlir::Operation* op);

}

#endif