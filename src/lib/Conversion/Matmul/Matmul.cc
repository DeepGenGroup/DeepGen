#include "Conversion/Matmul/Matmul.h"

namespace KernelCodeGen {
namespace Matmul {

bool haveBatch(mlir::affine::AffineForOp batchForOp) {
  // 判断矩阵乘法是否具有batch维度
  auto result = getLoopBoundAndStep(batchForOp);
  if (std::get<1>(result) == 1)
    return false;
  return true;
}

llvm::SmallVector<mlir::Value> amendOneDimBatch(mlir::func::FuncOp &funcOp, mlir::affine::AffineForOp &loopBatch) {
  // 当batch为1时，修改func中batch
  mlir::OpBuilder builder(funcOp.getContext());
  auto& bodyBlock = funcOp.front();
  mlir::ValueRange oldArgs = bodyBlock.getArguments();
  size_t argsNum = oldArgs.size();

  // 替换func的args
  llvm::SmallVector<mlir::Type> newTypes;
  for (auto oldArg : oldArgs) {
    auto oldType = oldArg.getType();
    auto memType =  mlir::dyn_cast<mlir::MemRefType>(oldType);
    auto oldShapes = memType.getShape();
    auto oldElemType = memType.getElementType();
    llvm::SmallVector<int64_t> newShapes(oldShapes.begin()+1, oldShapes.end());
    auto newType = mlir::MemRefType::get(mlir::ArrayRef<int64_t>(newShapes), 
                                         oldElemType, {}, memType.getMemorySpaceAsInt());
    newTypes.push_back(newType);
  }
  auto functionType = builder.getFunctionType(mlir::TypeRange(newTypes), mlir::TypeRange({}));
  funcOp.setFunctionType(functionType);

  // func block添加new args
  llvm::SmallVector<mlir::Location> locs(argsNum, builder.getUnknownLoc());
  bodyBlock.addArguments(newTypes, locs);
  mlir::ValueRange allArgs = bodyBlock.getArguments();
  llvm::SmallVector<mlir::Value> newArgs;
  llvm::SmallVector<mlir::Value> oldArgs_;
  for (int i=0; i<argsNum; i++) {
    newArgs.push_back(allArgs[argsNum + i]);
    oldArgs_.push_back(allArgs[i]);
  }

  // 替换使用了old args的func中op
  for (int i=0; i<argsNum; i++) {
    auto users = getValueUsers(oldArgs_[i]);
    for (auto user : users) {
      mlir::OpBuilder b(user);
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)) {
        auto oldOperands = loadOp.getMapOperands();
        llvm::SmallVector<mlir::Value> newOperands(oldOperands.begin()+1, oldOperands.end());
        auto newLoadOp = b.create<mlir::affine::AffineLoadOp>(loadOp.getLoc(), newArgs[i], newOperands);
        replaceAndErase(newLoadOp, loadOp);
      } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(user)){
        auto oldOperands = storeOp.getMapOperands();
        auto val = storeOp.getValue();
        llvm::SmallVector<mlir::Value> newOperands(oldOperands.begin()+1, oldOperands.end());
        b.create<mlir::affine::AffineStoreOp>(storeOp.getLoc(), val, newArgs[i], newOperands);
        storeOp.erase();
      }
    }
    bodyBlock.eraseArgument(0);
  }
  
  // 将batch forOp 删除
  mlir::affine::AffineForOp nextForOp;
  for (auto &op : loopBatch.getBody()->getOperations()) {
    if (auto castOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      nextForOp = castOp;
      break;
    }
  }
  nextForOp->moveAfter(loopBatch);
  loopBatch.erase();
  return newArgs;
}

mlir::AffineMap addBatchDimMap(mlir::OpBuilder builder, mlir::AffineMap map) {
  // 给batch添加一个维度
  auto dim0 = builder.getAffineDimExpr(0);
  llvm::SmallVector<mlir::AffineExpr> newExprs{dim0};
  auto oldExprs = map.getResults();
  for (auto oldExpr : oldExprs) {
    newExprs.push_back(shiftAffineExprDim(builder, oldExpr, 1));
  }
  return mlir::AffineMap::get(map.getNumDims() + 1, 0, llvm::ArrayRef<mlir::AffineExpr>(newExprs), builder.getContext());
}

}
}