#include "Operators/Operators.h"
#include "Common/Utils.h"
#include <filesystem>

namespace KernelCodeGen {


mlir::func::FuncOp buildFunction(mlir::OpBuilder& builder, 
                                 const std::string& funcName, 
                                 const std::string& OpType, 
                                 const std::vector<mlir::Type>& inputsTypes, 
                                 const std::vector<bool>& isTranspose,
                                 const std::vector<std::string>& paraDims,
                                 const int& outputArgNum) {
  llvm::ArrayRef<mlir::Type> inputsTypesArray(inputsTypes);
  auto functionType = builder.getFunctionType(mlir::TypeRange(inputsTypesArray), mlir::TypeRange({}));
  auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), llvm::StringRef(funcName), functionType);
  
  auto& region = funcOp->getRegion(0);
  if (!region.hasOneBlock()) {
    region.emplaceBlock();
  }
  auto& body =  funcOp.front(); //? region.front()  : ;
  llvm::SmallVector<mlir::Location> locs(inputsTypes.size(), builder.getUnknownLoc());
  body.addArguments(inputsTypes, locs);
  
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  funcOp->setAttr(std::string("func.op.type"), builder.getStringAttr(OpType));
  funcOp->setAttr(std::string(AttrVisibility), builder.getStringAttr("public"));
  auto intType = builder.getIntegerType(32);
  funcOp->setAttr(std::string("func.output.arg.num"), builder.getIntegerAttr(intType, outputArgNum));
  // parallel attr
  llvm::SmallVector<mlir::Attribute> strAttrs;
  for (auto paraDim : paraDims){
    strAttrs.push_back(builder.getStringAttr(paraDim));
  }
  funcOp->setAttr(PARALLELDIMS, builder.getArrayAttr(strAttrs));
  // transpose attr
  llvm::SmallVector<mlir::Attribute> intAttrs;
  for (auto isTran : isTranspose){
    auto intType = builder.getIntegerType(1);
    intAttrs.push_back(builder.getIntegerAttr(intType, isTran));
  }
  funcOp->setAttr(ARGTRAN, builder.getArrayAttr(intAttrs));
  // funcOp->setAttr(std::string(AttrKernelFunc), builder.getI32IntegerAttr(1));
  
  auto& entryBlock = funcOp.front();
  builder.setInsertionPointToStart(&entryBlock);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&entryBlock);
  return funcOp;
}

std::vector<mlir::Value> createBatchNestForOp(mlir::OpBuilder& builder, std::vector<int64_t> batchs) {
  // 生成有关batch的嵌套循环
  std::vector<mlir::Value> ivs_;
  mlir::SmallVector<int64_t, 3> lowerBounds;
  mlir::SmallVector<int64_t, 3> steps;
  mlir::SmallVector<int64_t, 3> upperBounds(batchs.begin(), batchs.end());
  for (int i=0; i<batchs.size(); i++) {
    lowerBounds.push_back(0);
    steps.push_back(1);
  }
  mlir::Location loc = builder.getUnknownLoc();
  mlir::affine::buildAffineLoopNest(builder, loc, lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange ivs) {
      for (auto iv : ivs) {
        ivs_.push_back(iv);
      }
    });
  mlir::Block* block = builder.getInsertionBlock();
  mlir::Operation* op = block->getParentOp();
  mlir::affine::AffineForOp innerForOp = nullptr;

  // int batchNum = 0;
  op->walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    // forOp->setAttr(BATCHNUM, builder.getIntegerAttr(builder.getI32Type(), batchNum));
    forOp->setAttr(FORDESC, builder.getStringAttr("batch"));
    innerForOp = forOp;
    // batchNum++;
  });
  builder.setInsertionPointToStart(block);
  if (innerForOp != nullptr){
    builder.setInsertionPointToStart(innerForOp.getBody());
  }
  return ivs_;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> splitShape(const std::vector<int64_t>& shape, int shapeLen) {
  // 将传入的dims的batch和shape区分出来
  // get batch shape
  std::vector<int64_t> bacths, shape_;
  for (int i=0; i<shape.size()-shapeLen; i++) {
    bacths.push_back(shape[i]);
  }
  // real shape
  for (int i=shape.size()-shapeLen; i<shape.size(); i++) {
    shape_.push_back(shape[i]);
  }
  return std::make_pair(bacths, shape_);
}

}