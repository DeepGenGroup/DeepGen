#include "Operators/Matmul.h"

namespace KernelCodeGen {
namespace Operators {

std::string Matmul::s_function = "Unknown";

void Matmul::buildNaiveExpress(mlir::ModuleOp module, 
                              const std::vector<std::vector<int64_t>>& inputShape, 
                              const std::vector<std::vector<int64_t>>& outputShape, 
                              const std::vector<std::string>& inputDType,
                              const std::vector<std::string>& outputDType,
                              const std::vector<bool>& isTranspose, 
                              const std::string& kernelName) {
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(module.getBody());
  auto ver = verify(inputShape, outputShape, inputDType, outputDType, isTranspose);
  if (ver.has_value()) {
    llvm::errs() << ver.value() << "\n";
    return ;
  }
  // get base args
  auto [batchs, mn] = splitShape(outputShape[0], 2);
  auto result = splitShape(inputShape[0], 2);
  int64_t k = result.second[1];
  if (isTranspose[0]) {
    result = splitShape(inputShape[0], 2);
    k = result.second[0];
  }
  auto typeC = tools::getDType(builder, outputDType[0]);

  // create funcOp
  mlir::func::FuncOp funcOp = createFunc(builder, inputShape, outputShape, inputDType, outputDType, isTranspose, kernelName);
  mlir::ValueRange operands = funcOp.getArguments();

  // create bacth nest forOp
  auto batchIvs = createBatchNestForOp(builder, batchs);
  
  // matmul nest 
  mlir::Location loc_ = builder.getUnknownLoc();
  mlir::SmallVector<int64_t, 3> lowerBounds = {0, 0};
  mlir::SmallVector<int64_t, 3> steps = {1, 1};
  mlir::SmallVector<int64_t, 3> upperBounds = {mn[0], mn[1]};
  mlir::affine::buildAffineLoopNest(builder, loc_, lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto row = ivs[0];
      auto col = ivs[1];

      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(loc, nestedBuilder.getFloatAttr(typeC, 0));

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value kiv, mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);

        auto indexA = getShapeOrIndex<mlir::Value>(batchIvs, {row, kiv}, isTranspose[0]);
        auto indexB = getShapeOrIndex<mlir::Value>(batchIvs, {kiv, col}, isTranspose[1]);

        auto ld_a = builder.create<mlir::affine::AffineLoadOp>(nestedLoc, /*A*/operands[0], mlir::ValueRange(indexA));
        auto ld_b = builder.create<mlir::affine::AffineLoadOp>(nestedLoc, /*B*/operands[1], mlir::ValueRange(indexB));
        auto mul = builder.create<mlir::arith::MulFOp>(nestedLoc, ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(nestedLoc, mul, iterArgs[0]);
        builder.create<mlir::affine::AffineYieldOp>(nestedLoc, add.getResult());
      };
      auto forK = nestedBuilder.create<mlir::affine::AffineForOp>(loc, /*lb*/0, k, 1, mlir::ValueRange({zero.getResult()}), kLoopBody);
      forK->setAttr(FORDESC, builder.getStringAttr(std::string{"k"}));
      auto indexC = getShapeOrIndex<mlir::Value>(batchIvs, {row, col}, false);
      nestedBuilder.create<mlir::affine::AffineStoreOp>(loc, forK.getResult(0), /*C*/operands[2], mlir::ValueRange(indexC));
  });
  // add attr 
  int index = 0;
  char dims[] = {'y', 'x'};
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    if (!forOp->getAttr(FORDESC) && index < 2) {
      forOp->setAttr(FORDESC, builder.getStringAttr(std::string{dims[index]}));
      index++;
    }
  });
}


std::optional<std::string> Matmul::verify(const std::vector<std::vector<int64_t>>& inputShape, 
                                          const std::vector<std::vector<int64_t>>& outputShape, 
                                          const std::vector<std::string>& inputDType,
                                          const std::vector<std::string>& outputDType,
                                          const std::vector<bool>& isTranspose) {
  if (inputShape.size() != 2 || outputShape.size() != 1) {
    std::string err{"The number of tensors does not meet the requirements."};
    return err;
  }
  if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
    std::string err{"The dimensions of shape and dtype are not equal."};
    return err;
  }
  // transpose
  if (isTranspose[0] == false && isTranspose[1] == false) {
    if (inputShape[0][inputShape[0].size()-1] != inputShape[1][inputShape[1].size()-2]) {
      std::string err{"The k dimensions are not equal"};
      return err;
    }
  } else if (isTranspose[0] == true && isTranspose[1] == true) {
    if (inputShape[0][inputShape[0].size()-2] != inputShape[1][inputShape[1].size()-1]) {
      std::string err{"The k dimensions are not equal"};
      return err;
    }
  } else if (isTranspose[0] == false && isTranspose[1] == true) {
    if (inputShape[0][inputShape[0].size()-1] != inputShape[1][inputShape[1].size()-1]) {
      std::string err{"The k dimensions are not equal"};
      return err;
    }
  } else {
    if (inputShape[0][inputShape[0].size()-2] != inputShape[1][inputShape[1].size()-2]) {
      std::string err{"The k dimensions are not equal"};
      return err;
    }
  }
  return std::nullopt;
}

mlir::func::FuncOp Matmul::createFunc(mlir::OpBuilder& builder, 
                                      const std::vector<std::vector<int64_t>>& inputShape, 
                                      const std::vector<std::vector<int64_t>>& outputShape, 
                                      const std::vector<std::string>& inputDType,
                                      const std::vector<std::string>& outputDType,
                                      const std::vector<bool>& isTranspose, 
                                      const std::string& kernelName) {
  std::vector<mlir::Type> inTypeArray;
  for(std::string type : inputDType) {
    mlir::Type mlirType = tools::getDType(builder, type);
    inTypeArray.push_back(mlirType);
  }
  mlir::Type outType = tools::getDType(builder, outputDType[0]);

  auto ms = MemorySpace::global;
  auto typeA = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[0]), inTypeArray[0], {}, static_cast<int>(ms));
  auto typeB = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[1]), inTypeArray[1], {}, static_cast<int>(ms));
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[0]), outType, {}, static_cast<int>(ms));
  Matmul::s_function = kernelName;

  return buildFunction(builder, kernelName, "Matmul", {typeA, typeB, typeC}, isTranspose, {"y", "x"}, 1);
}

}  // Operators
}  // KernelCodeGen


