#include "Operators/Softmax.h"

namespace KernelCodeGen {
namespace Operators {

std::string Softmax::s_function = "Unknown";

void Softmax::buildNaiveExpress(mlir::ModuleOp module, 
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
  auto [batchs, shape] = splitShape(outputShape[0], 2);
  auto type = tools::getDType(builder, outputDType[0]);
  mlir::func::FuncOp funcOp = createFunc(builder, inputShape, outputShape, inputDType, outputDType, isTranspose, kernelName);
  mlir::ValueRange operands = funcOp.getArguments();
  auto batchIvs = createBatchNestForOp(builder, batchs);

  // Softmax expressed as 2 x-loops inside 1 y-loop:
  //   x1: online softmax reduce (max + sum in single pass, numerically stable)
  //   x2: normalize (div by sum)
  // Each computation step inside x1 is clearly separated for readability.
  // Pipeline limitation: same y-loop cannot have >1 x-loop with independent iter vars
  // going through bufferizeLoopCarryVar. The online algorithm naturally combines max+sum
  // into one iter-var set, avoiding this constraint.
  auto yLoopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value row, mlir::ValueRange iterArgs) {
    auto negInf = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(type, -std::numeric_limits<float>::infinity()));
    auto zero = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(type, 0.0f));

    // x1: online softmax reduce — computes row max and row sum in a single pass
    auto x1LoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange iterMD) {
      auto index = getShapeOrIndex(batchIvs, {row, col}, isTranspose[0]);
      auto ld = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(index));
      // stage-max: update running max
      auto newMax = bb.create<mlir::arith::MaxNumFOp>(l, iterMD[0], ld);
      // stage-sum: correct old sum for new max, add new element
      auto sub1 = bb.create<mlir::arith::SubFOp>(l, iterMD[0], newMax);
      auto exp1 = bb.create<mlir::math::ExpOp>(l, sub1);
      auto scaledOldSum = bb.create<mlir::arith::MulFOp>(l, exp1, iterMD[1]);
      auto sub2 = bb.create<mlir::arith::SubFOp>(l, ld, newMax);
      auto exp2 = bb.create<mlir::math::ExpOp>(l, sub2);
      auto newSum = bb.create<mlir::arith::AddFOp>(l, scaledOldSum, exp2);
      bb.create<mlir::affine::AffineYieldOp>(l, mlir::ValueRange({newMax, newSum}));
    };
    auto x1loop = b.create<mlir::affine::AffineForOp>(loc, 0, shape[1], 1,
      mlir::ValueRange({negInf, zero}), x1LoopBody);
    x1loop->setAttr(FORDESC, builder.getStringAttr("x"));
    llvm::SmallVector<mlir::Attribute> iterDescs{builder.getStringAttr("Max"), builder.getStringAttr("Sum")};
    x1loop->setAttr(ITERVARDESC, builder.getArrayAttr(iterDescs));

    // x2: normalize — exp(elem - max) / sum
    auto x2LoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange iterArgs1) {
      auto ldIndex = getShapeOrIndex(batchIvs, {row, col}, isTranspose[0]);
      auto ld = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(ldIndex));
      auto sub = bb.create<mlir::arith::SubFOp>(l, ld, x1loop.getResult(0));
      auto exp = bb.create<mlir::math::ExpOp>(l, sub);
      auto div = bb.create<mlir::arith::DivFOp>(l, exp, x1loop.getResult(1));
      auto stIndex = getShapeOrIndex(batchIvs, {row, col}, false);
      bb.create<mlir::affine::AffineStoreOp>(l, div, operands[1], mlir::ValueRange(stIndex));
      bb.create<mlir::affine::AffineYieldOp>(l);
    };
    auto x2loop = b.create<mlir::affine::AffineForOp>(loc, 0, shape[1], 1,
      mlir::ValueRange({}), x2LoopBody);
    x2loop->setAttr(FORDESC, builder.getStringAttr("x"));

    b.create<mlir::affine::AffineYieldOp>(loc);
  };
  auto yloop = builder.create<mlir::affine::AffineForOp>(
    builder.getUnknownLoc(), 0, shape[0], 1, mlir::ValueRange({}), yLoopBody);
  yloop->setAttr(FORDESC, builder.getStringAttr("y"));
}


std::optional<std::string> Softmax::verify(const std::vector<std::vector<int64_t>>& inputShape, 
                                          const std::vector<std::vector<int64_t>>& outputShape, 
                                          const std::vector<std::string>& inputDType,
                                          const std::vector<std::string>& outputDType,
                                          const std::vector<bool>& isTranspose) {
  if (inputShape.size() != 1 || outputShape.size() != 1) {
    std::string err{"The number of tensors does not meet the requirements."};
    return err;
  }
  if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
    std::string err{"The dimensions of shape and dtype are not equal."};
    return err;
  }
  if (isTranspose[0]) {
    if (inputShape[0][inputShape[0].size()-1] != outputShape[0][outputShape[0].size()-2] ||
        inputShape[0][inputShape[0].size()-2] != outputShape[0][outputShape[0].size()-1]) {
      std::string err{"input shape is ont equal output shape."};
      return err;
    }
  } else {
    if (inputShape[0][inputShape[0].size()-1] != outputShape[0][outputShape[0].size()-1] ||
        inputShape[0][inputShape[0].size()-2] != outputShape[0][outputShape[0].size()-2]) {
      std::string err{"input shape is ont equal output shape."};
      return err;
    }
  }
  return std::nullopt;
}

mlir::func::FuncOp Softmax::createFunc(mlir::OpBuilder& builder, 
                                      const std::vector<std::vector<int64_t>>& inputShape, 
                                      const std::vector<std::vector<int64_t>>& outputShape, 
                                      const std::vector<std::string>& inputDType,
                                      const std::vector<std::string>& outputDType,
                                      const std::vector<bool>& isTranspose,
                                      const std::string& kernelName) {
  auto inputMlirType = tools::getDType(builder, inputDType[0]);
  auto outputMlirType = tools::getDType(builder, outputDType[0]);
  auto ms = MemorySpace::global;
  auto inputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[0]), inputMlirType, {}, static_cast<int>(ms));
  auto outputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[0]), outputMlirType, {}, static_cast<int>(ms));
  Softmax::s_function = kernelName;

  return buildFunction(builder, kernelName, "SoftMax", {inputType, outputType}, isTranspose, {"y"}, 1);
}

}  // Operators
}  // KernelCodeGen
