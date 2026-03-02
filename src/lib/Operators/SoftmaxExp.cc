#include "Operators/SoftmaxExp.h"
#include <limits>

namespace KernelCodeGen {
namespace Operators {

std::string SoftmaxExp::s_function = "Unknown";

void SoftmaxExp::buildNaiveExpress(mlir::ModuleOp module,
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
    return;
  }

  auto [batchs, shape] = splitShape(outputShape[0], 2);
  auto type = tools::getDType(builder, outputDType[0]);

  mlir::func::FuncOp funcOp = createFunc(builder, inputShape, outputShape,
                                          inputDType, outputDType, isTranspose, kernelName);
  mlir::ValueRange operands = funcOp.getArguments();
  // operands[0] = input scores
  // operands[1] = exp_output (same shape as input)
  // operands[2] = sum_output [sl, 1]
  auto batchIvs = createBatchNestForOp(builder, batchs);

  // Create zero index constant before y-loop so it's a valid affine symbol
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);

  auto yLoopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value row, mlir::ValueRange iterArgs) {
    auto negInf = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(type, -std::numeric_limits<float>::infinity()));
    auto zero = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(type, 0.0f));

    // x1: online softmax reduce — computes row max and row sum in a single pass
    auto x1LoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange iterMD) {
      auto index = getShapeOrIndex(batchIvs, {row, col}, isTranspose[0]);
      auto ld = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(index));
      auto newMax = bb.create<mlir::arith::MaxNumFOp>(l, iterMD[0], ld);
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

    // Store row sum to sum_output[row, 0]
    auto sumStoreIdx = getShapeOrIndex(batchIvs, {row, zeroIdx.getResult()}, false);
    b.create<mlir::affine::AffineStoreOp>(loc, x1loop.getResult(1), operands[2], mlir::ValueRange(sumStoreIdx));

    // x2: compute exp(elem - max) WITHOUT division by sum
    auto x2LoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange iterArgs1) {
      auto ldIndex = getShapeOrIndex(batchIvs, {row, col}, isTranspose[0]);
      auto ld = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(ldIndex));
      auto sub = bb.create<mlir::arith::SubFOp>(l, ld, x1loop.getResult(0));
      auto exp = bb.create<mlir::math::ExpOp>(l, sub);
      auto stIndex = getShapeOrIndex(batchIvs, {row, col}, false);
      bb.create<mlir::affine::AffineStoreOp>(l, exp, operands[1], mlir::ValueRange(stIndex));
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

std::optional<std::string> SoftmaxExp::verify(const std::vector<std::vector<int64_t>>& inputShape,
                                              const std::vector<std::vector<int64_t>>& outputShape,
                                              const std::vector<std::string>& inputDType,
                                              const std::vector<std::string>& outputDType,
                                              const std::vector<bool>& isTranspose) {
  if (inputShape.size() != 1 || outputShape.size() != 2) {
    return std::string{"SoftmaxExp requires 1 input and 2 outputs (exp_scores, sum)."};
  }
  if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
    return std::string{"The dimensions of shape and dtype are not equal."};
  }
  auto inCore = inputShape[0];
  auto outCore = outputShape[0];
  if (inCore.size() != outCore.size()) {
    return std::string{"Input and exp_output must have same rank."};
  }
  for (size_t i = 0; i < inCore.size(); i++) {
    if (inCore[i] != outCore[i]) {
      return std::string{"Input and exp_output shapes must match."};
    }
  }
  return std::nullopt;
}

mlir::func::FuncOp SoftmaxExp::createFunc(mlir::OpBuilder& builder,
                                          const std::vector<std::vector<int64_t>>& inputShape,
                                          const std::vector<std::vector<int64_t>>& outputShape,
                                          const std::vector<std::string>& inputDType,
                                          const std::vector<std::string>& outputDType,
                                          const std::vector<bool>& isTranspose,
                                          const std::string& kernelName) {
  auto inputMlirType = tools::getDType(builder, inputDType[0]);
  auto expOutputMlirType = tools::getDType(builder, outputDType[0]);
  auto sumOutputMlirType = tools::getDType(builder, outputDType[1]);
  auto ms = MemorySpace::global;
  auto inputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[0]), inputMlirType, {}, static_cast<int>(ms));
  auto expOutputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[0]), expOutputMlirType, {}, static_cast<int>(ms));
  auto sumOutputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[1]), sumOutputMlirType, {}, static_cast<int>(ms));
  SoftmaxExp::s_function = kernelName;
  return buildFunction(builder, kernelName, "SoftmaxExp", {inputType, expOutputType, sumOutputType}, isTranspose, {"y"}, 2);
}

}  // Operators
}  // KernelCodeGen
