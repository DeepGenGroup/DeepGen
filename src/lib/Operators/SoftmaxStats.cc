#include "Operators/SoftmaxStats.h"
#include <limits>

namespace KernelCodeGen {
namespace Operators {

std::string SoftmaxStats::s_function = "Unknown";

void SoftmaxStats::buildNaiveExpress(mlir::ModuleOp module,
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
  auto [batchs_in, shape_in] = splitShape(inputShape[0], 2);
  auto type = tools::getDType(builder, outputDType[0]);

  mlir::func::FuncOp funcOp = createFunc(builder, inputShape, outputShape,
                                          inputDType, outputDType, isTranspose, kernelName);
  mlir::ValueRange operands = funcOp.getArguments();
  // operands[0] = scores input  [sl, sl]
  // operands[1] = em output     [sl, 1]
  // operands[2] = denom output  [sl, 1]
  auto batchIvs = createBatchNestForOp(builder, batchs);

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
    auto x1loop = b.create<mlir::affine::AffineForOp>(loc, 0, shape_in[1], 1,
      mlir::ValueRange({negInf, zero}), x1LoopBody);
    x1loop->setAttr(FORDESC, builder.getStringAttr("x"));
    llvm::SmallVector<mlir::Attribute> iterDescs{builder.getStringAttr("Max"), builder.getStringAttr("Sum")};
    x1loop->setAttr(ITERVARDESC, builder.getArrayAttr(iterDescs));

    // em = exp(max)
    auto emVal = b.create<mlir::math::ExpOp>(loc, x1loop.getResult(0));
    auto emStoreIdx = getShapeOrIndex(batchIvs, {row, zeroIdx.getResult()}, false);
    b.create<mlir::affine::AffineStoreOp>(loc, emVal, operands[1], mlir::ValueRange(emStoreIdx));

    // denom = sum (which is sum(exp(scores - max)))
    auto denomStoreIdx = getShapeOrIndex(batchIvs, {row, zeroIdx.getResult()}, false);
    b.create<mlir::affine::AffineStoreOp>(loc, x1loop.getResult(1), operands[2], mlir::ValueRange(denomStoreIdx));

    b.create<mlir::affine::AffineYieldOp>(loc);
  };
  auto yloop = builder.create<mlir::affine::AffineForOp>(
    builder.getUnknownLoc(), 0, shape_in[0], 1, mlir::ValueRange({}), yLoopBody);
  yloop->setAttr(FORDESC, builder.getStringAttr("y"));
}

std::optional<std::string> SoftmaxStats::verify(const std::vector<std::vector<int64_t>>& inputShape,
                                                const std::vector<std::vector<int64_t>>& outputShape,
                                                const std::vector<std::string>& inputDType,
                                                const std::vector<std::string>& outputDType,
                                                const std::vector<bool>& isTranspose) {
  if (inputShape.size() != 1 || outputShape.size() != 2) {
    return std::string{"SoftmaxStats requires 1 input (scores) and 2 outputs (em, denom)."};
  }
  if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
    return std::string{"The dimensions of shape and dtype are not equal."};
  }
  return std::nullopt;
}

mlir::func::FuncOp SoftmaxStats::createFunc(mlir::OpBuilder& builder,
                                            const std::vector<std::vector<int64_t>>& inputShape,
                                            const std::vector<std::vector<int64_t>>& outputShape,
                                            const std::vector<std::string>& inputDType,
                                            const std::vector<std::string>& outputDType,
                                            const std::vector<bool>& isTranspose,
                                            const std::string& kernelName) {
  auto inputMlirType = tools::getDType(builder, inputDType[0]);
  auto emOutputMlirType = tools::getDType(builder, outputDType[0]);
  auto denomOutputMlirType = tools::getDType(builder, outputDType[1]);
  auto ms = MemorySpace::global;
  auto inputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[0]), inputMlirType, {}, static_cast<int>(ms));
  auto emOutputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[0]), emOutputMlirType, {}, static_cast<int>(ms));
  auto denomOutputType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[1]), denomOutputMlirType, {}, static_cast<int>(ms));
  SoftmaxStats::s_function = kernelName;
  return buildFunction(builder, kernelName, "SoftmaxStats", {inputType, emOutputType, denomOutputType}, isTranspose, {"y"}, 2);
}

}  // Operators
}  // KernelCodeGen
