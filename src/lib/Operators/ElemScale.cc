#include "Operators/ElemScale.h"
#include <cmath>

namespace KernelCodeGen {
namespace Operators {

std::string ElemScale::s_function = "Unknown";

void ElemScale::buildNaiveExpress(mlir::ModuleOp module,
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

  auto [batchs, outShape] = splitShape(outputShape[0], 2);
  auto [inBatchs, inShape] = splitShape(inputShape[0], 2);
  auto type = tools::getDType(builder, outputDType[0]);

  // scale = sqrt(head_dim), head_dim = first core dim of input
  int64_t scaleDim = inShape[0];
  float scaleVal = std::sqrt(static_cast<float>(scaleDim));

  mlir::func::FuncOp funcOp = createFunc(builder, inputShape, outputShape,
                                          inputDType, outputDType, isTranspose, kernelName);
  mlir::ValueRange operands = funcOp.getArguments();
  auto batchIvs = createBatchNestForOp(builder, batchs);

  mlir::Location loc = builder.getUnknownLoc();
  mlir::SmallVector<int64_t, 2> lowerBounds = {0, 0};
  mlir::SmallVector<int64_t, 2> steps = {1, 1};
  mlir::SmallVector<int64_t, 2> upperBounds = {outShape[0], outShape[1]};

  // y=sl, x=hd; reads input with transpose (Q[hd,sl] -> Q[col,row])
  mlir::affine::buildAffineLoopNest(builder, loc, lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nb, mlir::Location l, mlir::ValueRange ivs) {
      auto row = ivs[0];
      auto col = ivs[1];
      auto scaleConst = nb.create<mlir::arith::ConstantOp>(l, nb.getFloatAttr(type, scaleVal));
      auto ldIdx = getShapeOrIndex(batchIvs, {row, col}, isTranspose[0]);
      auto ld = nb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(ldIdx));
      auto div = nb.create<mlir::arith::DivFOp>(l, ld, scaleConst);
      auto stIdx = getShapeOrIndex(batchIvs, {row, col}, false);
      nb.create<mlir::affine::AffineStoreOp>(l, div, operands[1], mlir::ValueRange(stIdx));
  });

  int index = 0;
  char dims[] = {'y', 'x'};
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
    if (!forOp->getAttr(FORDESC) && index < 2) {
      forOp->setAttr(FORDESC, builder.getStringAttr(std::string{dims[index]}));
      index++;
    }
  });
}

std::optional<std::string> ElemScale::verify(const std::vector<std::vector<int64_t>>& inputShape,
                                             const std::vector<std::vector<int64_t>>& outputShape,
                                             const std::vector<std::string>& inputDType,
                                             const std::vector<std::string>& outputDType,
                                             const std::vector<bool>& isTranspose) {
  if (inputShape.size() != 1 || outputShape.size() != 1) {
    return std::string{"ElemScale requires exactly 1 input and 1 output tensor."};
  }
  if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
    return std::string{"The dimensions of shape and dtype are not equal."};
  }
  return std::nullopt;
}

mlir::func::FuncOp ElemScale::createFunc(mlir::OpBuilder& builder,
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
  ElemScale::s_function = kernelName;
  return buildFunction(builder, kernelName, "ElemScale", {inputType, outputType}, isTranspose, {"y"}, 1);
}

}  // Operators
}  // KernelCodeGen
