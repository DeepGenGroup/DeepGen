#include "Operators/RowDiv.h"

namespace KernelCodeGen {
namespace Operators {

std::string RowDiv::s_function = "Unknown";

void RowDiv::buildNaiveExpress(mlir::ModuleOp module,
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
  // operands[0] = data input  [sl, hd]
  // operands[1] = sum input   [sl, 1]
  // operands[2] = data output [sl, hd]
  auto batchIvs = createBatchNestForOp(builder, batchs);

  // Create zero index constant before y-loop so it's a valid affine symbol
  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);

  auto yLoopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value row, mlir::ValueRange iterArgs) {
    // Load sum for this row: sum[row, 0]
    auto sumLoadIdx = getShapeOrIndex(batchIvs, {row, zeroIdx.getResult()}, false);
    auto sumVal = b.create<mlir::affine::AffineLoadOp>(loc, operands[1], mlir::ValueRange(sumLoadIdx));

    // x: divide each element by the row sum
    auto xLoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange xIterArgs) {
      auto ldIdx = getShapeOrIndex(batchIvs, {row, col}, isTranspose[0]);
      auto ld = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(ldIdx));
      auto div = bb.create<mlir::arith::DivFOp>(l, ld, sumVal);
      auto stIdx = getShapeOrIndex(batchIvs, {row, col}, false);
      bb.create<mlir::affine::AffineStoreOp>(l, div, operands[2], mlir::ValueRange(stIdx));
      bb.create<mlir::affine::AffineYieldOp>(l);
    };
    auto xloop = b.create<mlir::affine::AffineForOp>(loc, 0, shape[1], 1,
      mlir::ValueRange({}), xLoopBody);
    xloop->setAttr(FORDESC, builder.getStringAttr("x"));

    b.create<mlir::affine::AffineYieldOp>(loc);
  };
  auto yloop = builder.create<mlir::affine::AffineForOp>(
    builder.getUnknownLoc(), 0, shape[0], 1, mlir::ValueRange({}), yLoopBody);
  yloop->setAttr(FORDESC, builder.getStringAttr("y"));
}

std::optional<std::string> RowDiv::verify(const std::vector<std::vector<int64_t>>& inputShape,
                                          const std::vector<std::vector<int64_t>>& outputShape,
                                          const std::vector<std::string>& inputDType,
                                          const std::vector<std::string>& outputDType,
                                          const std::vector<bool>& isTranspose) {
  if (inputShape.size() != 2 || outputShape.size() != 1) {
    return std::string{"RowDiv requires 2 inputs (data, sum) and 1 output."};
  }
  if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
    return std::string{"The dimensions of shape and dtype are not equal."};
  }
  return std::nullopt;
}

mlir::func::FuncOp RowDiv::createFunc(mlir::OpBuilder& builder,
                                      const std::vector<std::vector<int64_t>>& inputShape,
                                      const std::vector<std::vector<int64_t>>& outputShape,
                                      const std::vector<std::string>& inputDType,
                                      const std::vector<std::string>& outputDType,
                                      const std::vector<bool>& isTranspose,
                                      const std::string& kernelName) {
  auto dataMlirType = tools::getDType(builder, inputDType[0]);
  auto sumMlirType = tools::getDType(builder, inputDType[1]);
  auto outMlirType = tools::getDType(builder, outputDType[0]);
  auto ms = MemorySpace::global;
  auto dataType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[0]), dataMlirType, {}, static_cast<int>(ms));
  auto sumType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[1]), sumMlirType, {}, static_cast<int>(ms));
  auto outType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[0]), outMlirType, {}, static_cast<int>(ms));
  RowDiv::s_function = kernelName;
  return buildFunction(builder, kernelName, "RowDiv", {dataType, sumType, outType}, isTranspose, {"y"}, 1);
}

}  // Operators
}  // KernelCodeGen
