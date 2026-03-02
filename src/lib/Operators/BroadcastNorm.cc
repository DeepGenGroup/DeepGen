#include "Operators/BroadcastNorm.h"

namespace KernelCodeGen {
namespace Operators {

std::string BroadcastNorm::s_function = "Unknown";

void BroadcastNorm::buildNaiveExpress(mlir::ModuleOp module,
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
  // operands[0] = scores input  [sl, sl]
  // operands[1] = em input      [sl, 1]
  // operands[2] = denom input   [sl, 1]
  // operands[3] = p output      [sl, sl]
  auto batchIvs = createBatchNestForOp(builder, batchs);

  auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);

  auto yLoopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value row, mlir::ValueRange iterArgs) {
    // Identity copy: p_out[y,x] = scores_in[y,x]
    // After fusing, both map to midBuf → this becomes a noop (load+store same buf).
    // fuseParaForOps fuses it harmlessly with matmul1, producing a clean 2-yloop
    // structure. The optimizer adds the real normalize (exp/em/denom) on tileP.
    auto xLoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange xIterArgs) {
      auto ldIdx = getShapeOrIndex(batchIvs, {row, col}, isTranspose[0]);
      auto val = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(ldIdx));
      auto stIdx = getShapeOrIndex(batchIvs, {row, col}, false);
      bb.create<mlir::affine::AffineStoreOp>(l, val, operands[3], mlir::ValueRange(stIdx));
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

std::optional<std::string> BroadcastNorm::verify(const std::vector<std::vector<int64_t>>& inputShape,
                                                 const std::vector<std::vector<int64_t>>& outputShape,
                                                 const std::vector<std::string>& inputDType,
                                                 const std::vector<std::string>& outputDType,
                                                 const std::vector<bool>& isTranspose) {
  if (inputShape.size() != 3 || outputShape.size() != 1) {
    return std::string{"BroadcastNorm requires 3 inputs (scores, em, denom) and 1 output (p)."};
  }
  if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
    return std::string{"The dimensions of shape and dtype are not equal."};
  }
  return std::nullopt;
}

mlir::func::FuncOp BroadcastNorm::createFunc(mlir::OpBuilder& builder,
                                             const std::vector<std::vector<int64_t>>& inputShape,
                                             const std::vector<std::vector<int64_t>>& outputShape,
                                             const std::vector<std::string>& inputDType,
                                             const std::vector<std::string>& outputDType,
                                             const std::vector<bool>& isTranspose,
                                             const std::string& kernelName) {
  auto scoresMlirType = tools::getDType(builder, inputDType[0]);
  auto emMlirType = tools::getDType(builder, inputDType[1]);
  auto denomMlirType = tools::getDType(builder, inputDType[2]);
  auto outMlirType = tools::getDType(builder, outputDType[0]);
  auto ms = MemorySpace::global;
  auto scoresType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[0]), scoresMlirType, {}, static_cast<int>(ms));
  auto emType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[1]), emMlirType, {}, static_cast<int>(ms));
  auto denomType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(inputShape[2]), denomMlirType, {}, static_cast<int>(ms));
  auto outType = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(outputShape[0]), outMlirType, {}, static_cast<int>(ms));
  BroadcastNorm::s_function = kernelName;
  return buildFunction(builder, kernelName, "BroadcastNorm", {scoresType, emType, denomType, outType}, isTranspose, {"y"}, 1);
}

}  // Operators
}  // KernelCodeGen
