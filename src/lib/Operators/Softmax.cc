#include "Operators/Softmax.h"

namespace KernelCodeGen {
namespace Operators {

std::string Softmax::s_function = "Unknown";

void Softmax::buildNaiveExpress(mlir::ModuleOp module, 
  std::vector<int64_t> shape, 
  const std::string& dtype,
  const std::string& kernelName,
  bool isTranspose
  ) 
{
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(module.getBody());
  auto ver = verify(builder, shape, dtype);
  if (ver.has_value()) {
    llvm::errs() << ver.value() << "\n";
    return ;
  }
  // get base args
  auto result = splitShape(shape, 2);
  std::vector<int64_t> batchs = result.first, shape_ = result.second;
  auto type = tools::getDType(builder, dtype);

  // create funcOp
  mlir::func::FuncOp funcOp = createFunc(builder, batchs, shape_, dtype, kernelName, isTranspose);
  mlir::ValueRange operands = funcOp.getArguments();

  // create bacth nest forOp
  auto batchIvs = createBatchNestForOp(builder, batchs);

  // softmax forop
  auto rowLoopBody = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value row, mlir::ValueRange iterArgs) {
    // max = -FLT_MAX, sum = 0.0f
    auto max = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(type, -std::numeric_limits<float>::infinity()));
    auto sum = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(type, 0.0f));
    
    // compute max and sum
    auto col1LoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange iterMD) {
      auto index = getShapeOrIndex(batchIvs, {row, col}, isTranspose);
      auto ld = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(index));
      // newMax = max(elem, iterMD[0])
      auto newMax = bb.create<mlir::arith::MaxNumFOp>(l, iterMD[0], ld);
      // factor = exp(oldMax - newMax)
      auto sub1 = bb.create<mlir::arith::SubFOp>(l, iterMD[0], newMax);
      auto exp1 = bb.create<mlir::math::ExpOp>(l, sub1);
      // f * factor
      auto mul = bb.create<mlir::arith::MulFOp>(l, exp1, iterMD[1]);
      // exp(elem - newMax)
      auto sub2 = bb.create<mlir::arith::SubFOp>(l, ld, newMax);
      auto exp2 = bb.create<mlir::math::ExpOp>(l, sub2);
      // d * factor + exp(elem - newMax)
      auto newSum = bb.create<mlir::arith::AddFOp>(l, mul, exp2);
      bb.create<mlir::affine::AffineYieldOp>(l, mlir::ValueRange({newMax, newSum}));
    };
    auto loop = b.create<mlir::affine::AffineForOp>(b.getUnknownLoc(), 0, shape_[1], 1, mlir::ValueRange({max, sum}), col1LoopBody);

    // div forOp
    auto col2LoopBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value col, mlir::ValueRange iterArgs) {
      auto index = getShapeOrIndex(batchIvs, {row, col}, isTranspose);
      auto ld = bb.create<mlir::affine::AffineLoadOp>(l, operands[0], mlir::ValueRange(index));
      // exp(elem - m) / d
      auto sub = bb.create<mlir::arith::SubFOp>(l, ld, loop.getResult(0));
      auto exp = bb.create<mlir::math::ExpOp>(l, sub);
      auto div = bb.create<mlir::arith::DivFOp>(l, exp, loop.getResult(1));
      bb.create<mlir::affine::AffineYieldOp>(l);
    };
    b.create<mlir::affine::AffineForOp>(b.getUnknownLoc(), 0, shape_[1], 1, mlir::ValueRange({}), col2LoopBody);
    b.create<mlir::affine::AffineYieldOp>(loc);
  };
  builder.create<mlir::affine::AffineForOp>(builder.getUnknownLoc(), 0, shape_[0], 1, mlir::ValueRange({}), rowLoopBody);
}


std::optional<std::string> Softmax::verify(
  mlir::OpBuilder builder, 
  std::vector<int64_t> shape, 
  const std::string& dtype
)
{
  if (shape.size() < 2 || shape.size() > 4) {
    std::string err{"Shape size must is 2, 3 or 4."};
    return err;
  }
  auto type = tools::getDType(builder, dtype);
  if (type == nullptr) {
    std::string err{"No exist this data type."};
    return err;
  }
  return std::nullopt;
}

mlir::func::FuncOp Softmax::createFunc(
  mlir::OpBuilder& builder, 
  std::vector<int64_t> batchs, 
  std::vector<int64_t> shape, 
  const std::string& dtype, 
  const std::string& kernelName,
  bool isTranspose
  )
{
  auto mlirType = tools::getDType(builder, dtype);
  auto shape_ = getShapeOrIndex(batchs, {shape[0], shape[1]}, isTranspose);
  auto ms = MemorySpace::global;
  auto type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_), mlirType, {}, static_cast<int>(ms));
  Softmax::s_function = kernelName;

  return buildFunction(builder, kernelName, "SoftMax", {type}, 1);
}

}  // Operators
}  // KernelCodeGen


