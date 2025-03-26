#include "Operators/Matmul.h"

namespace KernelCodeGen {
namespace Operators {

std::string Matmul::s_function = "Unknown";

void Matmul::buildNaiveExpress(mlir::ModuleOp module, 
  std::vector<int64_t> shape, 
  const std::vector<std::string>& dtypes,
  const std::string& kernelName,
  bool isTransposeA, 
  bool isTransposeB
  ) 
{
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(module.getBody());
  auto ver = verify(builder, shape, dtypes);
  if (ver.has_value()) {
    llvm::errs() << ver.value() << "\n";
    return ;
  }
  // get base args
  auto result = splitShape(shape, 3);
  std::vector<int64_t> batchs = result.first, mnk = result.second;
  auto typeC = tools::getDType(builder, dtypes[2]);

  // create funcOp
  mlir::func::FuncOp funcOp = createFunc(builder, batchs, mnk, dtypes, kernelName, isTransposeA, isTransposeB);
  mlir::ValueRange operands = funcOp.getArguments();

  // create bacth nest forOp
  auto batchIvs = createBatchNestForOp(builder, batchs);
  
  // matmul nest 
  mlir::Location loc_ = builder.getUnknownLoc();
  mlir::SmallVector<int64_t, 3> lowerBounds = {0, 0};
  mlir::SmallVector<int64_t, 3> steps = {1, 1};
  mlir::SmallVector<int64_t, 3> upperBounds = {mnk[0], mnk[1]};
  mlir::affine::buildAffineLoopNest(builder, loc_, lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto row = ivs[0];
      auto col = ivs[1];

      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(loc, nestedBuilder.getFloatAttr(typeC, 0));

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value k, mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);

        auto indexA = getShapeOrIndex<mlir::Value>(batchIvs, {row, k}, isTransposeA);
        auto indexB = getShapeOrIndex<mlir::Value>(batchIvs, {k, col}, isTransposeB);

        auto ld_a = builder.create<mlir::affine::AffineLoadOp>(nestedLoc, /*A*/operands[0], mlir::ValueRange(indexA));
        auto ld_b = builder.create<mlir::affine::AffineLoadOp>(nestedLoc, /*B*/operands[1], mlir::ValueRange(indexB));
        auto mul = builder.create<mlir::arith::MulFOp>(nestedLoc, ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(nestedLoc, mul, iterArgs[0]);
        builder.create<mlir::affine::AffineYieldOp>(nestedLoc, add.getResult());
      };
      auto Cij = nestedBuilder.create<mlir::affine::AffineForOp>(loc, /*lb*/0, mnk[2], 1, mlir::ValueRange({zero.getResult()}), kLoopBody);
      auto indexC = getShapeOrIndex<mlir::Value>(batchIvs, {row, col}, false);
      nestedBuilder.create<mlir::affine::AffineStoreOp>(loc, Cij.getResult(0), /*C*/operands[2], mlir::ValueRange(indexC));
    });
}


std::optional<std::string> Matmul::verify(
  mlir::OpBuilder builder, std::vector<int64_t> shape, const std::vector<std::string>& dtypes) {
  if (shape.size() < 3 || shape.size() > 5) {
    std::string err{"Shape size must is 3, 4 or 5."};
    return err;
  }
  if(dtypes.size() != 3) {
    std::string err{"dtypes size must is 3."};
    return err;
  }
  for(auto dtype : dtypes){
    auto type = tools::getDType(builder, dtype);
    if (type == nullptr) {
      std::string err{"No exist this data type."};
      return err;
    }
  }
  return std::nullopt;
}

mlir::func::FuncOp Matmul::createFunc(
  mlir::OpBuilder& builder, 
  std::vector<int64_t> batchs, 
  std::vector<int64_t> shape, 
  const std::vector<std::string>& dtypes, 
  const std::string& kernelName,
  bool isTransposeA,
  bool isTransposeB
  )
{
  std::vector<mlir::Type> mlirTypeArray;
  for(auto type : dtypes) {
    auto mlirType = tools::getDType(builder, type);
    mlirTypeArray.push_back(mlirType);
  }
  auto shape_a = getShapeOrIndex(batchs, {shape[0], shape[2]}, isTransposeA);
  auto shape_b = getShapeOrIndex(batchs, {shape[2], shape[1]}, isTransposeB);
  auto shape_c = getShapeOrIndex(batchs, {shape[0], shape[1]}, false);
  auto ms = MemorySpace::global;
  auto typeA = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_a), mlirTypeArray[0], {}, static_cast<int>(ms));
  auto typeB = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_b), mlirTypeArray[1], {}, static_cast<int>(ms));
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_c), mlirTypeArray[2], {}, static_cast<int>(ms));
  Matmul::s_function = kernelName;

  return buildFunction(builder, kernelName, "Matmul", {typeA, typeB, typeC}, 1);
}

}  // Operators
}  // KernelCodeGen


