#include "Operators/Matmul.h"

namespace KernelCodeGen {
namespace Operators {

std::string Matmul::s_function = "Unknown";

void Matmul::buildNaiveExpress(mlir::ModuleOp module, 
  std::vector<int64_t> shape, 
  const std::vector<std::string>& dtypes,
  const std::string& kernelName,
  bool isTransposeA 
  ) 
{
  mlir::OpBuilder builder(module);
  auto ver = verify(builder, shape, dtypes);
  if (ver.has_value()) {
    llvm::errs() << ver.value() << "\n";
    return ;
  }

  int64_t m = shape[0];
  int64_t n = shape[1];
  int64_t k = shape[2];
  std::vector<mlir::Type> mlirTypeArray;
  for(auto type : dtypes){
    auto mlirType = tools::getDType(builder,type);
    mlirTypeArray.push_back(mlirType);
  }
  mlir::func::FuncOp funcOp = createFunc(module, shape, mlirTypeArray, kernelName,isTransposeA);
  auto& bodyBlock = funcOp.front();

  // auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&bodyBlock);
  mlir::ValueRange operands = bodyBlock.getArguments();

  mlir::Location loc_ = builder.getUnknownLoc();
  mlir::SmallVector<int64_t, 3> lowerBounds = {0,0};
  mlir::SmallVector<int64_t, 3> steps = {1,1};
  mlir::SmallVector<int64_t, 3> upperBounds = {m, n};
  mlir::affine::buildAffineLoopNest(builder, loc_, lowerBounds, upperBounds, steps,
    [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
      auto i = ivs[0];
      auto j = ivs[1];

      auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(loc, nestedBuilder.getFloatAttr(mlirTypeArray[2], 0));

      auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
        mlir::OpBuilder::InsertionGuard nestedGuard(builder);
        auto k = iv;
        mlir::affine::AffineLoadOp ld_a = nullptr;
        if(!isTransposeA){
          ld_a = builder.create<mlir::affine::AffineLoadOp>(nestedLoc, /*A*/operands[0], mlir::ValueRange({i, k}));
        }
        else{
          ld_a = builder.create<mlir::affine::AffineLoadOp>(nestedLoc, /*A*/operands[0], mlir::ValueRange({k, i}));
        }
        auto ld_b = builder.create<mlir::affine::AffineLoadOp>(nestedLoc, /*B*/operands[1], mlir::ValueRange({k, j}));
        auto mul = builder.create<mlir::arith::MulFOp>(nestedLoc, ld_a, ld_b);
        auto add = builder.create<mlir::arith::AddFOp>(nestedLoc, mul, iterArgs[0]);
        builder.create<mlir::affine::AffineYieldOp>(nestedLoc, add.getResult());
      };
      auto Cij = nestedBuilder.create<mlir::affine::AffineForOp>(loc, /*lowerBound*/0, k, /*step*/1, mlir::ValueRange({zero.getResult()}), kLoopBody);

      nestedBuilder.create<mlir::affine::AffineStoreOp>(loc, Cij.getResult(0), /*C*/operands[2], mlir::ValueRange({i, j}));
    }
  );
  // builder.restoreInsertionPoint(ip);
}


std::optional<std::string> Matmul::verify(
  mlir::OpBuilder builder, std::vector<int64_t> shape, const std::vector<std::string>& dtypes) {
  if (shape.size() != 3) {
    std::string err{"Shape size must is 3."};
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
  mlir::ModuleOp module, 
  std::vector<int64_t> shape, 
  const std::vector<mlir::Type>& dtype, 
  const std::string& kernelName,
  bool isTransposeA
  )
{
  int64_t m = shape[0];
  int64_t n = shape[1];
  int64_t k = shape[2];
  // std::cout << "[D]createFunc: m=" << m << ";n=" << n << ";k=" << k << "isTransposeA=" << isTransposeA <<std::endl;
  std::vector<int64_t> shape_a;
  if(!isTransposeA){
   shape_a = std::vector<int64_t>{m, k};
  }
  else{
   shape_a = std::vector<int64_t>{k, m};
  }
  auto shape_b = std::vector<int64_t>{k, n};
  auto shape_c = std::vector<int64_t>{m, n};
  auto ms = MemorySpace::global;
  auto typeA = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_a), dtype[0], {}, static_cast<int>(ms));
  auto typeB = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_b), dtype[1], {}, static_cast<int>(ms));
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_c), dtype[2], {}, static_cast<int>(ms));
  Matmul::s_function = kernelName;

  return buildFunction(module, kernelName, "Matmul", {typeA, typeB, typeC});
}

}  // Operators
}  // KernelCodeGen


