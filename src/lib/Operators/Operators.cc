#include "Operators/Operators.h"
#include "Common/Utils.h"
#include <filesystem>

namespace KernelCodeGen {


mlir::func::FuncOp buildFunction(mlir::ModuleOp module, const std::string& funcName, const std::string& OpName, 
                                 const std::vector<mlir::Type>& inputsTypes) {
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  llvm::ArrayRef<mlir::Type> inputsTypesArray(inputsTypes);
  auto functionType = builder.getFunctionType(mlir::TypeRange(inputsTypesArray), mlir::TypeRange({}));
  auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), llvm::StringRef(funcName), functionType);
  
  auto& region = funcOp->getRegion(0);
  if (!region.hasOneBlock()) {
    region.emplaceBlock();
  }
  auto& body =  funcOp.front(); //? region.front()  : ;
  int nums = static_cast<int>(inputsTypes.size());
  for (int i = 0; i < nums; i++ ) {
    body.addArguments(inputsTypes[i], builder.getUnknownLoc());
  }
  
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  funcOp->setAttr(std::string("func.op.name"), builder.getStringAttr(OpName));
  funcOp->setAttr(std::string(AttrVisibility), builder.getStringAttr("public"));
  funcOp->setAttr(std::string(AttrKernelFunc), builder.getI32IntegerAttr(1));
  
  auto& entryBlock = funcOp.front();
  builder.setInsertionPointToStart(&entryBlock);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  return funcOp;
}

}