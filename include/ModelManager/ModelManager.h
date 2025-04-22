#ifndef _ModelManager_h_
#define _ModelManager_h_

#include "Common/Utils.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include <vector>
#include "mlir/IR/MLIRContext.h"

namespace KernelCodeGen {

class ModelManager{
public :
    bool process(const std::string& filepath);
private:
    bool seperateMaingraph(mlir::ModuleOp* root, std::vector<mlir::ModuleOp*> submodules);
    // 图优化
    bool graphOptimize();
    // torchMLIR lower to Linalg
    bool torchMLIRLowerToLinalg();
    bool insertKernelNaiveExpressionsToRootModule();
    std::vector<mlir::ModuleOp> m_modules;

private:
    bool isRootFunction(mlir::func::FuncOp& mod);
    void markAsRootFunction(mlir::func::FuncOp & mod);
    mlir::ModuleOp* m_rootModule;
    mlir::MLIRContext m_ctx;

};

} // KernelCodeGen

#endif  // _ModelManager_h_