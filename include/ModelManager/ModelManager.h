#ifndef _ModelManager_h_
#define _ModelManager_h_

#include "Common/Utils.h"
#include <vector>

namespace KernelCodeGen {

class ModelManager{
public :
    explicit ModelManager(mlir::ModuleOp& mod);
    bool process();
private:

    // 将module中的func根据attr 拆分进多个module。每个module对应一个kernel，存入 m_kernels
    bool seperateKernelFuncIntoModules();
    // 图优化
    bool graphOptimize();
    // torchMLIR lower to Linalg
    bool torchMLIRLowerToLinalg();
    bool insertKernelNaiveExpressionsToRootModule();
    std::vector<mlir::ModuleOp> m_modules;

private:
    bool isRootFunction(mlir::func::FuncOp& mod);
    void markAsRootFunction(mlir::func::FuncOp & mod);
    mlir::ModuleOp m_rootModule;

};

} // KernelCodeGen

#endif  // _ModelManager_h_