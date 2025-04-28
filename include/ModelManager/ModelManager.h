#ifndef _ModelManager_h_
#define _ModelManager_h_

#include "Common/Utils.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include <vector>
#include "mlir/IR/MLIRContext.h"
#include <unordered_map>

namespace KernelCodeGen {

class DOMTreeNode {
public:
    mlir::Operation* op = nullptr;
    std::string opName;
    std::vector<DOMTreeNode*> parents;
    std::vector<DOMTreeNode*> childs;
    DOMTreeNode() = default;
    void addUniqueChild(DOMTreeNode* node);
    void addChild(DOMTreeNode* node);
    void addUniqueParent(DOMTreeNode* node);
};

class LowerInfo {
public:
    mlir::Operation* outmostForOp;
    std::vector< mlir::Operation*> outStoreOp;   // affine.store
    std::vector< mlir::Operation*> inputLoadOp;  // affine.load
};

class ModelManager{
public :
    bool process(const std::string& filepath);
private:
    bool seperateMaingraph(mlir::ModuleOp* root, std::vector<mlir::ModuleOp*> submodules);
    // 图优化
    bool graphOptimize();
    // 构建支配树
    bool getDOMTree(mlir::ModuleOp* graph);
    void buildDomNodes(mlir::func::FuncOp func);
    DOMTreeNode* buildNode(mlir::Operation* op, bool noparent ,bool nochild, std::unordered_map<mlir::Operation*, DOMTreeNode*>& hash);
    
    void analyzeRetOp(mlir::ModuleOp* mod);

    // torchMLIR lower to Linalg
    bool torchMLIRLowerToLinalg();
    bool insertKernelNaiveExpressionsToRootModule();
    std::vector<mlir::ModuleOp> m_modules;

private:
    bool isRootFunction(mlir::func::FuncOp& mod);
    void markAsRootFunction(mlir::func::FuncOp & mod);
    mlir::ModuleOp* m_rootModule;
    mlir::MLIRContext m_ctx;
    std::unordered_map<mlir::Operation*, DOMTreeNode*> m_domNodes ;


};

} // KernelCodeGen

#endif  // _ModelManager_h_