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

class AffineGroupInfo {
public:
    mlir::Operation* outmostForOp;
    std::vector< mlir::Operation*> storeOps;   // affine.store
    std::vector< mlir::Operation*> loadOps;  // affine.load
};

/// @brief 与ONNX模型交互，将模型接入torchIR-> stableHLO -> linalg
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
    // 从ReturnOp开始递归地分析，得到for循环结构之间的依赖关系
    void analyzeRetOp(mlir::ModuleOp* mod);
    // 
    void hoistAllocOp(mlir::ModuleOp* mod);
    void memrefShrinkDim(mlir::ModuleOp* mod);
    AffineGroupInfo* getInfoFromDefineOp(mlir::Operation* memrefStoreOp);

    // torchMLIR lower to Linalg
    bool torchMLIRLowerToLinalg();
    bool insertKernelNaiveExpressionsToRootModule();

    void lowerStableHLOToAffine(mlir::ModuleOp* mod);
    void lowerTorchToStableHLO(mlir::ModuleOp* mod);
    void lowerOnnxIRToTorch(mlir::ModuleOp* mod);

    void analyzeParallelizabilityOfAffine(mlir::ModuleOp* mod);
    void init();

    std::vector<mlir::ModuleOp> m_modules;

private:
    bool isRootFunction(mlir::func::FuncOp& mod);
    void markAsRootFunction(mlir::func::FuncOp & mod);

    mlir::Operation* getInnerMostParallelableLoop(mlir::Operation* innermostOp);
    mlir::ModuleOp* m_rootModule;
    std::unique_ptr<mlir::MLIRContext> m_ctx {nullptr};
    std::unordered_map<mlir::Operation*, DOMTreeNode*> m_domNodes ;
    std::unordered_map<std::string, std::string> m_onnxOpLocationMap;  // 存放 onnx.mlir中 location->op对应关系。

};



} // KernelCodeGen

#endif  // _ModelManager_h_