#include "ModelManager/ModelManager.h"
#include "KernelCodeGen.h"

namespace KernelCodeGen {

ModelManager::ModelManager(mlir::ModuleOp& mod) : 
    m_rootModule(mod)
{ 
    ;
}

bool ModelManager::process()
{
    torchMLIRLowerToLinalg();
    // mark 'forward' as root func
    m_rootModule.walk([&](mlir::func::FuncOp op){
        this->markAsRootFunction(op);
    });
    // optimize based on root func
    graphOptimize();
    // based on optimized graph, insert naive operator expression funcs into rootModule
    insertKernelNaiveExpressionsToRootModule();
    // move kernel funcs from rootFunc to new modules
    seperateKernelFuncIntoModules();
    return true;
}

/**
 * @brief 基于 root module & forward function ，将其中出现的 linalg.matmul 等算子替换为 func.call, 
 * 并在root module里插入对应的算子朴素表达function
 * 
 * @return true 
 * @return false 
 */
bool ModelManager::insertKernelNaiveExpressionsToRootModule()
{
    auto& ops = this->m_rootModule.getBody()->getOperations();
    mlir::OpBuilder b(m_rootModule.getContext());
    KernelCodeGenerator gen(Target::ROCm,"906");
    for(auto& op : ops){
        if(tools::isOpAttrEqualToString(&op,AttrKernelType,
            tools::KcgKernelTypeToString(KcgKernelType::matmul))){
            // auto kernelmod = gen.create<Operators::Matmul>(
            //     std::vector<int64_t>{M, N, K},
            //     std::vector<std::string>{dtypeA,dtypeB,dtypeC},
            //     name,isATranspose
            // );
            continue;
        }
        if(tools::isOpAttrEqualToString(&op,AttrKernelType,
            tools::KcgKernelTypeToString(KcgKernelType::conv2d))){
            // auto kernelmod = gen.create<Operators::Matmul>(
            //     std::vector<int64_t>{M, N, K},
            //     std::vector<std::string>{dtypeA,dtypeB,dtypeC},
            //     name,isATranspose
            // );
            continue;
        }
        if(tools::isOpAttrEqualToString(&op,AttrKernelType,
            tools::KcgKernelTypeToString(KcgKernelType::poolmax))){
            // auto kernelmod = gen.create<Operators::Matmul>(
            //     std::vector<int64_t>{M, N, K},
            //     std::vector<std::string>{dtypeA,dtypeB,dtypeC},
            //     name,isATranspose
            // );
            continue;
        }
    }
    return true;
}

bool ModelManager::seperateKernelFuncIntoModules()
{
    auto loc = m_rootModule.getLoc();
    int i=0;
    this->m_modules.push_back(m_rootModule);
    m_rootModule.walk([&](mlir::func::FuncOp op){
      if(isRootFunction(op)){
        ;
      }
      else{
        auto funName = op.getName();
        auto newMod = mlir::ModuleOp::create(loc,funName);
        mlir::OpBuilder b(newMod.getContext());
        b.setInsertionPointToEnd(newMod.getBody());
        auto dumpOp = b.create<mlir::arith::ConstantIntOp>(b.getUnknownLoc(),111,32);
        op->moveAfter(dumpOp);
        dumpOp.erase();
        m_modules.push_back(newMod);
      }
    });
    return true;
}

void ModelManager::markAsRootFunction(mlir::func::FuncOp & op){
    tools::opSetAttr(op,"kcg.isRoot","1");
}
bool ModelManager::isRootFunction(mlir::func::FuncOp & op){
    return op->hasAttr("kcg.isRoot");
}

/**
 * @brief 对forward函数里的算子进行walk，分析性质并添加属性用来标记其种类
 * 
 * @return true 
 * @return false 
 */
bool ModelManager::graphOptimize()
{
    // 处理完毕后或处理过程里，调用 generateKernelNaiveExpressions 在rootModule里插入朴素表达的funcop
    // todo ...
    // ... (graph optimize algorithm ) ...
    // optimize ok, add attr to operators
    for(auto& op : this->m_rootModule.getBody()->getOperations()){
        if(mlir::dyn_cast<mlir::linalg::MatmulOp>(op) != nullptr){
            tools::opSetAttr(&op, AttrKernelType, tools::KcgKernelTypeToString(KcgKernelType::matmul));
            continue;
        }
        if(mlir::dyn_cast<mlir::linalg::Conv2DNchwFchwOp>(op) != nullptr){
            tools::opSetAttr(&op, AttrKernelType, tools::KcgKernelTypeToString(KcgKernelType::conv2d));
            continue;
        }
        if(mlir::dyn_cast<mlir::linalg::PoolingNchwMaxOp>(op) != nullptr){
            tools::opSetAttr(&op, AttrKernelType, tools::KcgKernelTypeToString(KcgKernelType::poolmax));
            continue;
        }
    }
    return true;
}
bool ModelManager::torchMLIRLowerToLinalg()
{
    // lower MLIR to linalg
    return true;
}

}  // KernelCodeGen