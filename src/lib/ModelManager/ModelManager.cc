#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "ModelManager/ModelManager.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/ChloOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace KernelCodeGen
{
    bool ModelManager::seperateMaingraph(mlir::ModuleOp* root, std::vector<mlir::ModuleOp*> submodules)
    {
        mlir::OpBuilder builder(root->getContext());
        std::vector<std::string> opnames ;
        for(auto& funcOp : root->getOps()){
            funcOp.setAttr("graph.level",builder.getStringAttr("main"));
            auto _funcOp = mlir::dyn_cast<mlir::func::FuncOp>(funcOp);
            for(auto& region : _funcOp->getRegions()){
                for(auto& op : region.getOps()){
                    auto d = op.getName().getStringRef().data();
                    opnames.push_back(std::string{d});
                }
            }
        }
        std::cout << "==========\n" ;
        for(int i=0;i<opnames.size();++i){
            std::cout << opnames[i] << std::endl;
        }
        return true;
    }

    bool ModelManager::process(const std::string& filepath)
    {
        // Import Module from IR text
        m_ctx.loadDialect<func::FuncDialect, arith::ArithDialect,stablehlo::StablehloDialect, torch::Torch::TorchDialect, 
                        chlo::ChloDialect, torch::TorchConversion::TorchConversionDialect>();
        mlir::OwningOpRef<mlir::ModuleOp> mod = parseSourceFile<ModuleOp>(filepath, &m_ctx);
        auto _mod = mod.operator->();
        std::vector<mlir::ModuleOp*> submodules;
        auto ret = seperateMaingraph(_mod, submodules);
        mod->dump();
        return ret;
    }


};
