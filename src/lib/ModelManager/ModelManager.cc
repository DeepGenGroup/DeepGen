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
    
    void ModelManager::buildDomNodes(mlir::func::FuncOp func)
    {
        func.walk([&](mlir::Operation* op){
            auto it = m_domNodes.find(op);
            if(it == m_domNodes.end()){
                auto node = new DOMTreeNode{};
                node->op = op;
                node->opName = op->getName().getStringRef().data();
                m_domNodes.insert(std::make_pair(op,node));
            }
        });
        mlir::Operation* retOp = nullptr; 
        func.walk([&](mlir::func::ReturnOp op){
            if(retOp == nullptr && tools::getIntAttr(op.getParentOp(),"isRoot") > 0){
                retOp = op.getOperation();
            }
        });

        for(auto& pair : m_domNodes){
            auto operands = pair.first->getOperands();
            for(auto operand : operands){
                auto defop = operand.getDefiningOp();
                if(defop == nullptr){
                    pair.second->parents.push_back(nullptr);
                }
                else{
                    auto parentNode = m_domNodes.find(defop)->second;
                    pair.second->parents.push_back(parentNode);
                }
            }
        }
        auto it = m_domNodes.find(retOp);
        assert(it != m_domNodes.end());
        auto retNode = it->second;
        return;
    }
    
    void ModelManager::analyzeRetOp(mlir::ModuleOp* mod)
    {
        mlir::func::ReturnOp retOp;
        bool flag = true;
        mod->walk([&](mlir::func::ReturnOp op){
            if(flag){
                retOp = op;
                flag = false;
            }
        });
        auto val = retOp.getOperand(0);
        auto memrefOp = val.getDefiningOp();
        llvm::outs() << memrefOp->getName().getStringRef() << "\n"; llvm::outs().flush();

        for(auto user : memrefOp->getUsers()) {
            if(user != nullptr){
                std::string name = user->getName().getStringRef().data();
                llvm::outs() << "[user name]" << name << "\n"; llvm::outs().flush();
                if(name.find("store") != std::string::npos){
                    for(auto o : user->getOperands()){
                        auto defop = o.getDefiningOp();
                        if(defop != nullptr){
                            llvm::outs() <<"[defOps]"<< defop->getName().getStringRef() << " - " << defop->getLoc() << "\n" ; llvm::outs().flush();
                        }
                        else{
                            auto pb = o.getParentBlock();
                            if(pb != nullptr){
                                auto ivsOuterForOp = pb->getParentOp();
                                if(ivsOuterForOp != nullptr){
                                    llvm::outs() <<"[forOp]"<< ivsOuterForOp->getName().getStringRef() << " - " << ivsOuterForOp->getLoc() << "\n" ; llvm::outs().flush();
                                    auto parentOp = ivsOuterForOp->getParentOp();
                                    if(parentOp != nullptr){
                                        std::string parentName = parentOp->getName().getStringRef().data();
                                        llvm::outs() << "parentOp = " << parentName << "\n"; llvm::outs().flush();
                                        if (parentName.find("func.func") != std::string::npos){
                                            
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    bool ModelManager::getDOMTree(mlir::ModuleOp* graph){
        DOMTreeNode* node = nullptr;
        bool flag = true;
        std::unordered_map<mlir::Operation*, DOMTreeNode*> hashtable {} ;

        graph->walk([&](mlir::func::FuncOp func){
            // if(node != nullptr){
            //     return;
            // }
            if(!flag){
                return;
            }
            if(func.getName() == "main_graph"){
                tools::opSetAttr(func,"isRoot",1);
                // func->walk([&](mlir::func::ReturnOp retop){
                //     if(node != nullptr){
                //         return;
                //     }
                //     node = buildNode(retop.getOperand(0).getDefiningOp(),true,false, hashtable);
                // });
                buildDomNodes(func);
                flag = false;
            }
        });
    
        return true;
    }
    
    void DOMTreeNode::addUniqueChild(DOMTreeNode* node){
        auto it = std::find(this->childs.begin(),this->childs.end(),node);
        if(it == this->childs.end()){
            childs.push_back(node);
        }
    }

    void DOMTreeNode::addUniqueParent(DOMTreeNode* node){
        auto it = std::find(this->parents.begin(),this->parents.end(),node);
        if(it == this->parents.end()){
            parents.push_back(node);
        }
    }

    void DOMTreeNode::addChild(DOMTreeNode* node){
        childs.push_back(node);
    }

    // hasparent haschild 控制迭代方向，防止父子间死循环
    DOMTreeNode* ModelManager::buildNode(mlir::Operation* op , bool hasparent ,bool haschild, std::unordered_map<mlir::Operation*, DOMTreeNode*>& hash)
    {
        if(tools::getIntAttr(op,"isRoot") > 0) {
            return nullptr;
        }
        DOMTreeNode* currNode = nullptr;
        auto it = hash.find(op);
        if(it != hash.end()){
            currNode = it->second;
        }
        else{
            currNode = new DOMTreeNode();
            hash.insert(std::make_pair(op,currNode));
            currNode->op = op;
            auto nameref = op->getName().getStringRef().data();
            currNode->opName = nameref;
        }

        if(hasparent){
            auto operands = op->getOperands();
            int index = 0;
            for(auto in : operands){
                auto defop = in.getDefiningOp();
                if(defop == nullptr){
                    currNode->parents.push_back(nullptr);
                    // llvm::outs() << nameref << " operand "<< index++ << " defop==NULL, loc=" << op->getLoc() << "\n" ; llvm::outs().flush();
                    continue;
                }
                else{
                    auto defOpNode = buildNode(defop,true,false,hash);
                    if(defOpNode != nullptr){
                        currNode->parents.push_back(defOpNode);
                        // defOpNode->addUniqueChild(currNode);
                    }
                    index++;
                }
            }
        }
        if(haschild){
            auto outs = op->getResults();
            for(auto out : outs){
                auto users = out.getUsers();
                for(auto childOp : users){
                    auto childNode = buildNode(childOp,true,false,hash);
                    if(childNode != nullptr){
                        currNode->childs.push_back(childNode);
                        // childNode->addUniqueParent(currNode);
                    }
                }
            }
        }
        return currNode;
    }

    bool ModelManager::process(const std::string& filepath)
    {
        // Import Module from IR text
        m_ctx.loadDialect<func::FuncDialect, arith::ArithDialect,stablehlo::StablehloDialect, torch::Torch::TorchDialect, 
                        memref::MemRefDialect, affine::AffineDialect, math::MathDialect, 
                        chlo::ChloDialect, torch::TorchConversion::TorchConversionDialect>();
        mlir::OwningOpRef<mlir::ModuleOp> mod = parseSourceFile<ModuleOp>(filepath, &m_ctx);
        auto _mod = mod.operator->();
        analyzeRetOp(_mod);
        // getDOMTree(_mod);
        // std::vector<mlir::ModuleOp*> submodules;
        // auto ret = seperateMaingraph(_mod, submodules);
        // mod->dump();
        return true;
    }


};
