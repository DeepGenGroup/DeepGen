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
#include "torch-mlir/InitAll.h"
#include "llvm/Support/raw_ostream.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/InitAllExtensions.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"
#include "stablehlo/transforms/Passes.h"


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
    
    void ModelManager::memrefShrinkDim(mlir::ModuleOp* mod)
    {
        std::vector<mlir::memref::AllocOp> opToShrink;
        mod->walk([&](mlir::memref::AllocOp op){
            auto type = op.getResult().getType();
            auto mem = mlir::dyn_cast<mlir::MemRefType>(type);
            if(mem != nullptr){
                const auto& shape = mem.getShape();
                int totalSize = 1;
                for(auto i : shape){
                    totalSize *= i;
                }
                if(totalSize==1){
                    opToShrink.push_back(op);
                }
            }
        });
        mlir::OpBuilder builder{mod->getContext()};
        for(auto op : opToShrink){
            llvm::outs() << "replace : " << op.getLoc() << "\n"; llvm::outs().flush();
            auto oldtype = op.getResult().getType();
            auto mem = mlir::dyn_cast<mlir::MemRefType>(oldtype);
            auto etype = mem.getElementType();
            auto newop = builder.create<mlir::memref::AllocOp>(op.getLoc(), mlir::MemRefType::get({1},etype)) ;
            for(auto user : op.getResult().getUsers()){
                if(auto ptr = mlir::dyn_cast<mlir::affine::AffineLoadOp>(user)){
                    ptr.getAffineMap();
                }
            } 
        }
    }
    void ModelManager::hoistAllocOp(mlir::ModuleOp* mod)
    {
        mlir::Operation* firstAllocOp = nullptr;
        mod->walk<mlir::WalkOrder::PreOrder>([&](mlir::memref::AllocOp op){
            if(firstAllocOp == nullptr){
                firstAllocOp = op.getOperation();
            }
        });
        llvm::outs() << "First  loc = " << firstAllocOp->getLoc() << "\n" ; llvm::outs().flush();
        std::vector<mlir::Operation*> opToMove{};
        mod->walk([&](mlir::memref::AllocOp op){
            std::string parentOpName = op.getOperation()->getParentOp()->getName().getStringRef().data();
            if(parentOpName.find("func.func") != std::string::npos && op.getOperation() != firstAllocOp){
                opToMove.push_back(op.getOperation());
            }
        });
        for(auto op : opToMove){
            op->moveAfter(firstAllocOp);
        }

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
        std::vector<AffineGroupInfo*> infos;
        std::queue<AffineGroupInfo*> q;
        auto info = getInfoFromDefineOp(memrefOp);
        q.push(info);

        while (!q.empty())
        {
            auto info = q.front();
            q.pop();
            if(info != nullptr){
                infos.push_back(info);        
                for(auto loadop : info->loadOps){
                    for(auto user : loadop->getOperand(0).getUsers()){
                        std::string name = user->getName().getStringRef().data();
                        if(name.find("affine.store") != std::string::npos){
                            auto tempInfo = getInfoFromDefineOp(user->getOperand(1).getDefiningOp());
                            q.push(tempInfo);
                        }
                    }
                }
            }
        }
        for(auto info : infos){
            tools::_opSetDescription(info->outmostForOp,"outmost");
            
        }
        std::cout << "collected Info num = " << infos.size() << std::endl;
        return;
    }


    AffineGroupInfo* ModelManager::getInfoFromDefineOp(mlir::Operation* memrefDefOp)
    {
        AffineGroupInfo* ret = new AffineGroupInfo();
        if(memrefDefOp != nullptr){
            llvm::outs() << "--- getinfo : " << memrefDefOp->getName().getStringRef() << " : " << memrefDefOp->getLoc() << "\n"; llvm::outs().flush();
        }
        for(auto user : memrefDefOp->getUsers()) {
            if(user != nullptr){
                std::string name = user->getName().getStringRef().data();
                llvm::outs() << "[user name]" << name << "\n"; llvm::outs().flush();
                if(name.find("store") != std::string::npos){
                    ret->storeOps.push_back(user);
                    for(auto o : user->getOperands()){  // operands of affine.store(val, targetMem, indexes) op
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
                                            ret->outmostForOp = ivsOuterForOp;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if(ret->outmostForOp != nullptr){
            ret->outmostForOp->walk([&](mlir::affine::AffineLoadOp loadop){
                ret->loadOps.push_back(loadop.getOperation());
            });
        }
        return ret;
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

    void ModelManager::lowerOnnxIRToTorch(mlir::ModuleOp* mod){
        mlir::PassManager pm(m_ctx.get());
        // pm.addNestedPass<mlir::func::FuncOp>(mlir::torch::onnx_c::createTorchOnnxToTorchPass());
        mlir::torch::Torch::TorchLoweringPipelineOptions opt;
        mlir::torch::Torch::createTorchOnnxToTorchBackendPipeline(pm, opt);
        // mlir::torch::Torch::createTorchFunctionToTorchBackendPipeline(pm, opt);
        // mlir::torch::Torch::createTorchSimplificationPipeline(pm,opt);
        pm.run(mod->getOperation());
        llvm::outs() << "\n======== onnx->torch ===========\n";llvm::outs().flush();mod->dump();
        return;
    }


    void ModelManager::lowerTorchToStableHLO(mlir::ModuleOp* mod){
        
        mlir::PassManager pm(m_ctx.get());
        mlir::torch::TorchConversion::StablehloBackendPipelineOptions opt;
        mlir::torch::TorchConversion::createTorchBackendToStablehloBackendPipeline(pm, opt);
        pm.addNestedPass<mlir::func::FuncOp>(mlir::stablehlo::createStablehloCanonicalizeDynamismPass());
        pm.addPass(mlir::stablehlo::createStablehloAggressiveSimplificationPass());
        pm.run(mod->getOperation());
        llvm::outs() << "\n======== torch->stablehlo ===========\n";llvm::outs().flush();mod->dump();

        return;
    }

    void ModelManager::lowerStableHLOToAffine(mlir::ModuleOp* mod){
        mlir::PassManager pm(m_ctx.get());
        mlir::stablehlo::StablehloLegalizeToLinalgPassOptions stablehloToLinalgOpt;
        stablehloToLinalgOpt.enablePrimitiveOps = true;
        stablehloToLinalgOpt.enableSparseOps = false;
        pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass(stablehloToLinalgOpt));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.run(mod->getOperation());
        llvm::outs() << "\n======== stablehlo -> linalg =========== \n"; llvm::outs().flush();mod->dump();
        

        mlir::bufferization::OneShotBufferizePassOptions opt;
        opt.bufferizeFunctionBoundaries = true;
        opt.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
        opt.allowUnknownOps = true;

        pm.addPass(::mlir::bufferization::createOneShotBufferizePass(opt));
        pm.run(mod->getOperation());
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createLoopInvariantCodeMotionPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createRemoveDeadValuesPass());
        pm.addPass(mlir::createSymbolDCEPass());
        pm.run(mod->getOperation());
        llvm::outs() << "\n======== linalg->affine (with LICM) =========== \n"; llvm::outs().flush();mod->dump();
        
        return;
    }

    void ModelManager::init(){
        mlir::DialectRegistry registry;
        mlir::registerAllExtensions(registry);
        mlir::linalg::registerAllDialectInterfaceImplementations(registry);
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::torch::registerAllDialects(registry);
        mlir::torch::registerAllExtensions(registry);
        mlir::torch::registerOptionalInputDialects(registry);
        m_ctx = std::make_unique<mlir::MLIRContext>(registry);

        // Import Module from IR text
        m_ctx->loadDialect<func::FuncDialect, arith::ArithDialect,stablehlo::StablehloDialect, torch::Torch::TorchDialect, 
                        memref::MemRefDialect, affine::AffineDialect, math::MathDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                        chlo::ChloDialect, torch::TorchConversion::TorchConversionDialect>();
    }

    void ModelManager::analyzeParallelizabilityOfAffine(mlir::ModuleOp* mod){
        std::vector<mlir::Operation*> outmostForLoops;
        mlir::Operation* mainGraphFunc = nullptr;
        mod->walk([&](mlir::func::FuncOp funcop){
            if(mainGraphFunc != nullptr){
                return;
            }
            mainGraphFunc = funcop.getOperation();
        });
        mod->walk([&](mlir::affine::AffineForOp forop){
            auto parentop = forop.getOperation()->getParentOp();
            if(parentop == mainGraphFunc){
                outmostForLoops.push_back(forop.getOperation());
            }
        });
        llvm::outs() << "outmostLoop size = " << outmostForLoops.size() << "\n";llvm::outs().flush();
        for(auto outmostLoop : outmostForLoops){
            tools::opSetAttr(outmostLoop,"loopLevel","outmostLoop");
            bool isWriteAfterRead = false;
            outmostLoop->walk([&](mlir::affine::AffineStoreOp op){
                auto memLoc = op.getOperand(op.getMemRefOperandIndex());
                for(auto user : memLoc.getUsers()){  // 这里打印出的location 依然基于 onnx.mlir， 即可以通过loc来映射 forlops - onnx.operator
                    auto opParentLoc = op->getParentOp()->getLoc();
                    auto userParentLoc = user->getParentOp()->getLoc();
                    // llvm::outs() <<"Loc: " << user->getParentOp()->getName() << " | " << op->getParentOp()->getName() << "\n";llvm::outs().flush();
                    if(op.getOperation() == user){
                        continue;
                    }
                    if(op->getParentOp() == user->getParentOp()){
                        isWriteAfterRead = true;
                        tools::opSetAttr(op->getParentOp(),"canParallel","no");
                    }
                } 
            });
        }
        for(auto outmostLoop : outmostForLoops){
            // 添加位置标识
            std::string loc = tools::getLocationString(outmostLoop->getLoc());
            auto it = m_onnxOpLocationMap.find(loc);
            if(it != m_onnxOpLocationMap.end()){
                std::string onnxOpName = it->second;
                tools::opSetAttr(outmostLoop,"onnx.op",onnxOpName);
                tools::opSetAttr(outmostLoop,"onnx.loc",loc);
            }
            outmostLoop->walk([&](mlir::affine::AffineStoreOp op){
                auto innerMostLoop = op.getOperation()->getParentOp();
                auto xloop = getInnerMostParallelableLoop(innerMostLoop);
                auto yloop = xloop->getParentOp();
                tools::opSetAttr(xloop,"loop.desc","x");
                tools::opSetAttr(yloop,"loop.desc","y");
            });
        }

    }
    
    mlir::Operation* ModelManager::getInnerMostParallelableLoop(mlir::Operation* innermostOp){
        if(tools::isOpAttrEqualToString(innermostOp,"canParallel","no")){
            auto op = innermostOp->getParentOp();
            if(mlir::dyn_cast<mlir::affine::AffineForOp>(op)){
                return getInnerMostParallelableLoop(op);
            }
            return nullptr;
        }
        else{
            return innermostOp;
        }
    }

    bool ModelManager::process(const std::string& filepath)
    {
        init();
        mlir::OwningOpRef<mlir::ModuleOp> mod = parseSourceFile<ModuleOp>(filepath, m_ctx.get());
        auto _mod = mod.operator->();
        auto getOpName = [](Operation *op) -> std::string {
          std::string name = op->getName().getStringRef().str();
          if (name != "torch.operator")
            return name;
          // for unconverted onnx ops
          return mlir::dyn_cast<StringAttr>(op->getAttr("name"))
              .getValue()
              .str();
        };
        _mod->walk([&](mlir::Operation* op){
            auto p = std::make_pair(tools::getLocationString(op->getLoc()), getOpName(op));
            m_onnxOpLocationMap.insert(p);
        });
        for(const auto& p : m_onnxOpLocationMap){
            llvm::outs() << "onnxMap : " << p.first << " -> " << p.second << "\n";llvm::outs().flush();
        }
        // hoistAllocOp(_mod);
        lowerOnnxIRToTorch(_mod);
        lowerTorchToStableHLO(_mod);
        lowerStableHLOToAffine(_mod);
        // analyzeRetOp(_mod);
        analyzeParallelizabilityOfAffine(_mod);
        // memrefShrinkDim(_mod);
        
        // getDOMTree(_mod);
        // std::vector<mlir::ModuleOp*> submodules;
        // auto ret = seperateMaingraph(_mod, submodules);
        mod->dump();
        return true;
    }


};
