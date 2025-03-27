#ifndef _KernelCodeGen_h_
#define _KernelCodeGen_h_

#include "Operators/Operators.h"
#include "Operators/Matmul.h"
#include "Operators/Softmax.h"

#include "Conversion/FuseMap.h"
#include "Conversion/Optimizer.h"
#include "Conversion/LoweringPasses.h"

#include "Target/LLVMIRTranslation.h"
#include "Target/HSACOTranslation.h"
#include "Target/PTXTranslation.h"


namespace KernelCodeGen {

  class KernelCodeGenerator {
    using Config = std::map<std::string, std::vector<std::map<std::string, int>>>;
  public:
    KernelCodeGenerator(Target target_, const std::string& arch_) : target(target_), arch(arch_) {}

    KernelCodeGenerator(const KernelCodeGenerator& other);
    
    template <typename OperatorType, typename... Args> 
    void create(mlir::ModuleOp mod, Args &&...args) {
      mlir::MLIRContext* context = mod.getContext();
      context->getOrLoadDialect<mlir::affine::AffineDialect>();
      context->getOrLoadDialect<mlir::memref::MemRefDialect>();
      context->getOrLoadDialect<mlir::func::FuncDialect>();
      context->getOrLoadDialect<mlir::arith::ArithDialect>();
      context->getOrLoadDialect<mlir::gpu::GPUDialect>();
      context->getOrLoadDialect<mlir::vector::VectorDialect>();
      context->getOrLoadDialect<mlir::scf::SCFDialect>();
      context->getOrLoadDialect<mlir::math::MathDialect>();
      context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
      context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
      OperatorType::buildNaiveExpress(mod, std::forward<Args>(args)...);
    }

    std::vector<std::string> createModel(mlir::ModuleOp& mod, std::vector<KernelData> kernelList);

    bool fusing(mlir::ModuleOp& mod, std::vector<FuseKernelData> fkList);

    bool mapping(mlir::ModuleOp& mod);

    bool optimize(mlir::ModuleOp& mod, std::map<std::string, int> config);

    bool lowering(mlir::ModuleOp& mod, std::vector<int>& griddims, std::vector<int>& blockdims, int& shmbytes);

    std::string translate(mlir::ModuleOp& mod);
    
    template <typename OperatorType>
    std::string kernelFuncName(){
      return OperatorType::getKernelName();
    }

  private:
    Target target;
    const std::string arch;
  };

}
#endif