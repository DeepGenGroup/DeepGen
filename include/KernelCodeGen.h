#ifndef _KernelCodeGen_h_
#define _KernelCodeGen_h_

#include "Operators/Operators.h"
#include "Conversion/Optimizer.h"
#include "Conversion/LoweringPasses.h"
#include "Target/LLVMIRTranslation.h"
#include "Target/HSACOTranslation.h"
#include "Target/PTXTranslation.h"

#include "Operators/Matmul.h"
#include "Operators/Softmax.h"

namespace KernelCodeGen
{

  struct KernelData {
    /* kernel的数据结构 */
    std::string name;
    std::string type;
    std::vector<std::string> argNames;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::string> dtypes;
    std::vector<bool> isTrans;
    int outputArgNum;
  };

  struct FuseKernelData {
    /*融合kernel的list*/
    std::string fkName;
    std::string type;
    std::vector<std::string> fuseKernels;
    std::vector<std::vector<int64_t>> newArgsShape;
    std::vector<std::vector<int64_t>> newVarsShape;
    std::vector<std::string> newArgsDtype;
    std::vector<std::string> newVarsDtype;
    std::vector<std::map<std::string, int64_t>> newArgsIndex;
    std::vector<std::map<std::string, int64_t>> newVarsIndex;
    int outputArgNum;
  };
  

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