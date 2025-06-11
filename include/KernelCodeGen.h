#ifndef _KernelCodeGen_h_
#define _KernelCodeGen_h_

#include "Operators/Operators.h"
#include "Operators/Matmul.h"
#include "Operators/Softmax.h"

#include "Conversion/Fusing.h"
#include "Conversion/Mapping.h"
#include "Conversion/Optimize.h"
#include "Conversion/LoweringPasses.h"

#include "Target/LLVMIRTranslation.h"
#include "Target/HSACOTranslation.h"
#include "Target/PTXTranslation.h"


namespace KernelCodeGen {

  class KernelCodeGenerator {
  public:
    KernelCodeGenerator(Target target, const std::string& arch) : target(target), arch(arch) {
      this->initContext();
    }
    KernelCodeGenerator(const KernelCodeGenerator& other);
    KernelCodeGenerator() {
      this->initContext();
    };

    void initContext() {
      context.getOrLoadDialect<mlir::affine::AffineDialect>();
      context.getOrLoadDialect<mlir::memref::MemRefDialect>();
      context.getOrLoadDialect<mlir::func::FuncDialect>();
      context.getOrLoadDialect<mlir::arith::ArithDialect>();
      context.getOrLoadDialect<mlir::gpu::GPUDialect>();
      context.getOrLoadDialect<mlir::vector::VectorDialect>();
      context.getOrLoadDialect<mlir::scf::SCFDialect>();
      context.getOrLoadDialect<mlir::math::MathDialect>();
      context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
      context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    }
    
    void setPaltform(Target tg, const std::string& ac) {
      this->target = tg;
      this->arch = ac;
    }

    template <typename OperatorType, typename... Args> 
    void create(mlir::ModuleOp mod,
                const std::vector<std::vector<int64_t>>& intputShape,
                const std::vector<std::vector<int64_t>>& outputShape,
                const std::vector<std::string>& inputDType,
                const std::vector<std::string>& outputDType,
                const std::vector<bool>& isTranspose, 
                Args &&...args) {
      OperatorType::buildNaiveExpress(mod, intputShape, outputShape, inputDType, outputDType, isTranspose, std::forward<Args>(args)...);
    }

    mlir::ModuleOp createModule();

    std::vector<std::string> createKernels(mlir::ModuleOp& mod, std::vector<KernelData> kernelList);

    bool fusing(mlir::ModuleOp& mod, std::vector<FuseKernelData> fkList);

    std::vector<mlir::ModuleOp> splitModule(mlir::ModuleOp& mod);

    bool mapping(mlir::ModuleOp& mod, const std::map<std::string, std::map<std::string, int64_t>>& tileConfig);

    bool optimize(mlir::ModuleOp& mod, const std::map<std::string, std::map<std::string, int64_t>>& tuneConfig);

    bool lowering(mlir::ModuleOp& mod/*, std::vector<int>& griddims, std::vector<int>& blockdims, int& shmbytes*/);

    std::string translate(mlir::ModuleOp& mod);
    
    template <typename OperatorType>
    std::string kernelFuncName(){
      return OperatorType::getKernelName();
    }

  private:
    Target target;
    std::string arch;
    mlir::MLIRContext context;
  };

}
#endif