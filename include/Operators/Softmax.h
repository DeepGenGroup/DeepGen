#ifndef __Softmax_h__
#define __Softmax_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {
  
struct Softmax : Operator<Softmax> {
  static void buildNaiveExpress(mlir::ModuleOp module, 
                                const std::vector<std::vector<int64_t>>& inputShape, 
                                const std::vector<std::vector<int64_t>>& outputShape, 
                                const std::vector<std::string>& inputDType,
                                const std::vector<std::string>& outputDType,
                                const std::vector<bool>& isTranspose,
                                const std::string& kernelName);

  static std::optional<std::string> verify(const std::vector<std::vector<int64_t>>& inputShape, 
                                          const std::vector<std::vector<int64_t>>& outputShape, 
                                          const std::vector<std::string>& inputDType,
                                          const std::vector<std::string>& outputDType,
                                          const std::vector<bool>& isTranspose);
  
  static mlir::func::FuncOp createFunc(mlir::OpBuilder& builder, 
                                      const std::vector<std::vector<int64_t>>& inputShape, 
                                      const std::vector<std::vector<int64_t>>& outputShape, 
                                      const std::vector<std::string>& inputDType,
                                      const std::vector<std::string>& outputDType,
                                      const std::vector<bool>& isTranspose,
                                      const std::string& kernelName);

  static std::string s_function;

  static std::string getKernelName(){
    return s_function;
  }

};

}

}  // KernelCodeGen

#endif //  __Softmax_h__