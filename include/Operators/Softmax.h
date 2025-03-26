#ifndef __Softmax_h__
#define __Softmax_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {
  
struct Softmax : Operator<Softmax> {
  static void buildNaiveExpress(mlir::ModuleOp module, 
    std::vector<int64_t> shape, 
    const std::string& dtype,
    const std::string& kernelName,
    bool isTranspose = false
    );

  static std::optional<std::string> verify(mlir::OpBuilder builder, std::vector<int64_t> shape, const std::string& dtype);
  
  static mlir::func::FuncOp createFunc(mlir::OpBuilder& builder, 
    std::vector<int64_t> batchs, 
    std::vector<int64_t> shape, 
    const std::string& dtype,
    const std::string& kernelName,
    bool isTranspose = false
    );

  static std::string s_function;

  static std::string getKernelName(){
    return s_function;
  }

};

}

}  // KernelCodeGen

#endif //  __Softmax_h__