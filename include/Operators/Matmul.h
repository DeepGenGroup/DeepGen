#ifndef __Matmul_h__
#define __Matmul_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {
  
struct Matmul : Operator<Matmul> {
  static void buildNaiveExpress(mlir::ModuleOp module, 
    std::vector<int64_t> shape, 
    const std::vector<std::string>& dtypes,
    const std::string& kernelName,
    bool isTransposeA = false
    );

  static std::optional<std::string> verify(mlir::OpBuilder builder, std::vector<int64_t> shape, const std::vector<std::string>& dtype);
  
  static mlir::func::FuncOp createFunc(mlir::ModuleOp module, 
    std::vector<int64_t> shape, 
    const std::vector<mlir::Type>& dtype,
    const std::string& kernelName,
    bool isTransposeA = false
    );

  static std::string s_function;

  static std::string getKernelName(){
    return s_function;
  }

};

}

}  // KernelCodeGen

#endif //  __Matmul_h__