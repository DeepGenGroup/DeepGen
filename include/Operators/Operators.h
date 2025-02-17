#ifndef __Operators_h__
#define __Operators_h__

#include "Common/Utils.h"

namespace KernelCodeGen {

mlir::func::FuncOp buildFunction(mlir::ModuleOp module, const std::string& funcName, const std::string& OpName, 
                                 const std::vector<mlir::Type>& inputsTypes);

template <typename T>
struct Operator {
  template <typename... Args>
  static void buildNaiveExpress(mlir::ModuleOp module, Args &&...args) {
    T::buildNaiveExpress(module, std::forward<Args>(args)...);
  }

  static std::string getKernelName(){
    return T::getKernelName();
  }
};
}  // KernelCodeGen
#endif  // __Operators_h__