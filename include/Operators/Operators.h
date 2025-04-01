#ifndef __Operators_h__
#define __Operators_h__

#include "Common/Utils.h"

namespace KernelCodeGen {

mlir::func::FuncOp buildFunction(mlir::OpBuilder& builder, 
                                 const std::string& funcName, 
                                 const std::string& OpName, 
                                 const std::vector<mlir::Type>& inputsTypes, 
                                 const std::vector<std::string>& paraDims,
                                 const int& outputArgNum);

std::vector<mlir::Value> createBatchNestForOp(mlir::OpBuilder& builder, std::vector<int64_t> batchs);

template<typename T>
std::vector<T> getShapeOrIndex(const std::vector<T>& batchs, const std::vector<T>& shape, bool isTran) {
  // 生成最终的shape还可以生成最终的value的index
  std::vector<T> newShape;
  for (auto dim : batchs) {
    newShape.push_back(dim);
  }
  if (isTran) {
    for (int i=shape.size()-1; i>=0; i--) {
      newShape.push_back(shape[i]);
    }
  } else {
    for (auto dim : shape) {
      newShape.push_back(dim);
    }
  }
  return newShape;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> splitShape(const std::vector<int64_t>& shape, int shapeLen);

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