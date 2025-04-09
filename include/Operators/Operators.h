#ifndef __Operators_h__
#define __Operators_h__

#include "Common/Utils.h"

namespace KernelCodeGen {

mlir::func::FuncOp buildFunction(mlir::OpBuilder& builder, 
                                 const std::string& funcName, 
                                 const std::string& OpType, 
                                 const std::vector<mlir::Type>& inputsTypes, 
                                 const std::vector<bool>& isTranspose,
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
  static void buildNaiveExpress(mlir::ModuleOp module, 
                                const std::vector<std::vector<int64_t>>& intputShape,
                                const std::vector<std::vector<int64_t>>& outputShape,
                                const std::vector<std::string>& inputDType,
                                const std::vector<std::string>& outputDType,
                                const std::vector<bool>& isTranspose,
                                 Args &&...args) {
    mlir::OpBuilder builder(module);
    auto ver = verify(builder, intputShape, outputShape, inputDType, outputDType);
    if (ver.has_value()) {
      llvm::errs() << ver.value() << "\n";
      return ;
    }
    T::buildNaiveExpress(module, intputShape, outputShape, inputDType, outputDType, isTranspose, std::forward<Args>(args)...);
  }

  static std::optional<std::string> verify(mlir::OpBuilder builder,
                                          const std::vector<std::vector<int64_t>>& inputShape,
                                          const std::vector<std::vector<int64_t>>& outputShape,
                                          const std::vector<std::string>& inputDType,
                                          const std::vector<std::string>& outputDType) {
    // operator verify
    if (inputShape.size() != inputDType.size() || outputShape.size() != outputDType.size()) {
      std::string err{"The dimensions of shape and dtype are not equal."};
      return err;
    }
    // input
    for (int i=0; i<inputShape.size(); i++) {
      if (inputShape[i].size() < 2 || inputShape[i].size() > 4) {
        std::string err{"input tensor shape size must is 2, 3 or 4."};
        return err;
      }
      auto type = tools::getDType(builder, inputDType[i]);
      if (type == nullptr) {
        std::string err{"No exist this data type."};
        return err;
      }
    }
    // output
    for (int i=0; i<inputShape.size(); i++){
      if (outputShape[i].size() < 2 || outputShape[i].size() > 4) {
        std::string err{"output tensor shape size must is 2, 3 or 4."};
        return err;
      }
      auto type = tools::getDType(builder, outputDType[i]);
      if (type == nullptr) {
        std::string err{"No exist this data type."};
        return err;
      }
    }
    return std::nullopt;
  }

  static std::string getKernelName(){
    return T::getKernelName();
  }
};
}  // KernelCodeGen
#endif  // __Operators_h__