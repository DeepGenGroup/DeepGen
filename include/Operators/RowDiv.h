#ifndef __RowDiv_h__
#define __RowDiv_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {

// Row-wise division: output[y,x] = input[y,x] / sum[y,0]
// Divides each element by the corresponding row's scalar (e.g. softmax sum).
//   func args: data_input[sl,hd], sum_input[sl,1], output[sl,hd]
struct RowDiv : Operator<RowDiv> {
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
  static std::string getKernelName() { return s_function; }
};

}  // Operators
}  // KernelCodeGen

#endif  // __RowDiv_h__
