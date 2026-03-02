#ifndef __SoftmaxStats_h__
#define __SoftmaxStats_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {

// Online reduce over scores to produce softmax statistics.
// Outputs em = exp(row_max) and denom = sum(exp(scores - row_max)) per row.
//   func args: input[sl,sl], em_output[sl,1], denom_output[sl,1]
//   x1 loop: online reduce to get row max + row sum
//   store em_output[y,0] = exp(max)
//   store denom_output[y,0] = sum
struct SoftmaxStats : Operator<SoftmaxStats> {
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

#endif  // __SoftmaxStats_h__
