#ifndef __SoftmaxExp_h__
#define __SoftmaxExp_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {

// Partial softmax: computes exp(x - max) WITHOUT final division by sum.
// Outputs both exp_scores and the per-row sum for deferred division.
//   func args: input[sl,sl], exp_output[sl,sl], sum_output[sl,1]
//   x1 loop: online reduce to get row max + row sum
//   store sum to sum_output
//   x2 loop: exp(elem - max) -> exp_output (no /sum)
struct SoftmaxExp : Operator<SoftmaxExp> {
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

#endif  // __SoftmaxExp_h__
