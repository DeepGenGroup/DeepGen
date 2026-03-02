#ifndef __BroadcastNorm_h__
#define __BroadcastNorm_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {

// Broadcast normalization: p[y,x] = exp(scores[y,x]) / (em[y,0] * denom[y,0])
// Used as the middle operator in split-attention kernel 2.
//   func args: scores_in[sl,sl], em[sl,1], denom[sl,1], p_out[sl,sl]
struct BroadcastNorm : Operator<BroadcastNorm> {
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

#endif  // __BroadcastNorm_h__
