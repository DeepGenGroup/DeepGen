#ifndef __ElemScale_h__
#define __ElemScale_h__
#include "Operators/Operators.h"

namespace KernelCodeGen {
namespace Operators {

// Element-wise scale: output[y,x] = input[x,y] / sqrt(scaleDim)
// Transposes the input while scaling, so Q[hd,sl] -> scaled_Q[sl,hd].
// The scale factor is derived from the first core dimension of inputShape (head_dim).
struct ElemScale : Operator<ElemScale> {
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

#endif  // __ElemScale_h__
