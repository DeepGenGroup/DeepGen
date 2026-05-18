#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "llvm/ADT/SetVector.h"

#include <vector>

namespace mlir::frisk {

// MLIR 的 Pass 生成机制要求在包含 .inc 文件之前，必须定义一个特定的宏来启用该 Pass 的基类定义。
#define GEN_PASS_DEF_FRISKOVERLAP
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

} // namespace mlir::frisk

namespace mlir::frisk {
namespace {

struct ExecGroup {
  enum class HardwareUnit { TMA, TensorCore, CudaCore, Unknown };

  HardwareUnit unit = HardwareUnit::Unknown;
  SmallVector<Operation *> ops;
  llvm::SetVector<Value> inputBuffers;
  llvm::SetVector<Value> outputBuffers;

  // Append op to this group only if it maps to the same execution unit.
  bool append(Operation *op) {
    HardwareUnit opUnit = classifyOperation(op);
    if (opUnit == HardwareUnit::Unknown)
      return false;
    if (!ops.empty() && opUnit != unit)
      return false;

    if (ops.empty())
      unit = opUnit;

    ops.push_back(op);

    for (Value read : getReadBuffers(op))
      if (!producedBuffers.contains(read))
        inputBuffers.insert(read);

    for (Value write : getWriteBuffers(op)) {
      outputBuffers.insert(write);
      producedBuffers.insert(write);
    }

    // Views/results generated inside this group are internal produced buffers.
    for (Value result : op->getResults())
      if (isa<MemRefType>(result.getType()))
        producedBuffers.insert(result);

    return true;
  }

private:
  llvm::SetVector<Value> producedBuffers;

  static HardwareUnit classifyOperation(Operation *op) {
    if (isa<BufferViewOp, CopyOp>(op))
      return HardwareUnit::TMA;
    if (isa<GemmOp, GemmWaitOp>(op))
      return HardwareUnit::TensorCore;
    if (isa<BlockOp, ReduceOp, FillOp>(op))
      return HardwareUnit::CudaCore;
    return HardwareUnit::Unknown;
  }

  static SmallVector<Value> getReadBuffers(Operation *op) {
    SmallVector<Value> reads;
    if (auto bufferView = dyn_cast<BufferViewOp>(op)) {
      reads.push_back(bufferView.getSource());
      return reads;
    }
    if (auto copy = dyn_cast<CopyOp>(op)) {
      reads.push_back(copy.getSrc());
      return reads;
    }
    if (auto gemm = dyn_cast<GemmOp>(op)) {
      reads.push_back(gemm.getA());
      reads.push_back(gemm.getB());
      return reads;
    }
    if (auto reduce = dyn_cast<ReduceOp>(op)) {
      reads.push_back(reduce.getSrc());
      return reads;
    }
    if (auto wait = dyn_cast<GemmWaitOp>(op)) {
      reads.push_back(wait.getBuffer());
      return reads;
    }
    return reads;
  }

  static SmallVector<Value> getWriteBuffers(Operation *op) {
    SmallVector<Value> writes;
    if (auto bufferView = dyn_cast<BufferViewOp>(op)) {
      writes.push_back(bufferView.getView());
      return writes;
    }
    if (auto copy = dyn_cast<CopyOp>(op)) {
      writes.push_back(copy.getDst());
      return writes;
    }
    if (auto gemm = dyn_cast<GemmOp>(op)) {
      writes.push_back(gemm.getC());
      return writes;
    }
    if (auto reduce = dyn_cast<ReduceOp>(op)) {
      writes.push_back(reduce.getDst());
      return writes;
    }
    if (auto fill = dyn_cast<FillOp>(op)) {
      writes.push_back(fill.getMemref());
      return writes;
    }
    return writes;
  }
};

class OverlapPass : public ::mlir::frisk::impl::friskOverlapBase<OverlapPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    (void)context;
  }

  void collectGroup(frisk::KernelOp kernelOp) {
    
  }

private:
  std::vector<ExecGroup> tc_groups, cc_groups, tma_groups;
};

} // namespace

std::unique_ptr<Pass> createOverlapPass() {
  return std::make_unique<OverlapPass>();
}

} // namespace mlir::frisk
