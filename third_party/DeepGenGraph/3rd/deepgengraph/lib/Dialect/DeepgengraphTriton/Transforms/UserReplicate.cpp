#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::deepgengraph::triton {

#define GEN_PASS_DEF_DEEPGENGRAPHTRITONUSERREPLICATE
#include "deepgengraph/Dialect/DeepgengraphTriton/Transforms/Passes.h.inc"

} // namespace mlir::deepgengraph::triton

namespace mlir::deepgengraph {
namespace triton {
namespace {

struct UserReplicatePattern : public OpRewritePattern<DeviceKernelOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DeviceKernelOp device_kernel_op, PatternRewriter &rewriter) const override {
    auto body = device_kernel_op.getBody();
    Operation *multi_use_op = nullptr;
    body->walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        if (result.use_empty() || result.hasOneUse()) {
          return WalkResult::advance();
        } else {
          multi_use_op = op;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (multi_use_op != nullptr) {
      rewriter.setInsertionPointAfter(multi_use_op);
      for (size_t i = 0; i < multi_use_op->getNumResults(); ++i) {
        auto res = multi_use_op->getResult(i);
        while (!res.hasOneUse() && !res.use_empty()) {
          auto use = res.getUses().begin();
          auto new_op = rewriter.clone(*multi_use_op);
          use->assign(new_op->getResult(i));
        }
      }
      return success();
    }
    return failure();
  }
};

class UserReplicatePass : public ::mlir::deepgengraph::triton::impl::DeepgengraphTritonUserReplicateBase<UserReplicatePass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<UserReplicatePattern>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createUserReplicatePass() { return std::make_unique<UserReplicatePass>(); }

} // namespace triton
} // namespace mlir::deepgengraph
