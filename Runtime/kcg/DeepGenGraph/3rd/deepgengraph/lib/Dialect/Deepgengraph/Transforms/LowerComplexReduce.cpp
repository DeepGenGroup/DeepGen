#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/Deepgengraph/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::deepgengraph {

#define GEN_PASS_DEF_DEEPGENGRAPHLOWERCOMPLEXREDUCE
#include "deepgengraph/Dialect/Deepgengraph/Transforms/Passes.h.inc"

} // namespace mlir::deepgengraph

namespace mlir::deepgengraph {

namespace {

  // 展开 softmax 算子成基础op
struct SoftmaxPattern : public OpRewritePattern<SoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SoftmaxOp op, PatternRewriter &rewriter) const override {
    // softmax -> exp, reduce, div
    auto src = op.getOperand();
    auto dst = op.getResult();

    auto reduce_dim = op.getReduceDimensionAttr();
    auto keep_dim = BoolAttr::get(op.getContext(), true);
    auto reduce_type = ReduceTypeAttr::get(op.getContext(), ReduceType::ADD);

    auto exp_op = rewriter.create<ExpOp>(op.getLoc(), src);
    auto reduce_op = rewriter.create<ReduceOp>(op.getLoc(), exp_op.getResult(), reduce_dim, reduce_type, keep_dim);
    auto div_op = rewriter.create<DivOp>(op.getLoc(), exp_op.getResult(), reduce_op.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), div_op.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

// normalize 展开
struct NormalizePattern : public OpRewritePattern<NormalizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NormalizeOp op, PatternRewriter &rewriter) const override {
    // Lp = 1: normalize -> abs, reduce, div
    auto src = op.getOperand();
    auto dst = op.getResult();

    auto reduce_dim = op.getReduceDimensionAttr();
    int64_t lp = op.getLpAttr().getValue().getSExtValue();
    if (lp == 1) {
      auto keep_dim = BoolAttr::get(op.getContext(), true);
      auto reduce_type = ReduceTypeAttr::get(op.getContext(), ReduceType::ADD);

      auto abs_op = rewriter.create<AbsOp>(op.getLoc(), src);
      auto reduce_op = rewriter.create<ReduceOp>(op.getLoc(), abs_op.getResult(), reduce_dim, reduce_type, keep_dim);
      auto div_op = rewriter.create<DivOp>(op.getLoc(), src, reduce_op.getResult());
      rewriter.replaceAllUsesWith(op.getResult(), div_op.getResult());
      rewriter.eraseOp(op);
    } else {
      llvm_unreachable("lp != 1");
    }
    return success();
  }
};

class LowerComplexReducePass : public ::mlir::deepgengraph::impl::DeepgengraphLowerComplexReduceBase<LowerComplexReducePass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    FunctionOpInterface op = getOperation();

    patterns.add<SoftmaxPattern>(context);
    patterns.add<NormalizePattern>(context);
    if (applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerComplexReducePass() { return std::make_unique<LowerComplexReducePass>(); }

} // namespace mlir::deepgengraph