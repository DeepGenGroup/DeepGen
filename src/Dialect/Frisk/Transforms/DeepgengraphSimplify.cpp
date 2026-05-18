#include "Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"

#include "Dialect/Frisk/IR/FriskEnums.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/ThreadImp/IR/ThreadImpDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlir/Parser/Parser.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_DEEPGENGRAPHSIMPLIFY 
#include "Dialect/Frisk/Transforms/Passes.h.inc"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

// 假设你的 Dialect namespace 是 deepgengraph_triton 和 deepgengraph
struct InlineDeviceKernelPattern : public mlir::OpRewritePattern<deepgengraph::triton::DeviceKernelOp> {
  using OpRewritePattern<deepgengraph::triton::DeviceKernelOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(deepgengraph::triton::DeviceKernelOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&](){
      op->getParentOp()->setAttr("grid", op.getGridAttr());
    });
    // 1. 获取 device_kernel 内部的 Region 和唯一的 Block
    mlir::Region &region = op.getRegion();
    if (region.empty()) {
      return mlir::failure();
    }
    mlir::Block &bodyBlock = region.front();

    std::vector<deepgengraph::triton::DeviceYieldOp> invalidOps;
    op->walk([&](deepgengraph::triton::DeviceYieldOp yieldOp){
      invalidOps.push_back(yieldOp);
    });
    for(auto e : invalidOps){
      rewriter.eraseOp(e);
    }

    // 2. 准备替换 Block Arguments 的 Values
    llvm::SmallVector<mlir::Value> replacements;
    mlir::Location loc = op.getLoc();

    // 2a. 替换 Grid 坐标参数 (%arg3, %arg4, %arg5)
    // 【注意】因为失去了 device_kernel，你需要生成新的 Op 来获取 grid 坐标。
    // 这里假设你的 Dialect 中有类似 deepgengraph::BlockIdOp 的操作。
    // 如果你是要生成 scf.parallel 循环，你需要在这里构建循环并获取 ivs。
    
    mlir::Value gridX = rewriter.create<gpu::BlockIdOp>(loc, ::mlir::gpu::Dimension::x); 
    mlir::Value gridY = rewriter.create<gpu::BlockIdOp>(loc, /*dim=*/::mlir::gpu::Dimension::y);
    mlir::Value gridZ = rewriter.create<gpu::BlockIdOp>(loc, /*dim=*/::mlir::gpu::Dimension::z);
    // mlir::Value gridX = rewriter.create<gpu::BlockIdOp>(loc, /*dim=*/0); 
    // mlir::Value gridY = rewriter.create<gpu::BlockIdOp>(loc, /*dim=*/1);
    // mlir::Value gridZ = rewriter.create<gpu::BlockIdOp>(loc, /*dim=*/2);
    replacements.push_back(gridX);
    replacements.push_back(gridY);
    replacements.push_back(gridZ);

    // 2b. 替换指针参数 (%arg6, %arg7, %arg8, %arg9)
    // 直接将 device_kernel 接收的 args ([%0, %1, %2, %3]) 传入
    for (mlir::Value arg : op.getArgs()) {
      replacements.push_back(arg);
    }

    // 校验参数数量是否匹配
    if (replacements.size() != bodyBlock.getNumArguments()) {
      return op->emitError("Block arguments size does not match replacements size.");
    }

    // 3. 将 bodyBlock 展开到 device_kernel 所在的位置
    // 这一步会自动将内部用到 %arg3~%arg9 的地方替换为我们上面准备好的 replacements
    rewriter.inlineBlockBefore(&bodyBlock, op, replacements);

    // 4. 删除原有的 device_kernel 操作

    rewriter.eraseOp(op);
    return mlir::success();
  }
};


struct ModifyKernelSignaturePattern : public mlir::OpConversionPattern<deepgengraph::KernelOp> {
  using OpConversionPattern<deepgengraph::KernelOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    deepgengraph::KernelOp op,
    OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const override 
  {
    using namespace deepgengraph::triton;
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs = {}, ArrayRef<DictionaryAttr> arg_attrs = {});
    auto oldType = op.getFunctionType();
    std::vector<Type> inputs = oldType.getInputs();
    std::vector<Type> outputs = {};
    std::vector<Type> oldOutputTypes = oldType.getResults();
    for(auto ty : oldType.getResults()){
      inputs.push_back(ty);
    }

    auto newFuncType = rewriter.getFunctionType(inputs, outputs);
    TypeConverter::SignatureConversion conversion(oldType.getNumInputs());
    for(auto i = 0;i < oldType.getNumInputs(); ++i){
      conversion.addInputs(i, newFuncType.getInputs()[i]);
    }
    auto outIdx = oldType.getNumResults();
    for(auto ty : oldType.getResults()){
      conversion.addInputs(ty);
    }
    rewriter.applySignatureConversion(&op.getBody().front(), conversion);
    rewriter.modifyOpInPlace(op, [&](){
      op.setFunctionType(newFuncType);
    });
    std::vector<int32_t> outIdxAray;
    for(int i=0;i<outIdx;++i){
      outIdxAray.push_back(i + oldType.getNumInputs());
    }
    op->setAttr("out_idx", rewriter.getI32ArrayAttr(outIdxAray));
    
    auto terminator = op.getBody().front().getTerminator();
    if(auto returnOp = mlir::dyn_cast_or_null<deepgengraph::ReturnOp>(terminator)){
      rewriter.setInsertionPoint(returnOp);
      // 创建一个新的没有操作数的 ReturnOp
      rewriter.create<deepgengraph::ReturnOp>(returnOp.getLoc()); 
      // 删掉旧的 ReturnOp
      rewriter.eraseOp(returnOp);
    }
    // replace empty_ptr ops with ptr_of
    std::vector<EmptyPointerOp> ops;
    op->walk([&](EmptyPointerOp emptyOp){
      ops.push_back(emptyOp);
    });
    
    for(auto oldEmptyOp : ops){
      for(int i=0;i<oldOutputTypes.size();++i){
        if(oldEmptyOp.getTensorType() == oldOutputTypes[i]){
          // remark as invalid type
          oldOutputTypes[i] = mlir::IndexType::get(op->getContext());
          rewriter.setInsertionPoint(oldEmptyOp);
          auto newop = rewriter.create<deepgengraph::triton::PointerOfOp>(oldEmptyOp->getLoc(), op.getFunctionBody().getArgument(i + oldType.getNumInputs()) );
          rewriter.replaceOp(oldEmptyOp, newop);
        }
      }
    }
    return success();
  }
};

// deepgengraph_triton.block_load 加入详细信息，以免后续op转换后丢失
class BlockLoadOpAddInfoPattern : public OpRewritePattern<deepgengraph::triton::BlockLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockLoadOp blockLoadOp,
    PatternRewriter &rewriter) const override
  {
    using namespace deepgengraph::triton;
    if(blockLoadOp->hasAttr("argId")){
      return failure();
    }
    auto blockPtrOf = blockLoadOp.getSrcPointer().getDefiningOp<BlockPointerOfOp>();
    if(blockPtrOf == nullptr){
      if(auto blockArg = mlir::dyn_cast<BlockArgument>(blockLoadOp.getSrcPointer())){
        auto id = blockArg.getArgNumber();
        auto parentOp= blockArg.getParentBlock()->getParentOp();
        if(auto scfFor = mlir::dyn_cast<scf::ForOp>(parentOp)){
          blockPtrOf = scfFor.getInitArgs()[id-1].getDefiningOp<BlockPointerOfOp>();
        }
      }
    }
    assert(blockPtrOf != nullptr);
    auto pointerOfOp = blockPtrOf.getBasePointer().getDefiningOp<deepgengraph::triton::PointerOfOp>();
    int argId = -1;
    if(pointerOfOp != nullptr ){
      auto arg = mlir::cast<BlockArgument>(pointerOfOp.getOperand());
      argId = arg.getArgNumber();
    }
    blockLoadOp->setAttr("block_shape", blockPtrOf.getBlockShapeAttr());
    blockLoadOp->setAttr("stride", blockPtrOf.getStrideAttr());
    blockLoadOp->setAttr("order", blockPtrOf.getOrderAttr());
    blockLoadOp->setAttr("offset", blockPtrOf.getOffsetAttr());
    blockLoadOp->setAttr("argId", rewriter.getI32IntegerAttr(argId)); 
    blockPtrOf->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    pointerOfOp->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    return mlir::success();
  }
};

// deepgengraph_triton.block_store 加入详细信息，以免后续op转换后丢失
class BlockStoreOpAddInfoPattern : public OpRewritePattern<deepgengraph::triton::BlockStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockStoreOp blockStoreOp,
    PatternRewriter &rewriter) const override
  {
    using namespace deepgengraph::triton;
    if(blockStoreOp->hasAttr("argId")){
      return failure();
    }
    auto blockPtrOf = blockStoreOp.getDstPointer().getDefiningOp<BlockPointerOfOp>();
    if(blockPtrOf == nullptr){
      if(auto blockArg = mlir::dyn_cast<BlockArgument>(blockStoreOp.getDstPointer())){
        auto id = blockArg.getArgNumber();
        auto parentOp= blockArg.getParentBlock()->getParentOp();
        if(auto scfFor = mlir::dyn_cast<scf::ForOp>(parentOp)){
          blockPtrOf = scfFor.getInitArgs()[id-1].getDefiningOp<BlockPointerOfOp>();
        }
      }
    }
    assert(blockPtrOf != nullptr);
    auto pointerOfOp = blockPtrOf.getBasePointer().getDefiningOp<deepgengraph::triton::PointerOfOp>();
    int argId = -1;
    if(pointerOfOp != nullptr ){
      auto arg = mlir::cast<BlockArgument>(pointerOfOp.getOperand());
      argId = arg.getArgNumber();
    }
    blockStoreOp->setAttr("block_shape", blockPtrOf.getBlockShapeAttr());
    blockStoreOp->setAttr("stride", blockPtrOf.getStrideAttr());
    blockStoreOp->setAttr("order", blockPtrOf.getOrderAttr());
    blockStoreOp->setAttr("offset", blockPtrOf.getOffsetAttr());
    blockStoreOp->setAttr("argId", rewriter.getI32IntegerAttr(argId)); 
    blockPtrOf->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    pointerOfOp->setAttr("argId", rewriter.getI32IntegerAttr(argId));
    return mlir::success();
  }
};

// deepgengraph_triton.block_store 加入详细信息，以免后续op转换后丢失
class BlockAdvanceOpAddInfoPattern : public OpRewritePattern<deepgengraph::triton::BlockAdvanceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op,
    PatternRewriter &rewriter) const override
  {
    using namespace deepgengraph::triton;
    if(op->hasAttr("argId")){
      return failure();
    }
    auto blockPtrOf = op.getOperand().getDefiningOp<BlockPointerOfOp>();
    if(blockPtrOf == nullptr){
      if(auto blockArg = mlir::dyn_cast<BlockArgument>(op.getOperand())){
        auto id = blockArg.getArgNumber();
        auto parentOp= blockArg.getParentBlock()->getParentOp();
        if(auto scfFor = mlir::dyn_cast<scf::ForOp>(parentOp)){
          blockPtrOf = scfFor.getInitArgs()[id-1].getDefiningOp<BlockPointerOfOp>();
        }
      }
    }
    assert(blockPtrOf != nullptr);
    auto pointerOfOp = blockPtrOf.getBasePointer().getDefiningOp<deepgengraph::triton::PointerOfOp>();
    int argId = -1;
    if(pointerOfOp != nullptr ){
      auto arg = mlir::cast<BlockArgument>(pointerOfOp.getOperand());
      argId = arg.getArgNumber();
    }
    op->setAttr("block_shape", blockPtrOf.getBlockShapeAttr());
    op->setAttr("stride", blockPtrOf.getStrideAttr());
    op->setAttr("order", blockPtrOf.getOrderAttr());
    op->setAttr("offset", blockPtrOf.getOffsetAttr());
    // 表示与参数列表中 global mem 的对应关系
    op->setAttr("argId", rewriter.getI32IntegerAttr(argId)); 
    // 表示其对应的blockptrof 需要滑动，来取得 gm中的不同数据
    blockPtrOf->setAttr("will_move", rewriter.getBoolAttr(true));
    return mlir::success();
  }
};



class DeepgengraphSimplifyPass : public impl::DeepgengraphSimplifyBase<DeepgengraphSimplifyPass> {
public:
  void runOnOperation() override {
    mlir::RewritePatternSet ps1(&getContext());
    mlir::RewritePatternSet ps2(&getContext());
    mlir::RewritePatternSet ps3(&getContext());
    // 1. inline deviceKernelOp into kernelOp
    ps1.add<InlineDeviceKernelPattern>(&getContext());
    TypeConverter tc;
    ConversionTarget target(getContext());
    
    // 2. move return value into kernelOp input arg list
    ps2.add<ModifyKernelSignaturePattern >(tc, &getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(ps1)))) {
      signalPassFailure();
    }
    
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
    target.addDynamicallyLegalOp<deepgengraph::KernelOp>(
      [](deepgengraph::KernelOp op){
        return op.getFunctionType().getNumResults() == 0;
    });
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(ps2)))) {
      signalPassFailure();
    }
    // 3. add info on load/store/advance op to identify arg usage
    ps3.add<BlockAdvanceOpAddInfoPattern, BlockLoadOpAddInfoPattern, BlockStoreOpAddInfoPattern>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(ps3)))) {
      signalPassFailure();
    }

    return;
  }
};

std::unique_ptr<Pass> createDeepgenGraphSimplifyPass(){
  return std::make_unique<DeepgengraphSimplifyPass>(); 
}

} // namespace mlir::deepgengraph
