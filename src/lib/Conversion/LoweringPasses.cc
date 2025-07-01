#include "Conversion/LoweringPasses.h"
#include <dlfcn.h>
#include <filesystem>

using namespace mlir;
namespace KernelCodeGen {

// ===================================================================
//                  amend memerf alloca Addrspace       
// ===================================================================
// 将memref.alloca的地址空间进行修改，local=0为cuda/loacl=5为rocm
struct AmendAllocaOpAddrSpace : public OpRewritePattern<memref::AllocaOp> {
  AmendAllocaOpAddrSpace(MLIRContext *context, Target target)
    : OpRewritePattern(context), target(target) {}

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp, PatternRewriter &rewriter) const override {
    MemRefType originalType = allocaOp.getType();
    int requiredSpace = (target == Target::ROCm) ? 5 : 0;
    if (static_cast<int>(originalType.getMemorySpaceAsInt()) == requiredSpace) {
      return failure();
    }
    MLIRContext *ctx = allocaOp.getContext();
    Attribute memorySpaceAttr = IntegerAttr::get(IntegerType::get(ctx, 64), requiredSpace);
    MemRefType newType = MemRefType::get(originalType.getShape(), originalType.getElementType(), 
                                         originalType.getLayout(),memorySpaceAttr);

    rewriter.setInsertionPoint(allocaOp);
    auto newAlloca = rewriter.create<memref::AllocaOp>(allocaOp.getLoc(), newType);
    rewriter.replaceOp(allocaOp, newAlloca.getResult());
    return success();
  }
  private:
    Target target;
};

struct AmendAllocaOpAddrSpacePass : public PassWrapper<AmendAllocaOpAddrSpacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AmendAllocaOpAddrSpacePass)

  AmendAllocaOpAddrSpacePass(Target target) : target(target) {}
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AmendAllocaOpAddrSpace>(&getContext(), target);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
  private:
    Target target;
};



// ===================================================================
//                  affine parallel to gpu index        
// ===================================================================
// 将affine的parallelOp 转成Gpu的block/threadIdOp表示，func添加grid/block size作为属性
struct ParallelToGPULowering : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp parallelOp, PatternRewriter &rewriter) const final {
    constexpr gpu::Dimension dims[] = {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    // 替换 parallelOp 为 gpu::BlockIdOp
    std::vector<int32_t> ubs;
    auto gpuidx = parallelOp->getAttr(AttrGPUIndex);
    auto gpuidxAttr = mlir::dyn_cast<mlir::StringAttr>(gpuidx);
    auto upperBounds = parallelOp.getUpperBoundsMap().getConstantResults();
    SmallVector<Value, 3> ids;
    for (unsigned i = 0; i < parallelOp.getNumDims(); ++i) {
      if (gpuidxAttr.getValue().str() == THREADIDX) {
        auto threadId = rewriter.create<gpu::ThreadIdOp>(parallelOp.getLoc(), dims[i]);
        ids.push_back(threadId);
      } else {
        auto blockId = rewriter.create<gpu::BlockIdOp>(parallelOp.getLoc(), dims[i]);
        ids.push_back(blockId);
      }
      ubs.push_back(static_cast<int32_t>(upperBounds[i]));
    }

    // 将func设置block和thread的上界属性
    func::FuncOp funcOp = nullptr;
    Operation* parentOp = parallelOp->getParentOp();
    while (parentOp) {
      if (funcOp = mlir::dyn_cast<func::FuncOp>(parentOp)) { break; }
      parentOp = parentOp->getParentOp();
    }
    if (funcOp == nullptr) {
      llvm::errs() << "The ParentOp of scf::ParallelOp must is FuncOp!\n";
      assert(false);
    }
    
    auto attr = rewriter.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(ubs));
    if (gpuidxAttr.getValue().str() == THREADIDX) {
      funcOp->setAttr(AttrBlockDim, attr);
    } else {
      funcOp->setAttr(AttrGridDim, attr);
    }
    
    // 替换使用循环变量的操作
    auto ivs = parallelOp.getIVs();
    for (unsigned i = 0; i < ivs.size(); ++i) {
      ivs[i].replaceAllUsesWith(ids[i]);
    }

    // 内层操作移出内层 p  collect op
    SmallVector<Operation *, 4> opsToMove;
    for (Operation &op : parallelOp.getBody()->getOperations()) {
      if (!dyn_cast<affine::AffineYieldOp>(op)) {
        opsToMove.push_back(&op);
      }
    }
    // 内层操作移出内层 p 
    Operation *tempOp = ids.back().getDefiningOp();
    for (Operation *op : opsToMove) {
      op->moveAfter(tempOp);
      tempOp = op;
    }
    rewriter.eraseOp(parallelOp);

    return success();
  }
};

// 将affine的parallelOp 转成Gpu的block/threadIdOp表示的pass
struct ParallelToGPUPass : public PassWrapper<ParallelToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelToGPUPass)
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    // target.addIllegalOp<gpu::BlockIdOp, gpu::ThreadIdOp>();
    target.addIllegalOp<affine::AffineParallelOp>();
    target.addLegalDialect<gpu::GPUDialect>();

    patterns.add<ParallelToGPULowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))){
      return signalPassFailure();
    }

  }
};


// ===================================================================
//                 gpu index to nvvm/rocdl index 
// ===================================================================
// 将 GUP 的IdOp转成 rocdl/nvvm的IdOp，读取func的attr加到新的IdOp上
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct GPUIndexIntrinsicOpLowering : public ConvertOpToLLVMPattern<Op> {
private:
  unsigned indexBitwidth;
  StringRef boundsAttrName;

public:
  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName("") {}

  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter,
                                       StringRef boundsAttrName)
      :  ConvertOpToLLVMPattern<Op>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()),
        boundsAttrName(boundsAttrName) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Operation *newOp;
    switch (op.getDimension()) {
    case gpu::Dimension::x:
      newOp = rewriter.create<XOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    case gpu::Dimension::y:
      newOp = rewriter.create<YOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    case gpu::Dimension::z:
      newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, indexBitwidth));
      break;
    }

    Operation *function;
    if (auto Func = op->template getParentOfType<func::FuncOp>())
      function = Func;
    if (auto llvmFunc = op->template getParentOfType<LLVM::LLVMFuncOp>())
      function = llvmFunc;
    if (!boundsAttrName.empty() && function) {
      if (auto attr = function->template getAttrOfType<DenseI32ArrayAttr>(
              boundsAttrName)) {
        int32_t maximum = attr[static_cast<uint32_t>(op.getDimension())];
        newOp->setAttr("range", rewriter.getDenseI32ArrayAttr({0, maximum}));
      }
    }

    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct GPUShuffleOpToROCDLLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value initShflValue = adaptor.getValue();
    Type shflType = initShflValue.getType();
    // TODO: Add support for non 32-bit shuffle values.
    if (!shflType.isIntOrFloat() || shflType.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(op, "only 32-bit int/float types are supported");

    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();

    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    Value zero_ = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    Value minus1 = rewriter.create<arith::ConstantIntOp>(loc, -1, 32);
    Value mbcntLo = rewriter.create<ROCDL::MbcntLoOp>(loc, int32Type, ValueRange{minus1, zero_});
    Value srcLaneId = rewriter.create<ROCDL::MbcntHiOp>(loc, int32Type, ValueRange{minus1, mbcntLo});  // tid % warpsize

    Value width = adaptor.getWidth();  // 这是自己设置的16
    Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 1);
    Value negwidth = rewriter.create<LLVM::SubOp>(loc, int32Type, width, one);
    Value maskAnd = rewriter.create<LLVM::AndOp>(loc, int32Type, srcLaneId, negwidth);  // -64
    Value add = rewriter.create<LLVM::AddOp>(loc, int32Type, maskAnd, adaptor.getOffset());

    // Value width = adaptor.getWidth();  // 这是自己设置的16
    // Value zero = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 0);
    // Value negwidth = rewriter.create<LLVM::SubOp>(loc, int32Type, zero, width);
    // Value add = rewriter.create<LLVM::AddOp>(loc, int32Type, srcLaneId, width);
    // Value widthOrZeroIfOutside = rewriter.create<LLVM::AndOp>(loc, int32Type, add, negwidth);  // -64
    Value dstLane;
    // TODO: Add support for gpu::ShuffleMode::UP and gpu::ShuffleMode::DOWN.
    // TODO: Use ds_swizzle for XOR when step/offsets are constants for better
    // perf.
    switch (op.getMode()) {
    case gpu::ShuffleMode::DOWN:
      dstLane = rewriter.create<LLVM::AddOp>(loc, int32Type, srcLaneId, adaptor.getOffset());  // tid % warpsize + offset
      break;
    case gpu::ShuffleMode::XOR:
      dstLane = rewriter.create<LLVM::XOrOp>(loc, int32Type, srcLaneId, adaptor.getOffset());
      break;
    case gpu::ShuffleMode::IDX:
      dstLane = adaptor.getOffset();
      break;
    default:
      return failure();
    }
    // "tid % warpsize + offset < -64  == false"
    Value isActiveSrcLane = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, add, width);
    Value selectDstLane = rewriter.create<LLVM::SelectOp>(loc, isActiveSrcLane, srcLaneId, dstLane); // tid % warpsize + offset / tid % warpsize
    Value two = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 2);
    Value dwordAlignedDstLane = rewriter.create<LLVM::ShlOp>(loc, int32Type, selectDstLane, two);  // dstlane * 4
    if (shflType.isF32()) {
      initShflValue = rewriter.create<LLVM::BitcastOp>(loc, int32Type, initShflValue);
    }
    Value shflValue = rewriter.create<ROCDL::DsBpermuteOp>(loc, int32Type, dwordAlignedDstLane, initShflValue);
    if (shflType.isF32()) {
      shflValue = rewriter.create<LLVM::BitcastOp>(loc, shflType, shflValue);
    }
    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    return success();
  }
};

struct GPUShuffleOpToNVVMLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto valueTy = adaptor.getValue().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto predTy = IntegerType::get(rewriter.getContext(), 1);

    // Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 1);
    Value minusOne = rewriter.create<LLVM::ConstantOp>(loc, int32Type, -1);
    // Value thirtyTwo = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 32);
    // Value numLeadInactiveLane = rewriter.create<LLVM::SubOp>(loc, int32Type, thirtyTwo, adaptor.getWidth());
    // Bit mask of active lanes: `(-1) >> (32 - activeWidth)`.
    // Value activeMask = rewriter.create<LLVM::LShrOp>(loc, int32Type, minusOne, numLeadInactiveLane);
    // Value maskAndClamp;
    // if (op.getMode() == gpu::ShuffleMode::UP) {
    //   // Clamp lane: `32 - activeWidth`
    //   maskAndClamp = numLeadInactiveLane;
    // } else {
    //   // Clamp lane: `activeWidth - 1`
    //   maskAndClamp = rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.getWidth(), one);
    // }
    Value segmaskAndClamp;
    auto constOp = adaptor.getWidth().getDefiningOp<arith::ConstantOp>();
    auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue());
    auto width = intAttr.getInt();
    // llvm::outs() << "witdh: " << width;
    if (width < 32) {
      segmaskAndClamp = rewriter.create<LLVM::ConstantOp>(loc, int32Type, ((32 - width) << 8) + 31);
    } else {
      segmaskAndClamp = rewriter.create<LLVM::ConstantOp>(loc, int32Type, width - 1);;
    }
    
    bool predIsUsed = !op->getResult(1).use_empty();
    UnitAttr returnValueAndIsValidAttr = nullptr;
    Type resultTy = valueTy;
    if (predIsUsed) {
      returnValueAndIsValidAttr = rewriter.getUnitAttr();
      resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {valueTy, predTy});
    }
    NVVM::ShflKind nvvmMode;
    switch (op.getMode()) {
      case gpu::ShuffleMode::XOR:
        nvvmMode = NVVM::ShflKind::bfly;
        break;
      case gpu::ShuffleMode::UP:
        nvvmMode =  NVVM::ShflKind::up;
        break;
      case gpu::ShuffleMode::DOWN:
        nvvmMode =  NVVM::ShflKind::down;
        break;
      case gpu::ShuffleMode::IDX:
        nvvmMode =  NVVM::ShflKind::idx;
        break;
      default:
        return failure();
    }
    // Value shfl = rewriter.create<NVVM::ShflOp>(loc, resultTy, activeMask, adaptor.getValue(), adaptor.getOffset(),
        // maskAndClamp, nvvmMode, returnValueAndIsValidAttr);
    Value shfl = rewriter.create<NVVM::ShflOp>(loc, resultTy, minusOne, adaptor.getValue(), adaptor.getOffset(),
        segmaskAndClamp, nvvmMode, returnValueAndIsValidAttr);
    if (predIsUsed) {
      Value shflValue = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 0);
      Value isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 1);
      rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    } else {
      rewriter.replaceOp(op, {shfl, nullptr});
    }
    return success();
  }
};

// 将gpu barrier转成rocdl的barrier
struct GPUBarrierToROCDLLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ROCDL::BarrierOp>(brOp);
    return success();
  }
};

// 将gpu barrier转成NVVM的barrier0
struct GPUBarrierToNVVMLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<NVVM::Barrier0Op>(brOp);
    return success();
  }
};

// 将上述 3 个重写加到这个pass中
struct GPUToROCDLOrNVVMPass : public PassWrapper<GPUToROCDLOrNVVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToROCDLOrNVVMPass)

  explicit GPUToROCDLOrNVVMPass(Target target_, unsigned indexBitwidth_) : 
                                target(target_), indexBitwidth(indexBitwidth_) {};
  private:
    Target target;
    unsigned indexBitwidth;
    StringRef amdgcnDataLayout =
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:"
    "32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:"
    "64-S32-A5-G1-ni:7:8:9";

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
    registry.insert<NVVM::NVVMDialect>();
  }
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget Codetarget(getContext());
    LowerToLLVMOptions options(&getContext());
    if (target == Target::ROCm) {
      options.dataLayout = llvm::DataLayout(amdgcnDataLayout);
    }
    options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter typeConverter(&getContext(), options);

    Codetarget.addIllegalDialect<gpu::GPUDialect>();
    Codetarget.addLegalDialect<LLVM::LLVMDialect, ROCDL::ROCDLDialect, NVVM::NVVMDialect>();

    if (target == Target::ROCm) {
      Codetarget.addLegalDialect<arith::ArithDialect>();
      // populateGpuToROCDLConversionPatterns(typeConverter, patterns, gpu::amd::Runtime::HIP);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp, 
                                               ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>>(typeConverter, AttrGridDim);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                               ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>>(typeConverter, AttrBlockDim);
      patterns.add<GPUShuffleOpToROCDLLowering>(typeConverter);
      patterns.add<GPUBarrierToROCDLLowering>(&getContext());
      populateMathToROCDLConversionPatterns(typeConverter, patterns);  // exp仅支持fp64/16
    } else if (target == Target::CUDA) {
      // populateGpuToNVVMConversionPatterns(typeConverter, patterns, 10);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp, 
                                               NVVM::BlockIdYOp, NVVM::BlockIdZOp>>(typeConverter, AttrGridDim);
      patterns.add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                               NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>>(typeConverter, AttrBlockDim);
      patterns.add<GPUShuffleOpToNVVMLowering>(typeConverter);
      patterns.add<GPUBarrierToNVVMLowering>(&getContext());
      populateLibDeviceConversionPatterns(typeConverter, patterns, /*benefit*/10);  // 大部分只支持fp32/fp16
    }

    if (failed(applyPartialConversion(getOperation(), Codetarget, std::move(patterns)))){
      return signalPassFailure();
    }

  }
};


// ===================================================================
//         LLVMFuncOp add attribute (nvvm.kernel, nvvm.maxnid)
// ===================================================================
// LLVMFunc 添加 nvvm.kernel 和 nvvm.maxnid 属性给func
struct LLVMFuncOpAddGPUAttrPass : public PassWrapper<LLVMFuncOpAddGPUAttrPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMFuncOpAddGPUAttrPass)
  explicit LLVMFuncOpAddGPUAttrPass(Target target_) : target(target_) {};
  private:
    Target target;
  void runOnOperation() override {
    auto module = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(module);
    module.walk<WalkOrder::PreOrder>([&](LLVM::LLVMFuncOp funcOp) {
      if (target == Target::CUDA) {
        funcOp->setAttr(AttrCUDAKernelFunc, builder.getIntegerAttr(builder.getI1Type(), 1));
      } else {
        funcOp->setAttr(AttrROCmKernelFunc, builder.getIntegerAttr(builder.getI1Type(), 1));
      }
      // funcOp->setAttr(AttrMaxBlockThreads, builder.getI32ArrayAttr(256));  // 这个固定了
    });
  }
};


// ====================================================================
//                         Affine ForOp Unroll
// ====================================================================
// affine 循环展开
struct AffineUnrollPass : public PassWrapper<AffineUnrollPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineUnrollPass)

  void runOnOperation() override {
    getOperation().walk([&] (affine::AffineForOp forOp){
      if (auto unrollName = forOp->getAttr("affine.loop")) {
        auto unrollAttr = mlir::dyn_cast<mlir::StringAttr>(unrollName);
        if (unrollAttr.getValue().str() == "unroll") {
          auto unrollNumAttr = forOp->getAttr("affine.unroll.num");
          auto unrollNum = mlir::dyn_cast<mlir::IntegerAttr>(unrollNumAttr).getInt();
          int64_t ub = forOp.getConstantUpperBound();
          int64_t lb = forOp.getConstantLowerBound();
          int64_t step = forOp.getStep().getLimitedValue();
          int64_t loopNum =  (ub - lb) / step;
          if (loopNum > unrollNum) {
            auto ret = affine::loopUnrollJamByFactor(forOp, unrollNum);
            if(failed(ret)){
              return signalPassFailure();
            }
          } else {
            auto ret = affine::loopUnrollFull(forOp);
            if(failed(ret)){
              return signalPassFailure();
            }
          }
        }
      }
    });
  }
};


// =====================================================================
//                  Vecotr Dialect To LLVM Dialect
// =====================================================================
// 将memref lowering到llvm上，因为 passes.h.inc中的base类没有提供可以选择indexBitWidth的options，所以自己写了一个
struct VectorToLLVMPass : public PassWrapper<VectorToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorToLLVMPass)

  VectorToLLVMPass(unsigned indexBitWidth_=32) : indexBitWidth(indexBitWidth_) {};

  unsigned indexBitWidth;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    options.overrideIndexBitwidth(indexBitWidth);
    bool force32BitVectorIndices = false;
    if (indexBitWidth == 32) {
      force32BitVectorIndices = true;
    }

    LLVMTypeConverter converter(&getContext(), options);
    mlir::populateVectorToLLVMConversionPatterns(converter, patterns, false, force32BitVectorIndices);
    // mlir::populateVectorToLLVMMatrixConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};


// =====================================================================
//    Add external libs to moduelOp, which will be added in attribute
// =====================================================================
struct AddExternalLibPass : public PassWrapper<AddExternalLibPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddExternalLibPass)

  AddExternalLibPass(Target target_, const std::string& arch_) : target(target_), arch(arch_) {
    if (target == Target::CUDA) 
      libPath = std::string(CUDA_BITCODE_PATH);
    else if (target == Target::ROCm) 
      libPath = std::string(HIP_BITCODE_PATH);
  };

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    const auto& bcfiles = getBitcodefiles();
    llvm::SmallVector<NamedAttribute, 2> attrs;
    for (size_t i = 0; i < bcfiles.size(); ++i) {
      auto name = StringAttr::get(module->getContext(), bcfiles[i].first);
      auto path = StringAttr::get(module->getContext(), bcfiles[i].second);
      NamedAttribute attr(name, path);
      attrs.push_back(attr);
    }
    DictionaryAttr dict = DictionaryAttr::get(module->getContext(), attrs);
    module.getOperation()->setAttr(AttrExternLib, dict);
  }

private:
  Target target;
  const std::string arch;
  std::string libPath;

  bool isBCFile(const std::string& extname,const std::string& fileName){
    bool is_oclcVersion = false;
    bool is_gfxArchMatch = false;
    std::vector<std::string> nameList;
    if (target == Target::CUDA) {
      nameList = {
        "libdevice.10.bc"
      };
    } else if (target == Target::ROCm) {
      nameList = {
        "opencl.bc", "ocml.bc", "ockl.bc", "oclc_finite_only_off.bc", "oclc_daz_opt_on.bc", "oclc_abi_version_400.bc", 
        "oclc_correctly_rounded_sqrt_on.bc", "oclc_unsafe_math_off.bc", "oclc_wavefrontsize64_on.bc"
      };
    }
    if(extname != ".bc"){
      return false;
    }
    for(const auto& name : nameList){
      if(fileName == name){
         return true;
      }
    }
    if(fileName.find("oclc_isa_version") != fileName.npos && fileName.find(arch) != fileName.npos){
      return true;
    }
    return false;
  }

  std::vector<std::pair<std::string,std::string>> getBitcodefiles() {
    std::vector<std::pair<std::string,std::string>> files;
    int index = 0;
    for (const auto& entry : std::filesystem::directory_iterator(libPath)) {
      if (entry.is_regular_file()) {
        const auto& fileName = entry.path().filename().string();
        auto extname = entry.path().extension().string();
        if(isBCFile(extname, fileName)){
          auto pair = std::make_pair("library_"+std::to_string(index++), libPath+"/"+fileName);
          files.push_back(std::move(pair));
        }
      }
    }
    // assert(files.size() == 10);
    return files;
  }
};


// =====================================================================
//              replace memref::AllocOp to LLVM::GlobalOp
// =====================================================================
// replace alloc<shared> to getGlobalOp
struct ReplaceAllocOpToGetGlobalOp : public PassWrapper<ReplaceAllocOpToGetGlobalOp, OperationPass<ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceAllocOpToGetGlobalOp)
   ReplaceAllocOpToGetGlobalOp() = default;
   void runOnOperation() override {
     auto module = getOperation();
    int i = 0;
     std::vector<MemRefType> memTypesToAdd {};
     module.walk<WalkOrder::PreOrder>([&](memref::AllocOp allocOp) {
      auto memspace = allocOp.getResult().getType().getMemorySpaceAsInt();
      if(memspace == (int)KernelCodeGen::MemorySpace::shared){
        OpBuilder builder(allocOp);
        OpBuilder b(module);
        b.setInsertionPointToStart(module.getBody());
        auto globalOp = b.create<memref::GlobalOp>(
          b.getUnknownLoc(),
          SHM_VAR_NAME(i),
          b.getStringAttr("public"),
          allocOp.getResult().getType(),
          Attribute(),
          false,
          IntegerAttr()
          );
        globalOp.setAlignment(KCG_ALIGNBYTE);  // 对齐到 4*sizeof(float) 字节，以增加访问效率
        auto newop = builder.create<memref::GetGlobalOp>(
          builder.getUnknownLoc(),allocOp.getResult().getType(),SHM_VAR_NAME(i));
        allocOp.getResult().replaceAllUsesWith(newop);
        allocOp.erase();
        ++i;
      }
     });
   }
}; 


// =====================================================================
//               Change the size of LLVM::GlobalOp to 0
// =====================================================================
// 将globalshm的尺寸修改为0
struct SetShmSizeZeroPass : public PassWrapper<SetShmSizeZeroPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetShmSizeZeroPass)

  SetShmSizeZeroPass() {};

  void runOnOperation() override {
    auto mod = getOperation();
    std::vector<mlir::LLVM::GlobalOp> globalOps = {};
    mod.walk([&](mlir::LLVM::GlobalOp op) {
      auto type = op.getGlobalType();
      auto arrTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type);
      auto newType = mlir::LLVM::LLVMArrayType::get(arrTy.getElementType(),0);
      auto builder = mlir::OpBuilder(op);
      //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type, bool isConstant, Linkage linkage, StringRef name, Attribute value, uint64_t alignment = 0, unsigned addrSpace = 0, bool dsoLocal = false, bool thread_local_ = false, SymbolRefAttr comdat = {}, ArrayRef<NamedAttribute> attrs = {});
      uint64_t align = 0;
      if(op.getAlignment()){
        align = op.getAlignment().value();
      }
      auto newOp = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(),
        newType,
        op.getConstant(),
        op.getLinkage(),
        op.getSymName(),
        op.getValueAttr(),
        align,
        op.getAddrSpace(),
        op.getDsoLocal(),
        op.getThreadLocal_()
      );
      auto useRange = op.getSymbolUses(mod);
      auto result = op.replaceAllSymbolUses(newOp.getSymNameAttr(),mod);
      assert(result.succeeded() && "setshmsizezeroPass failed");
      op.erase();
      globalOps.push_back(newOp);
    });
    if (globalOps.size()) {
      SmallVector<Operation*> gtps;
      mod.walk([&](mlir::LLVM::AddressOfOp op) {
        auto ptr = mlir::dyn_cast_if_present<mlir::LLVM::LLVMPointerType>(op.getRes().getType());
        if(ptr.getAddressSpace() == (int)KernelCodeGen::MemorySpace::shared) {
          // auto parentOp = mlir::dyn_cast<LLVM::LLVMFuncOp>(op->getParentOp());
          // if (!parentOp) assert(false);
          // auto dataflowTypeAttr = mlir::dyn_cast<mlir::StringAttr>(parentOp->getAttr("func.dataflow.type"));
          // std::string type_ = dataflowTypeAttr.getValue().str();
          for (auto user : op.getResult().getUsers()) {
            if (auto getptr = mlir::dyn_cast<LLVM::GEPOp>(user)) {
              getptr.replaceAllUsesWith(op.getResult());
              gtps.push_back(user);
            }
          }
        }
      });
      for (auto gtp : gtps) gtp->erase();
    }
  }
};


// =====================================================================
//    Multiple allocOp/allocaOp are merged into one allocOp/allocaOp
// =====================================================================
// 将alloc和alloca操作合并，生成一个memref
struct CombineMemrefPass : public PassWrapper<CombineMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CombineMemrefPass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast<func::FuncOp>(&op)) {
        combineAllocOrAllocaOp<memref::AllocOp>(funcOp);
        // combineAllocOrAllocaOp<memref::AllocaOp>(funcOp);
      }
    }
  }

  template <typename LoadOrStoreOp>
  AffineMap moreDimToOneDimMap(LoadOrStoreOp op, int64_t startIndex, llvm::ArrayRef<int64_t> shapes, MLIRContext* context) {
    auto oldAffineMap = op.getAffineMap();
    auto oldExprs = oldAffineMap.getResults();
    OpBuilder b(context);
    AffineExpr expr = b.getAffineConstantExpr(0);
    // affine_map<(x,y) -> (f(x,y), g(x,y))>
    for (size_t i=0; i<oldExprs.size(); i++) {
      int num = 1;
      for (size_t j=i+1; j<shapes.size(); j++) {
        num *= shapes[j];
      }
      expr = expr + oldExprs[i] * num;
    }
    expr = startIndex + expr;
    auto map = AffineMap::get(oldAffineMap.getNumDims(), 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), context);
    return map;
  }

  
  template<typename AllocOrAllocaOp>
  void combineAllocOrAllocaOp(func::FuncOp &funcOp) {
    int64_t memSizeAB = 0;
    int64_t memSizeC = 0;
    llvm::DenseMap<AllocOrAllocaOp, int64_t> indexMap;
    AllocOrAllocaOp firstOp = nullptr;
    MemRefType type;
    // 记录 allocop的mem尺寸起始位置
    funcOp.walk<WalkOrder::PreOrder>([&](AllocOrAllocaOp allocOp) {
      if(tools::isOpAttrEqualToString(allocOp,AttrDescription,"smC")){
        type = mlir::dyn_cast<MemRefType>(allocOp.getResult().getType());
        int64_t temp = 1;
        for (auto shape : type.getShape()) {
          temp *= shape;
        }
        indexMap.try_emplace(allocOp, memSizeC);
        memSizeC = temp;
      }
      else{
        if (memSizeAB == 0) {
          firstOp = allocOp;
        }
        indexMap.try_emplace(allocOp, memSizeAB);
        type = mlir::dyn_cast<MemRefType>(allocOp.getResult().getType());
        int64_t temp = 1;
        for (auto shape : type.getShape()) {
          temp *= shape;
        }
        memSizeAB += temp;
      }
    });
    if (!memSizeAB){ return; }
    memSizeAB = memSizeC > memSizeAB ? memSizeC : memSizeAB;
    OpBuilder b(firstOp);
    auto newType = MemRefType::get({memSizeAB}, type.getElementType(), {}, type.getMemorySpaceAsInt());
    auto newAllocOp = b.create<AllocOrAllocaOp>(firstOp.getLoc(), newType);
    newAllocOp.setAlignment(KCG_ALIGNBYTE);
    for (const auto& pair : indexMap) {
      Value result = pair.first->getResult(0);
      auto t = mlir::dyn_cast<MemRefType>(result.getType());
      auto shapes = t.getShape();  // 每次alloc的shape

      SmallVector<Operation *> users;

      for (auto user : result.getUsers()) {  // collect users
        users.push_back(user);
      }
      // auto users = result.getUsers();
      // 替换 user 位置的op
      for (auto user : users) {
        if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(user)) {
          auto map = moreDimToOneDimMap(loadOp, pair.second, shapes, loadOp->getContext());
          b.setInsertionPointAfter(loadOp);
          auto newLoadOp = b.create<affine::AffineLoadOp>(loadOp.getLoc(), newAllocOp.getResult(), map, loadOp.getMapOperands());
          loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
          loadOp.erase();

        } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
          auto map = moreDimToOneDimMap(storeOp, pair.second, shapes, storeOp->getContext());
          b.setInsertionPointAfter(storeOp);
          b.create<affine::AffineStoreOp>(storeOp.getLoc(), storeOp.getValue(), newAllocOp.getResult(), map, storeOp.getMapOperands());
          storeOp.erase();

        } else if (auto vectorLoadOp = mlir::dyn_cast<affine::AffineVectorLoadOp>(user)) {
          auto map = moreDimToOneDimMap(vectorLoadOp, pair.second, shapes, vectorLoadOp->getContext());
          b.setInsertionPointAfter(vectorLoadOp);
          auto newVectorLoadOp = b.create<affine::AffineVectorLoadOp>(vectorLoadOp.getLoc(), vectorLoadOp.getVectorType(), 
                                                              newAllocOp.getResult(), map, vectorLoadOp.getMapOperands());
          vectorLoadOp.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
          vectorLoadOp.erase();

        } else if (auto vectorStoreOp = mlir::dyn_cast<affine::AffineVectorStoreOp>(user)) {
          auto map = moreDimToOneDimMap(vectorStoreOp, pair.second, shapes, vectorStoreOp->getContext());
          b.setInsertionPointAfter(vectorStoreOp);
          b.create<affine::AffineVectorStoreOp>(vectorStoreOp.getLoc(), vectorStoreOp.getValue(), 
                                            newAllocOp.getResult(), map, vectorStoreOp.getMapOperands());
          vectorStoreOp.erase();
        }
      }
      pair.first->erase();
    }
  }
};


// =====================================================================
//                            Unused Passes
// =====================================================================
// *** 弃用 *** 将affine.parallel 提取为kernel的形式。外层内层分别替换dim变量为blockid和threadid
struct ExtractAffineParallelPass : public PassWrapper<ExtractAffineParallelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractAffineParallelPass)

  void runOnOperation() override {
    // constexpr gpu::Dimension dims[] = {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    constexpr gpu::Dimension dims[] = {gpu::Dimension::y, gpu::Dimension::x};
    auto mod = mlir::dyn_cast<ModuleOp>(getOperation());
    mlir::func::FuncOp kernel = nullptr;
    // 定位 funcop, innerParallel, outerParallel, 两层parallel间的首个op
    for(auto &op : mod.getOps()){
      kernel = mlir::dyn_cast<mlir::func::FuncOp>(op);
      if(kernel != nullptr){
        break;
      }
    }
    assert(kernel != nullptr);
    mlir::affine::AffineParallelOp outerParallel = nullptr;
    mlir::affine::AffineParallelOp innerParallel = nullptr;
    for(auto &op : kernel.getOps()){
      outerParallel = mlir::dyn_cast<mlir::affine::AffineParallelOp>(op);
      if(outerParallel != nullptr){
        break;
      }
    }
    assert(outerParallel != nullptr);
    mlir::Operation* betweenParallelFirstOp = nullptr;
    for(auto &op : outerParallel.getOps()){
      if(betweenParallelFirstOp == nullptr){
        betweenParallelFirstOp = &op;
      }
      innerParallel = mlir::dyn_cast<mlir::affine::AffineParallelOp>(op);
      if(innerParallel != nullptr){
        break;
      }
    }
    assert(innerParallel != nullptr);
    mlir::OpBuilder builder(betweenParallelFirstOp);
    // collect dim const and set attr
    auto getIntUpperBounds = [](mlir::affine::AffineParallelOp& parallel,std::vector<int>& ret) 
    {
      for(auto e : parallel.getUpperBoundsMap().getConstantResults()) {
        ret.push_back(e);
      }
    };
    std::vector<int> in{},out{};
    getIntUpperBounds(outerParallel,out);
    getIntUpperBounds(innerParallel,in);
    llvm::ArrayRef<int32_t> gridDim = out;
    llvm::ArrayRef<int32_t> blockDim = in;
    kernel->setAttr("func.grid.dim", builder.getDenseI32ArrayAttr(gridDim));
    kernel->setAttr("func.block.dim", builder.getDenseI32ArrayAttr(blockDim));

    mlir::SmallVector<mlir::Value,4> blockIdOps;
    mlir::SmallVector<mlir::Value,4>  threadIdOps;
    for(int i=0;i<outerParallel.getNumDims();++i){
      auto bid = builder.create<mlir::gpu::BlockIdOp>(builder.getUnknownLoc(),dims[i]);
      blockIdOps.push_back(bid);
    }
    for(int i=0;i<innerParallel.getNumDims();++i){
      auto tid = builder.create<mlir::gpu::ThreadIdOp>(builder.getUnknownLoc(), gpu::Dimension::x);
      threadIdOps.push_back(tid);
    }
    auto argCnt = innerParallel.getBody()->getArguments().size();
    llvm::outs() << "argCnt = " << argCnt << "\n";llvm::outs().flush();
    for(int i=0;i<argCnt;++i){
      Value inarg = innerParallel.getBody()->getArgument(i);
      Value replacement = threadIdOps[i];
      inarg.replaceAllUsesWith(replacement);
    }
    auto argCnt1 = outerParallel.getBody()->getArguments().size();
    llvm::outs() << "argCnt1 = " << argCnt1 << "\n";llvm::outs().flush();
    for(int i=0;i<argCnt1;++i){
      Value arg = outerParallel.getBody()->getArgument(i);
      Value replacement = blockIdOps[i];
      arg.replaceAllUsesWith(replacement);
    }
    // 处理循环体中的内容

    SmallVector<Operation*> innerOpsToMove;
    for (auto &op : innerParallel.getBody()->getOperations()) {
      // 如果需要，将操作移动到新的位置
      innerOpsToMove.push_back(&op);
    }
    Operation* innerTmpOp = innerParallel;
    for(auto op : innerOpsToMove){
      if(mlir::dyn_cast<mlir::affine::AffineYieldOp>(op) == nullptr){
        op->moveAfter(innerTmpOp);
        innerTmpOp = op;
      }
    }

    innerParallel.erase();

    SmallVector<Operation*> outerOpsToMove;
    for (auto &op : outerParallel.getBody()->getOperations()) {
      // 如果需要，将操作移动到新的位置
      outerOpsToMove.push_back(&op);
    }
    Operation* outTMpOp = outerParallel;
    for(auto op : outerOpsToMove){
      if(mlir::dyn_cast<mlir::affine::AffineYieldOp>(op) == nullptr){
        op->moveAfter(outTMpOp);
        outTMpOp = op;
      }
    }
    // 删除 `affine.parallel` 操作
    outerParallel.erase();
  }
};

// *** 弃用 *** 将scf的parallelOp 转成Gpu的block/threadIdOp表示，func添加grid/block size作为属性
struct SCFParallelToGPULowering : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp outerParallelOp, PatternRewriter &rewriter) const final {
    constexpr gpu::Dimension dims[] = {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    auto &ops = outerParallelOp.getBody()->getOperations();
    if (ops.empty()){
      assert(false && "SCFParallel->GPU : outerParallelOp not exist!");
      return failure();
    }
    outerParallelOp.getLowerBound();
    scf::ParallelOp innerParallelOp = nullptr;
    for (Operation &op : ops) {
      innerParallelOp = dyn_cast<scf::ParallelOp>(&op);
      if (innerParallelOp)
        break;
    }
    if (!innerParallelOp){
      assert(false && "SCFParallel->GPU : innerParallelOp not exist!");
      return failure();
    }

    auto outerUpperBounds = outerParallelOp.getUpperBound();
    auto innerUpperBounds = innerParallelOp.getUpperBound();

    std::vector<int32_t> blockUpperBounds;
    std::vector<int32_t> threadUpperBounds;

    // 替换外层 parallelOp 为 gpu::BlockIdOp
    Location loc = outerParallelOp.getLoc();
    SmallVector<Value, 3> blockIds;
    for (unsigned i = 0; i < outerParallelOp.getNumLoops(); ++i) {
      auto blockId = rewriter.create<gpu::BlockIdOp>(loc, dims[i]);
      blockIds.push_back(blockId);

      auto constOp = outerUpperBounds[i].getDefiningOp<arith::ConstantOp>();
      auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue());
      blockUpperBounds.push_back(intAttr.getInt());
    }

    // 替换内层 parallelOp 为 gpu::ThreadIdOp
    rewriter.setInsertionPoint(innerParallelOp);
    SmallVector<Value, 3> threadIds;
    for (unsigned i = 0; i < innerParallelOp.getNumLoops(); ++i) {
      auto threadId = rewriter.create<gpu::ThreadIdOp>(loc, dims[i]);
      threadIds.push_back(threadId);

      auto constOp = innerUpperBounds[i].getDefiningOp<arith::ConstantOp>();
      auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue());
      threadUpperBounds.push_back(intAttr.getInt());
    }

    // 将func设置block和thread的上界属性
    auto parentOp = outerParallelOp->getParentOp();
    auto funcOp = mlir::dyn_cast<func::FuncOp>(parentOp);
    if (funcOp == nullptr) {
      llvm::errs() << "The ParentOp of scf::ParallelOp must is FuncOp!\n";
      assert(false);
    }
    funcOp->setAttr("func.grid.dim", rewriter.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(blockUpperBounds)));
    funcOp->setAttr("func.block.dim", rewriter.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(threadUpperBounds)));

    // 替换使用外层和内层循环变量的操作
    auto outerInductionVars = outerParallelOp.getInductionVars();
    for (unsigned i = 0; i < outerInductionVars.size(); ++i) {
      outerInductionVars[i].replaceAllUsesWith(blockIds[i]);
    }

    auto innerInductionVars = innerParallelOp.getInductionVars();
    for (unsigned i = 0; i < innerInductionVars.size(); ++i) {
      innerInductionVars[i].replaceAllUsesWith(threadIds[i]);
    }

    // 内层操作移出内层 p  collect op
    SmallVector<Operation *, 4> innerOpsToMove;
    for (Operation &op : innerParallelOp.getBody()->getOperations()) {
      if (!dyn_cast<scf::YieldOp>(op)) {
        innerOpsToMove.push_back(&op);
      }
    }
    // 内层操作移出内层 p 
    Operation *innerTempOp = threadIds.back().getDefiningOp();
    for (Operation *op : innerOpsToMove) {
      op->moveAfter(innerTempOp);
      innerTempOp = op;
    }
    rewriter.eraseOp(innerParallelOp);

    // 外 collect op
    SmallVector<Operation *, 4> outerOpsToMove;
    for (Operation &op : outerParallelOp.getBody()->getOperations()) {
      if (!dyn_cast<scf::YieldOp>(op)) {
        outerOpsToMove.push_back(&op);
      }
    }
    // move
    Operation *outerTempOp = blockIds.back().getDefiningOp();
    for (Operation *op : outerOpsToMove) {
      op->moveAfter(outerTempOp);
      outerTempOp = op;
    }
    rewriter.eraseOp(outerParallelOp);

    return success();
  }
};

// ***弃用***，自己写了一个从gpu到rocdl的pass，转了idop和BarrierOp，主要是添加属性range
struct ROCDLIdOpModifyPass : public PassWrapper<ROCDLIdOpModifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ROCDLIdOpModifyPass)
  
  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    for (Operation &op : module.getBody()->getOperations()) {
      auto funcOp = mlir::dyn_cast<func::FuncOp>(&op);
      if (funcOp == nullptr) {
        llvm::errs() << "there is other operations which is not funcOp in the module!\n";
        assert(false);
      }
      auto blockDims = mlir::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr("func.grid.dim"));
      auto threadDims = mlir::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr("func.block.dim"));


      funcOp.walk([&](Operation *op) {
        if (auto blockIdXOp = mlir::dyn_cast<ROCDL::BlockIdXOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[0]}));
        } else if (auto blockIdYOp = mlir::dyn_cast<ROCDL::BlockIdYOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[1]}));
        } else if (auto blockIdZOp = mlir::dyn_cast<ROCDL::BlockIdZOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[2]}));
        } else if (auto threadIdXOp = mlir::dyn_cast<ROCDL::ThreadIdXOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[0]}));
        } else if (auto threadIdYOp = mlir::dyn_cast<ROCDL::ThreadIdYOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[1]}));
        } else if (auto threadIdZOp = mlir::dyn_cast<ROCDL::ThreadIdZOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[2]}));
        }
      });
    }
  }
};

// ***弃用***，去除多余的unrealized_conversion_cast操作/因为内置有去除函数
struct EraseRedundantUnCCastPass : public PassWrapper<EraseRedundantUnCCastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseRedundantUnCCastPass)
  
  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    SmallVector<std::pair<Operation*, Operation*>> pairOps;
    SmallVector<Operation*> noChOps;
    module.walk([&](Operation *op){
      if (auto uccOp = mlir::dyn_cast<UnrealizedConversionCastOp>(op)) {
        for (auto &use: uccOp.getResult(0).getUses()) {
          Operation *nextOp = use.getOwner();
          if (isa<UnrealizedConversionCastOp>(nextOp))
            pairOps.push_back(std::make_pair(op, nextOp));
          break;
        }
        if (uccOp.use_empty()) {
          noChOps.push_back(op);
        }
      }
    });
    for (auto pairOp: pairOps) {
      auto firstOp = mlir::dyn_cast<UnrealizedConversionCastOp>(pairOp.first);
      auto secondOp = mlir::dyn_cast<UnrealizedConversionCastOp>(pairOp.second);
      if (firstOp.getOperand(0).getType() == secondOp.getResult(0).getType()) {
        secondOp.getResult(0).replaceAllUsesWith(firstOp.getOperand(0));
      }
      pairOp.second->erase();
      pairOp.first->erase();
    }
    for (auto noChOp: noChOps) {
      // llvm::outs() << *noChOp << "\n";
      noChOp->erase();
    }
  }
};

// ***弃用***，将arith的constantOp（index）转成i64
struct ConvertArithIndexToI64Pass : public PassWrapper<ConvertArithIndexToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertArithIndexToI64Pass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    module.walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        Type constantType = constantOp.getValue().getType();
        if (constantType.isIndex()) {
          auto indexValue = mlir::dyn_cast<IntegerAttr>(constantOp.getValue()).getInt();
          OpBuilder builder(op);
          auto i64Op = builder.create<arith::ConstantOp>(constantOp.getLoc(), builder.getI64IntegerAttr(indexValue));
          auto indexVal = builder.create<arith::IndexCastOp>(i64Op.getLoc(), builder.getIndexType(), i64Op);
          constantOp.getResult().replaceAllUsesWith(indexVal.getResult());
          constantOp.erase();
        }
      }
    });
  }
};

// ***弃用*** 没有使用，和上面一样的功能
struct ConvertArithConstantIndexToI64 : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constOp, PatternRewriter &rewriter) const final {
    Type constType = constOp.getValue().getType();
    if (constType.isIndex()) {
      auto indexValue = mlir::dyn_cast<IntegerAttr>(constOp.getValue()).getInt();
      auto i64Op = rewriter.create<arith::ConstantOp>(constOp.getLoc(), rewriter.getI64IntegerAttr(indexValue));
      auto indexVal = rewriter.create<arith::IndexCastOp>(i64Op.getLoc(), rewriter.getIndexType(), i64Op);
      rewriter.replaceOp(constOp, indexVal);
      return success();
    } else {
      return failure();
    }
  }
};

// ***弃用***
struct ConvertIndexToI64Pass : public PassWrapper<ConvertIndexToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertIndexToI64Pass)
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    patterns.add<ConvertArithConstantIndexToI64>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      return signalPassFailure();
  }
};

// ***弃用*** 使用函数代替了没有写模式匹配了，因为一直报错
struct ReplacePtrtointAndCallOp : public OpRewritePattern<LLVM::PtrToIntOp> {
  using OpRewritePattern<LLVM::PtrToIntOp>::OpRewritePattern;

  explicit ReplacePtrtointAndCallOp(MLIRContext *context, LLVM::LLVMFuncOp newFuncOp_) : 
            OpRewritePattern<LLVM::PtrToIntOp>(context), newFuncOp(newFuncOp_) {};

  LogicalResult matchAndRewrite(LLVM::PtrToIntOp ptrOp, PatternRewriter &rewriter) const final {
    // get ptrop & new ptrop
    Value argPtr = ptrOp.getArg();
    auto newPtrOp = rewriter.create<LLVM::PtrToIntOp>(ptrOp.getLoc(), rewriter.getIntegerType(64), argPtr);
    auto users = ptrOp.getResult().getUsers();
    for (auto user : users) {
      if (auto callOp = mlir::dyn_cast<LLVM::CallOp>(user)) {
        rewriter.setInsertionPointAfter(callOp);
        auto newCallOp = rewriter.create<LLVM::CallOp>(callOp->getLoc(), newFuncOp, ValueRange({newPtrOp.getResult()}));
        // callOp.getResult().replaceAllUsesWith(newCallOp.getResult());
        // callOp.erase();
        rewriter.replaceOp(callOp, newCallOp);
        // rewriter.eraseOp(callOp);
      }
    }
    // ptrOp.erase();
    rewriter.replaceOp(ptrOp, newPtrOp);
    return success();
  }

  private:
  LLVM::LLVMFuncOp newFuncOp;
};

// ***弃用*** 替换malloc(i32) -> malloc(i64) / ptrtointOp & callOp 也替换成符合i64的操作
struct MallocFuncOpArgTypeI32ToI64Pass : public PassWrapper<MallocFuncOpArgTypeI32ToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MallocFuncOpArgTypeI32ToI64Pass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    auto reuslt = createI64MallocFuncOp(module);
    replacePtrtointAndCallOp(module, reuslt.first);   // create i64 malloc
    deleteI32MallocFuncOp(reuslt.second);   // erase i32 malloc
  }

  private:

  void replacePtrtointAndCallOp(ModuleOp &module, LLVM::LLVMFuncOp &mallocFuncOp) {
    if (!mallocFuncOp) return;
    SmallVector<LLVM::PtrToIntOp> ptrOps;
    module.walk([&](LLVM::PtrToIntOp ptrOp) {
      ptrOps.push_back(ptrOp);
    });
    
    for (auto ptrOp: ptrOps) {
      Value argPtr = ptrOp.getArg();
      OpBuilder buidler(ptrOp);
      auto newPtrOp = buidler.create<LLVM::PtrToIntOp>(ptrOp.getLoc(), buidler.getIntegerType(64), argPtr);

      auto users = ptrOp.getResult().getUsers();
      for (auto user : users) {
        if (auto callOp = mlir::dyn_cast<LLVM::CallOp>(user)) {
          buidler.setInsertionPointAfter(callOp);
          auto newCallOp = buidler.create<LLVM::CallOp>(callOp->getLoc(), mallocFuncOp, ValueRange({newPtrOp.getResult()}));
          callOp.getResult().replaceAllUsesWith(newCallOp.getResult());
          callOp.erase();
          // rewriter.replaceOp(callOp, newCallOp);
          // rewriter.eraseOp(callOp);
        }
      }
      ptrOp.erase();
      // rewriter.replaceOp(ptrOp, newPtrOp);
      // llvm::outs() <<newPtrOp << "\n";
    }
  }

  std::pair<LLVM::LLVMFuncOp, LLVM::LLVMFuncOp> createI64MallocFuncOp(ModuleOp &module) {
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast<LLVM::LLVMFuncOp>(&op)) {
        if (funcOp.getName() != "malloc") {
          continue;
        } else {
          LLVM::LLVMFunctionType funcType = funcOp.getFunctionType();
          if (funcType.getNumParams() == 1 && funcType.getParams()[0].isInteger(64)) {
            return std::make_pair(funcOp, nullptr);   // malloc本身就是i64的func

          } else if (funcType.getNumParams() == 1 && funcType.getParams()[0].isInteger(32)) {
            OpBuilder b(module.getBodyRegion());
            MLIRContext *context = module->getContext();
            auto newFuncType = LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(context), 
                                      {IntegerType::get(context, 64)}, /*opaquePointers*/false);
            auto newFuncOp = b.create<LLVM::LLVMFuncOp>(module->getLoc(), "malloc", newFuncType);
            return std::make_pair(newFuncOp, funcOp);   // malloc是i32的func，创建新的true
          }
        }
      }
    }
    return std::make_pair(nullptr, nullptr);
  }

  void deleteI32MallocFuncOp(LLVM::LLVMFuncOp &funcOp) {
    if (funcOp){
      funcOp.erase();
    }
  }
};

// *** Unused *** 将memref ，展平 并设置align 16（4*sizeof float）。结果目前存在错误，C变成对称阵（C_ij==C_ji）
struct FlattenMemrefPass : public PassWrapper<FlattenMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenMemrefPass)

  void runOnOperation() override {
    auto module = mlir::dyn_cast<ModuleOp>(getOperation());
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto funcOp = mlir::dyn_cast<func::FuncOp>(&op)) {
        flattenAllocOp<memref::AllocOp>(funcOp);
        flattenAllocOp<memref::AllocaOp>(funcOp);
      }
    }
  }

  template <typename LoadOrStoreOp>
  AffineMap moreDimToOneDimMap(LoadOrStoreOp op, int64_t startIndex, llvm::ArrayRef<int64_t> shapes, MLIRContext* context) {
    auto oldAffineMap = op.getAffineMap();
    auto oldExprs = oldAffineMap.getResults();
    OpBuilder b(context);
    AffineExpr expr = b.getAffineConstantExpr(0);
    for (size_t i=0; i<oldExprs.size(); i++) {
      int num = 1;
      for (size_t j=i+1; j<shapes.size(); j++) {
        num *= shapes[j];
      }
      expr = expr + oldExprs[i] * num;
    }
    if(startIndex != 0){
      expr = startIndex + expr;
    }
    auto map = AffineMap::get(oldAffineMap.getNumDims(), oldAffineMap.getNumSymbols(), llvm::ArrayRef<mlir::AffineExpr>({expr}), context);
    return map;
  }

  
  template<typename AllocOrAllocaOp>
  void flattenAllocOp(func::FuncOp &funcOp) {
    funcOp.walk<WalkOrder::PreOrder>([&](AllocOrAllocaOp oldOp) {
      Value result = oldOp->getResult(0);
      auto resType = mlir::dyn_cast<MemRefType>(result.getType());
      auto resShape = resType.getShape();  // 每次alloc的shape

      int64_t len = 1;
      for(auto d : resShape){
        len *= d;
      }
      SmallVector<Operation *> users;
      for (auto user : result.getUsers()) {  // collect users
        users.push_back(user);
      }
      auto b = mlir::OpBuilder(oldOp);
      auto newType = MemRefType::get({len}, resType.getElementType(), {}, resType.getMemorySpaceAsInt());
      auto newAllocOp = b.create<AllocOrAllocaOp>(oldOp.getLoc(), newType);
      newAllocOp.setAlignment(KCG_ALIGNBYTE);
      SmallVector<Operation *> opToDelete {};

      // 替换 user 位置的op
      for (auto user : users) {
        if (auto loadOp = mlir::dyn_cast<affine::AffineLoadOp>(user)) {
          auto map = moreDimToOneDimMap(loadOp, 0, resShape, loadOp->getContext());
          b.setInsertionPointAfter(loadOp);
          auto newLoadOp = b.create<affine::AffineLoadOp>(loadOp.getLoc(), newAllocOp.getResult(), map, loadOp.getMapOperands());
          loadOp.getResult().replaceAllUsesWith(newLoadOp.getResult());
          opToDelete.push_back(loadOp);
        } else if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
          auto map = moreDimToOneDimMap(storeOp, 0, resShape, storeOp->getContext());
          b.setInsertionPointAfter(storeOp);
          b.create<affine::AffineStoreOp>(storeOp.getLoc(), storeOp.getValue(), newAllocOp.getResult(), map, storeOp.getMapOperands());
          opToDelete.push_back(storeOp);
        } else if (auto vectorLoadOp = mlir::dyn_cast<affine::AffineVectorLoadOp>(user)) {
          auto map = moreDimToOneDimMap(vectorLoadOp, 0, resShape, vectorLoadOp->getContext());
          b.setInsertionPointAfter(vectorLoadOp);
          auto newVectorLoadOp = b.create<affine::AffineVectorLoadOp>(vectorLoadOp.getLoc(), vectorLoadOp.getVectorType(), 
                                                              newAllocOp.getResult(), map, vectorLoadOp.getMapOperands());
          vectorLoadOp.getResult().replaceAllUsesWith(newVectorLoadOp.getResult());
          opToDelete.push_back(vectorLoadOp);


        } else if (auto vectorStoreOp = mlir::dyn_cast<affine::AffineVectorStoreOp>(user)) {
          auto map = moreDimToOneDimMap(vectorStoreOp, 0, resShape, vectorStoreOp->getContext());
          b.setInsertionPointAfter(vectorStoreOp);
          b.create<affine::AffineVectorStoreOp>(vectorStoreOp.getLoc(), vectorStoreOp.getValue(), 
                                            newAllocOp.getResult(), map, vectorStoreOp.getMapOperands());
          opToDelete.push_back(vectorStoreOp);
        }
        else {
          assert(false && " KCG Unimplement Cast!!");
        }
      }
      for(auto userOp : opToDelete){
        userOp->erase();
      }
      oldOp.erase();
    });

  }
};

// 添加 gpu.printfop
struct AddDebugLogPass : public PassWrapper<AddDebugLogPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddDebugLogPass)

  void runOnOperation() override {
    auto mod = mlir::dyn_cast<mlir::ModuleOp>(getOperation());
    mod.walk([&](mlir::func::FuncOp func){
      auto subOps = func.getOps();
      for(auto & op : subOps){
        if(auto desc = op.getAttr("kcg.debug")){
          // mlir::StringAttr printLog = mlir::dyn_cast<mlir::StringAttr>(desc);
          // mlir::OpBuilder b(&op);
          // b.setInsertionPointAfter(&op);
          // b.create<mlir::gpu::PrintfOp>(op.getLoc(),printLog,mlir::ValueRange());
        }
      }
    });
  }
};

// ###################################################################################################


// ===============================================================
//                           Using Passes
// ===============================================================
std::unique_ptr<OperationPass<ModuleOp>> createAmendAllocaOpAddrSpacePass(Target target) {
  return std::make_unique<AmendAllocaOpAddrSpacePass>(target);
}

std::unique_ptr<OperationPass<ModuleOp>> createParallelToGPUPass() {
  return std::make_unique<ParallelToGPUPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createGPUToROCDLOrNVVMPass(Target target, unsigned indexBitwidth) {
  return std::make_unique<GPUToROCDLOrNVVMPass>(target, indexBitwidth);
}

std::unique_ptr<OperationPass<ModuleOp>> createLLVMFuncOpAddGPUAttrPass(Target target) {
  return std::make_unique<LLVMFuncOpAddGPUAttrPass>(target);
}

std::unique_ptr<OperationPass<ModuleOp>> createAffineUnrollPass() {
  return std::make_unique<AffineUnrollPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createVectorToLLVMPass(unsigned indexBitWidth) {
  return std::make_unique<VectorToLLVMPass>(indexBitWidth);
}

std::unique_ptr<OperationPass<ModuleOp>> createAddExternalLibPass(Target target, const std::string& arch) {
  return std::make_unique<AddExternalLibPass>(target, arch);
}

std::unique_ptr<OperationPass<ModuleOp>> ReplaceAllocToGetglobalPass() {
  return std::make_unique<ReplaceAllocOpToGetGlobalOp>();
}

std::unique_ptr<OperationPass<ModuleOp>> createGlobalShmSetZeroPass() {
  return std::make_unique<SetShmSizeZeroPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createCombineMemrefPass() {
  return std::make_unique<CombineMemrefPass>();
}
// ================================================================

// ***弃用***
std::unique_ptr<OperationPass<ModuleOp>> createExtractAffineParallelPass() {
  return std::make_unique<ExtractAffineParallelPass>();
}
// ***弃用***
std::unique_ptr<OperationPass<ModuleOp>> createAddDebugLogPass() {
  return std::make_unique<AddDebugLogPass>();
}
// ***弃用***，自己写了一个从gpu到rocdl的pass，转了idop和BarrierOp
std::unique_ptr<OperationPass<ModuleOp>> createROCDLIdOpModifyPass() {
  return std::make_unique<ROCDLIdOpModifyPass>();
}
// ***弃用***
std::unique_ptr<OperationPass<ModuleOp>> createEraseRedundantUnCCastPass() {
  return std::make_unique<EraseRedundantUnCCastPass>();
}
// ***弃用***
std::unique_ptr<OperationPass<ModuleOp>> createConvertArithIndexToI64Pass() {
  return std::make_unique<ConvertArithIndexToI64Pass>();
}
// ***弃用***
std::unique_ptr<OperationPass<ModuleOp>> createMallocFuncOpArgTypeI32ToI64Pass() {
  return std::make_unique<MallocFuncOpArgTypeI32ToI64Pass>();
}
// *** Unused ***
std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemrefPass() {
  return std::make_unique<FlattenMemrefPass>();
}

}