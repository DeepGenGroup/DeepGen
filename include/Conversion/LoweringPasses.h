#ifndef _LoweringPasses_h_
#define _LoweringPasses_h_

#include "mlir/Pass/Pass.h"
#include "Common/Utils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

// lowering
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/Passes.h.inc"
#include "config.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"

// conversion
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
namespace KernelCodeGen {

/// Rewriting that unrolls SourceOp to scalars if it's operating on vectors.
template <typename SourceOp>
struct ScalarizeVectorOpLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor, 
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    const LLVMTypeConverter &converter = *this->getTypeConverter();
    TypeRange operandTypes(operands);
    if (llvm::none_of(operandTypes, llvm::IsaPred<VectorType>)) {
      return rewriter.notifyMatchFailure(op, "expected vector operand");
    }
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return rewriter.notifyMatchFailure(op, "expected no region/successor");
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected single result");
    VectorType vectorType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "expected vector result");

    Location loc = op->getLoc();
    Value result = rewriter.create<LLVM::UndefOp>(loc, vectorType);
    Type indexType = converter.convertType(rewriter.getIndexType());
    StringAttr name = op->getName().getIdentifier();
    Type elementType = vectorType.getElementType();

    for (int64_t i = 0; i < vectorType.getNumElements(); ++i) {
      Value index = rewriter.create<LLVM::ConstantOp>(loc, indexType, i);
      auto extractElement = [&](Value operand) -> Value {
        if (!isa<VectorType>(operand.getType()))
          return operand;
        return rewriter.create<LLVM::ExtractElementOp>(loc, operand, index);
      };
      auto scalarOperands = llvm::map_to_vector(operands, extractElement);
      Operation *scalarOp = rewriter.create(loc, name, scalarOperands, elementType, op->getAttrs());
      result = rewriter.create<LLVM::InsertElementOp>(loc, result, scalarOp->getResult(0), index);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLowering(LLVMTypeConverter &lowering, StringRef f32Func, 
                                StringRef f64Func, StringRef f32ApproxFunc)
      : ConvertOpToLLVMPattern<SourceOp>(lowering), 
      f32Func(f32Func), f64Func(f64Func), f32ApproxFunc(f32ApproxFunc) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor, 
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    static_assert(std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>, SourceOp>::value,
                  "expected op with same operand and result types");

    SmallVector<Value, 1> castedOperands;
    for (Value operand : adaptor.getOperands())
      castedOperands.push_back(maybeCast(operand, rewriter));

    Type resultType = castedOperands.front().getType();
    Type funcType = getFunctionType(resultType, castedOperands);
    StringRef funcName = getFunctionName(cast<LLVM::LLVMFunctionType>(funcType).getReturnType(), op.getFastmath());
    if (funcName.empty())
      return failure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp = rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, castedOperands);

    if (resultType == adaptor.getOperands().front().getType()) {
      rewriter.replaceOp(op, {callOp.getResult()});
      return success();
    }

    Value truncated = rewriter.create<LLVM::FPTruncOp>(op->getLoc(), adaptor.getOperands().front().getType(), callOp.getResult());
    rewriter.replaceOp(op, {truncated});
    return success();
  }

private:
  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    Type type = operand.getType();
    if (!isa<Float16Type>(type))
      return operand;
    return rewriter.create<LLVM::FPExtOp>(operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
  }

  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  StringRef getFunctionName(Type type, arith::FastMathFlags flag) const {
    if (isa<Float32Type>(type)) {
      if (((uint32_t)arith::FastMathFlags::afn & (uint32_t)flag) && !f32ApproxFunc.empty())
        return f32ApproxFunc;
      else
        return f32Func;
    }
    if (isa<Float64Type>(type))
      return f64Func;
    return "";
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, Type funcType, Operation *op) const {
    using LLVM::LLVMFuncOp;
    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);
    mlir::OpBuilder b(op->getParentOfType<FunctionOpInterface>());
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }

  const std::string f32Func;
  const std::string f64Func;
  const std::string f32ApproxFunc;
};


// std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertGPUPrintToLLVMPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAddDebugLogPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAmendAllocaOpAddrSpacePass(Target target);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createParallelToGPUPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExtractAffineParallelPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGPUToROCDLOrNVVMPass(Target target, unsigned indexBitwidth);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLLVMFuncOpAddGPUAttrPass(Target target);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createROCDLIdOpModifyPass();  // no use

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEraseRedundantUnCCastPass();  // no use

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertArithIndexToI64Pass();  // no use

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAffineUnrollPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createVectorToLLVMPass(unsigned indexBitWidth);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGlobalShmSetZeroPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMallocFuncOpArgTypeI32ToI64Pass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAddExternalLibPass(Target target, const std::string& arch);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> ReplaceAllocToGetglobalPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCombineMemrefPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFlattenMemrefPass();
}

#endif