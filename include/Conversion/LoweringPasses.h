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