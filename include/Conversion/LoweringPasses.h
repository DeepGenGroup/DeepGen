#ifndef _LoweringPasses_h_
#define _LoweringPasses_h_

#include "mlir/Pass/Pass.h"
#include "Common/Utils.h"

namespace KernelCodeGen {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertGPUPrintToLLVMPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAddDebugLogPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExtractAffineParallelPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createParallelToROCDLPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createROCDLIdOpModifyPass();  // no use

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEraseRedundantUnCCastPass();  // no use

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConvertArithIndexToI64Pass();  // no use

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAffineFullUnrollPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createVectorToLLVMPass(unsigned indexBitWidth);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGlobalShmSetZeroPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMallocFuncOpArgTypeI32ToI64Pass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAddExternalLibPass(const std::string& libsPath, const std::string& gfx_arch);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> ReplaceAllocToGetglobalPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCombineMemrefPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFlattenMemrefPass();
}

#endif