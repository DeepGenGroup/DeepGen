#ifndef _Matmul_h_
#define _Matmul_h_
#include "Conversion/General/GeneralFuncs.h"

namespace KernelCodeGen {
namespace Matmul {

bool haveBatch(mlir::affine::AffineForOp batchForOp);

llvm::SmallVector<mlir::Value> amendOneDimBatch(mlir::func::FuncOp &funcOp, mlir::affine::AffineForOp &loopBatch);

mlir::AffineMap addBatchDimMap(mlir::OpBuilder builder, mlir::AffineMap map);

}
}
#endif // _Matmul_h_