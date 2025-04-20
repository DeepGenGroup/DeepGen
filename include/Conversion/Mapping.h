#ifndef _Mapping_h_
#define _Mapping_h_

#include <tuple>
#include <iostream>
#include "Operators/Operators.h"
#include "Conversion/General/Rewriter.h"
#include "Common/Utils.h"

namespace KernelCodeGen {

void blockForOpShiftDown(std::vector<mlir::affine::AffineForOp>& blockForOps);

mlir::affine::AffineParallelOp fuseParallelOp(mlir::OpBuilder builder, 
                                              std::vector<mlir::affine::AffineParallelOp> parallelOps);

void eraseSingleIterForOps(mlir::func::FuncOp funcOp);

}

#endif // _Mapping_h_