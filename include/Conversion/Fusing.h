#ifndef _Fusing_h_
#define _Fusing_h_

#include <tuple>
#include <iostream>
#include "Operators/Operators.h"
#include "Conversion/General/Rewriter.h"
#include "Common/Utils.h"

namespace KernelCodeGen {

struct KernelData {
  /* kernel的数据结构 */
  std::string name;
  std::string type;
  std::vector<std::string> argNames;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::string> dtypes;
  std::vector<bool> isTrans;
  int outputArgNum;
};

struct FuseKernelData {
  /*融合kernel的list*/
  std::string name;
  std::string type;
  std::vector<std::string> fuseKernels;
  std::vector<std::vector<int64_t>> funcArgShapes;
  std::vector<std::vector<int64_t>> midVarShapes;
  std::vector<std::string> funcArgDtypes;
  std::vector<std::string> midVarDtypes;
  std::vector<std::map<std::string, std::vector<int64_t>>> funcArgIndex;
  std::vector<std::map<std::string, std::vector<int64_t>>> midVarIndex;
  std::vector<bool> isTranspose;
  std::vector<std::string> paraDims;
  int outputArgNum;
};

std::vector<std::vector<mlir::affine::AffineForOp>> getBatchFors(const std::vector<mlir::func::FuncOp>& fks); 

std::tuple<mlir::func::FuncOp, 
std::vector<mlir::Value>, 
std::vector<mlir::Value>> createFuseFuncAndMidMems(mlir::OpBuilder& builder, 
                                                   FuseKernelData fkd);

std::vector<std::vector<mlir::Value>> collectOldMems(const std::vector<std::map<std::string, std::vector<int64_t>>>& newMemsIndex, 
                                                     const std::vector<mlir::func::FuncOp>& fks);

void moveOperation(mlir::func::FuncOp funcOp, 
                   std::vector<mlir::func::FuncOp> fks, 
                   const std::vector<mlir::Value>& funcArgs, 
                   const std::vector<mlir::Value>& midVars, 
                   const std::vector<std::vector<mlir::Value>>& argToArgs, 
                   const std::vector<std::vector<mlir::Value>>& midToArgs);

void normalizeParaForOp(std::vector<mlir::affine::AffineForOp>& yloops);

void separateParaForOps(mlir::func::FuncOp funcOp);

void fuseParaForOps(mlir::func::FuncOp funcOp);

}

#endif // _Fusing_h_