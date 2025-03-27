#ifndef _FuseMap_h_
#define _FuseMap_h_

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
  std::string fkName;
  std::string type;
  std::vector<std::string> fuseKernels;
  std::vector<std::vector<int64_t>> newArgsShape;
  std::vector<std::vector<int64_t>> newVarsShape;
  std::vector<std::string> newArgsDtype;
  std::vector<std::string> newVarsDtype;
  std::vector<std::map<std::string, int64_t>> newArgsIndex;
  std::vector<std::map<std::string, int64_t>> newVarsIndex;
  int outputArgNum;
};

std::vector<mlir::func::FuncOp> getKernelFuncOps(mlir::ModuleOp mod, std::vector<std::string> kernelNames);

std::tuple<mlir::func::FuncOp, std::vector<mlir::Value>, std::vector<mlir::Value>> 
  createFuseFuncAndMidMems(mlir::OpBuilder& builder, FuseKernelData fkd);

std::vector<std::vector<mlir::Value>> collectOldMems(std::vector<std::map<std::string, int64_t>> newMemsIndex, std::vector<mlir::func::FuncOp> fks);

void fuseKernels(mlir::func::FuncOp funcOp, std::vector<mlir::func::FuncOp> fks, std::vector<mlir::Value> newArgs, std::vector<mlir::Value> newVars, 
                 std::vector<std::vector<mlir::Value>> oldArgs, std::vector<std::vector<mlir::Value>> oldVars);

}

#endif // _FuseMap_h_