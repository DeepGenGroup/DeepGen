#ifndef _Mapping_h_
#define _Mapping_h_

#include <tuple>
#include <iostream>
#include "Operators/Operators.h"
#include "Conversion/General/Rewriter.h"
#include "Common/Utils.h"

namespace KernelCodeGen {

template<typename operation>
std::vector<operation> collectOps(mlir::func::FuncOp funcOp, 
                                  const std::string& attrName, 
                                  const std::string& forDesc) {
  // 获取func中任意operation
  std::vector<operation> ops;
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](operation op) {
    if (auto desc = op->getAttr(attrName)) {
      auto descAttr = mlir::dyn_cast<mlir::StringAttr>(desc);
      auto descStr = descAttr.getValue().str();
      if (descStr == forDesc) {
        ops.push_back(op);
      }
    }
  });
  return ops;
}

std::vector<std::string> getArrayStringAttr(mlir::Operation* op, 
                                            std::string attrName);

void normalizeParaForOp(std::vector<mlir::affine::AffineForOp> &yloops, 
                        std::vector<std::map<std::string, int64_t>> &paraCfg);

void blockForOpShiftDown(std::vector<mlir::affine::AffineForOp>& blockForOps);

mlir::affine::AffineParallelOp fuseParallelOp(mlir::OpBuilder builder, 
                                              std::vector<mlir::affine::AffineParallelOp> parallelOps);

}

#endif // _Mapping_h_