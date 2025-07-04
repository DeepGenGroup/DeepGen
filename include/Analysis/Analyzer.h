#ifndef _Analyzer_h_
#define _Analyzer_h_

#include "Common/Utils.h"
#include <vector>

namespace KernelCodeGen {

struct CompareLoop {
  int operator()(const mlir::affine::AffineForOp& x, const mlir::affine::AffineForOp& y) const {
    mlir::Operation* x_ptr = x;
    mlir::Operation* y_ptr = y;
    auto x_hashCode = reinterpret_cast<size_t>(x_ptr);
    auto y_hashCode = reinterpret_cast<size_t>(y_ptr);
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};

struct CompareFunc {
  int operator()(const mlir::func::FuncOp& x, const mlir::func::FuncOp& y) const {
    mlir::Operation* x_ptr = x;
    mlir::Operation* y_ptr = y;
    auto x_hashCode = reinterpret_cast<size_t>(x_ptr);
    auto y_hashCode = reinterpret_cast<size_t>(y_ptr);
    if (x_hashCode >= y_hashCode) return 0;
    else return 1;
  }
};

struct ValueCompare {
  bool operator()(const mlir::Value& val0, const mlir::Value& val1) const {
    return val0.getAsOpaquePointer() < val1.getAsOpaquePointer();
  }
};

namespace Analyzer {
  std::vector<int64_t> getParallelNumber(mlir::affine::AffineParallelOp parallelLevel, int64_t& totalNumber);
  int64_t getThreadPerBlock(mlir::affine::AffineParallelOp parallelLevel);
  std::vector<mlir::affine::AffineForOp> collectFuncLoops(mlir::func::FuncOp funcOp);
  std::map<std::string, std::string> collectNameTypeMap(mlir::ModuleOp& module);
  std::set<std::string> collectFuncTypes(mlir::ModuleOp& module);
  int getThreadsPerCTA(mlir::ModuleOp module); 

  std::vector<mlir::Value> getParallelIdx(mlir::affine::AffineParallelOp parallelLevel);
  mlir::affine::AffineForOp findRootLoop(mlir::Operation* op);
  mlir::Block* getClostScopeOp(mlir::Operation* op);
}

}  // KernelCodeGen

#endif  // _Analyzer_h_