#ifndef _Optimizer_h_
#define _Optimizer_h_

#include "Analysis/Analyzer.h"
#include "Conversion/General/Rewriter.h"
#include "Common/Utils.h"

#include <unordered_map>

namespace KernelCodeGen {

struct Optimizer {
  virtual bool applicable(mlir::ModuleOp& module) = 0;
  virtual void applyOptimzer(mlir::ModuleOp& module, std::map<std::string, int> config) = 0;

  bool operator==(const Optimizer& other) {
    return name == other.name;
  }
  std::string name;
};

struct MatmulConfigUtils{
  explicit MatmulConfigUtils(const std::map<std::string, int>& config);
  int BM ;
  int BN ;
  int BK ;
  int TM ;
  int TN ;
  int BLOCK_Y;  // blockDimY
  int BLOCK_X;  // blockDimX
  int THREAD_NUM;  // block 内的线程数
  int GLOB_LOAD_ROW_WIDTH_A;
  int GLOB_LOAD_ROW_WIDTH_B;
  int BLOCK_REPEAT_A;  // 离散化时，warp处理区域的分散次数
  int WARP_REPEAT_A;  // 离散化时，thread处理区域的分散次数
  int BLOCK_REPEAT_B;
  int WARP_REPEAT_B;
  int GLOB_STORE_ROW_WIDTH;
  int GLOB_STORE_COL_THREAD_NUM;
  int GLOB_STORE_ROW_THREAD_NUM;
  int GLOB_LOAD_COL_THREAD_NUM_A;
  int GLOB_LOAD_COL_THREAD_NUM_B;
  int GLOB_LOAD_ROW_THREAD_NUM_A;
  int GLOB_LOAD_ROW_THREAD_NUM_B;
};

struct MatmulOptimizer : Optimizer {

  MatmulOptimizer() {
    this->name = std::move(std::string("Matmul"));
  }

  virtual bool applicable(mlir::ModuleOp& module) override;
  virtual void applyOptimzer(mlir::ModuleOp& module, std::map<std::string, int> config) override;

  // mlir::AffineMap getAffineMap(const std::string& mapIdentifier, mlir::OpBuilder& builder, std::map<std::string, int> config);
  mlir::AffineMap getGlobToTempMapA(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getGlobToTempMapB(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getTempToSharedMapSMA(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getTempToSharedMapSMB(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getSharedToRegMapSMA(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getSharedToRegMapSMB(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getCalculateMapReg(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getRegToGlobMapC(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getRegToSharedMapSMC(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getReduceMapSMC(mlir::OpBuilder& builder, const std::map<std::string, int>& config);
  mlir::AffineMap getReduceMapRegC(mlir::OpBuilder& builder, const std::map<std::string, int>& config);

  void clear() {
    matmuls.clear();
    matmulLoops.clear();
    matmulBuffers.clear();
  }

  // using the outermost loop represent a matmul.
  std::set<mlir::func::FuncOp, CompareFunc> matmuls;

  // Map: from outermost loop to all loops in the matmul(loopM->[loopM, loopN, loopK]).
  std::map<mlir::func::FuncOp, std::vector<mlir::affine::AffineForOp>, CompareFunc> matmulLoops;


  // Memory: A, B, C
  struct MemoryBuffer {
    mlir::Value A;
    mlir::Value B;
    mlir::Value C;
  };

  // loopM->[A, B, C]
  std::map<mlir::func::FuncOp, MemoryBuffer, CompareFunc> matmulBuffers;
  
};



}  // KernelCodeGen

#endif // _Optimizer_h_