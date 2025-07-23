#ifndef _Optimizer_h_
#define _Optimizer_h_

#include "Analysis/Analyzer.h"
#include "Conversion/General/Rewriter.h"
#include "Common/Utils.h"

#include <unordered_map>

namespace KernelCodeGen {

struct Optimizer {
  virtual bool applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) = 0;
  virtual void applyOptimzer(mlir::func::FuncOp& funcOp) = 0;

  bool operator==(const Optimizer& other) {
    return name == other.name;
  }
  llvm::SmallVector<mlir::AffineExpr> getExprs(mlir::OpBuilder builder, int dimCount) {
    llvm::SmallVector<mlir::AffineExpr> dims;
    for (int i=0; i<dimCount; i++) {
      dims.push_back(builder.getAffineDimExpr(i));
    }
    return dims;
  }
  std::string name;
};

// ======================================== matmul ==================================================
struct  MatmulOptimizer : Optimizer {
  MatmulOptimizer() {
    this->name = std::move(std::string("Matmul"));
  }
  virtual bool applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) override;
  virtual void applyOptimzer(mlir::func::FuncOp& funcOp) override;

  // affine map
  std::array<int64_t, 6> getCfgDatas(const std::string& bufType);
  std::array<mlir::AffineExpr, 2> getGlobToSmExprs(const llvm::SmallVector<mlir::AffineExpr>& dims, const std::array<int64_t, 6>& args);
  mlir::AffineMap getGlobToTempMap(mlir::OpBuilder& builder, const std::string& bufType);
  mlir::AffineMap getTempToSmMap(mlir::OpBuilder& builder, const std::string& bufType);
  
  mlir::AffineMap getSmToRegMap(mlir::OpBuilder& builder, const std::string& bufType);
  mlir::AffineMap getCalculateMap(mlir::OpBuilder& builder);

  std::array<mlir::AffineExpr, 2> getRegCStoreExprs(const llvm::SmallVector<mlir::AffineExpr>& dims);
  mlir::AffineMap getRegCToGlobMap(mlir::OpBuilder& builder);
  mlir::AffineMap getRegCToSmMap(mlir::OpBuilder& builder);

  std::array<mlir::AffineExpr, 2> getReduceExprs(const llvm::SmallVector<mlir::AffineExpr>& dims);
  mlir::AffineMap getReduceSmCToRegMap(mlir::OpBuilder& builder);
  mlir::AffineMap getReduceRegCToGlobMap(mlir::OpBuilder& builder);
  // number vars
  std::map<std::string, int64_t> cfg;
  mlir::affine::AffineForOp yTileForOp, xTileForOp, kForOp;
  mlir::affine::AffineParallelOp blockIdx, threadIdx;
  mlir::Value byIdx, bxIdx, A, B, C;

  // function base data
  std::vector<int64_t> batchs;
  int64_t M, N, K;
  mlir::MemRefType typeA, typeB, typeC;
  bool isTranB, isTranA;

  // config args
  int64_t gridShapeX, gridShapeY;
  int64_t threadNum, blockRepeatM, blockRepeatN, warpRepeatM, warpRepeatN;
  int64_t globLoadRowWidthA, globLoadRowWidthB, globStoreRowWidth;  // 不连续load/stroe
  int64_t globLoadAllWidthA, globLoadAllWidthB, globStoreAllWidth;  // 连续load/store
  int64_t globLoadTotalWidthA, globLoadTotalWidthB, globStoreTotalWidth;  // 每个线程load/store元素个数

  // other func
  void computeTuneArgs();
  void parseFuncArgs(mlir::func::FuncOp funcOp);
  std::array<mlir::Value, 6> createBasicBuffers();
  std::array<mlir::Value, 2> createSplitUBuffers();
};


// ====================================== flash attention ================================
struct FlashAttnOptimizer : Optimizer {
  FlashAttnOptimizer() {
    this->name = std::move(std::string("FlashAttn"));
  }
  virtual bool applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) override;
  virtual void applyOptimzer(mlir::func::FuncOp& funcOp) override;

  // affine map
  std::array<int64_t, 7> getCfgDatas(const std::string& bufType);
  std::array<mlir::AffineExpr, 2> getGlobToSmExprs(const llvm::SmallVector<mlir::AffineExpr>& dims, 
                                                                    const std::array<int64_t, 7>& args);
  mlir::AffineMap getGlobQKToTempQKMap(mlir::OpBuilder& builder, const std::string& bufType);
  mlir::AffineMap getGlobVToTempVMap(mlir::OpBuilder& builder);
  mlir::AffineMap getTempToSmMap(mlir::OpBuilder& builder, const std::string& bufType);
  
  std::array<int64_t, 8> getSmCfgDatas(const std::string& bufType);
  mlir::AffineMap getSmQKVToRegQKVMap(mlir::OpBuilder& builder, const std::string& bufType);
  mlir::AffineMap getSmPToRegPMap(mlir::OpBuilder& builder);

  mlir::AffineMap getSmToRegUpdateTtileOMap(mlir::OpBuilder& builder);
  mlir::AffineMap getRegUpdateTtileOMap(mlir::OpBuilder& builder);

  mlir::AffineMap getCalculateMap(mlir::OpBuilder& builder, std::string calculatetype);

  mlir::AffineMap getRegSumAndMaxMap(mlir::OpBuilder& builder);
  mlir::AffineMap getTilePToSmPMap(mlir::OpBuilder& builder);
  mlir::AffineMap getTileOToGlobOMap(mlir::OpBuilder& builder);

  mlir::AffineMap getBlockLevelSmMap(mlir::OpBuilder& builder);
  mlir::AffineMap getBlockLevelRegMap(mlir::OpBuilder& builder);
  
  // number vars
  std::map<std::string, int64_t> cfg;
  mlir::Value midBuf, sumBuf, maxBuf;
  mlir::affine::AffineForOp initBufFor;
  std::vector<mlir::affine::AffineForOp> xBlockFors, xTileForOps, yTileForOps, kForOps;
  mlir::affine::AffineParallelOp blockIdx, threadIdx;
  mlir::Value byIdx, Q, K, V, O;

  // function base data
  mlir::MemRefType typeQ, typeK, typeV, typeO, typeMid;
  int64_t batchSize, headNum, seqLen, headDim;
  bool isTranQ, isTranK, isTranV;

  // config args
  int64_t threadNum, blockPY, blockPX, blockOY, blockOX;
  int64_t blockRepeatQ, blockRepeatK, warpRepeatQ, warpRepeatK;
  int64_t blockRepeatP, blockRepeatV, warpRepeatP, warpRepeatV;
  int64_t globLoadTotalWidthQ, globLoadTotalWidthK, globLoadTotalWidthV;
  int64_t globLoadRowWidthQ, globLoadRowWidthK, globLoadRowWidthV;
  int64_t globLoadAllWidthQ, globLoadAllWidthK, globLoadAllWidthV;

  // other func
  void computeTuneArgs();
  void parseFuncArgs(mlir::func::FuncOp funcOp);
  std::pair<std::array<mlir::Value, 5>, std::array<mlir::Value, 7>> createBasicBuffers();

  // 1111
  std::vector<mlir::affine::AffineForOp> reduceAndBraodcast(mlir::Operation* localtionOp,
                                                     const std::vector<mlir::Value>& regBufs,
                                                     const std::vector<mlir::Value>& smBufs);
  
  void updateTileO(mlir::Operation* localtionOp, 
                   mlir::Value smFactor, 
                   mlir::Value tileO, 
                   std::string bufDesc);

  void moveMemrefDefineAhead(mlir::Operation* threadParallelOp);
};

}  // KernelCodeGen

#endif // _Optimizer_h_