#include "Conversion/Optimize.h"

namespace KernelCodeGen {


// ======================================= global to sm =========================================
std::array<int64_t, 6> MatmulOptimizer::getCfgDatas(const std::string& bufType) {
  // 有些属性相同，但是表示AB不同矩阵的config数据将其统一返回

  int64_t blockTileY = cfg.at("BLOCK_SIZE_M"), blockTileX = cfg.at("BLOCK_SIZE_K");
  int64_t isTran = this->isTranA, globLoadWidth = cfg.at("GLOB_LOAD_WIDTH_A");
  int64_t globLoadAllWidth = globLoadAllWidthA;
  int64_t globLoadRowWidth = globLoadRowWidthA;
  if (isTran) {
    blockTileY = cfg.at("BLOCK_SIZE_K"); blockTileX = cfg.at("BLOCK_SIZE_M");
  }
  if (bufType == "B") {
    blockTileY = cfg.at("BLOCK_SIZE_K"); blockTileX = cfg.at("BLOCK_SIZE_N");
    isTran = this->isTranB; globLoadWidth = cfg.at("GLOB_LOAD_WIDTH_B");
    globLoadAllWidth = globLoadAllWidthB;
    globLoadRowWidth = globLoadRowWidthB;
    if (isTran) {
      blockTileY = cfg.at("BLOCK_SIZE_N"); blockTileX = cfg.at("BLOCK_SIZE_K");
    }
  }
  return {blockTileY, blockTileX, isTran, globLoadWidth, globLoadAllWidth, globLoadRowWidth};
}

std::array<mlir::AffineExpr, 2> MatmulOptimizer::getGlobToSmExprs(const llvm::SmallVector<mlir::AffineExpr>& dims, 
                                                                  const std::array<int64_t, 6>& args) {
  // 因为glob到temp和temp到sm到map有一部分相同，所以可以共用一个函数
  // dims = {tid, iter}
  mlir::AffineExpr tyIdx, txIdx;
  auto [blockTileY, blockTileX, isTran, globLoadWidth, globLoadAllWidth, globLoadRowWidth] = args;
  // thread level
  if (cfg.at("LOAD_CONTINUOUS")) {
    auto tidIndex = (dims[1] * globLoadAllWidth) + (dims[0] * globLoadWidth);
    auto tIdx = tools::mapUtils::reshapeThreadBlock(tidIndex, {blockTileY, blockTileX});
    tyIdx = tIdx[0]; txIdx = tIdx[1];  // (iter * globLoadAllWidth + tid * globLoadWidth) / btx
  } else {
    auto threadRowNum = this->threadNum / blockTileY; // threadNum / blockSizeM  | threadNum / blockSizeK
    auto tIdx = tools::mapUtils::reshapeThreadBlock(dims[0], {blockTileY, threadRowNum});
    tyIdx = tIdx[0];  // tid / threadRowNum
    txIdx = dims[1] * globLoadRowWidth + tIdx[1] * globLoadWidth;  // iter * globLoadRowWidth + (tid % threadRowNum) * globLoadWidth
  }
  return {tyIdx, txIdx};
}

mlir::AffineMap MatmulOptimizer::getGlobToTempMap(mlir::OpBuilder& builder, const std::string& bufType) {
  // glob load data to temp reg
  int dimCount = 5 + this->batchs.size();
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by, bx, tid, k, iter}
  auto args = getCfgDatas(bufType);
  // block level
  mlir::AffineExpr row, col;
  if (bufType == "A") {
    row = dims[dimCount-5];                     // blockIdx.y
    col = dims[dimCount-2];                     // k * BK
    if (args[2]) {
      row = dims[dimCount-2];                   // k * BK
      col = dims[dimCount-5];                   // blockIdx.y
    }
  } else {
    row = dims[dimCount-2];
    col = dims[dimCount-4];  
    if (args[2]) {
      row = dims[dimCount-4];
      col = dims[dimCount-2];
    }
  }
  // thread level
  auto [tyIdx, txIdx] = getGlobToSmExprs({dims[dimCount-3], dims[dimCount-1]}, args);
  // create exprs
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<this->batchs.size(); i++) {
    exprs.push_back(dims[i]);  // batch
  }
  exprs.push_back(row + tyIdx);
  exprs.push_back(col + txIdx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap MatmulOptimizer::getTempToSmMap(mlir::OpBuilder& builder, const std::string& bufType) {
  // 获取从temp reg 到sm的map
  int dimCount = 2;
  auto dims = getExprs(builder, dimCount); // {tid, iter}
  // init datas
  auto args = getCfgDatas(bufType);
  auto [tyIdx, txIdx] = getGlobToSmExprs(dims, args);
  if ((bufType == "A" && !args[2]) || (bufType == "B" && args[2])) {
    auto temp = tyIdx;
    tyIdx = txIdx + builder.getAffineDimExpr(dimCount);
    txIdx = temp;
    dimCount++;
  }
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(tyIdx);
  exprs.push_back(txIdx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

// ================================== sm to reg and calculate ====================================
mlir::AffineMap MatmulOptimizer::getSmToRegMap(mlir::OpBuilder& builder, const std::string& bufType) {
  // sm to reg
  int dimCount = 4;
  auto dims = getExprs(builder, dimCount);  // {tid, bk, blockRepIter, warpRepIter}
  auto tids = tools::mapUtils::reshapeThreadBlock(dims[0], {cfg["LOCAL_SPLIT_U"], this->threadNum/cfg["LOCAL_SPLIT_U"]});
  auto widx = tools::mapUtils::wapr_y(tids[1], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_X"]);
  auto lidx = tools::mapUtils::lane_y(tids[1], cfg["WARP_SIZE"], cfg.at("WARP_LAYOUT_X"));
  int64_t blockLayout = cfg["BLOCK_LAYOUT_Y"], warpLayout = cfg["WARP_LAYOUT_Y"];
  int64_t blockScatter = cfg["BLOCK_SCATTER_WIDTH_M"], warpScatter = cfg["WARP_SCATTER_WIDTH_M"];
  if (bufType == "B" ) {
    widx = tools::mapUtils::wapr_x(tids[1], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_X"]);
    lidx = tools::mapUtils::lane_x(tids[1], cfg["WARP_SIZE"], cfg.at("WARP_LAYOUT_X"));
    blockLayout = cfg["BLOCK_LAYOUT_X"], warpLayout = cfg["WARP_LAYOUT_X"];
    blockScatter = cfg["BLOCK_SCATTER_WIDTH_N"], warpScatter = cfg["WARP_SCATTER_WIDTH_N"];
  }
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(dims[1] + tids[0]);
  exprs.push_back((dims[2] * blockLayout + widx) * warpLayout * blockScatter + (dims[3] * warpLayout + lidx) * warpScatter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getCalculateMap(mlir::OpBuilder& builder) {
  // reg 内积
  auto iter = builder.getAffineDimExpr(0);
  llvm::SmallVector<mlir::AffineExpr> exprs{iter};
  return mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

// ======================================= store regc ============================================
std::array<mlir::AffineExpr, 2> MatmulOptimizer::getRegCStoreExprs(const llvm::SmallVector<mlir::AffineExpr>& dims) {
  // {tid, blockRepIterA, blockRepIterB, warpRepIterA, warpRepIterB, iterA, iterB}
  auto warp_y = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_X"]);
  auto warp_x = tools::mapUtils::wapr_x(dims[0], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_X"]);
  auto lane_y = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_X"]);
  auto lane_x = tools::mapUtils::lane_x(dims[0], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_X"]);
  // create exprs
  auto ty = (dims[1] * cfg["BLOCK_LAYOUT_Y"] + warp_y * cfg["BLOCK_SCATTER_WIDTH_M"]) * cfg["WARP_LAYOUT_Y"] + 
             dims[3] * cfg["WARP_LAYOUT_Y"] + lane_y * cfg["WARP_SCATTER_WIDTH_M"] + dims[5];
  auto tx = (dims[2] * cfg["BLOCK_LAYOUT_X"] + warp_x * cfg["BLOCK_SCATTER_WIDTH_N"]) * cfg["WARP_LAYOUT_X"] + 
             dims[4] * cfg["WARP_LAYOUT_X"] + lane_x * cfg["WARP_SCATTER_WIDTH_N"] + dims[6];
  return {ty, tx};
}

mlir::AffineMap MatmulOptimizer::getRegCToGlobMap(mlir::OpBuilder& builder) {
  // regc to globc
  int dimCount = 9 + this->batchs.size();
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by, bx, tid, blockRepIterA, blockRepIterB, warpRepIterA, warpRepIterB, iterA, iterB}
  llvm::SmallVector<mlir::AffineExpr> exprs, locDims(dims.end()-7, dims.end());
  auto [ty, tx] = getRegCStoreExprs(locDims);      
  // create expr
  for (int i=0; i<this->batchs.size(); i++) {
    exprs.push_back(dims[i]);
  }
  exprs.push_back(dims[dimCount-9] + ty);  // blockIdx.y
  exprs.push_back(dims[dimCount-8] + tx);  // blockIdx.x
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getRegCToSmMap(mlir::OpBuilder& builder) {
  // regc to sm
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);  // {tid, blockRepIterA, blockRepIterB, warpRepIterA, warpRepIterB, iterA, iterB}
  llvm::SmallVector<mlir::AffineExpr> exprs, locDims(dims);
  auto tids = tools::mapUtils::reshapeThreadBlock(dims[0], {cfg["LOCAL_SPLIT_U"], this->threadNum/cfg["LOCAL_SPLIT_U"]});
  locDims[0] = tids[1];  // tid == tid%layer_thread_num
  auto [ty, tx] = getRegCStoreExprs(locDims);
  // create exprs
  exprs.push_back(tids[0]);   // 区分shaed
  exprs.push_back(ty);
  exprs.push_back(tx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

// =========================================== splitu =============================================
std::array<mlir::AffineExpr, 2> MatmulOptimizer::getReduceExprs(const llvm::SmallVector<mlir::AffineExpr>& dims) {
  // {tid, iter}
  mlir::AffineExpr ty, tx;
  if (cfg["STORE_CONTINUOUS"]) {
    auto tidIndex = dims[1] * this->globStoreAllWidth + dims[0] * cfg["GLOB_STORE_WIDTH"];
    auto tIdx = tools::mapUtils::reshapeThreadBlock(tidIndex, {cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]});
    ty = tIdx[0]; tx = tIdx[1];
  } else {
    auto threadRowNum = this->threadNum / cfg["BLOCK_SIZE_M"];
    auto tIdx = tools::mapUtils::reshapeThreadBlock(dims[0], {cfg["BLOCK_SIZE_M"], threadRowNum});
    ty = tIdx[0]; tx = dims[2] * this->globStoreRowWidth + tIdx[1] * cfg["GLOB_STORE_WIDTH"];
  }
  return {ty, tx};
}

mlir::AffineMap MatmulOptimizer::getReduceSmCToRegMap(mlir::OpBuilder& builder) {
  // smC to reg to reduce
  int dimCount = 3;
  auto dims = getExprs(builder, dimCount); // {tid, iterSplitU, iter}
  llvm::SmallVector<mlir::AffineExpr> exprs, locDims{dims[0], dims[2]};
  auto [ty, tx] = getReduceExprs(locDims);
  exprs.push_back(dims[1]);
  exprs.push_back(ty);
  exprs.push_back(tx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getReduceRegCToGlobMap(mlir::OpBuilder& builder) {
  int dimCount = 4 + this->batchs.size();
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by, bx, tid, iter}
  llvm::SmallVector<mlir::AffineExpr> exprs, locDims{dims[dimCount-2], dims[dimCount-1]};
  auto [ty, tx] = getReduceExprs(locDims);
  // create expr
  for (int i=0; i<this->batchs.size(); i++) {
    exprs.push_back(dims[i]);
  }
  exprs.push_back(dims[dimCount-4] + ty);  // blockIdx.y * BM
  exprs.push_back(dims[dimCount-3] + tx);  // blockIdx.x * BN
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

// ===================================== 解析和计算必要的信息 ========================================

void MatmulOptimizer::computeTuneArgs() {
  // gird的布局
  this->gridShapeY = this->M / cfg.at("BLOCK_SIZE_M");
  this->gridShapeX = this->N / cfg.at("BLOCK_SIZE_N");
  // 线程的个数
  this->threadNum = (cfg.at("BLOCK_SIZE_M") * cfg.at("BLOCK_SIZE_N") / cfg.at("THREAD_SIZE_M") / cfg.at("THREAD_SIZE_N")) * cfg.at("LOCAL_SPLIT_U");
  // 离散化重复的次数
  this->blockRepeatM = cfg.at("THREAD_SIZE_M") / cfg.at("BLOCK_SCATTER_WIDTH_M");
  this->blockRepeatN = cfg.at("THREAD_SIZE_N") / cfg.at("BLOCK_SCATTER_WIDTH_N");
  this->warpRepeatM = cfg.at("BLOCK_SCATTER_WIDTH_M") / cfg.at("WARP_SCATTER_WIDTH_M");
  this->warpRepeatN = cfg.at("BLOCK_SCATTER_WIDTH_N") / cfg.at("WARP_SCATTER_WIDTH_N");
  // 每个线程需要从glob加载的数据总量
  this->globLoadTotalWidthA = cfg.at("BLOCK_SIZE_M") * cfg.at("BLOCK_SIZE_K") / this->threadNum;
  this->globLoadTotalWidthB = cfg.at("BLOCK_SIZE_N") * cfg.at("BLOCK_SIZE_K") / this->threadNum;
  // glob非连续load时（未转置），block行加载的数据量（线程数量必须大于除数）
  this->globLoadRowWidthA = this->threadNum / cfg.at("BLOCK_SIZE_M") * cfg.at("GLOB_LOAD_WIDTH_A");
  this->globLoadRowWidthB = this->threadNum / cfg.at("BLOCK_SIZE_K") * cfg.at("GLOB_LOAD_WIDTH_B");
  if (isTranA) {
    this->globLoadRowWidthA = this->threadNum / cfg.at("BLOCK_SIZE_K") * cfg.at("GLOB_LOAD_WIDTH_A");
  }
  if (isTranB) {
    this->globLoadRowWidthB = this->threadNum / cfg.at("BLOCK_SIZE_N") * cfg.at("GLOB_LOAD_WIDTH_B");
  }
  // glob连续load时，一个block加载的数据总量
  this->globLoadAllWidthA = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_A");
  this->globLoadAllWidthB = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_B");
  // 含有splitu参数时，store的方式（连续/非连续）
  if (cfg.at("LOCAL_SPLIT_U") > 1) {
    this->globStoreTotalWidth = cfg.at("BLOCK_SIZE_M") * cfg.at("BLOCK_SIZE_N") / this->threadNum;
    this->globStoreRowWidth = this->threadNum / cfg.at("BLOCK_SIZE_M") * cfg.at("GLOB_STORE_WIDTH");
    this->globStoreAllWidth = this->threadNum * cfg.at("GLOB_STORE_WIDTH");
  }
}

void MatmulOptimizer::parseFuncArgs(mlir::func::FuncOp funcOp) {
  // 解析kernel函数的参数基本信息
  typeA = mlir::dyn_cast<mlir::MemRefType>(A.getType());
  typeB = mlir::dyn_cast<mlir::MemRefType>(B.getType());
  typeC = mlir::dyn_cast<mlir::MemRefType>(C.getType());
  // get transpose args
  std::vector<bool> isTrans;
  auto transArr = funcOp->getAttr(ARGTRAN);
  auto transArrAttr = mlir::dyn_cast<mlir::ArrayAttr>(transArr);
  for (auto tran : transArrAttr) {
    auto tranAttr = mlir::dyn_cast<mlir::IntegerAttr>(tran);
    isTrans.push_back(tranAttr.getInt());
  }
  isTranA = isTrans[0]; isTranB = isTrans[1];
  // get mnk/batch
  auto shapeC = typeC.getShape();
  auto shapeA = typeA.getShape();
  M = shapeC[shapeC.size()-2]; N = shapeC[shapeC.size()-1];
  K = shapeA[shapeA.size()-1];
  if (isTranA) {
    K = shapeA[shapeA.size()-2];
  }
  for (int i=0; i<shapeC.size()-2; i++) {
    batchs.push_back(shapeC[i]);
  }
}

bool MatmulOptimizer::applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) {
  // 应用优化器，收集需要的必要信息
  cfg = config;
  mlir::ValueRange operands = funcOp.getArguments();
  A = operands[0]; B = operands[1]; C = operands[2];
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (auto parallelOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(op)) {
      auto gpuIdx = getStrAttr(parallelOp, AttrGPUIndex);
      if (gpuIdx == BLOCKIDX) {
        this->blockIdx = parallelOp;
      } else if (gpuIdx == THREADIDX) {
        this->threadIdx = parallelOp;
      }
    } else if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      auto forDesc = getStrAttr(forOp, FORDESC);
      if (forDesc == "ttiley") {
        this->yTileForOp = forOp;
      } else if (forDesc == "ttilex") {
        this->xTileForOp = forOp;
      } else if (forDesc == "k") {
        this->kForOp = forOp;
      }
    } else if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
      auto applyDesc = getStrAttr(applyOp, APPLYDESC);
      if (applyDesc == "blocky") {
        this->byIdx = applyOp.getResult();
      } else if (applyDesc == "blockx") {
        this->bxIdx = applyOp.getResult();
      }
    }
  });
  parseFuncArgs(funcOp);  // parseFuncArgs
  computeTuneArgs();  // compute config args
  return true;
}

// ====================================== create buffer ===============================================
std::array<mlir::Value, 6> MatmulOptimizer::createBasicBuffers() {
  // create all buffers
  auto dtypeA = typeA.getElementType();
  auto dtypeB = typeB.getElementType();
  // create registers buffers
  std::vector<std::vector<int64_t>> regShapes{
    {cfg.at("THREAD_SIZE_M")}, {cfg.at("THREAD_SIZE_N")}, {globLoadTotalWidthA}, {globLoadTotalWidthB}
  };
  std::vector<mlir::Type> regDTypes{dtypeA, dtypeB, dtypeA, dtypeB};
  std::vector<std::string> regDescs{"regA", "regB", "tempA", "tempB"};
  auto reg = Rewriter::allocBuffers(regShapes, regDTypes, MemorySpace::local, regDescs, this->threadIdx);
  // create shared memory buffers
  std::vector<std::vector<int64_t>> smShapes{
    {cfg.at("BLOCK_SIZE_K"), cfg.at("BLOCK_SIZE_M")}, {cfg.at("BLOCK_SIZE_K"), cfg.at("BLOCK_SIZE_N")}
  };
  auto sm = Rewriter::allocBuffers(smShapes, {dtypeA, dtypeB}, MemorySpace::shared, {"smA", "smB"}, blockIdx);
  return {sm[0], sm[1], reg[0], reg[1], reg[2], reg[3]};
}

std::array<mlir::Value, 2> MatmulOptimizer::createSplitUBuffers() {
  // create splitU buffer
  auto dtypeC = typeC.getElementType();
  std::vector<std::vector<int64_t>> regShapes{{cfg.at("THREAD_SIZE_M") * cfg.at("THREAD_SIZE_N")}};
  std::vector<std::vector<int64_t>> smShapes{{cfg.at("LOCAL_SPLIT_U"), cfg.at("BLOCK_SIZE_M"), cfg.at("BLOCK_SIZE_N")}};
  auto reg = Rewriter::allocBuffers(regShapes, {dtypeC}, MemorySpace::local, {"regC"}, this->threadIdx);
  auto sm = Rewriter::allocBuffers(smShapes, {dtypeC}, MemorySpace::shared, {"smC"}, blockIdx);
  return {sm[0], reg[0]};
}

// ========================================= applyOptimzer ===============================================
void MatmulOptimizer::applyOptimzer(mlir::func::FuncOp& funcOp) {
  mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(funcOp->getParentOp());
  mlir::OpBuilder builder(module);
  // bufferize + splitK + reorderK
  std::vector<mlir::affine::AffineForOp> tileCLoops{yTileForOp, xTileForOp};
  auto regC = Rewriter::bufferizeLoopCarryVar(kForOp, tileCLoops, MemorySpace::local, {"regC"});
  LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

  auto k_axes = Rewriter::split(kForOp, {cfg.at("LOCAL_SPLIT_U"), cfg.at("BLOCK_SIZE_K")});
  auto k_outer = k_axes[0], k_mider = k_axes[1], k_inner = k_axes[2];
  LOG_DEBUG("===== after split =======\n",module);

  std::vector<mlir::affine::AffineParallelOp> blockLevel{this->threadIdx};
  Rewriter::addLoopsToParallel({k_inner}, blockLevel, true);
  this->threadIdx = blockLevel[0];
  LOG_DEBUG("===== after addLoopsToParallel =======\n", module);

  Rewriter::reorder({k_outer, k_mider, yTileForOp, xTileForOp});
  LOG_DEBUG("===== after reorder =======\n",module);
  // buffer
  auto [smA, smB, regA, regB, tempA, tempB] = createBasicBuffers();
  LOG_DEBUG("===== after alloc_buffer =======\n",module);

  // block mapping
  if (cfg["BLOCK_MAPPING"] > 0) {
    std::vector<int64_t> gridShape{this->gridShapeY, this->gridShapeX};
    std::vector<int64_t> blockTiles{cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]};
    auto result =  Rewriter::blockMapping(blockIdx, blockTiles, gridShape, cfg["BLOCK_MAPPING"]);
    this->byIdx = result[0]; this->bxIdx = result[1];
    LOG_DEBUG("===== after blockMapping =======\n",module);
  }
  // ====================================== load and store =======================================
  // splitu and fuse forop into parallelop
  auto bIdx = Analyzer::getParallelIdx(blockIdx);
  auto tIdx = Analyzer::getParallelIdx(this->threadIdx);
  // {b1, b2, by, bx, tid}
  llvm::SmallVector<mlir::Value> operands(bIdx.begin(), bIdx.end()-1);
  operands.push_back(byIdx); operands.push_back(bxIdx); operands.push_back(tIdx[0]);

  // glob load to temp reg
  auto loadTileAMap = getGlobToTempMap(builder, "A");
  auto loadTileBMap = getGlobToTempMap(builder, "B");   // {b1, b2, by, bx, tid}
  llvm::SmallVector<mlir::Value> gttoperands(operands);
  gttoperands.push_back(k_outer.getInductionVar());
  auto loadTileA = Rewriter::loadToRegisters(A, tempA, loadTileAMap, gttoperands, {cfg["GLOB_LOAD_WIDTH_A"]}, k_outer, Position::begin, "globToRegA");
  auto loadTileB = Rewriter::loadToRegisters(B, tempB, loadTileBMap, gttoperands, {cfg["GLOB_LOAD_WIDTH_B"]}, loadTileA, Position::after, "globToRegB");
  LOG_DEBUG("===== after read A/B =======\n",module);
  // temp reg load to sm
  auto storeTileAMap = getTempToSmMap(builder, "A");
  auto storeTileBMap = getTempToSmMap(builder, "B");   // {tid, iter}
  auto storeTileA = Rewriter::loadFromRegisters(tempA, smA, storeTileAMap, {tIdx[0]}, {cfg["GLOB_LOAD_WIDTH_A"]}, loadTileB, Position::after, "tempToSmA");
  auto storeTileB = Rewriter::loadFromRegisters(tempB, smB, storeTileBMap, {tIdx[0]}, {cfg["GLOB_LOAD_WIDTH_B"]}, storeTileA, Position::after, "tempToSmB");
  auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
  auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);
  LOG_DEBUG("===== write A/B =======\n",module);
  // sm load to cal reg
  auto loadFragAMap = getSmToRegMap(builder, "A");
  auto loadFragBMap = getSmToRegMap(builder, "B");    // {tid, bk, blockRepIter, warpRepIter}
  llvm::SmallVector<mlir::Value> stroperands{tIdx[0], k_mider.getInductionVar()};  
  std::vector<int64_t> widthsA{cfg["BLOCK_SCATTER_WIDTH_M"], cfg["WARP_SCATTER_WIDTH_M"]};
  std::vector<int64_t> widthsB{cfg["BLOCK_SCATTER_WIDTH_N"], cfg["WARP_SCATTER_WIDTH_N"]};
  auto loadFragA = Rewriter::loadToRegisters(smA, regA, loadFragAMap, stroperands, widthsA, k_mider, Position::begin, "smToRegA");
  auto loadFragB = Rewriter::loadToRegisters(smB, regB, loadFragBMap, stroperands, widthsB, loadFragA, Position::after, "smToRegB");
  LOG_DEBUG("===== read sh_A/B =======\n",module);
  // Calculate 
  auto calMap = getCalculateMap(builder);  // {iter}
  Rewriter::cache_read(xTileForOp, A, regA, calMap, {yTileForOp.getInductionVar()});
  Rewriter::cache_read(xTileForOp, B, regB, calMap, {xTileForOp.getInductionVar()});
  LOG_DEBUG("===== load regA & cache_read =======\n",module);
  // ==============================================================================================
  // split store c for
  auto writeCbody = Rewriter::get_write(this->threadIdx, C);
  assert(writeCbody.size() == 1);
  auto m_inner_axes = Rewriter::split(writeCbody[0][0], widthsA);
  auto n_inner_axes = Rewriter::split(writeCbody[0][1], widthsB);
  auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1], m_inner_2 = m_inner_axes[2];
  auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1], n_inner_2 = n_inner_axes[2];
  Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1, m_inner_2, n_inner_2});
  LOG_DEBUG("===== load split & reorder regC to C =======\n",module);

  mlir::Value smC, regC_;
  if (cfg["LOCAL_SPLIT_U"] > 1) {
    auto LSUBarrier = Rewriter::barrier(m_inner_0, Position::before);
    auto buffer = createSplitUBuffers();
    smC = buffer[0]; regC_ = buffer[1];
    // storeOp globc to regc
    auto regCToSmMap = getRegCToSmMap(builder);  
    llvm::SmallVector<mlir::Value> rtsOperands{tIdx[0]}; 
    for (int i=0; i<3; i++) {
      rtsOperands.push_back(m_inner_axes[i].getInductionVar());
      rtsOperands.push_back(n_inner_axes[i].getInductionVar());
    }
    Rewriter::cache_write(m_inner_0, C, smC, regCToSmMap, rtsOperands);
    LOG_DEBUG("===== load cache_write regC to C =======\n",module);
    // smC to reg to reduce
    auto smCToRegMap = getReduceSmCToRegMap(builder);
    auto [rLoop0, rLoop1] = Rewriter::splitUReduce(smC, regC_, smCToRegMap, {tIdx[0]}, cfg["LOCAL_SPLIT_U"], cfg["GLOB_STORE_WIDTH"], m_inner_0, Position::after);
    LOG_DEBUG("===== load splitUReduce =======\n",module);

    auto regCToGlobMap = getReduceRegCToGlobMap(builder);
    llvm::SmallVector<mlir::Value> rtgOperands(operands);
    Rewriter::splitUWrite(regC_, C, regCToGlobMap, rtgOperands, cfg["LOCAL_SPLIT_U"], cfg["GLOB_STORE_WIDTH"], rLoop1, Position::after, "");
    auto storeBarrier1 = Rewriter::barrier(m_inner_0, Position::after);
    LOG_DEBUG("===== load write to C =======\n",module);
  } else {
    auto regCToGlobMap = getRegCToGlobMap(builder);
    llvm::SmallVector<mlir::Value> rtgOperands(operands);
    for (int i=0; i<3; i++) {
      rtgOperands.push_back(m_inner_axes[i].getInductionVar());
      rtgOperands.push_back(n_inner_axes[i].getInductionVar());
    }
    Rewriter::cache_write(m_inner_0, C, C, regCToGlobMap, rtgOperands);  // 只是改了一下map
    LOG_DEBUG("===== load cache_write regC to C =======\n",module);
  }
  Rewriter::vectorize(n_inner_2, cfg["WARP_SCATTER_WIDTH_N"]);
  LOG_DEBUG("===== vectorize =======\n",module);
  
  // mlir::affine::AffineForOp regRearForOp;
  // std::vector<mlir::affine::AffineForOp> shPerfetchRegForOp, perfetchSharedForOp, regPerfetchRegForOp;
  // if (cfg["SHARED_PREFETCH"]) {
  //   std::vector<mlir::affine::AffineForOp> shLoadRegForOps{loadTileA, loadTileB}, loadSharedForOps{storeTileA, storeTileB};
  //   std::vector<mlir::Value> smBufs{smA, smB};
  //   auto shResult = Rewriter::sharedMemroyPrefetch(k_outer, shLoadRegForOps, loadSharedForOps, k_mider, smBufs);
  //   smA = shResult.first[smA], smB = shResult.first[smB];
  //   shPerfetchRegForOp = shResult.second.first;
  //   perfetchSharedForOp = shResult.second.second;
  //   loadTileA = shLoadRegForOps[0], loadTileB = shLoadRegForOps[1];
  //   storeTileA = loadSharedForOps[0], storeTileB = loadSharedForOps[1];
  //   LOG_DEBUG("===== sharedMemroyPrefetch =======\n",module);
  // }

  // if (cfg["REG_PREFETCH"]) {
  //   std::vector<mlir::affine::AffineForOp> regLoadRegForOps{loadFragA, loadFragB};
  //   std::vector<mlir::Value> regBufs{regA, regB};
  //   auto regResult = Rewriter::registersPrefetch(k_mider, regLoadRegForOps, yTileForOp, regBufs);
  //   regA = regResult.first[regA], regB = regResult.first[regB];
  //   regPerfetchRegForOp = regResult.second.first;
  //   regRearForOp = regResult.second.second;
  //   loadFragA = regLoadRegForOps[0], loadFragB = regLoadRegForOps[1];
  //   LOG_DEBUG("===== registersPrefetch =======\n",module);
  // }

  // if (cfg["SHARED_PREFETCH"] && cfg["REG_PREFETCH"]) {
  //   Rewriter::doublePerfetchAdjust(perfetchSharedForOp, shPerfetchRegForOp, regPerfetchRegForOp, regRearForOp, {smA, smB}, {regA, regB});
  //   LOG_DEBUG("===== doublePerfetchAdjust =======\n",module);
  // }

  mlir::affine::AffineForOp regRearForOp;
  std::vector<mlir::affine::AffineForOp> pfLdRegForOps, pfLdSMForOps, pfLdRegForOps_;
  if (cfg["SHARED_PREFETCH"]) {
    std::vector<mlir::affine::AffineForOp> LdRegForOps{loadTileA, loadTileB}, ldSMForOps{storeTileA, storeTileB};
    std::vector<mlir::Value> smBufs{smA, smB};
    auto smResult = Rewriter::sharedMemroyPrefetch(k_outer, LdRegForOps, ldSMForOps, k_mider, smBufs);
    smA = smBufs[0], smB = smBufs[1];
    loadTileA = LdRegForOps[0], loadTileB = LdRegForOps[1];
    storeTileA = ldSMForOps[0], storeTileB = ldSMForOps[1];
    pfLdRegForOps = smResult.first; pfLdSMForOps = smResult.second;
    LOG_DEBUG("===== sharedMemroyPrefetch =======\n",module);
  }

  if (cfg["REG_PREFETCH"]) {
    std::vector<mlir::affine::AffineForOp> regLdRegForOps{loadFragA, loadFragB};
    std::vector<mlir::Value> regBufs{regA, regB};
    auto regResult = Rewriter::registersPrefetch(k_mider, regLdRegForOps, yTileForOp, regBufs);
    regA = regBufs[0], regB = regBufs[1];
    loadFragA = regLdRegForOps[0], loadFragB = regLdRegForOps[1];
    pfLdRegForOps_ = regResult.first; regRearForOp = regResult.second;
    LOG_DEBUG("===== registersPrefetch =======\n",module);
  }

  if (cfg["SHARED_PREFETCH"] && cfg["REG_PREFETCH"]) {
    Rewriter::doubleBufferAdjust(pfLdSMForOps, pfLdRegForOps, pfLdRegForOps_, regRearForOp);
    LOG_DEBUG("===== doublePerfetchAdjust =======\n",module);
  }

  if (cfg["LOCAL_SPLIT_U"] > 1) {
    Rewriter::bufferCombine({smA, smB}, {smC}, "smABC");
    Rewriter::bufferCombine({regC[0]}, {regC_}, "regC");
    LOG_DEBUG("===== bufferCombine =======\n",module);
  }

  Rewriter::unrollAttribute(module, cfg["UNROLL_NUM"]);
  LOG_DEBUG("===== unrollAttribute =======\n",module);
}


}
