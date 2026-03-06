#include "Conversion/Optimize.h"
#include <cmath>
#include <limits>

namespace KernelCodeGen {

// ======================================= global to sm =========================================
std::array<int64_t, 7> GemmNormColSumOptimizer::getCfgDatas(const std::string& bufType) {
  // In transposed tiling: K is "A" (outer/prologue), Q is "B" (inner/loop)
  // "K" here maps to the role of Q in GemmStats, "Q" maps to K in GemmStats
  int64_t blockTileY = cfg.at("Br"), blockTileX = cfg.at("Slice1");
  int64_t isTran = this->isTranK, globLoadWidth = cfg.at("GLOB_LOAD_WIDTH_Q");
  int64_t globLoadAllWidth = globLoadAllWidthK;
  int64_t globLoadRowWidth = globLoadRowWidthK;
  int64_t loadContinuous = cfg.at("LOAD_CONTINUOUS_P");
  if (isTran) {
    blockTileY = cfg.at("Slice1"); blockTileX = cfg.at("Br");
  }
  if (bufType == "Q") {
    blockTileY = cfg.at("Slice1"); blockTileX = cfg.at("Bc");
    isTran = this->isTranQ; globLoadWidth = cfg.at("GLOB_LOAD_WIDTH_K");
    globLoadAllWidth = globLoadAllWidthQ;
    globLoadRowWidth = globLoadRowWidthQ;
    if (isTran) {
      blockTileY = cfg.at("Bc"); blockTileX = cfg.at("Slice1");
    }
  }
  return {blockTileY, blockTileX, isTran, globLoadWidth, globLoadAllWidth, globLoadRowWidth, loadContinuous};
}

std::array<mlir::AffineExpr, 2> GemmNormColSumOptimizer::getGlobToSmExprs(const llvm::SmallVector<mlir::AffineExpr>& dims,
                                                                          const std::array<int64_t, 7>& args) {
  mlir::AffineExpr tyIdx, txIdx;
  auto [blockTileY, blockTileX, isTran, globLoadWidth, globLoadAllWidth, globLoadRowWidth, loadContinuous] = args;
  if (loadContinuous) {
    auto tidIndex = (dims[1] * globLoadAllWidth) + (dims[0] * globLoadWidth);
    auto tIdx = tools::mapUtils::reshapeThreadBlock(tidIndex, {blockTileY, blockTileX});
    tyIdx = tIdx[0]; txIdx = tIdx[1];
  } else {
    auto threadRowNum = this->threadNum / blockTileY;
    auto tIdx = tools::mapUtils::reshapeThreadBlock(dims[0], {blockTileY, threadRowNum});
    tyIdx = tIdx[0];
    txIdx = dims[1] * globLoadRowWidth + tIdx[1] * globLoadWidth;
  }
  return {tyIdx, txIdx};
}

mlir::AffineMap GemmNormColSumOptimizer::getGlobQKToTempQKMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by, bx, tid, k, iter}
  auto args = getCfgDatas(bufType);
  mlir::AffineExpr row, col;
  row = dims[dimCount-5];                      // by
  col = dims[dimCount-2];                      // k
  if (args[2]) {
    row = dims[dimCount-2];
    col = dims[dimCount-5];
  }
  if (bufType == "Q") {
    row = dims[dimCount-2];                    // k
    col = dims[dimCount-4];                    // bx
    if (args[2]) {
      row = dims[dimCount-4];
      col = dims[dimCount-2];
    }
  }
  auto [tyIdx, txIdx] = getGlobToSmExprs({dims[dimCount-3], dims[dimCount-1]}, args);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<2; i++) {
    exprs.push_back(dims[i]);
  }
  exprs.push_back(row + tyIdx);
  exprs.push_back(col + txIdx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap GemmNormColSumOptimizer::getTempToSmMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 2;
  auto dims = getExprs(builder, dimCount); // {tid, iter}
  auto args = getCfgDatas(bufType);
  auto [tyIdx, txIdx] = getGlobToSmExprs(dims, args);
  if ((bufType == "K" && !args[2]) || (bufType == "Q" && args[2])) {
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

mlir::AffineMap GemmNormColSumOptimizer::getTempToSmKPrologueMap(mlir::OpBuilder& builder) {
  int dimCount = 3;
  auto dims = getExprs(builder, dimCount);
  auto args = getCfgDatas("K");
  auto [tyIdx, txIdx] = getGlobToSmExprs({dims[0], dims[2]}, args);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  if (args[2]) {
    exprs.push_back(tyIdx + dims[1]);
    exprs.push_back(txIdx);
  } else {
    exprs.push_back(txIdx);
    exprs.push_back(tyIdx + dims[1]);
  }
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

// ================================== sm to reg and calculate ====================================
std::array<int64_t, 8> GemmNormColSumOptimizer::getSmCfgDatas(const std::string& bufType) {
  int64_t blockLayoutY = cfg["BLOCK_LAYOUT_P_Y"], blockLayoutX = cfg["BLOCK_LAYOUT_P_X"];
  int64_t warpLayoutY = cfg["WARP_LAYOUT_P_Y"], warpLayoutX = cfg["WARP_LAYOUT_P_X"];
  int64_t blockScatterY = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatterY = cfg["WARP_SCATTER_WIDTH_Q"];
  int64_t blockScatterX = cfg["BLOCK_SCATTER_WIDTH_K"], warpScatterX = cfg["WARP_SCATTER_WIDTH_K"];
  return {blockLayoutY, blockLayoutX, warpLayoutY, warpLayoutX, blockScatterY, blockScatterX, warpScatterY, warpScatterX};
}

mlir::AffineMap GemmNormColSumOptimizer::getSmQKVToRegQKVMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 4;
  auto dims = getExprs(builder, dimCount);  // {tid, bk, blockRepIter, warpRepIter}
  auto args = getSmCfgDatas(bufType);
  mlir::AffineExpr widx, lidx;
  int64_t blockLayout, warpLayout, blockScatter, warpScatter;
  if (bufType == "K") {
    widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], args[1]);
    lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], args[3]);
    blockLayout = args[0], warpLayout = args[2], blockScatter = args[4], warpScatter = args[6];
  } else if (bufType == "Q") {
    widx = tools::mapUtils::wapr_x(dims[0], cfg["WARP_SIZE"], args[1]);
    lidx = tools::mapUtils::lane_x(dims[0], cfg["WARP_SIZE"], args[3]);
    blockLayout = args[1], warpLayout = args[3], blockScatter = args[5], warpScatter = args[7];
  }
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(dims[1]);
  exprs.push_back((dims[2] * blockLayout + widx) * warpLayout * blockScatter + (dims[3] * warpLayout + lidx) * warpScatter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap GemmNormColSumOptimizer::getSmKPrologueToRegKMap(mlir::OpBuilder& builder) {
  int dimCount = 5;
  auto dims = getExprs(builder, dimCount);
  auto args = getSmCfgDatas("K");
  auto widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], args[1]);
  auto lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], args[3]);
  int64_t blockLayout = args[0], warpLayout = args[2], blockScatter = args[4], warpScatter = args[6];
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(dims[1] + dims[2]);
  exprs.push_back((dims[3] * blockLayout + widx) * warpLayout * blockScatter + (dims[4] * warpLayout + lidx) * warpScatter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap GemmNormColSumOptimizer::getCalculateMap(mlir::OpBuilder& builder, std::string calculatetype) {
  if (calculatetype == "matmul") {
    auto iter = builder.getAffineDimExpr(0);
    llvm::SmallVector<mlir::AffineExpr> exprs{iter};
    return mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
  }
  auto itery = builder.getAffineDimExpr(0);
  auto iterx = builder.getAffineDimExpr(1);
  llvm::SmallVector<mlir::AffineExpr> exprs{itery, iterx};
  return mlir::AffineMap::get(2, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

// ======================================= row_sum write =========================================
mlir::AffineMap GemmNormColSumOptimizer::getRowSumWriteMap(mlir::OpBuilder& builder) {
  // {b1, b2, by, tid, blockRepK, warpRepK, width} -> {b1, b2, by + row_in_block}
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);
  auto widx = tools::mapUtils::wapr_y(dims[3], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
  auto lidx = tools::mapUtils::lane_y(dims[3], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);
  int64_t blockLayout = cfg["BLOCK_LAYOUT_P_Y"], warpLayout = cfg["WARP_LAYOUT_P_Y"];
  int64_t blockScatter = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatter = cfg["WARP_SCATTER_WIDTH_Q"];
  mlir::AffineExpr blockExpr = (dims[4] * blockLayout + widx * blockScatter) * warpLayout;
  mlir::AffineExpr warpLevel = (dims[5] * warpLayout + lidx * warpScatter);
  mlir::AffineExpr row_in_block = blockExpr + warpLevel + dims[6];
  llvm::SmallVector<mlir::AffineExpr> exprs{dims[0], dims[1], dims[2] + row_in_block};
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

// ==================================== parsing and tuning =====================================

void GemmNormColSumOptimizer::computeTuneArgs() {
  this->blockPY = cfg.at("Br") / cfg.at("PTr");
  this->blockPX = cfg.at("Bc") / cfg.at("PTc");
  this->threadNum = blockPY * blockPX;
  this->blockRepeatK = cfg.at("PTr") / cfg.at("BLOCK_SCATTER_WIDTH_Q");
  this->blockRepeatQ = cfg.at("PTc") / cfg.at("BLOCK_SCATTER_WIDTH_K");
  this->warpRepeatK = cfg.at("BLOCK_SCATTER_WIDTH_Q") / cfg.at("WARP_SCATTER_WIDTH_Q");
  this->warpRepeatQ = cfg.at("BLOCK_SCATTER_WIDTH_K") / cfg.at("WARP_SCATTER_WIDTH_K");
  this->globLoadTotalWidthK = cfg.at("Br") * cfg.at("Slice1") / this->threadNum;
  this->globLoadTotalWidthQ = cfg.at("Bc") * cfg.at("Slice1") / this->threadNum;
  this->globLoadRowWidthK = this->threadNum / cfg.at("Br") * cfg.at("GLOB_LOAD_WIDTH_Q");
  this->globLoadRowWidthQ = this->threadNum / cfg.at("Slice1") * cfg.at("GLOB_LOAD_WIDTH_K");
  if (isTranK) {
    this->globLoadRowWidthK = this->threadNum / cfg.at("Slice1") * cfg.at("GLOB_LOAD_WIDTH_Q");
  }
  if (isTranQ) {
    this->globLoadRowWidthQ = this->threadNum / cfg.at("Bc") * cfg.at("GLOB_LOAD_WIDTH_K");
  }
  this->globLoadAllWidthK = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_Q");
  this->globLoadAllWidthQ = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_K");
}

void GemmNormColSumOptimizer::parseFuncArgs(mlir::func::FuncOp funcOp) {
  typeK = mlir::dyn_cast<mlir::MemRefType>(K.getType());
  typeQ = mlir::dyn_cast<mlir::MemRefType>(Q.getType());
  typeEm = mlir::dyn_cast<mlir::MemRefType>(Em.getType());
  typeDenom = mlir::dyn_cast<mlir::MemRefType>(Denom.getType());
  typeRowSumOut = mlir::dyn_cast<mlir::MemRefType>(RowSumOut.getType());
  typeMid = mlir::dyn_cast<mlir::MemRefType>(midBuf.getType());
  std::vector<bool> isTrans;
  auto transArr = funcOp->getAttr(ARGTRAN);
  auto transArrAttr = mlir::dyn_cast<mlir::ArrayAttr>(transArr);
  for (auto tran : transArrAttr) {
    auto tranAttr = mlir::dyn_cast<mlir::IntegerAttr>(tran);
    isTrans.push_back(tranAttr.getInt());
  }
  isTranK = isTrans[0]; isTranQ = isTrans[1];
  auto shapeRowSum = typeRowSumOut.getShape();
  batchSize = shapeRowSum[0]; headNum = shapeRowSum[1];
  seqLen = shapeRowSum[2];
  auto shapeK = typeK.getShape();
  headDim = isTranK ? shapeK[2] : shapeK[3];
}

bool GemmNormColSumOptimizer::applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) {
  this->cfg = config;
  mlir::ValueRange operands = funcOp.getArguments();
  // funcArgs: [K, Q_t, Em, Denom, RowSumOut]
  this->K = operands[0]; this->Q = operands[1];
  this->Em = operands[2]; this->Denom = operands[3]; this->RowSumOut = operands[4];
  funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (auto parallelOp = mlir::dyn_cast<mlir::affine::AffineParallelOp>(op)) {
      auto gpuIdx = getStrAttr(parallelOp, AttrGPUIndex);
      if (gpuIdx == BLOCKIDX) {
        blockIdx = parallelOp;
      } else if (gpuIdx == THREADIDX) {
        this->threadIdx = parallelOp;
      }
    } else if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
      auto forDesc = getStrAttr(forOp, FORDESC);
      if (forDesc == "blockx") {
        this->xBlockFors.push_back(forOp);
      } else if (forDesc == "ttiley") {
        this->yTileForOps.push_back(forOp);
      } else if (forDesc == "ttilex") {
        this->xTileForOps.push_back(forOp);
      } else if (forDesc == "k") {
        this->kForOps.push_back(forOp);
      }
    } else if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
      auto applyDesc = getStrAttr(applyOp, APPLYDESC);
      if (applyDesc == "blocky") {
        this->byIdx = applyOp.getResult();
      }
    } else if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
      auto allocDesc = getStrAttr(allocOp, AttrBufDescription);
      if (allocDesc.find("midBuf") != std::string::npos) {
        midBuf = allocOp.getResult();
      }
    }
  });
  parseFuncArgs(funcOp);
  computeTuneArgs();
  return true;
}

void GemmNormColSumOptimizer::moveMemrefDefineAhead(mlir::Operation* threadParallelOp){
  auto parallelop = mlir::dyn_cast<mlir::affine::AffineParallelOp>(threadParallelOp);
  assert(parallelop != nullptr);
  mlir::affine::AffineForOp firstForOp {};
  std::vector<mlir::Operation*> opsToMove {};
  for(auto& childop : parallelop->getRegion(0).getOps()) {
    firstForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(childop);
    if(firstForOp != nullptr){
      break;
    }
  }
  assert(firstForOp != nullptr);
  parallelop.walk([&](mlir::memref::AllocaOp op){
    opsToMove.push_back(op.getOperation());
  });
  parallelop.walk([&](mlir::memref::AllocOp op){
    opsToMove.push_back(op.getOperation());
  });
  for(auto op : opsToMove){
    op->moveBefore(firstForOp);
  }
}

// ================================== applyOptimzer ========================================

void GemmNormColSumOptimizer::applyOptimzer(mlir::func::FuncOp& funcOp) {
  mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(funcOp->getParentOp());
  mlir::OpBuilder builder(module);
  bool wave64SafeRowSum = cfg.count("WARP_SIZE") && cfg.at("WARP_SIZE") > 32;
  bool useWave64DirectScore =
      wave64SafeRowSum && cfg.count("CAUSAL_MASK") && cfg.at("CAUSAL_MASK");

  // tileP bufferize
  std::vector<mlir::affine::AffineForOp> tilePLoops{yTileForOps[0], xTileForOps[0]};
  auto tileP = Rewriter::bufferizeLoopCarryVar(kForOps[0], tilePLoops, MemorySpace::local, {"tileP"});
  LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

  // k1 split and reorder
  auto k1 = Rewriter::split(kForOps[0], {cfg.at("Slice1")});
  auto k1_outer = k1[0], k1_inner = k1[1];
  Rewriter::reorder({k1_outer, k1_inner, yTileForOps[0], xTileForOps[0]});
  LOG_DEBUG("===== after split & reorder K1 =======\n",module);

  // Buffer allocation
  // In transposed tiling: K is "A" (prologue), Q is "B" (inner loop)
  auto dtypeK = typeK.getElementType();
  auto dtypeQ = typeQ.getElementType();
  auto dtypeMid = typeMid.getElementType();

  std::vector<std::vector<int64_t>> smShapes{
    {cfg.at("Slice1"), cfg.at("Br")}, {cfg.at("Slice1"), cfg.at("Bc")}
  };
  std::vector<mlir::Type> smType{dtypeK, dtypeQ};
  std::vector<std::string> smDescs{"smK", "smQ"};
  auto sm = Rewriter::allocBuffers(smShapes, smType, MemorySpace::shared, smDescs, blockIdx);
  auto smK = sm[0], smQ = sm[1];
  mlir::Value smQFix;
  if (useWave64DirectScore) {
    smQFix = Rewriter::allocBuffers({{cfg.at("Hd")}},
                                    {dtypeQ},
                                    MemorySpace::shared, {"smQFix"}, blockIdx)[0];
  }

  std::vector<std::vector<int64_t>> regShapes{
    {globLoadTotalWidthK}, {globLoadTotalWidthQ}, {cfg.at("PTr")}, {cfg.at("PTc")}
  };
  std::vector<mlir::Type> regDTypes{dtypeK, dtypeQ, dtypeK, dtypeQ};
  std::vector<std::string> regDescs{"tempK", "tempQ", "regK", "regQ"};
  auto reg = Rewriter::allocBuffers(regShapes, regDTypes, MemorySpace::local, regDescs, this->xBlockFors[0]);
  auto tempK = reg[0], tempQ = reg[1], regK = reg[2], regQ = reg[3];
  mlir::Value scoreAccWave64;
  if (useWave64DirectScore) {
    scoreAccWave64 = Rewriter::allocBuffers(
        {{1}},
        {dtypeMid},
        MemorySpace::local, {"scoreAccWave64"}, this->xBlockFors[0])[0];
  }
  LOG_DEBUG("===== after alloc_buffer =======\n",module);

  // smKFull allocation (K is pre-loaded like Q in GemmStats)
  mlir::Value smKFull;
  if (cfg["Slice1"] == cfg["Hd"]) {
    smKFull = smK;
  } else {
    smKFull = Rewriter::allocBuffers({{cfg["Hd"], cfg["Br"]}},
                                      {typeK.getElementType()},
                                      MemorySpace::shared, {"smKFull"}, blockIdx)[0];
  }
  LOG_DEBUG("===== after alloc smKFull =======\n",module);

  // regRowSum allocation for row_sum along PTr
  auto regRowSum = Rewriter::allocBuffers({{cfg.at("PTr")}},
                                           {dtypeMid},
                                           MemorySpace::local, {"regRowSum"}, blockIdx)[0];
  // Initialize regRowSum to zero before the blockx loop
  {
    mlir::OpBuilder initBuilder = getBuilder(xBlockFors[0], Position::before);
    auto loc = initBuilder.getUnknownLoc();
    auto zeroConst = initBuilder.create<mlir::arith::ConstantOp>(loc, initBuilder.getFloatAttr(dtypeMid, 0.0));
    auto initBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value iv, mlir::ValueRange iterArgs) {
      bb.create<mlir::affine::AffineStoreOp>(l, zeroConst, regRowSum, mlir::ValueRange{iv});
      bb.create<mlir::affine::AffineYieldOp>(l);
    };
    initBuilder.create<mlir::affine::AffineForOp>(loc, 0, cfg["PTr"], 1, mlir::ValueRange{}, initBody);
  }
  LOG_DEBUG("===== after regRowSum init =======\n",module);

  auto bIdx = Analyzer::getParallelIdx(this->blockIdx);
  auto tIdx = Analyzer::getParallelIdx(this->threadIdx);
  int batchCount = (int)bIdx.size() - 1;

  // ====== K PROLOGUE: load entire K tile into smKFull[Hd×Br] before the blockx loop ======
  auto loadTileKMap = getGlobQKToTempQKMap(builder, "K");
  auto prologueStoreKMap = getTempToSmKPrologueMap(builder);
  {
    mlir::OpBuilder prologueBuilder = getBuilder(xBlockFors[0], Position::before);
    auto loc = prologueBuilder.getUnknownLoc();
    auto c0 = prologueBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    llvm::SmallVector<int64_t> plbs{0}, pubs{cfg["Hd"]}, psteps{cfg["Slice1"]};
    auto [prologueForOps, prologueIvs] = createNestedLoops(prologueBuilder, plbs, pubs, psteps);
    auto k_prologue = prologueForOps[0];

    llvm::SmallVector<mlir::Value> kPrologueGlobOps(bIdx.begin(), bIdx.end()-1);
    kPrologueGlobOps.push_back(byIdx);
    kPrologueGlobOps.push_back(c0);
    kPrologueGlobOps.push_back(tIdx[0]);
    kPrologueGlobOps.push_back(prologueIvs[0]);

    auto loadTileK_p = Rewriter::loadToRegisters(K, tempK, loadTileKMap, kPrologueGlobOps,
                                                  {cfg["GLOB_LOAD_WIDTH_Q"]}, k_prologue, Position::begin, "");

    mlir::affine::AffineForOp lastBeforeStoreK_p = loadTileK_p;
    if (cfg.count("SCALE_Q") && cfg.at("SCALE_Q")) {
      auto dtype = typeK.getElementType();
      float scaleVal = 1.0f / std::sqrt(static_cast<float>(headDim));
      mlir::OpBuilder sb(loadTileK_p->getBlock(), ++mlir::Block::iterator(loadTileK_p.getOperation()));
      auto sLoc = sb.getUnknownLoc();
      auto scaleConst = sb.create<mlir::arith::ConstantOp>(sLoc, sb.getFloatAttr(dtype, scaleVal));
      auto scaleBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value iv, mlir::ValueRange iterArgs) {
        auto ld = bb.create<mlir::affine::AffineLoadOp>(l, tempK, mlir::ValueRange{iv});
        auto scaled = bb.create<mlir::arith::MulFOp>(l, ld, scaleConst);
        bb.create<mlir::affine::AffineStoreOp>(l, scaled, tempK, mlir::ValueRange{iv});
        bb.create<mlir::affine::AffineYieldOp>(l);
      };
      lastBeforeStoreK_p = sb.create<mlir::affine::AffineForOp>(sLoc, 0, globLoadTotalWidthK, 1, mlir::ValueRange{}, scaleBody);
      LOG_DEBUG("===== fused K scale into prologue =======\n",module);
    }

    Rewriter::loadFromRegisters(tempK, smKFull, prologueStoreKMap,
                                {tIdx[0], prologueIvs[0]},
                                {cfg["GLOB_LOAD_WIDTH_Q"]}, lastBeforeStoreK_p, Position::after, "");
    Rewriter::barrier(k_prologue, Position::after);
  }
  LOG_DEBUG("===== K prologue done =======\n",module);

  // ====== Q loading: Q is loaded inside k1_outer (inner dimension) ======
  auto loadTileQMap = getGlobQKToTempQKMap(builder, "Q");
  if (useWave64DirectScore) {
    mlir::OpBuilder qFixBuilder = getBuilder(k1_outer, Position::before);
    auto loc = qFixBuilder.getUnknownLoc();
    auto fixColConst =
        qFixBuilder.create<mlir::arith::ConstantIndexOp>(loc, cfg.at("WARP_SCATTER_WIDTH_K"));
    auto d0 = qFixBuilder.getAffineDimExpr(0);
    auto validTidSet = mlir::IntegerSet::get(
        1, 0, {qFixBuilder.getAffineConstantExpr(headDim - 1) - d0}, {false});
    auto qFixIf = qFixBuilder.create<mlir::affine::AffineIfOp>(
        loc, validTidSet, mlir::ValueRange{tIdx[0]}, false);
    mlir::OpBuilder thenB = mlir::OpBuilder::atBlockBegin(qFixIf.getThenBlock());

    auto fixColGlobal = thenB.create<mlir::arith::AddIOp>(
        loc, xBlockFors[0].getInductionVar(), fixColConst);
    llvm::SmallVector<mlir::Value> qIdxs;
    qIdxs.reserve(batchCount + 2);
    for (int bi = 0; bi < batchCount; ++bi) {
      qIdxs.push_back(bIdx[bi]);
    }
    qIdxs.push_back(tIdx[0]);
    qIdxs.push_back(fixColGlobal);
    auto qFixVal = thenB.create<mlir::memref::LoadOp>(loc, Q, qIdxs);
    thenB.create<mlir::affine::AffineStoreOp>(loc, qFixVal, smQFix, mlir::ValueRange{tIdx[0]});

    auto afterQFix = getBuilder(qFixIf, Position::after);
    Rewriter::barrier(afterQFix);
    LOG_DEBUG("===== Q fix-column prologue done =======\n",module);
  }

  llvm::SmallVector<mlir::Value> operands(bIdx.begin(), bIdx.end()-1);
  operands.push_back(byIdx); operands.push_back(xBlockFors[0].getInductionVar()); operands.push_back(tIdx[0]);
  llvm::SmallVector<mlir::Value> qGlobOperands(operands);
  qGlobOperands.push_back(k1_outer.getInductionVar());
  auto loadTileQ = Rewriter::loadToRegisters(Q, tempQ, loadTileQMap, qGlobOperands, {cfg["GLOB_LOAD_WIDTH_K"]}, k1_outer, Position::begin, "");
  LOG_DEBUG("===== after read Q =======\n",module);
  auto storeTileQMap = getTempToSmMap(builder, "Q");
  auto storeTileQ = Rewriter::loadFromRegisters(tempQ, smQ, storeTileQMap, {tIdx[0]}, {cfg["GLOB_LOAD_WIDTH_K"]}, loadTileQ, Position::after, "");
  auto prefix = Rewriter::barrier(loadTileQ, Position::before);
  auto suffix = Rewriter::barrier(storeTileQ, Position::after);
  LOG_DEBUG("===== write Q =======\n",module);

  // sm K (pre-loaded) and sm Q to registers
  auto loadFragKMap = getSmKPrologueToRegKMap(builder);
  auto loadFragQMap = getSmQKVToRegQKVMap(builder, "Q");
  llvm::SmallVector<mlir::Value> kFragOperands{tIdx[0], k1_outer.getInductionVar(), k1_inner.getInductionVar()};
  llvm::SmallVector<mlir::Value> qFragOperands{tIdx[0], k1_inner.getInductionVar()};
  std::vector<int64_t> widthsK{cfg["BLOCK_SCATTER_WIDTH_Q"], cfg["WARP_SCATTER_WIDTH_Q"]};
  std::vector<int64_t> widthsQ{cfg["BLOCK_SCATTER_WIDTH_K"], cfg["WARP_SCATTER_WIDTH_K"]};
  auto loadFragK = Rewriter::loadToRegisters(smKFull, regK, loadFragKMap, kFragOperands, widthsK, k1_inner, Position::begin, "");
  auto loadFragQ = Rewriter::loadToRegisters(smQ, regQ, loadFragQMap, qFragOperands, widthsQ, loadFragK, Position::after, "");
  LOG_DEBUG("===== read sh_K/Q =======\n",module);

  // matmul1 micro-kernel: regK × regQ → tileP (P^T = K^T @ Q)
  auto calMap = getCalculateMap(builder, "matmul");
  Rewriter::cache_read(xTileForOps[0], K, regK, calMap, {yTileForOps[0].getInductionVar()});
  Rewriter::cache_read(xTileForOps[0], Q, regQ, calMap, {xTileForOps[0].getInductionVar()});
  LOG_DEBUG("===== load regK & cache_read =======\n",module);

  auto tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
  auto txds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttilexDown");
  std::vector<mlir::affine::AffineForOp> forOps{tyds[0], txds[0]};
  Rewriter::separateNoOpRelyForOp(forOps);
  LOG_DEBUG("===== separateNoOpRelyForOp =======\n",module);

  // CAUSAL_MASK on tileP (transposed: mask where globalRow_K > globalCol_Q)
  {
    bool doMask = cfg.count("CAUSAL_MASK") && cfg.at("CAUSAL_MASK");
    if (doMask) {
      tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
      auto dtype = typeMid.getElementType();
      mlir::OpBuilder sb(tyds[0]);
      auto loc = sb.getUnknownLoc();

      mlir::Value negInf = sb.create<mlir::arith::ConstantOp>(loc, sb.getFloatAttr(dtype, -1.0e30));

      llvm::SmallVector<int64_t> lbs{0, 0}, ubs{cfg["PTr"], cfg["PTc"]}, steps{1, 1};
      auto [maskForOps, ivs] = createNestedLoops(sb, lbs, ubs, steps);
      sb.setInsertionPointToStart(maskForOps.back().getBody());

      auto ld = sb.create<mlir::affine::AffineLoadOp>(loc, tileP[0], ivs);
      mlir::Value val = ld;

      // Compute globalRow (K side, outer by) and globalCol (Q side, inner bx)
      int dimCount = 5;
      auto dims = getExprs(sb, dimCount);
      auto iE = dims[0], jE = dims[1], tidE = dims[2], byE = dims[3], bxE = dims[4];

      int64_t BSW_Q = cfg["BLOCK_SCATTER_WIDTH_Q"], WSW_Q = cfg["WARP_SCATTER_WIDTH_Q"];
      int64_t BLP_Y = cfg["BLOCK_LAYOUT_P_Y"], WLP_Y = cfg["WARP_LAYOUT_P_Y"];
      int64_t BLP_X = cfg["BLOCK_LAYOUT_P_X"], WLP_X = cfg["WARP_LAYOUT_P_X"];
      int64_t WARP_SZ = cfg["WARP_SIZE"];
      int64_t BSW_K = cfg["BLOCK_SCATTER_WIDTH_K"], WSW_K = cfg["WARP_SCATTER_WIDTH_K"];

      // globalRow from K (outer, by): PTr direction (i)
      auto warp_y = tools::mapUtils::wapr_y(tidE, WARP_SZ, BLP_X);
      auto lane_y = tools::mapUtils::lane_y(tidE, WARP_SZ, WLP_X);
      auto rowInBr = (iE.floorDiv(BSW_Q) * BLP_Y + warp_y) * WLP_Y * BSW_Q
                     + ((iE % BSW_Q).floorDiv(WSW_Q) * WLP_Y + lane_y) * WSW_Q + iE % WSW_Q;
      auto globalRow = byE + rowInBr;

      // globalCol from Q (inner, bx): PTc direction (j)
      auto warp_x = tools::mapUtils::wapr_x(tidE, WARP_SZ, BLP_X);
      auto lane_x = tools::mapUtils::lane_x(tidE, WARP_SZ, WLP_X);
      auto colInBc = (jE.floorDiv(BSW_K) * BLP_X + warp_x) * WLP_X * BSW_K
                     + ((jE % BSW_K).floorDiv(WSW_K) * WLP_X + lane_x) * WSW_K + jE % WSW_K;
      auto globalCol = bxE + colInBc;

      // Transposed mask: P^T[i_k, j_q] should be masked when i_k > j_q
      // (original P[j_q, i_k] masked when i_k > j_q, i.e. col > row)
      mlir::Value tid = this->threadIdx.getIVs()[0];
      mlir::Value bx = xBlockFors[0].getInductionVar();
      llvm::SmallVector<mlir::Value> mapOps{ivs[0], ivs[1], tid, byIdx, bx};

      auto rowMap = mlir::AffineMap::get(dimCount, 0, {globalRow}, sb.getContext());
      auto colMap = mlir::AffineMap::get(dimCount, 0, {globalCol}, sb.getContext());

      auto rowVal = sb.create<mlir::affine::AffineApplyOp>(loc, rowMap, mapOps);
      auto colVal = sb.create<mlir::affine::AffineApplyOp>(loc, colMap, mapOps);

      auto cmp = sb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt,
                                                  rowVal.getResult(), colVal.getResult());
      val = sb.create<mlir::arith::SelectOp>(loc, cmp, negInf, val);

      sb.create<mlir::affine::AffineStoreOp>(loc, val, tileP[0], ivs);
      LOG_DEBUG("===== causal mask (transposed) =======\n",module);
    }
  }

  // ===== NORMALIZE tileP + accumulate regRowSum =====
  // Optimized: outer loop j loads Em/Denom once, inner loop i reuses the factor.
  // Em/Denom only depend on j (inner S_q dim), so we avoid PTr redundant global loads.
  //   for j in [0, PTc):
  //     factor = em(j) * denom(j)        // 1 global load per j
  //     for i in [0, PTr):
  //       regRowSum[i] += exp(tileP[i][j]) / factor
  {
    tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
    auto dtype = typeMid.getElementType();
    mlir::OpBuilder nb(tyds[0]);
    auto loc = nb.getUnknownLoc();

    // Outer loop: j in [0, PTc) — load Em/Denom once per j
    llvm::SmallVector<int64_t> jlbs{0}, jubs{cfg["PTc"]}, jsteps{1};
    auto [jForOps, jIvs] = createNestedLoops(nb, jlbs, jubs, jsteps);
    nb.setInsertionPointToStart(jForOps.back().getBody());

    int emMapDims = batchCount + 3; // batch dims + bx + tid + flat_j
    auto emDimExprs = getExprs(nb, emMapDims);
    auto bxExpr = emDimExprs[batchCount];
    auto tidExpr = emDimExprs[batchCount + 1];
    auto flatJExpr = emDimExprs[batchCount + 2];

    int64_t BSW_K_n = cfg["BLOCK_SCATTER_WIDTH_K"];
    int64_t WSW_K_n = cfg["WARP_SCATTER_WIDTH_K"];
    int64_t WARP_SZ = cfg["WARP_SIZE"];
    int64_t BLP_X = cfg["BLOCK_LAYOUT_P_X"];
    int64_t WLP_X = cfg["WARP_LAYOUT_P_X"];

    auto warp_x_e = tools::mapUtils::wapr_x(tidExpr, WARP_SZ, BLP_X);
    auto lane_x_e = tools::mapUtils::lane_x(tidExpr, WARP_SZ, WLP_X);

    auto blockRepQ_e = flatJExpr.floorDiv(BSW_K_n);
    auto warpRepQ_e = (flatJExpr % BSW_K_n).floorDiv(WSW_K_n);
    auto scatterQ_e = flatJExpr % WSW_K_n;

    auto colInBc = (blockRepQ_e * BLP_X + warp_x_e) * WLP_X * BSW_K_n
                   + (warpRepQ_e * WLP_X + lane_x_e) * WSW_K_n + scatterQ_e;

    auto emRank = typeEm.getRank();
    llvm::SmallVector<mlir::AffineExpr> emResults;
    for (int i = 0; i < batchCount; i++) {
      emResults.push_back(emDimExprs[i]);
    }
    emResults.push_back(bxExpr + colInBc);
    while ((int)emResults.size() < emRank) {
      emResults.push_back(nb.getAffineConstantExpr(0));
    }
    auto emMap = mlir::AffineMap::get(emMapDims, 0,
                     llvm::ArrayRef<mlir::AffineExpr>(emResults), nb.getContext());

    llvm::SmallVector<mlir::Value> emOps;
    for (int i = 0; i < batchCount; i++) {
      emOps.push_back(bIdx[i]);
    }
    emOps.push_back(xBlockFors[0].getInductionVar());
    emOps.push_back(threadIdx.getIVs()[0]);
    emOps.push_back(jIvs[0]);

    auto emVal = nb.create<mlir::affine::AffineLoadOp>(loc, Em, emMap, emOps);
    auto denomVal = nb.create<mlir::affine::AffineLoadOp>(loc, Denom, emMap, emOps);
    auto factor = nb.create<mlir::arith::MulFOp>(loc, emVal, denomVal);

    // Inner loop: i in [0, PTr) — reuse factor, accumulate regRowSum[i]
    llvm::SmallVector<int64_t> ilbs{0}, iubs{cfg["PTr"]}, isteps{1};
    auto [iForOps, iIvs] = createNestedLoops(nb, ilbs, iubs, isteps);
    nb.setInsertionPointToStart(iForOps.back().getBody());

    // ROCm wave64 still shows residual row_sum leakage on one local Q slot.
    // The fallback score for that slot is recomputed here with exact row/col.
    if (useWave64DirectScore) {
      int dimCount = 5;
      auto dims = getExprs(nb, dimCount);
      auto iE = dims[0], jE = dims[1], tidE = dims[2], byE = dims[3], bxE = dims[4];

      int64_t BSW_Q_m = cfg["BLOCK_SCATTER_WIDTH_Q"];
      int64_t WSW_Q_m = cfg["WARP_SCATTER_WIDTH_Q"];
      int64_t BLP_Y_m = cfg["BLOCK_LAYOUT_P_Y"];
      int64_t WLP_Y_m = cfg["WARP_LAYOUT_P_Y"];
      int64_t BLP_X_m = cfg["BLOCK_LAYOUT_P_X"];
      int64_t WLP_X_m = cfg["WARP_LAYOUT_P_X"];
      int64_t WARP_SZ_m = cfg["WARP_SIZE"];
      int64_t BSW_K_m = cfg["BLOCK_SCATTER_WIDTH_K"];
      int64_t WSW_K_m = cfg["WARP_SCATTER_WIDTH_K"];

      auto warp_y_m = tools::mapUtils::wapr_y(tidE, WARP_SZ_m, BLP_X_m);
      auto lane_y_m = tools::mapUtils::lane_y(tidE, WARP_SZ_m, WLP_X_m);
      auto rowInBr_m = (iE.floorDiv(BSW_Q_m) * BLP_Y_m + warp_y_m) * WLP_Y_m * BSW_Q_m
                     + ((iE % BSW_Q_m).floorDiv(WSW_Q_m) * WLP_Y_m + lane_y_m) * WSW_Q_m + iE % WSW_Q_m;
      auto globalRow_m = byE + rowInBr_m;

      auto warp_x_m = tools::mapUtils::wapr_x(tidE, WARP_SZ_m, BLP_X_m);
      auto lane_x_m = tools::mapUtils::lane_x(tidE, WARP_SZ_m, WLP_X_m);
      auto colInBc_m = (jE.floorDiv(BSW_K_m) * BLP_X_m + warp_x_m) * WLP_X_m * BSW_K_m
                     + ((jE % BSW_K_m).floorDiv(WSW_K_m) * WLP_X_m + lane_x_m) * WSW_K_m + jE % WSW_K_m;
      auto globalCol_m = bxE + colInBc_m;

      auto rowMap_m = mlir::AffineMap::get(dimCount, 0, {globalRow_m}, nb.getContext());
      auto colMap_m = mlir::AffineMap::get(dimCount, 0, {globalCol_m}, nb.getContext());
      llvm::SmallVector<mlir::Value> maskOps{
          iIvs[0], jIvs[0], threadIdx.getIVs()[0], byIdx, xBlockFors[0].getInductionVar()};
      auto rowVal_m = nb.create<mlir::affine::AffineApplyOp>(loc, rowMap_m, maskOps);
      auto colVal_m = nb.create<mlir::affine::AffineApplyOp>(loc, colMap_m, maskOps);
      auto invalid_m = nb.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, rowVal_m.getResult(), colVal_m.getResult());

      auto zero_m = nb.create<mlir::arith::ConstantOp>(loc, nb.getFloatAttr(dtype, 0.0));
      auto scoreIdx = nb.create<mlir::arith::ConstantIndexOp>(loc, 0);
      llvm::SmallVector<mlir::Value> tilePIdx{iIvs[0], jIvs[0]};
      auto pVal_m = nb.create<mlir::affine::AffineLoadOp>(loc, tileP[0], tilePIdx);
      nb.create<mlir::affine::AffineStoreOp>(loc, pVal_m, scoreAccWave64, mlir::ValueRange{scoreIdx});

      auto setJ0Lane1 = [&]() {
        auto d0 = nb.getAffineDimExpr(0);
        auto d1 = nb.getAffineDimExpr(1);
        llvm::SmallVector<mlir::AffineExpr> exprs{d0, (d1 % WLP_X_m) - 1};
        llvm::SmallVector<bool, 2> flags{true, true};
        return mlir::IntegerSet::get(/*dimCount*/2, /*symbolCount*/0, exprs, flags);
      }();
      auto slotIf = nb.create<mlir::affine::AffineIfOp>(
          loc, setJ0Lane1,
          llvm::SmallVector<mlir::Value>{jIvs[0], threadIdx.getIVs()[0]}, false);
      mlir::OpBuilder thenB = mlir::OpBuilder::atBlockBegin(slotIf.getThenBlock());
      thenB.create<mlir::affine::AffineStoreOp>(loc, zero_m, scoreAccWave64, mlir::ValueRange{scoreIdx});
      auto rowInBrMap_m = mlir::AffineMap::get(dimCount, 0, {rowInBr_m}, thenB.getContext());
      auto rowInBrVal_m =
          thenB.create<mlir::affine::AffineApplyOp>(loc, rowInBrMap_m, maskOps);

      llvm::SmallVector<int64_t> hlbs{0}, hubs{headDim}, hsteps{1};
      auto [hForOps, hIvs] = createNestedLoops(thenB, hlbs, hubs, hsteps);
      thenB.setInsertionPointToStart(hForOps.back().getBody());

      auto kVal_m = thenB.create<mlir::affine::AffineLoadOp>(
          loc, smKFull, mlir::ValueRange{hIvs[0], rowInBrVal_m.getResult()});
      auto qVal_m = thenB.create<mlir::affine::AffineLoadOp>(
          loc, smQFix, mlir::ValueRange{hIvs[0]});
      auto prod_m = thenB.create<mlir::arith::MulFOp>(loc, kVal_m, qVal_m);
      auto oldScore_m =
          thenB.create<mlir::affine::AffineLoadOp>(loc, scoreAccWave64, mlir::ValueRange{scoreIdx});
      auto newScore_m = thenB.create<mlir::arith::AddFOp>(loc, oldScore_m, prod_m);
      thenB.create<mlir::affine::AffineStoreOp>(loc, newScore_m, scoreAccWave64, mlir::ValueRange{scoreIdx});

      mlir::OpBuilder afterIf = getBuilder(slotIf, Position::after);
      auto finalScore_m =
          afterIf.create<mlir::affine::AffineLoadOp>(loc, scoreAccWave64, mlir::ValueRange{scoreIdx});
      auto expVal_m = afterIf.create<mlir::math::ExpOp>(loc, finalScore_m);
      auto normalized_m = afterIf.create<mlir::arith::DivFOp>(loc, expVal_m, factor);
      auto selected_m =
          afterIf.create<mlir::arith::SelectOp>(loc, invalid_m, zero_m, normalized_m);
      auto oldSum_m =
          afterIf.create<mlir::affine::AffineLoadOp>(loc, regRowSum, mlir::ValueRange{iIvs[0]});
      auto newSum_m = afterIf.create<mlir::arith::AddFOp>(loc, oldSum_m, selected_m);
      afterIf.create<mlir::affine::AffineStoreOp>(loc, newSum_m, regRowSum, mlir::ValueRange{iIvs[0]});
    } else {
      llvm::SmallVector<mlir::Value> tilePIdx{iIvs[0], jIvs[0]};
      mlir::Value pVal = nb.create<mlir::affine::AffineLoadOp>(loc, tileP[0], tilePIdx);
      auto expVal = nb.create<mlir::math::ExpOp>(loc, pVal);
      auto normalizedLocal = nb.create<mlir::arith::DivFOp>(loc, expVal, factor);
      auto oldSum = nb.create<mlir::affine::AffineLoadOp>(loc, regRowSum, mlir::ValueRange{iIvs[0]});
      auto newSum = nb.create<mlir::arith::AddFOp>(loc, oldSum, normalizedLocal);
      nb.create<mlir::affine::AffineStoreOp>(loc, newSum, regRowSum, mlir::ValueRange{iIvs[0]});
    }
  }
  LOG_DEBUG("===== normalize + accumulate regRowSum (Em/Denom hoisted) =======\n",module);

  // Erase remaining ttileyDown write-back loops (they store tileP → midBuf, no longer needed)
  {
    auto tydsFinal = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
    for (auto tyd : tydsFinal) {
      tyd.erase();
    }
  }

  // Cleanup: erase midBuf (GEMM intermediate) if no remaining uses
  if (this->midBuf.use_empty()) {
    this->midBuf.getDefiningOp()->erase();
  }
  LOG_DEBUG("===== cleanup done =======\n",module);

  // ====== Warp shuffle reduction + guarded write to global RowSumOut ======
  // Use a single builder to ensure correct insertion order:
  //   1. Shuffle reduce regRowSum across lane_x (WLP_X threads per row)
  //   2. Only lane_x==0 threads write the final result to global memory
  {
    int64_t WLP_X = cfg["WARP_LAYOUT_P_X"];
    int64_t WARP_SZ = cfg["WARP_SIZE"];
    int64_t PTr_ = cfg["PTr"];

    mlir::OpBuilder wb = getBuilder(xBlockFors[0], Position::after);
    auto loc = wb.getUnknownLoc();

    // Step 1: warp shuffle reduction across lane_x
    if (WLP_X > 1) {
      auto widthI32 = wb.create<mlir::arith::ConstantOp>(loc, wb.getI32IntegerAttr(WLP_X));

      for (int64_t dist = 1; dist < WLP_X; dist *= 2) {
        auto distI32 = wb.create<mlir::arith::ConstantOp>(loc, wb.getI32IntegerAttr(dist));
        for (int64_t r = 0; r < PTr_; ++r) {
          auto rIdx = wb.create<mlir::arith::ConstantIndexOp>(loc, r);
          auto val = wb.create<mlir::affine::AffineLoadOp>(
              loc, regRowSum, mlir::ValueRange{rIdx});
          auto shfl = wb.create<mlir::gpu::ShuffleOp>(
              loc, val.getResult(), distI32, widthI32, mlir::gpu::ShuffleMode::DOWN);
          auto sum = wb.create<mlir::arith::AddFOp>(loc, val.getResult(), shfl.getResult(0));
          wb.create<mlir::affine::AffineStoreOp>(
              loc, sum, regRowSum, mlir::ValueRange{rIdx});
        }
      }
      LOG_DEBUG("===== warp shuffle reduction for regRowSum =======\n",module);
    }

    // Step 2: guard write-back — only lane_x==0 threads have the correct full sum
    if (WLP_X > 1) {
      mlir::Value tid = this->threadIdx.getIVs()[0];
      auto d0 = wb.getAffineDimExpr(0);
      auto modExpr = d0 % WLP_X;
      auto intSet = mlir::IntegerSet::get(1, 0, {modExpr}, {true});
      auto ifOp = wb.create<mlir::affine::AffineIfOp>(
          loc, intSet, mlir::ValueRange{tid}, false);
      wb.setInsertionPointToStart(ifOp.getThenBlock());
    }

    // Step 3: write regRowSum to global RowSumOut
    auto rowSumMap = getRowSumWriteMap(builder);
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto d2 = builder.getAffineDimExpr(2);
    auto regIdxExpr = d0 * cfg["BLOCK_SCATTER_WIDTH_Q"] + d1 * cfg["WARP_SCATTER_WIDTH_Q"] + d2;
    auto regIdxMap = mlir::AffineMap::get(3, 0, {regIdxExpr}, wb.getContext());

    llvm::SmallVector<int64_t> lbs{0, 0, 0};
    llvm::SmallVector<int64_t> ubs{blockRepeatK, warpRepeatK, cfg["WARP_SCATTER_WIDTH_Q"]};
    llvm::SmallVector<int64_t> steps{1, 1, 1};
    auto [writeForOps, writeIvs] = createNestedLoops(wb, lbs, ubs, steps);
    wb.setInsertionPointToStart(writeForOps.back().getBody());

    auto regIdx = wb.create<mlir::affine::AffineApplyOp>(loc, regIdxMap, writeIvs);
    auto sumVal = wb.create<mlir::affine::AffineLoadOp>(loc, regRowSum, mlir::ValueRange{regIdx.getResult()});

    auto bivs = this->blockIdx.getIVs();
    llvm::SmallVector<mlir::Value> writeOperands(bivs.rbegin(), bivs.rend()-1);
    writeOperands.push_back(byIdx);
    writeOperands.push_back(tIdx[0]);
    for (auto& iv : writeIvs) writeOperands.push_back(iv);

    wb.create<mlir::affine::AffineStoreOp>(loc, sumVal, RowSumOut, rowSumMap, writeOperands);
  }
  LOG_DEBUG("===== write regRowSum to global =======\n",module);

  // moveMemrefDefineAhead
  mlir::affine::AffineParallelOp threadParallelOp;
  funcOp.walk([&](mlir::affine::AffineParallelOp p){
    auto attr = p.getOperation()->getAttr(AttrGPUIndex);
    auto stringattr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if(std::string(stringattr.data()) == THREADIDX){
      threadParallelOp = p;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  moveMemrefDefineAhead(threadParallelOp.getOperation());
  LOG_DEBUG("===== moveMemrefDefineAhead =======\n",module);

  // Prefetch support (Q loading only, K is pre-loaded)
  mlir::affine::AffineForOp regRearForOp;
  std::vector<mlir::affine::AffineForOp> pfLdRegForOps, pfLdSMForOps, pfLdRegForOps_;
  if (cfg["SHARED_PREFETCH_P"]) {
    std::vector<mlir::affine::AffineForOp> LdRegForOps{loadTileQ};
    std::vector<mlir::affine::AffineForOp> ldSMForOps{storeTileQ};
    std::vector<mlir::Value> smBufs{smQ};
    int64_t prefetchStep = cfg.at("Slice1");
    auto smResult = Rewriter::sharedMemroyPrefetch(k1_outer, LdRegForOps, ldSMForOps, k1_inner, smBufs);
    smQ = smBufs[0];
    loadTileQ = LdRegForOps[0];
    storeTileQ = ldSMForOps[0];
    pfLdRegForOps = smResult.first; pfLdSMForOps = smResult.second;
    LOG_DEBUG("===== sharedMemroyPrefetch (Q only) =======\n",module);

    mlir::Value newKOuterIv = k1_outer.getInductionVar();
    auto patchSmKFullMap = [&](auto loadOp) {
      if (loadOp.getMemRef() != smKFull) return;
      auto mapOps = loadOp.getMapOperands();
      for (unsigned i = 0; i < mapOps.size(); ++i) {
        if (mapOps[i] == newKOuterIv) {
          auto oldMap = loadOp.getAffineMap();
          auto *ctx  = oldMap.getContext();
          auto dimExpr     = mlir::getAffineDimExpr(i, ctx);
          auto shiftedExpr = dimExpr - prefetchStep;
          llvm::SmallVector<mlir::AffineExpr> newResults;
          for (auto expr : oldMap.getResults())
            newResults.push_back(expr.replace(dimExpr, shiftedExpr));
          auto newMap = mlir::AffineMap::get(oldMap.getNumDims(),
                            oldMap.getNumSymbols(), newResults, ctx);
          loadOp->setAttr("map", mlir::AffineMapAttr::get(newMap));
          break;
        }
      }
    };
    loadFragK.walk([&](mlir::affine::AffineVectorLoadOp vlop) { patchSmKFullMap(vlop); });
    loadFragK.walk([&](mlir::affine::AffineLoadOp lop)        { patchSmKFullMap(lop); });
    LOG_DEBUG("===== fix smKFull k_outer IV shift =======\n",module);
  }

  if (cfg["REG_PREFETCH_P"]) {
    std::vector<mlir::affine::AffineForOp> regLdRegForOps{loadFragK, loadFragQ};
    std::vector<mlir::Value> regBufs{regK, regQ};
    auto regResult = Rewriter::registersPrefetch(k1_inner, regLdRegForOps, yTileForOps[0], regBufs);
    regK = regBufs[0], regQ = regBufs[1];
    loadFragK = regLdRegForOps[0], loadFragQ = regLdRegForOps[1];
    pfLdRegForOps_ = regResult.first; regRearForOp = regResult.second;
    LOG_DEBUG("===== registersPrefetch =======\n",module);
  }

  if (cfg["SHARED_PREFETCH_P"] && cfg["REG_PREFETCH_P"]) {
    Rewriter::doubleBufferAdjust(pfLdSMForOps, pfLdRegForOps, pfLdRegForOps_, regRearForOp);
    LOG_DEBUG("===== doublePerfetchAdjust =======\n",module);
  }

  // Causal tile-level guard (transposed): wrap xBlockFors body in affine.if
  // Condition: bx - by + Bc - 1 >= 0 (i.e. bx < by + Bc, skip blocks where all K rows > all Q rows)
  if (cfg.count("CAUSAL_MASK") && cfg.at("CAUSAL_MASK")) {
    auto loc = builder.getUnknownLoc();
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    // d0 = bx (Q/inner block offset), d1 = by (K/outer block offset)
    // Condition: bx - by + Bc - 1 >= 0
    auto constraint = d0 - d1 + (int64_t)(cfg["Bc"] - 1);
    auto intSet = mlir::IntegerSet::get(2, 0, {constraint}, {false});

    builder.setInsertionPointToStart(xBlockFors[0].getBody());
    auto ifOp = builder.create<mlir::affine::AffineIfOp>(
        loc, intSet,
        mlir::ValueRange{xBlockFors[0].getInductionVar(), byIdx},
        false);

    auto* loopBlock = xBlockFors[0].getBody();
    auto* yieldOp = loopBlock->getTerminator();
    auto* ifYield = ifOp.getThenBlock()->getTerminator();
    std::vector<mlir::Operation*> opsToMove;
    for (auto& op : *loopBlock) {
      if (&op != ifOp.getOperation() && &op != yieldOp)
        opsToMove.push_back(&op);
    }
    for (auto* op : opsToMove) {
      op->moveBefore(ifYield);
    }
    LOG_DEBUG("===== causal tile-level guard (transposed) =======\n",module);
  }

  Rewriter::unrollAttribute(module, cfg["UNROLL_NUM"]);
  LOG_DEBUG("===== unrollAttribute =======\n",module);
}

}
