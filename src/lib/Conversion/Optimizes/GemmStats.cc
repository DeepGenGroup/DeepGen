#include "Conversion/Optimize.h"
#include <cmath>
#include <limits>

namespace KernelCodeGen {

// ======================================= global to sm =========================================
std::array<int64_t, 7> GemmStatsOptimizer::getCfgDatas(const std::string& bufType) {
  int64_t blockTileY = cfg.at("Br"), blockTileX = cfg.at("Slice1");
  int64_t isTran = this->isTranQ, globLoadWidth = cfg.at("GLOB_LOAD_WIDTH_Q");
  int64_t globLoadAllWidth = globLoadAllWidthQ;
  int64_t globLoadRowWidth = globLoadRowWidthQ;
  int64_t loadContinuous = cfg.at("LOAD_CONTINUOUS_P");
  if (isTran) {
    blockTileY = cfg.at("Slice1"); blockTileX = cfg.at("Br");
  }
  if (bufType == "K") {
    blockTileY = cfg.at("Slice1"); blockTileX = cfg.at("Bc");
    isTran = this->isTranK; globLoadWidth = cfg.at("GLOB_LOAD_WIDTH_K");
    globLoadAllWidth = globLoadAllWidthK;
    globLoadRowWidth = globLoadRowWidthK;
    if (isTran) {
      blockTileY = cfg.at("Bc"); blockTileX = cfg.at("Slice1");
    }
  }
  return {blockTileY, blockTileX, isTran, globLoadWidth, globLoadAllWidth, globLoadRowWidth, loadContinuous};
}

std::array<mlir::AffineExpr, 2> GemmStatsOptimizer::getGlobToSmExprs(const llvm::SmallVector<mlir::AffineExpr>& dims,
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

mlir::AffineMap GemmStatsOptimizer::getGlobQKToTempQKMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by, bx, tid, k, iter}
  auto args = getCfgDatas(bufType);
  mlir::AffineExpr row, col;
  row = dims[dimCount-5];                      // by
  col = dims[dimCount-2];                      // k - QK(slice1)
  if (args[2]) {
    row = dims[dimCount-2];                    // k
    col = dims[dimCount-5];                    // by
  }
  if (bufType == "K") {
    row = dims[dimCount-2];                      // k - QK(slice1)
    col = dims[dimCount-4];                      // bx
    if (args[2]) {
      row = dims[dimCount-4];                    // bx
      col = dims[dimCount-2];                    // k
    }
  }
  auto [tyIdx, txIdx] = getGlobToSmExprs({dims[dimCount-3], dims[dimCount-1]}, args);  // tid, iter
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<2; i++) {
    exprs.push_back(dims[i]);  // batch
  }
  exprs.push_back(row + tyIdx);
  exprs.push_back(col + txIdx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap GemmStatsOptimizer::getTempToSmMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 2;
  auto dims = getExprs(builder, dimCount); // {tid, iter}
  auto args = getCfgDatas(bufType);
  auto [tyIdx, txIdx] = getGlobToSmExprs(dims, args);
  if ((bufType == "Q" && !args[2]) || (bufType == "K" && args[2])) {
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

mlir::AffineMap GemmStatsOptimizer::getTempToSmQPrologueMap(mlir::OpBuilder& builder) {
  int dimCount = 3;
  auto dims = getExprs(builder, dimCount);
  auto args = getCfgDatas("Q");
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
std::array<int64_t, 8> GemmStatsOptimizer::getSmCfgDatas(const std::string& bufType) {
  int64_t blockLayoutY = cfg["BLOCK_LAYOUT_P_Y"], blockLayoutX = cfg["BLOCK_LAYOUT_P_X"];
  int64_t warpLayoutY = cfg["WARP_LAYOUT_P_Y"], warpLayoutX = cfg["WARP_LAYOUT_P_X"];
  int64_t blockScatterY = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatterY = cfg["WARP_SCATTER_WIDTH_Q"];
  int64_t blockScatterX = cfg["BLOCK_SCATTER_WIDTH_K"], warpScatterX = cfg["WARP_SCATTER_WIDTH_K"];
  return {blockLayoutY, blockLayoutX, warpLayoutY, warpLayoutX, blockScatterY, blockScatterX, warpScatterY, warpScatterX};
}

mlir::AffineMap GemmStatsOptimizer::getSmQKVToRegQKVMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 4;
  auto dims = getExprs(builder, dimCount);  // {tid, bk, blockRepIter, warpRepIter}
  auto args = getSmCfgDatas(bufType);
  mlir::AffineExpr widx, lidx;
  int64_t blockLayout, warpLayout, blockScatter, warpScatter;
  if (bufType == "Q") {
    widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], args[1]);
    lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], args[3]);
    blockLayout = args[0], warpLayout = args[2], blockScatter = args[4], warpScatter = args[6];
  } else if (bufType == "K") {
    widx = tools::mapUtils::wapr_x(dims[0], cfg["WARP_SIZE"], args[1]);
    lidx = tools::mapUtils::lane_x(dims[0], cfg["WARP_SIZE"], args[3]);
    blockLayout = args[1], warpLayout = args[3], blockScatter = args[5], warpScatter = args[7];
  }
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(dims[1]);
  exprs.push_back((dims[2] * blockLayout + widx) * warpLayout * blockScatter + (dims[3] * warpLayout + lidx) * warpScatter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap GemmStatsOptimizer::getSmQPrologueToRegQMap(mlir::OpBuilder& builder) {
  int dimCount = 5;
  auto dims = getExprs(builder, dimCount);
  auto args = getSmCfgDatas("Q");
  auto widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], args[1]);
  auto lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], args[3]);
  int64_t blockLayout = args[0], warpLayout = args[2], blockScatter = args[4], warpScatter = args[6];
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(dims[1] + dims[2]);
  exprs.push_back((dims[3] * blockLayout + widx) * warpLayout * blockScatter + (dims[4] * warpLayout + lidx) * warpScatter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap GemmStatsOptimizer::getCalculateMap(mlir::OpBuilder& builder, std::string calculatetype) {
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

mlir::AffineMap GemmStatsOptimizer::getRegSumAndMaxMap(mlir::OpBuilder& builder) {
  auto iter = builder.getAffineDimExpr(0);
  llvm::SmallVector<mlir::AffineExpr> exprs{iter};
  return mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

// ======================================= softmax block level ====================================
mlir::AffineMap GemmStatsOptimizer::getBlockLevelSmMap(mlir::OpBuilder& builder) {
  int dimCount = 4;  // {tid, blockrepeatq, warprepeatq, width}
  auto dims = getExprs(builder, dimCount);
  auto widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
  auto lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);
  int64_t blockLayout = cfg["BLOCK_LAYOUT_P_Y"], warpLayout = cfg["WARP_LAYOUT_P_Y"];
  int64_t blockScatter = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatter = cfg["WARP_SCATTER_WIDTH_Q"];
  mlir::AffineExpr blockExpr = (dims[1] * blockLayout + widx * blockScatter) * warpLayout;
  mlir::AffineExpr warpLevel = (dims[2] * warpLayout + lidx * warpScatter);
  llvm::SmallVector<mlir::AffineExpr> exprs{blockExpr + warpLevel + dims[3]};
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap GemmStatsOptimizer::getBlockLevelRegMap(mlir::OpBuilder& builder) {
  int dimCount = 3;  // {blockrepeatq, warprepeatq, width}
  auto dims = getExprs(builder, dimCount);
  int64_t blockScatter = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatter = cfg["WARP_SCATTER_WIDTH_Q"];
  mlir::AffineExpr expr = dims[0] + dims[1] + dims[2];
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext());
}

// ======================================= em/denom write =========================================
mlir::AffineMap GemmStatsOptimizer::getEmDenomWriteMap(mlir::OpBuilder& builder) {
  // {b1, b2, by, tid, blockRepQ, warpRepQ, width} -> {b1, b2, by + row_in_block, 0}
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);
  auto widx = tools::mapUtils::wapr_y(dims[3], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
  auto lidx = tools::mapUtils::lane_y(dims[3], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);
  int64_t blockLayout = cfg["BLOCK_LAYOUT_P_Y"], warpLayout = cfg["WARP_LAYOUT_P_Y"];
  int64_t blockScatter = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatter = cfg["WARP_SCATTER_WIDTH_Q"];
  mlir::AffineExpr blockExpr = (dims[4] * blockLayout + widx * blockScatter) * warpLayout;
  mlir::AffineExpr warpLevel = (dims[5] * warpLayout + lidx * warpScatter);
  mlir::AffineExpr row_in_block = blockExpr + warpLevel + dims[6];
  auto zero = builder.getAffineConstantExpr(0);
  llvm::SmallVector<mlir::AffineExpr> exprs{dims[0], dims[1], dims[2] + row_in_block, zero};
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

// ==================================== parsing and tuning =====================================

void GemmStatsOptimizer::computeTuneArgs() {
  this->blockPY = cfg.at("Br") / cfg.at("PTr");
  this->blockPX = cfg.at("Bc") / cfg.at("PTc");
  this->threadNum = blockPY * blockPX;
  this->blockRepeatQ = cfg.at("PTr") / cfg.at("BLOCK_SCATTER_WIDTH_Q");
  this->blockRepeatK = cfg.at("PTc") / cfg.at("BLOCK_SCATTER_WIDTH_K");
  this->warpRepeatQ = cfg.at("BLOCK_SCATTER_WIDTH_Q") / cfg.at("WARP_SCATTER_WIDTH_Q");
  this->warpRepeatK = cfg.at("BLOCK_SCATTER_WIDTH_K") / cfg.at("WARP_SCATTER_WIDTH_K");
  this->globLoadTotalWidthQ = cfg.at("Br") * cfg.at("Slice1") / this->threadNum;
  this->globLoadTotalWidthK = cfg.at("Bc") * cfg.at("Slice1") / this->threadNum;
  this->globLoadRowWidthQ = this->threadNum / cfg.at("Br") * cfg.at("GLOB_LOAD_WIDTH_Q");
  this->globLoadRowWidthK = this->threadNum / cfg.at("Slice1") * cfg.at("GLOB_LOAD_WIDTH_K");
  if (isTranQ) {
    this->globLoadRowWidthQ = this->threadNum / cfg.at("Slice1") * cfg.at("GLOB_LOAD_WIDTH_Q");
  }
  if (isTranK) {
    this->globLoadRowWidthK = this->threadNum / cfg.at("Bc") * cfg.at("GLOB_LOAD_WIDTH_K");
  }
  this->globLoadAllWidthQ = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_Q");
  this->globLoadAllWidthK = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_K");
}

void GemmStatsOptimizer::parseFuncArgs(mlir::func::FuncOp funcOp) {
  typeQ = mlir::dyn_cast<mlir::MemRefType>(Q.getType());
  typeK = mlir::dyn_cast<mlir::MemRefType>(K.getType());
  typeEmOut = mlir::dyn_cast<mlir::MemRefType>(EmOut.getType());
  typeDenomOut = mlir::dyn_cast<mlir::MemRefType>(DenomOut.getType());
  typeMid = mlir::dyn_cast<mlir::MemRefType>(midBuf.getType());
  std::vector<bool> isTrans;
  auto transArr = funcOp->getAttr(ARGTRAN);
  auto transArrAttr = mlir::dyn_cast<mlir::ArrayAttr>(transArr);
  for (auto tran : transArrAttr) {
    auto tranAttr = mlir::dyn_cast<mlir::IntegerAttr>(tran);
    isTrans.push_back(tranAttr.getInt());
  }
  isTranQ = isTrans[0]; isTranK = isTrans[1];
  auto shapeEmOut = typeEmOut.getShape();
  batchSize = shapeEmOut[0]; headNum = shapeEmOut[1];
  seqLen = shapeEmOut[2];
  auto shapeQ = typeQ.getShape();
  headDim = isTranQ ? shapeQ[2] : shapeQ[3];
}

bool GemmStatsOptimizer::applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) {
  this->cfg = config;
  mlir::ValueRange operands = funcOp.getArguments();
  this->Q = operands[0]; this->K = operands[1]; this->EmOut = operands[2]; this->DenomOut = operands[3];
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
      } else if (forDesc == "initBuf") {
        this->initBufFor = forOp;
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
      } else if (allocDesc == "Sum") {
        sumBuf = allocOp.getResult();
      } else if (allocDesc == "Max") {
        maxBuf = allocOp.getResult();
      }
    }
  });
  parseFuncArgs(funcOp);
  computeTuneArgs();
  return true;
}

// ======================================== reduce and broadcast =================================
std::vector<mlir::affine::AffineForOp> GemmStatsOptimizer::reduceAndBraodcast(mlir::Operation* localtionOp,
                                                                              const std::vector<mlir::Value>& regBufs,
                                                                              const std::vector<mlir::Value>& smBufs) {
  auto onlineSoftmax = [&](mlir::OpBuilder &b,
                           std::vector<mlir::Value> lds,
                           std::vector<mlir::Value> oldLds) -> std::vector<mlir::Value> {
      auto ldMax = lds[0], ldSum = lds[1], oldMax = oldLds[0], oldSum = oldLds[1];
      // max(ldMax, oldMax)
      auto newMax = b.create<mlir::arith::MaxNumFOp>(b.getUnknownLoc(), ldMax, oldMax);
      // exp(ldMax - newMax)
      auto sub1 = b.create<mlir::arith::SubFOp>(b.getUnknownLoc(), ldMax, newMax);
      auto exp1 = b.create<mlir::math::ExpOp>(b.getUnknownLoc(), sub1);
      // exp(oldMax - newMax)
      auto sub2 = b.create<mlir::arith::SubFOp>(b.getUnknownLoc(), oldMax, newMax);
      auto exp2 = b.create<mlir::math::ExpOp>(b.getUnknownLoc(), sub2);
      // ldSum * exp(ldMax - newMax) + oldSum * exp(oldMax - newMax)
      auto mul1 = b.create<mlir::arith::MulFOp>(b.getUnknownLoc(), ldSum, exp1);
      auto mul2 = b.create<mlir::arith::MulFOp>(b.getUnknownLoc(), oldSum, exp2);
      auto add = b.create<mlir::arith::AddFOp>(b.getUnknownLoc(), mul1, mul2);
      return std::vector<mlir::Value>{newMax, add, exp2};
  };
  int64_t ydim = cfg["PTr"], width = this->blockPX;
  mlir::Value tid = this->threadIdx.getIVs()[0];
  mlir::OpBuilder builder = getBuilder(localtionOp, Position::after);
  auto warpLevelForOp = warpReduce(builder, ydim, width, regBufs, onlineSoftmax);
  auto blockLevelForOp = blockReduce(builder, ydim, width, tid, regBufs, smBufs, onlineSoftmax);
  auto bcForOp = warpBroadcast(builder, ydim, width, /*buf*/{regBufs[0]}, /*index*/0);
  return std::vector<mlir::affine::AffineForOp>{warpLevelForOp, blockLevelForOp, bcForOp};
}

void GemmStatsOptimizer::moveMemrefDefineAhead(mlir::Operation* threadParallelOp){
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

void GemmStatsOptimizer::applyOptimzer(mlir::func::FuncOp& funcOp) {
  mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(funcOp->getParentOp());
  mlir::OpBuilder builder(module);

  // tileP bufferize (no tileO for GemmStats)
  std::vector<mlir::affine::AffineForOp> tilePLoops{yTileForOps[0], xTileForOps[0]};
  auto tileP = Rewriter::bufferizeLoopCarryVar(kForOps[0], tilePLoops, MemorySpace::local, {"tileP"});
  LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

  // k1 split and reorder
  auto k1 = Rewriter::split(kForOps[0], {cfg.at("Slice1")});
  auto k1_outer = k1[0], k1_inner = k1[1];
  Rewriter::reorder({k1_outer, k1_inner, yTileForOps[0], xTileForOps[0]});
  LOG_DEBUG("===== after split & reorder K1 =======\n",module);

  // Inline buffer allocation (smQ, smK, smFactor + tempQ, tempK, regQ, regK)
  auto dtypeQ = typeQ.getElementType();
  auto dtypeK = typeK.getElementType();
  auto dtypeMid = typeMid.getElementType();

  std::vector<std::vector<int64_t>> smShapes{
    {cfg.at("Slice1"), cfg.at("Br")}, {cfg.at("Slice1"), cfg.at("Bc")}, {cfg.at("Br")}
  };
  std::vector<mlir::Type> smType{dtypeQ, dtypeK, dtypeMid};
  std::vector<std::string> smDescs{"smQ", "smK", "smFactor"};
  auto sm = Rewriter::allocBuffers(smShapes, smType, MemorySpace::shared, smDescs, blockIdx);
  auto smQ = sm[0], smK = sm[1], smFactor = sm[2];

  std::vector<std::vector<int64_t>> regShapes{
    {globLoadTotalWidthQ}, {globLoadTotalWidthK}, {cfg.at("PTr")}, {cfg.at("PTc")}
  };
  std::vector<mlir::Type> regDTypes{dtypeQ, dtypeK, dtypeQ, dtypeK};
  std::vector<std::string> regDescs{"tempQ", "tempK", "regQ", "regK"};
  auto reg = Rewriter::allocBuffers(regShapes, regDTypes, MemorySpace::local, regDescs, this->xBlockFors[0]);
  auto tempQ = reg[0], tempK = reg[1], regQ = reg[2], regK = reg[3];
  LOG_DEBUG("===== after alloc_buffer =======\n",module);

  // smQFull allocation
  mlir::Value smQFull;
  if (cfg["Slice1"] == cfg["Hd"]) {
    smQFull = smQ;
  } else {
    smQFull = Rewriter::allocBuffers({{cfg["Hd"], cfg["Br"]}},
                                      {typeQ.getElementType()},
                                      MemorySpace::shared, {"smQFull"}, blockIdx)[0];
  }
  LOG_DEBUG("===== after alloc smQFull =======\n",module);

  auto bIdx = Analyzer::getParallelIdx(this->blockIdx);
  auto tIdx = Analyzer::getParallelIdx(this->threadIdx);

  // ====== Q PROLOGUE: load entire Q tile into smQFull[Hd×Br] before the key-block loop ======
  auto loadTileQMap = getGlobQKToTempQKMap(builder, "Q");
  auto prologueStoreQMap = getTempToSmQPrologueMap(builder);
  {
    mlir::OpBuilder prologueBuilder = getBuilder(xBlockFors[0], Position::before);
    auto loc = prologueBuilder.getUnknownLoc();
    auto c0 = prologueBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    llvm::SmallVector<int64_t> plbs{0}, pubs{cfg["Hd"]}, psteps{cfg["Slice1"]};
    auto [prologueForOps, prologueIvs] = createNestedLoops(prologueBuilder, plbs, pubs, psteps);
    auto k_prologue = prologueForOps[0];

    llvm::SmallVector<mlir::Value> qPrologueGlobOps(bIdx.begin(), bIdx.end()-1);
    qPrologueGlobOps.push_back(byIdx);
    qPrologueGlobOps.push_back(c0);
    qPrologueGlobOps.push_back(tIdx[0]);
    qPrologueGlobOps.push_back(prologueIvs[0]);

    auto loadTileQ_p = Rewriter::loadToRegisters(Q, tempQ, loadTileQMap, qPrologueGlobOps,
                                                  {cfg["GLOB_LOAD_WIDTH_Q"]}, k_prologue, Position::begin, "");

    mlir::affine::AffineForOp lastBeforeStoreQ_p = loadTileQ_p;
    if (cfg.count("SCALE_Q") && cfg.at("SCALE_Q")) {
      auto dtype = typeQ.getElementType();
      float scaleVal = 1.0f / std::sqrt(static_cast<float>(headDim));
      mlir::OpBuilder sb(loadTileQ_p->getBlock(), ++mlir::Block::iterator(loadTileQ_p.getOperation()));
      auto sLoc = sb.getUnknownLoc();
      auto scaleConst = sb.create<mlir::arith::ConstantOp>(sLoc, sb.getFloatAttr(dtype, scaleVal));
      auto scaleBody = [&](mlir::OpBuilder &bb, mlir::Location l, mlir::Value iv, mlir::ValueRange iterArgs) {
        auto ld = bb.create<mlir::affine::AffineLoadOp>(l, tempQ, mlir::ValueRange{iv});
        auto scaled = bb.create<mlir::arith::MulFOp>(l, ld, scaleConst);
        bb.create<mlir::affine::AffineStoreOp>(l, scaled, tempQ, mlir::ValueRange{iv});
        bb.create<mlir::affine::AffineYieldOp>(l);
      };
      lastBeforeStoreQ_p = sb.create<mlir::affine::AffineForOp>(sLoc, 0, globLoadTotalWidthQ, 1, mlir::ValueRange{}, scaleBody);
      LOG_DEBUG("===== fused Q scale into prologue =======\n",module);
    }

    Rewriter::loadFromRegisters(tempQ, smQFull, prologueStoreQMap,
                                {tIdx[0], prologueIvs[0]},
                                {cfg["GLOB_LOAD_WIDTH_Q"]}, lastBeforeStoreQ_p, Position::after, "");
    Rewriter::barrier(k_prologue, Position::after);
  }
  LOG_DEBUG("===== Q prologue done =======\n",module);

  // ====== K loading: only K is loaded inside k1_outer (Q is pre-loaded) ======
  llvm::SmallVector<mlir::Value> operands(bIdx.begin(), bIdx.end()-1);
  operands.push_back(byIdx); operands.push_back(xBlockFors[0].getInductionVar()); operands.push_back(tIdx[0]);
  auto loadTileKMap = getGlobQKToTempQKMap(builder, "K");
  llvm::SmallVector<mlir::Value> kGlobOperands(operands);
  kGlobOperands.push_back(k1_outer.getInductionVar());
  auto loadTileK = Rewriter::loadToRegisters(K, tempK, loadTileKMap, kGlobOperands, {cfg["GLOB_LOAD_WIDTH_K"]}, k1_outer, Position::begin, "");
  LOG_DEBUG("===== after read K =======\n",module);
  // temp K to shared K
  auto storeTileKMap = getTempToSmMap(builder, "K");   // {tid, iter}
  auto storeTileK = Rewriter::loadFromRegisters(tempK, smK, storeTileKMap, {tIdx[0]}, {cfg["GLOB_LOAD_WIDTH_K"]}, loadTileK, Position::after, "");
  auto prefix = Rewriter::barrier(loadTileK, Position::before);
  auto suffix = Rewriter::barrier(storeTileK, Position::after);
  LOG_DEBUG("===== write K =======\n",module);

  // sm Q (pre-loaded with k_outer offset) and sm K to registers
  auto loadFragQMap = getSmQPrologueToRegQMap(builder);
  auto loadFragKMap = getSmQKVToRegQKVMap(builder, "K");    // {tid, bk, blockRepIter, warpRepIter}
  llvm::SmallVector<mlir::Value> qFragOperands{tIdx[0], k1_outer.getInductionVar(), k1_inner.getInductionVar()};
  llvm::SmallVector<mlir::Value> kFragOperands{tIdx[0], k1_inner.getInductionVar()};
  std::vector<int64_t> widthsQ{cfg["BLOCK_SCATTER_WIDTH_Q"], cfg["WARP_SCATTER_WIDTH_Q"]};
  std::vector<int64_t> widthsK{cfg["BLOCK_SCATTER_WIDTH_K"], cfg["WARP_SCATTER_WIDTH_K"]};
  auto loadFragQ = Rewriter::loadToRegisters(smQFull, regQ, loadFragQMap, qFragOperands, widthsQ, k1_inner, Position::begin, "");
  auto loadFragK = Rewriter::loadToRegisters(smK, regK, loadFragKMap, kFragOperands, widthsK, loadFragQ, Position::after, "");
  LOG_DEBUG("===== read sh_Q/K =======\n",module);

  // matmul1 micro-kernel: regQ × regK → tileP
  auto calMap = getCalculateMap(builder, "matmul");  // {iter}
  Rewriter::cache_read(xTileForOps[0], Q, regQ, calMap, {yTileForOps[0].getInductionVar()});
  Rewriter::cache_read(xTileForOps[0], K, regK, calMap, {xTileForOps[0].getInductionVar()});
  LOG_DEBUG("===== load regQ & cache_read =======\n",module);

  // separateNoOpRelyForOp
  auto tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
  auto txds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttilexDown");
  std::vector<mlir::affine::AffineForOp> forOps{tyds[0], txds[0]};
  Rewriter::separateNoOpRelyForOp(forOps);
  LOG_DEBUG("===== separateNoOpRelyForOp =======\n",module);

  // SCALE_SCORES (optional): tileP[i][j] *= 1/sqrt(headDim)
  if (cfg.count("SCALE_SCORES") && cfg.at("SCALE_SCORES")) {
    tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
    auto dtype = typeMid.getElementType();
    float scaleVal = 1.0f / std::sqrt(static_cast<float>(headDim));
    mlir::OpBuilder sb(tyds[0]);
    auto loc = sb.getUnknownLoc();
    auto scaleConst = sb.create<mlir::arith::ConstantOp>(loc, sb.getFloatAttr(dtype, scaleVal));
    llvm::SmallVector<int64_t> lbs{0, 0}, ubs{cfg["PTr"], cfg["PTc"]}, steps{1, 1};
    auto [scaleForOps, scaleIvs] = createNestedLoops(sb, lbs, ubs, steps);
    sb.setInsertionPointToStart(scaleForOps.back().getBody());
    auto ld = sb.create<mlir::affine::AffineLoadOp>(loc, tileP[0], scaleIvs);
    auto scaled = sb.create<mlir::arith::MulFOp>(loc, ld, scaleConst);
    sb.create<mlir::affine::AffineStoreOp>(loc, scaled, tileP[0], scaleIvs);
    LOG_DEBUG("===== scale scores after mm1 =======\n",module);
  }

  // Fused SOFTCAP_TANH + CAUSAL_MASK (single loop over tileP)
  {
    bool doSoftcap = cfg.count("SOFTCAP_TANH") && cfg.at("SOFTCAP_TANH");
    bool doMask = cfg.count("CAUSAL_MASK") && cfg.at("CAUSAL_MASK");
    if (doSoftcap || doMask) {
      tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
      auto dtype = typeMid.getElementType();
      mlir::OpBuilder sb(tyds[0]);
      auto loc = sb.getUnknownLoc();

      mlir::Value invScaleConst, scaleConst, negInf;
      if (doSoftcap) {
        invScaleConst = sb.create<mlir::arith::ConstantOp>(loc, sb.getFloatAttr(dtype, 1.0f / 50.0f));
        scaleConst = sb.create<mlir::arith::ConstantOp>(loc, sb.getFloatAttr(dtype, 50.0f));
      }
      if (doMask) {
        negInf = sb.create<mlir::arith::ConstantOp>(loc, sb.getFloatAttr(dtype, -1.0e30));
      }

      llvm::SmallVector<int64_t> lbs{0, 0}, ubs{cfg["PTr"], cfg["PTc"]}, steps{1, 1};
      auto [forOps, ivs] = createNestedLoops(sb, lbs, ubs, steps);
      sb.setInsertionPointToStart(forOps.back().getBody());

      auto ld = sb.create<mlir::affine::AffineLoadOp>(loc, tileP[0], ivs);
      mlir::Value val = ld;

      if (doSoftcap) {
        auto divided = sb.create<mlir::arith::MulFOp>(loc, val, invScaleConst);
        auto tanhed = sb.create<mlir::math::TanhOp>(loc, divided);
        val = sb.create<mlir::arith::MulFOp>(loc, tanhed, scaleConst);
      }

      if (doMask) {
        int dimCount = 5;
        auto dims = getExprs(sb, dimCount);
        auto iE = dims[0], jE = dims[1], tidE = dims[2], byE = dims[3], bxE = dims[4];

        int64_t BSW_Q = cfg["BLOCK_SCATTER_WIDTH_Q"], WSW_Q = cfg["WARP_SCATTER_WIDTH_Q"];
        int64_t BLP_Y = cfg["BLOCK_LAYOUT_P_Y"], WLP_Y = cfg["WARP_LAYOUT_P_Y"];
        int64_t BLP_X = cfg["BLOCK_LAYOUT_P_X"], WLP_X = cfg["WARP_LAYOUT_P_X"];
        int64_t WARP_SZ = cfg["WARP_SIZE"];
        int64_t BSW_K = cfg["BLOCK_SCATTER_WIDTH_K"], WSW_K = cfg["WARP_SCATTER_WIDTH_K"];

        auto warp_y = tools::mapUtils::wapr_y(tidE, WARP_SZ, BLP_X);
        auto lane_y = tools::mapUtils::lane_y(tidE, WARP_SZ, WLP_X);
        auto rowInBr = (iE.floorDiv(BSW_Q) * BLP_Y + warp_y) * WLP_Y * BSW_Q
                       + ((iE % BSW_Q).floorDiv(WSW_Q) * WLP_Y + lane_y) * WSW_Q + iE % WSW_Q;
        auto globalRow = byE + rowInBr;

        auto warp_x = tools::mapUtils::wapr_x(tidE, WARP_SZ, BLP_X);
        auto lane_x = tools::mapUtils::lane_x(tidE, WARP_SZ, WLP_X);
        auto colInBc = (jE.floorDiv(BSW_K) * BLP_X + warp_x) * WLP_X * BSW_K
                       + ((jE % BSW_K).floorDiv(WSW_K) * WLP_X + lane_x) * WSW_K + jE % WSW_K;
        auto globalCol = bxE + colInBc;

        auto rowMap = mlir::AffineMap::get(dimCount, 0, {globalRow}, sb.getContext());
        auto colMap = mlir::AffineMap::get(dimCount, 0, {globalCol}, sb.getContext());

        mlir::Value tid = this->threadIdx.getIVs()[0];
        mlir::Value bx = xBlockFors[0].getInductionVar();
        llvm::SmallVector<mlir::Value> mapOps{ivs[0], ivs[1], tid, byIdx, bx};

        auto rowVal = sb.create<mlir::affine::AffineApplyOp>(loc, rowMap, mapOps);
        auto colVal = sb.create<mlir::affine::AffineApplyOp>(loc, colMap, mapOps);

        auto cmp = sb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt,
                                                    colVal.getResult(), rowVal.getResult());
        val = sb.create<mlir::arith::SelectOp>(loc, cmp, negInf, val);
      }

      sb.create<mlir::affine::AffineStoreOp>(loc, val, tileP[0], ivs);
      LOG_DEBUG("===== fused softcap+mask =======\n",module);
    }
  }

  // smMax/smSum/regMax/regSum initialization
  // smMax/smSum: at threadIdx scope (shared, init once per block)
  // regMax/regSum: ALSO at threadIdx scope (local, init once per thread BEFORE blockx loop)
  // This lets the online reduce accumulate across ALL blockx iterations.
  auto smMaxAndSum = Rewriter::createHierarchyInitBuf(initBufFor, {cfg["Br"]}, threadIdx, MemorySpace::shared);
  auto regMaxAndSum = Rewriter::createHierarchyInitBuf(initBufFor, {cfg["PTr"]}, threadIdx, MemorySpace::local);
  auto smMax = smMaxAndSum[0], smSum = smMaxAndSum[1], regMax = regMaxAndSum[0], regSum = regMaxAndSum[1];
  LOG_DEBUG("===== createSMAndRegInitBuf =======\n",module);

  // Cache read/write for max/sum: replace maxBuf/sumBuf with regMax/regSum
  auto sumAndMaxRegMap = getRegSumAndMaxMap(builder);
  tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
  txds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttilexDown");
  Rewriter::cache_read(txds[0], maxBuf, regMax, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  Rewriter::cache_read(txds[0], sumBuf, regSum, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  Rewriter::cache_write(txds[0], maxBuf, regMax, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  Rewriter::cache_write(txds[0], sumBuf, regSum, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  LOG_DEBUG("===== amend thread level load and store of max and sum =======\n",module);

  // reduceAndBraodcast: AFTER blockx loop (not inside it!)
  // Each thread has accumulated partial max/sum across all blockx iterations.
  // Now combine across threads via warp/block reduce.
  auto softmaxForOps = reduceAndBraodcast(xBlockFors[0].getOperation(), {regMax, regSum}, {smMax, smSum, smFactor});
  LOG_DEBUG("===== add warp level and block level ops of max and sum =======\n",module);

  // blockLevel reduce split and map patching
  auto blForOps = Rewriter::split(softmaxForOps[1], widthsQ);
  auto blSmMap = getBlockLevelSmMap(builder);
  auto blRegMap = getBlockLevelRegMap(builder);
  llvm::SmallVector<mlir::Value> blRegOperands;
  for (auto bl : blForOps) { blRegOperands.push_back(bl.getInductionVar()); }
  llvm::SmallVector<mlir::Value> blSmOperands(blRegOperands);
  blSmOperands.insert(blSmOperands.begin(), tIdx[0]);
  Rewriter::cache_read(blForOps.back(), smMax, smMax, blSmMap, blSmOperands);
  Rewriter::cache_read(blForOps.back(), smSum, smSum, blSmMap, blSmOperands);
  Rewriter::cache_read(blForOps.back(), regMax, regMax, blRegMap, blRegOperands);
  Rewriter::cache_read(blForOps.back(), regSum, regSum, blRegMap, blRegOperands);
  Rewriter::cache_write(blForOps.back(), smMax, smMax, blSmMap, blSmOperands);
  Rewriter::cache_write(blForOps.back(), smSum, smSum, blSmMap, blSmOperands);
  Rewriter::cache_write(blForOps.back(), smFactor, smFactor, blSmMap, blSmOperands);
  Rewriter::cache_write(blForOps.back(), regMax, regMax, blRegMap, blRegOperands);
  LOG_DEBUG("===== split block level forop and amend map of reg and sm =======\n",module);

  // ====== Write em and denom to global memory ======
  {
    auto emDenomMap = getEmDenomWriteMap(builder);
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto d2 = builder.getAffineDimExpr(2);
    auto regIdxExpr = d0 * cfg["BLOCK_SCATTER_WIDTH_Q"] + d1 * cfg["WARP_SCATTER_WIDTH_Q"] + d2;
    auto regIdxMap = mlir::AffineMap::get(3, 0, {regIdxExpr}, builder.getContext());

    mlir::OpBuilder emBuilder = getBuilder(blForOps.front(), Position::after);
    auto loc = emBuilder.getUnknownLoc();

    llvm::SmallVector<int64_t> lbs{0, 0, 0};
    llvm::SmallVector<int64_t> ubs{blockRepeatQ, warpRepeatQ, cfg["WARP_SCATTER_WIDTH_Q"]};
    llvm::SmallVector<int64_t> steps{1, 1, 1};
    auto [writeForOps, writeIvs] = createNestedLoops(emBuilder, lbs, ubs, steps);
    emBuilder.setInsertionPointToStart(writeForOps.back().getBody());

    auto regIdx = emBuilder.create<mlir::affine::AffineApplyOp>(loc, regIdxMap, writeIvs);

    // em = exp(regMax[regIdx])
    auto maxVal = emBuilder.create<mlir::affine::AffineLoadOp>(loc, regMax, mlir::ValueRange{regIdx.getResult()});
    auto emVal = emBuilder.create<mlir::math::ExpOp>(loc, maxVal);

    // denom = regSum[regIdx]
    auto sumVal = emBuilder.create<mlir::affine::AffineLoadOp>(loc, regSum, mlir::ValueRange{regIdx.getResult()});

    // Build operands: {b1, b2, by, tid, blockRepQ, warpRepQ, width}
    auto bivs = this->blockIdx.getIVs();
    llvm::SmallVector<mlir::Value> emOperands(bivs.rbegin(), bivs.rend()-1);
    emOperands.push_back(byIdx);
    emOperands.push_back(tIdx[0]);
    for (auto& iv : writeIvs) emOperands.push_back(iv);

    emBuilder.create<mlir::affine::AffineStoreOp>(loc, emVal, EmOut, emDenomMap, emOperands);
    emBuilder.create<mlir::affine::AffineStoreOp>(loc, sumVal, DenomOut, emDenomMap, emOperands);
  }
  LOG_DEBUG("===== write em and denom to global =======\n",module);

  // Cleanup: erase initBufFor, midBuf, sumBuf, maxBuf
  this->initBufFor.erase();
  this->midBuf.getDefiningOp()->erase();
  this->sumBuf.getDefiningOp()->erase();
  this->maxBuf.getDefiningOp()->erase();
  LOG_DEBUG("===== cleanup done =======\n",module);

  // moveMemrefDefineAhead
  mlir::affine::AffineParallelOp threadParallelOp;
  funcOp.walk([&](mlir::affine::AffineParallelOp p){
    auto attr = p.getOperation()->getAttr(AttrGPUIndex);
    auto stringattr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if(std::string(stringattr.data()) == THREADIDX){
      threadParallelOp = p;
      return mlir::WalkResult::interrupt();
    }
  });
  moveMemrefDefineAhead(threadParallelOp.getOperation());
  LOG_DEBUG("===== moveMemrefDefineAhead =======\n",module);

  // Prefetch support (K loading only, Q is pre-loaded)
  mlir::affine::AffineForOp regRearForOp;
  std::vector<mlir::affine::AffineForOp> pfLdRegForOps, pfLdSMForOps, pfLdRegForOps_;
  if (cfg["SHARED_PREFETCH_P"]) {
    std::vector<mlir::affine::AffineForOp> LdRegForOps{loadTileK}, ldSMForOps{storeTileK};
    std::vector<mlir::Value> smBufs{smK};
    int64_t prefetchStep = cfg.at("Slice1");
    auto smResult = Rewriter::sharedMemroyPrefetch(k1_outer, LdRegForOps, ldSMForOps, k1_inner, smBufs);
    smK = smBufs[0];
    loadTileK = LdRegForOps[0];
    storeTileK = ldSMForOps[0];
    pfLdRegForOps = smResult.first; pfLdSMForOps = smResult.second;
    LOG_DEBUG("===== sharedMemroyPrefetch (K only) =======\n",module);

    // Fix smQFull k_outer IV shift after prefetch changes the loop range
    mlir::Value newKOuterIv = k1_outer.getInductionVar();
    auto patchSmQFullMap = [&](auto loadOp) {
      if (loadOp.getMemRef() != smQFull) return;
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
    loadFragQ.walk([&](mlir::affine::AffineVectorLoadOp vlop) { patchSmQFullMap(vlop); });
    loadFragQ.walk([&](mlir::affine::AffineLoadOp lop)        { patchSmQFullMap(lop); });
    LOG_DEBUG("===== fix smQFull k_outer IV shift =======\n",module);
  }

  if (cfg["REG_PREFETCH_P"]) {
    std::vector<mlir::affine::AffineForOp> regLdRegForOps{loadFragQ, loadFragK};
    std::vector<mlir::Value> regBufs{regQ, regK};
    auto regResult = Rewriter::registersPrefetch(k1_inner, regLdRegForOps, yTileForOps[0], regBufs);
    regQ = regBufs[0], regK = regBufs[1];
    loadFragQ = regLdRegForOps[0], loadFragK = regLdRegForOps[1];
    pfLdRegForOps_ = regResult.first; regRearForOp = regResult.second;
    LOG_DEBUG("===== registersPrefetch =======\n",module);
  }

  if (cfg["SHARED_PREFETCH_P"] && cfg["REG_PREFETCH_P"]) {
    Rewriter::doubleBufferAdjust(pfLdSMForOps, pfLdRegForOps, pfLdRegForOps_, regRearForOp);
    LOG_DEBUG("===== doublePerfetchAdjust =======\n",module);
  }

  // Causal tile-level guard: wrap xBlockFors body in affine.if (bx < by + Br)
  if (cfg.count("CAUSAL_MASK") && cfg.at("CAUSAL_MASK")) {
    auto loc = builder.getUnknownLoc();
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto constraint = d0 - d1 + (int64_t)(cfg["Br"] - 1);
    auto intSet = mlir::IntegerSet::get(2, 0, {constraint}, {false});

    builder.setInsertionPointToStart(xBlockFors[0].getBody());
    auto ifOp = builder.create<mlir::affine::AffineIfOp>(
        loc, intSet,
        mlir::ValueRange{byIdx, xBlockFors[0].getInductionVar()},
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
    LOG_DEBUG("===== causal tile-level guard =======\n",module);
  }

  Rewriter::unrollAttribute(module, cfg["UNROLL_NUM"]);
  LOG_DEBUG("===== unrollAttribute =======\n",module);
}

}
