#include "Conversion/Optimize.h"
#include <cmath>
#include <limits>

namespace KernelCodeGen {

// ======================================= global to sm =========================================
std::array<int64_t, 7> FlashAttnSplitK2Optimizer::getCfgDatas(const std::string& bufType) {
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
  } else if (bufType == "V") {
    blockTileY = cfg.at("Slice2"); blockTileX = cfg.at("Hd");
    isTran = this->isTranV; globLoadWidth = cfg.at("GLOB_LOAD_WIDTH_V");
    globLoadAllWidth = globLoadAllWidthV;
    globLoadRowWidth = globLoadRowWidthV;
    loadContinuous = cfg.at("LOAD_CONTINUOUS_O");
    if (isTran) {
      blockTileY = cfg.at("Hd"); blockTileX = cfg.at("Slice2");
    }
  }
  return {blockTileY, blockTileX, isTran, globLoadWidth, globLoadAllWidth, globLoadRowWidth, loadContinuous};
}

std::array<mlir::AffineExpr, 2> FlashAttnSplitK2Optimizer::getGlobToSmExprs(const llvm::SmallVector<mlir::AffineExpr>& dims,
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

mlir::AffineMap FlashAttnSplitK2Optimizer::getGlobQKToTempQKMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by, bx, tid, k, iter}
  auto args = getCfgDatas(bufType);
  mlir::AffineExpr row, col;
  row = dims[dimCount-5];
  col = dims[dimCount-2];
  if (args[2]) {
    row = dims[dimCount-2];
    col = dims[dimCount-5];
  }
  if (bufType == "K") {
    row = dims[dimCount-2];
    col = dims[dimCount-4];
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

mlir::AffineMap FlashAttnSplitK2Optimizer::getGlobVToTempVMap(mlir::OpBuilder& builder) {
  int dimCount = 6;
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by(bx), tid, k, iter}
  auto args = getCfgDatas("V");
  mlir::AffineExpr row = dims[dimCount-4] + dims[dimCount-2];
  auto [tyIdx, txIdx] = getGlobToSmExprs({dims[dimCount-3], dims[dimCount-1]}, args);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<2; i++) {
    exprs.push_back(dims[i]);
  }
  exprs.push_back(row + tyIdx);
  exprs.push_back(txIdx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap FlashAttnSplitK2Optimizer::getTempToSmMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 2;
  auto dims = getExprs(builder, dimCount); // {tid, iter}
  auto args = getCfgDatas(bufType);
  auto [tyIdx, txIdx] = getGlobToSmExprs(dims, args);
  if ((bufType == "Q" && !args[2]) || (bufType == "K" && args[2]) || (bufType == "V" && args[2])) {
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

mlir::AffineMap FlashAttnSplitK2Optimizer::getTempToSmQPrologueMap(mlir::OpBuilder& builder) {
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
std::array<int64_t, 8> FlashAttnSplitK2Optimizer::getSmCfgDatas(const std::string& bufType) {
  int64_t blockLayoutY = cfg["BLOCK_LAYOUT_P_Y"], blockLayoutX = cfg["BLOCK_LAYOUT_P_X"];
  int64_t warpLayoutY = cfg["WARP_LAYOUT_P_Y"], warpLayoutX = cfg["WARP_LAYOUT_P_X"];
  int64_t blockScatterY = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatterY = cfg["WARP_SCATTER_WIDTH_Q"];
  int64_t blockScatterX = cfg["BLOCK_SCATTER_WIDTH_K"], warpScatterX = cfg["WARP_SCATTER_WIDTH_K"];
  if (bufType == "P" || bufType == "V") {
    blockLayoutY = cfg["BLOCK_LAYOUT_O_Y"], blockLayoutX = cfg["BLOCK_LAYOUT_O_X"];
    warpLayoutY = cfg["WARP_LAYOUT_O_Y"], warpLayoutX = cfg["WARP_LAYOUT_O_X"];
    blockScatterY = cfg["BLOCK_SCATTER_WIDTH_P"], warpScatterY = cfg["WARP_SCATTER_WIDTH_P"];
    blockScatterX = cfg["BLOCK_SCATTER_WIDTH_V"], warpScatterX = cfg["WARP_SCATTER_WIDTH_V"];
  }
  return {blockLayoutY, blockLayoutX, warpLayoutY, warpLayoutX, blockScatterY, blockScatterX, warpScatterY, warpScatterX};
}

mlir::AffineMap FlashAttnSplitK2Optimizer::getSmQKVToRegQKVMap(mlir::OpBuilder& builder, const std::string& bufType) {
  int dimCount = 4;
  auto dims = getExprs(builder, dimCount);  // {tid, bk, blockRepIter, warpRepIter}
  auto args = getSmCfgDatas(bufType);
  mlir::AffineExpr widx, lidx;
  int64_t blockLayout, warpLayout, blockScatter, warpScatter;
  if (bufType == "Q") {
    widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], args[1]);
    lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], args[3]);
    blockLayout = args[0], warpLayout = args[2], blockScatter = args[4], warpScatter = args[6];
  } else if (bufType == "K" || bufType == "V") {
    widx = tools::mapUtils::wapr_x(dims[0], cfg["WARP_SIZE"], args[1]);
    lidx = tools::mapUtils::lane_x(dims[0], cfg["WARP_SIZE"], args[3]);
    blockLayout = args[1], warpLayout = args[3], blockScatter = args[5], warpScatter = args[7];
  }
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(dims[1]);
  exprs.push_back((dims[2] * blockLayout + widx) * warpLayout * blockScatter + (dims[3] * warpLayout + lidx) * warpScatter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap FlashAttnSplitK2Optimizer::getSmQPrologueToRegQMap(mlir::OpBuilder& builder) {
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

mlir::AffineMap FlashAttnSplitK2Optimizer::getSmPToRegPMap(mlir::OpBuilder& builder) {
  int dimCount = 6;
  auto dims = getExprs(builder, dimCount);  // {tid, kmid, bk, blockRepIter, warpRepIter, iter}
  auto args = getSmCfgDatas("P");
  int64_t blockLayout = args[0], warpLayout = args[2], blockScatter = args[4], warpScatter = args[6];
  mlir::AffineExpr widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], args[1]);
  mlir::AffineExpr lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], args[3]);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back((dims[3] * blockLayout + widx) * warpLayout * blockScatter + (dims[4] * warpLayout + lidx) * warpScatter + dims[5]);
  exprs.push_back(dims[1] + dims[2]);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::affine::AffineForOp FlashAttnSplitK2Optimizer::generateShufflePToRegP(
    mlir::OpBuilder& builder, mlir::Value tileP, mlir::Value regP,
    mlir::Value tid, mlir::Value k2_midder_iv, mlir::Value k2_inner_iv,
    mlir::affine::AffineForOp k2_inner) {
  int64_t WLPX = cfg["WARP_LAYOUT_P_X"];
  int64_t WSWK = cfg["WARP_SCATTER_WIDTH_K"];
  int64_t WARP_SZ = cfg["WARP_SIZE"];
  int64_t stride = WLPX * WSWK;
  int64_t PTr_ = cfg["PTr"];

  builder.setInsertionPointToStart(k2_inner.getBody());
  auto loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();

  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto d2 = builder.getAffineDimExpr(2);
  auto kExpr = d0 + d1;

  auto laneY   = (d2 % WARP_SZ).floorDiv(WLPX);
  auto ownerLx = (kExpr % stride).floorDiv(WSWK);
  auto srcExpr = laneY * WLPX + ownerLx;
  auto srcMap  = mlir::AffineMap::get(3, 0, {srcExpr}, builder.getContext());
  auto srcIdx  = builder.create<mlir::affine::AffineApplyOp>(
      loc, srcMap, mlir::ValueRange{k2_midder_iv, k2_inner_iv, tid});
  auto srcI32  = builder.create<mlir::arith::IndexCastOp>(loc, i32Ty, srcIdx.getResult());
  auto widthI32 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32IntegerAttr(WARP_SZ));

  auto colExpr = kExpr.floorDiv(stride) * WSWK + kExpr % WSWK;
  auto colMap  = mlir::AffineMap::get(2, 0, {colExpr}, builder.getContext());
  auto colIdx  = builder.create<mlir::affine::AffineApplyOp>(
      loc, colMap, mlir::ValueRange{k2_midder_iv, k2_inner_iv});

  auto wrapBody = [&](mlir::OpBuilder &b, mlir::Location l,
                      mlir::Value /*iv*/, mlir::ValueRange) {
    for (int64_t r = 0; r < PTr_; ++r) {
      auto rConst = b.create<mlir::arith::ConstantIndexOp>(l, r);
      auto pElem = b.create<mlir::affine::AffineLoadOp>(
          l, tileP, mlir::ValueRange{rConst, colIdx.getResult()});
      auto shuffled = b.create<mlir::gpu::ShuffleOp>(
          l, pElem.getResult(), srcI32, widthI32, mlir::gpu::ShuffleMode::IDX);
      b.create<mlir::affine::AffineStoreOp>(l, shuffled.getResult(0), regP,
                                            mlir::ValueRange{rConst});
    }
    b.create<mlir::affine::AffineYieldOp>(l);
  };
  return builder.create<mlir::affine::AffineForOp>(
      loc, 0, 1, 1, mlir::ValueRange{}, wrapBody);
}

mlir::AffineMap FlashAttnSplitK2Optimizer::getCalculateMap(mlir::OpBuilder& builder, std::string calculatetype) {
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

// ======================================= store tile ============================================
mlir::AffineMap FlashAttnSplitK2Optimizer::getTilePToSmPMap(mlir::OpBuilder& builder) {
  // {tid, blockRepIterQ, blockRepIterK, warpRepIterQ, warpRepIterK, iterQ, iterK}
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);
  auto warp_y = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
  auto warp_x = tools::mapUtils::wapr_x(dims[0], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
  auto lane_y = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);
  auto lane_x = tools::mapUtils::lane_x(dims[0], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);
  auto ty = (dims[1] * cfg["BLOCK_LAYOUT_P_Y"] + warp_y * cfg["BLOCK_SCATTER_WIDTH_Q"]) * cfg["WARP_LAYOUT_P_Y"] +
             dims[3] * cfg["WARP_LAYOUT_P_Y"] + lane_y * cfg["WARP_SCATTER_WIDTH_Q"] + dims[5];
  auto tx = (dims[2] * cfg["BLOCK_LAYOUT_P_X"] + warp_x * cfg["BLOCK_SCATTER_WIDTH_K"]) * cfg["WARP_LAYOUT_P_X"] +
             dims[4] * cfg["WARP_LAYOUT_P_X"] + lane_x * cfg["WARP_SCATTER_WIDTH_K"] + dims[6];

  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(ty);
  exprs.push_back(tx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap FlashAttnSplitK2Optimizer::getTileOToGlobOMap(mlir::OpBuilder& builder) {
  // {b1, b2, by, tid, blockRepIterQ, blockRepIterK, warpRepIterQ, warpRepIterK, iterQ, iterK}
  int dimCount = 10;
  auto dims = getExprs(builder, dimCount);
  auto warp_y = tools::mapUtils::wapr_y(dims[3], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_O_X"]);
  auto warp_x = tools::mapUtils::wapr_x(dims[3], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_O_X"]);
  auto lane_y = tools::mapUtils::lane_y(dims[3], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_O_X"]);
  auto lane_x = tools::mapUtils::lane_x(dims[3], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_O_X"]);
  auto ty = (dims[4] * cfg["BLOCK_LAYOUT_O_Y"] + warp_y * cfg["BLOCK_SCATTER_WIDTH_P"]) * cfg["WARP_LAYOUT_O_Y"] +
             dims[6] * cfg["WARP_LAYOUT_O_Y"] + lane_y * cfg["WARP_SCATTER_WIDTH_P"] + dims[8];
  auto tx = (dims[5] * cfg["BLOCK_LAYOUT_O_X"] + warp_x * cfg["BLOCK_SCATTER_WIDTH_V"]) * cfg["WARP_LAYOUT_O_X"] +
             dims[7] * cfg["WARP_LAYOUT_O_X"] + lane_x * cfg["WARP_SCATTER_WIDTH_V"] + dims[9];

  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(dims[0]);
  exprs.push_back(dims[1]);
  exprs.push_back(dims[2] + ty);
  exprs.push_back(tx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

// ==================================== parse / config / applicable =====================================

void FlashAttnSplitK2Optimizer::computeTuneArgs() {
  this->blockPY = cfg.at("Br") / cfg.at("PTr");
  this->blockPX = cfg.at("Bc") / cfg.at("PTc");
  this->blockOY = cfg.at("Br") / cfg.at("OTr");
  this->blockOX = cfg.at("Hd") / cfg.at("OTc");
  this->threadNum = blockPY * blockPX;
  this->blockRepeatQ = cfg.at("PTr") / cfg.at("BLOCK_SCATTER_WIDTH_Q");
  this->blockRepeatK = cfg.at("PTc") / cfg.at("BLOCK_SCATTER_WIDTH_K");
  this->warpRepeatQ = cfg.at("BLOCK_SCATTER_WIDTH_Q") / cfg.at("WARP_SCATTER_WIDTH_Q");
  this->warpRepeatK = cfg.at("BLOCK_SCATTER_WIDTH_K") / cfg.at("WARP_SCATTER_WIDTH_K");
  this->blockRepeatP = cfg.at("OTr") / cfg.at("BLOCK_SCATTER_WIDTH_P");
  this->blockRepeatV = cfg.at("OTc") / cfg.at("BLOCK_SCATTER_WIDTH_V");
  this->warpRepeatP = cfg.at("BLOCK_SCATTER_WIDTH_P") / cfg.at("WARP_SCATTER_WIDTH_P");
  this->warpRepeatV = cfg.at("BLOCK_SCATTER_WIDTH_V") / cfg.at("WARP_SCATTER_WIDTH_V");
  this->globLoadTotalWidthQ = cfg.at("Br") * cfg.at("Slice1") / this->threadNum;
  this->globLoadTotalWidthK = cfg.at("Bc") * cfg.at("Slice1") / this->threadNum;
  this->globLoadTotalWidthV = cfg.at("Hd") * cfg.at("Slice2") / this->threadNum;
  this->globLoadRowWidthQ = this->threadNum / cfg.at("Br") * cfg.at("GLOB_LOAD_WIDTH_Q");
  this->globLoadRowWidthK = this->threadNum / cfg.at("Slice1") * cfg.at("GLOB_LOAD_WIDTH_K");
  this->globLoadRowWidthV = this->threadNum / cfg.at("Slice2") * cfg.at("GLOB_LOAD_WIDTH_V");
  if (isTranQ) {
    this->globLoadRowWidthQ = this->threadNum / cfg.at("Slice1") * cfg.at("GLOB_LOAD_WIDTH_Q");
  }
  if (isTranK) {
    this->globLoadRowWidthK = this->threadNum / cfg.at("Bc") * cfg.at("GLOB_LOAD_WIDTH_K");
  }
  if (isTranV) {
    this->globLoadRowWidthV = this->threadNum / cfg.at("Hd") * cfg.at("GLOB_LOAD_WIDTH_V");
  }
  this->globLoadAllWidthQ = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_Q");
  this->globLoadAllWidthK = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_K");
  this->globLoadAllWidthV = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_V");

  this->useShuffleP = false;
  if (cfg.count("SHUFFLE_P") && cfg.at("SHUFFLE_P") == 1) {
    bool layoutOk = (
      cfg.at("BLOCK_LAYOUT_P_X") == 1 &&
      cfg.at("PTr") == cfg.at("OTr") &&
      cfg.at("BLOCK_LAYOUT_P_Y") == cfg.at("BLOCK_LAYOUT_O_Y") &&
      cfg.at("BLOCK_LAYOUT_P_X") == cfg.at("BLOCK_LAYOUT_O_X") &&
      cfg.at("WARP_LAYOUT_P_Y") == cfg.at("WARP_LAYOUT_O_Y") &&
      cfg.at("WARP_LAYOUT_P_X") == cfg.at("WARP_LAYOUT_O_X") &&
      cfg.at("BLOCK_SCATTER_WIDTH_Q") == cfg.at("BLOCK_SCATTER_WIDTH_P") &&
      cfg.at("WARP_SCATTER_WIDTH_Q") == cfg.at("WARP_SCATTER_WIDTH_P")
    );
    if (layoutOk) {
      this->useShuffleP = true;
      llvm::errs() << "[opt] useShuffleP=1: skipping smP, using warp shuffle\n";
    } else {
      llvm::errs() << "[opt] SHUFFLE_P=1 requested but layout constraints not met, falling back to smP\n";
    }
  }

  this->useSplitKPV = false;
  if (cfg.count("SPLITK_PV") && cfg.at("SPLITK_PV") == 1) {
    bool layoutOk = (
      cfg.at("BLOCK_LAYOUT_P_X") == 1 &&
      cfg.at("PTr") == cfg.at("OTr") &&
      cfg.at("BLOCK_LAYOUT_P_Y") == cfg.at("BLOCK_LAYOUT_O_Y") &&
      cfg.at("BLOCK_LAYOUT_P_X") == cfg.at("BLOCK_LAYOUT_O_X") &&
      cfg.at("WARP_LAYOUT_P_Y") == cfg.at("WARP_LAYOUT_O_Y") &&
      cfg.at("WARP_LAYOUT_P_X") == cfg.at("WARP_LAYOUT_O_X") &&
      cfg.at("BLOCK_SCATTER_WIDTH_Q") == cfg.at("BLOCK_SCATTER_WIDTH_P") &&
      cfg.at("WARP_SCATTER_WIDTH_Q") == cfg.at("WARP_SCATTER_WIDTH_P")
    );
    if (layoutOk) {
      this->useSplitKPV = true;
      this->useShuffleP = false;
      llvm::errs() << "[opt] useSplitKPV=1: skipping smP, using split-K PV with tileO reduction\n";
    } else {
      llvm::errs() << "[opt] SPLITK_PV=1 requested but layout constraints not met, falling back to smP\n";
    }
  }
}

void FlashAttnSplitK2Optimizer::parseFuncArgs(mlir::func::FuncOp funcOp) {
  typeQ = mlir::dyn_cast<mlir::MemRefType>(Q.getType());
  typeK = mlir::dyn_cast<mlir::MemRefType>(K.getType());
  typeV = mlir::dyn_cast<mlir::MemRefType>(V.getType());
  typeEm = mlir::dyn_cast<mlir::MemRefType>(Em.getType());
  typeDenom = mlir::dyn_cast<mlir::MemRefType>(Denom.getType());
  typeO = mlir::dyn_cast<mlir::MemRefType>(O.getType());
  typeMid = mlir::dyn_cast<mlir::MemRefType>(midBuf.getType());
  std::vector<bool> isTrans;
  auto transArr = funcOp->getAttr(ARGTRAN);
  auto transArrAttr = mlir::dyn_cast<mlir::ArrayAttr>(transArr);
  for (auto tran : transArrAttr) {
    auto tranAttr = mlir::dyn_cast<mlir::IntegerAttr>(tran);
    isTrans.push_back(tranAttr.getInt());
  }
  isTranQ = isTrans[0]; isTranK = isTrans[1]; isTranV = isTrans[2];
  auto shapeO = typeO.getShape();
  batchSize = shapeO[0]; headNum = shapeO[1];
  seqLen = shapeO[2]; headDim = shapeO[3];
}

bool FlashAttnSplitK2Optimizer::applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) {
  this->cfg = config;
  mlir::ValueRange operands = funcOp.getArguments();
  this->Q = operands[0]; this->K = operands[1]; this->V = operands[2];
  this->Em = operands[3]; this->Denom = operands[4]; this->O = operands[5];
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

std::pair<std::array<mlir::Value, 4>, std::array<mlir::Value, 7>> FlashAttnSplitK2Optimizer::createBasicBuffers() {
  auto dtypeQ = typeQ.getElementType();
  auto dtypeK = typeK.getElementType();
  auto dtypeV = typeV.getElementType();
  auto dtypeMid = typeMid.getElementType();

  std::vector<std::vector<int64_t>> smShapes{
    {cfg.at("Slice1"), cfg.at("Br")}, {cfg.at("Slice1"), cfg.at("Bc")},
    {cfg.at("Slice2"), cfg.at("Hd")}, {cfg.at("Br"), cfg.at("Bc")}
  };
  std::vector<mlir::Type> smType{dtypeQ, dtypeK, dtypeV, dtypeMid};
  std::vector<std::string> smDescs{"smQ", "smK", "smV", "smP"};
  auto sm = Rewriter::allocBuffers(smShapes, smType, MemorySpace::shared, smDescs, blockIdx);
  std::array<mlir::Value, 4> sm_;
  std::copy(sm.begin(), sm.end(), sm_.begin());

  std::vector<std::vector<int64_t>> regShapes{
    {globLoadTotalWidthQ}, {globLoadTotalWidthK}, {globLoadTotalWidthV},
    {cfg.at("PTr")}, {cfg.at("PTc")}, {cfg.at("OTr")}, {cfg.at("OTc")}
  };
  std::vector<mlir::Type> regDTypes{dtypeQ, dtypeK, dtypeV, dtypeQ, dtypeK, dtypeMid, dtypeV};
  std::vector<std::string> regDescs{"tempQ", "tempK", "tempV", "regQ", "regK", "regP", "regV"};
  auto reg = Rewriter::allocBuffers(regShapes, regDTypes, MemorySpace::local, regDescs, this->xBlockFors[0]);
  std::array<mlir::Value, 7> reg_;
  std::copy(reg.begin(), reg.end(), reg_.begin());
  return {sm_, reg_};
}

void FlashAttnSplitK2Optimizer::moveMemrefDefineAhead(mlir::Operation* threadParallelOp) {
  auto parallelop = mlir::dyn_cast<mlir::affine::AffineParallelOp>(threadParallelOp);
  assert(parallelop != nullptr);
  mlir::affine::AffineForOp firstForOp {};
  std::vector<mlir::Operation*> opsToMove {};
  for (auto& childop : parallelop->getRegion(0).getOps()) {
    firstForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(childop);
    if (firstForOp != nullptr) {
      break;
    }
  }
  assert(firstForOp != nullptr);
  parallelop.walk([&](mlir::memref::AllocaOp op) {
    opsToMove.push_back(op.getOperation());
  });
  parallelop.walk([&](mlir::memref::AllocOp op) {
    opsToMove.push_back(op.getOperation());
  });
  for (auto op : opsToMove) {
    op->moveBefore(firstForOp);
  }
}

// ================================== applyOptimzer ========================================

void FlashAttnSplitK2Optimizer::applyOptimzer(mlir::func::FuncOp& funcOp) {
  mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(funcOp->getParentOp());
  mlir::OpBuilder builder(module);

  // ===== a. tileP + tileO bufferize, k split/reorder =====
  std::vector<mlir::affine::AffineForOp> tilePLoops{yTileForOps[0], xTileForOps[0]};
  auto tileP = Rewriter::bufferizeLoopCarryVar(kForOps[0], tilePLoops, MemorySpace::local, {"tileP"});
  std::vector<mlir::affine::AffineForOp> tileOLoops{yTileForOps[1], xTileForOps[1]};
  auto tileO = Rewriter::bufferizeLoopCarryVar(kForOps[1], tileOLoops, MemorySpace::local, {"tileO"});
  LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

  auto k1 = Rewriter::split(kForOps[0], {cfg.at("Slice1")});
  auto k1_outer = k1[0], k1_inner = k1[1];
  Rewriter::reorder({k1_outer, k1_inner, yTileForOps[0], xTileForOps[0]});
  auto k2 = Rewriter::split(kForOps[1], {cfg.at("Bc"), cfg.at("Slice2")});
  auto k2_outer = k2[0], k2_midder = k2[1], k2_inner = k2[2];
  Rewriter::reorder({k2_outer, k2_midder, k2_inner, yTileForOps[1], xTileForOps[1]});
  LOG_DEBUG("===== after split & reorder all K =======\n",module);

  // ===== b. createBasicBuffers (4 shared, 7 reg) =====
  auto [sm, reg] = createBasicBuffers();
  auto [smQ, smK, smV, smP] = sm;
  auto [tempQ, tempK, tempV, regQ, regK, regP, regV] = reg;
  LOG_DEBUG("===== after alloc_buffer =======\n",module);

  // ===== c. smQFull allocation =====
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

  // ===== d. Q prologue with SCALE_Q =====
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

  // ===== e. K load pipeline =====
  llvm::SmallVector<mlir::Value> operands(bIdx.begin(), bIdx.end()-1);
  operands.push_back(byIdx); operands.push_back(xBlockFors[0].getInductionVar()); operands.push_back(tIdx[0]);
  auto loadTileKMap = getGlobQKToTempQKMap(builder, "K");
  llvm::SmallVector<mlir::Value> kGlobOperands(operands);
  kGlobOperands.push_back(k1_outer.getInductionVar());
  auto loadTileK = Rewriter::loadToRegisters(K, tempK, loadTileKMap, kGlobOperands, {cfg["GLOB_LOAD_WIDTH_K"]}, k1_outer, Position::begin, "");
  LOG_DEBUG("===== after read K =======\n",module);

  auto storeTileKMap = getTempToSmMap(builder, "K");
  auto storeTileK = Rewriter::loadFromRegisters(tempK, smK, storeTileKMap, {tIdx[0]}, {cfg["GLOB_LOAD_WIDTH_K"]}, loadTileK, Position::after, "");
  auto prefix = Rewriter::barrier(loadTileK, Position::before);
  auto suffix = Rewriter::barrier(storeTileK, Position::after);
  LOG_DEBUG("===== write K =======\n",module);

  // ===== f. matmul1 micro-kernel =====
  auto loadFragQMap = getSmQPrologueToRegQMap(builder);
  auto loadFragKMap = getSmQKVToRegQKVMap(builder, "K");
  llvm::SmallVector<mlir::Value> qFragOperands{tIdx[0], k1_outer.getInductionVar(), k1_inner.getInductionVar()};
  llvm::SmallVector<mlir::Value> kFragOperands{tIdx[0], k1_inner.getInductionVar()};
  std::vector<int64_t> widthsQ{cfg["BLOCK_SCATTER_WIDTH_Q"], cfg["WARP_SCATTER_WIDTH_Q"]};
  std::vector<int64_t> widthsK{cfg["BLOCK_SCATTER_WIDTH_K"], cfg["WARP_SCATTER_WIDTH_K"]};
  auto loadFragQ = Rewriter::loadToRegisters(smQFull, regQ, loadFragQMap, qFragOperands, widthsQ, k1_inner, Position::begin, "");
  auto loadFragK = Rewriter::loadToRegisters(smK, regK, loadFragKMap, kFragOperands, widthsK, loadFragQ, Position::after, "");
  LOG_DEBUG("===== read sh_Q/K =======\n",module);

  auto calMap = getCalculateMap(builder, "matmul");
  Rewriter::cache_read(xTileForOps[0], Q, regQ, calMap, {yTileForOps[0].getInductionVar()});
  Rewriter::cache_read(xTileForOps[0], K, regK, calMap, {xTileForOps[0].getInductionVar()});
  LOG_DEBUG("===== load regQ & cache_read =======\n",module);

  // ===== g. separateNoOpRelyForOp =====
  auto tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
  auto txds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttilexDown");
  std::vector<mlir::affine::AffineForOp> forOps{tyds[0], txds[0]};
  Rewriter::separateNoOpRelyForOp(forOps);
  LOG_DEBUG("===== separateNoOpRelyForOp =======\n",module);

  // ===== h. SCALE_SCORES =====
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

  // ===== NORMALIZE tileP using Em/Denom (replaces softmax steps 2-3) =====
  {
    tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
    auto dtype = typeMid.getElementType();
    mlir::OpBuilder nb(tyds[0]);
    auto loc = nb.getUnknownLoc();

    llvm::SmallVector<int64_t> nlbs{0, 0}, nubs{cfg["PTr"], cfg["PTc"]}, nsteps{1, 1};
    auto [normForOps, normIvs] = createNestedLoops(nb, nlbs, nubs, nsteps);
    nb.setInsertionPointToStart(normForOps.back().getBody());

    auto pVal = nb.create<mlir::affine::AffineLoadOp>(loc, tileP[0], normIvs);
    auto expVal = nb.create<mlir::math::ExpOp>(loc, pVal);

    // Build affine map to compute global row for Em/Denom load.
    // Decompose flat row index into block/warp scatter and combine with thread position.
    int batchCount = (int)bIdx.size() - 1;
    int emMapDims = batchCount + 3; // batch dims + by + tid + flat_i
    auto emDimExprs = getExprs(nb, emMapDims);
    auto byExpr = emDimExprs[batchCount];
    auto tidExpr = emDimExprs[batchCount + 1];
    auto flatIExpr = emDimExprs[batchCount + 2];

    auto warp_y_e = tools::mapUtils::wapr_y(tidExpr, cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
    auto lane_y_e = tools::mapUtils::lane_y(tidExpr, cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);

    int64_t BSW_Q_n = cfg["BLOCK_SCATTER_WIDTH_Q"];
    int64_t WSW_Q_n = cfg["WARP_SCATTER_WIDTH_Q"];
    auto blockRepQ_e = flatIExpr.floorDiv(BSW_Q_n);
    auto warpRepQ_e = (flatIExpr % BSW_Q_n).floorDiv(WSW_Q_n);
    auto scatterQ_e = flatIExpr % WSW_Q_n;

    auto rowInBr = (blockRepQ_e * cfg["BLOCK_LAYOUT_P_Y"] + warp_y_e) * cfg["WARP_LAYOUT_P_Y"] * BSW_Q_n
                   + (warpRepQ_e * cfg["WARP_LAYOUT_P_Y"] + lane_y_e) * WSW_Q_n + scatterQ_e;

    auto emRank = typeEm.getRank();
    llvm::SmallVector<mlir::AffineExpr> emResults;
    for (int i = 0; i < batchCount; i++) {
      emResults.push_back(emDimExprs[i]);
    }
    emResults.push_back(byExpr + rowInBr);
    while ((int)emResults.size() < emRank) {
      emResults.push_back(nb.getAffineConstantExpr(0));
    }
    auto emMap = mlir::AffineMap::get(emMapDims, 0,
                     llvm::ArrayRef<mlir::AffineExpr>(emResults), nb.getContext());

    llvm::SmallVector<mlir::Value> emOps;
    for (int i = 0; i < batchCount; i++) {
      emOps.push_back(bIdx[i]);
    }
    emOps.push_back(byIdx);
    emOps.push_back(threadIdx.getIVs()[0]);
    emOps.push_back(normIvs[0]);

    auto emVal = nb.create<mlir::affine::AffineLoadOp>(loc, Em, emMap, emOps);
    auto denomVal = nb.create<mlir::affine::AffineLoadOp>(loc, Denom, emMap, emOps);

    auto factor = nb.create<mlir::arith::MulFOp>(loc, emVal, denomVal);
    auto normalized = nb.create<mlir::arith::DivFOp>(loc, expVal, factor);
    nb.create<mlir::affine::AffineStoreOp>(loc, normalized, tileP[0], normIvs);
  }
  LOG_DEBUG("===== normalize tileP with Em/Denom =======\n",module);

  // ===== i. tileP to smP (with shuffle/split-k logic) =====
  // With BroadcastNorm fusing to 2 yloops, there's only 1 ttileyDown (matmul1's store-back).
  // This IS the tileP→smP store loop (no separate softmax rear).
  tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
  txds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttilexDown");
  auto qs = Rewriter::split(tyds[0], widthsQ);
  auto ks = Rewriter::split(txds[0], widthsK);
  auto qs_outer = qs[0], qs_midder = qs[1], qs_inner = qs[2];
  auto ks_outer = ks[0], ks_midder = ks[1], ks_inner = ks[2];
  Rewriter::reorder({qs_outer, ks_outer, qs_midder, ks_midder, qs_inner, ks_inner});
  if (!useShuffleP && !useSplitKPV) {
    auto tilePToSmPMap = getTilePToSmPMap(builder);
    llvm::SmallVector<mlir::Value> rtsOperands{tIdx[0]};
    for (int i=0; i<3; i++) {
      rtsOperands.push_back(qs[i].getInductionVar());
      rtsOperands.push_back(ks[i].getInductionVar());
    }
    Rewriter::cache_write(ks_inner, midBuf, smP, tilePToSmPMap, rtsOperands);
    Rewriter::vectorize(ks_inner, cfg["WARP_SCATTER_WIDTH_K"]);
    Rewriter::barrier(qs_outer, Position::after);
  }
  LOG_DEBUG("===== split & reorder tileP to smP =======\n",module);

  // ===== j. fuseForOps for xBlockFors =====
  // BroadcastNorm creates a 3rd blockx loop (between matmul1 and matmul2).
  // Fuse all blockx loops + k2_outer into one loop.
  std::vector<std::vector<mlir::affine::AffineForOp>> bfors;
  for (auto& bf : xBlockFors) {
    bfors.push_back({bf});
  }
  bfors.push_back({k2_outer});
  auto xBlockFor = fuseForOps(bfors)[0];

  // No softmax rear to move/cache_read (BroadcastNorm fuses directly into matmul1+matmul2)
  LOG_DEBUG("===== fuse blockx for op done =======\n",module);


  // ===== m. V load pipeline =====
  auto loadTileVMap = getGlobVToTempVMap(builder);
  llvm::SmallVector<mlir::Value> gvoperands(bIdx.begin(), bIdx.end()-1);
  gvoperands.push_back(xBlockFor.getInductionVar()); gvoperands.push_back(tIdx[0]); gvoperands.push_back(k2_midder.getInductionVar());
  auto loadTileV = Rewriter::loadToRegisters(V, tempV, loadTileVMap, gvoperands, {cfg["GLOB_LOAD_WIDTH_V"]}, k2_midder, Position::begin, "");
  LOG_DEBUG("===== after read V =======\n",module);

  auto storeTileVMap = getTempToSmMap(builder, "V");
  auto storeTileV = Rewriter::loadFromRegisters(tempV, smV, storeTileVMap, {tIdx[0]}, {cfg["GLOB_LOAD_WIDTH_V"]}, loadTileV, Position::after, "");
  auto prefix_ = Rewriter::barrier(loadTileV, Position::before);
  auto suffix_ = Rewriter::barrier(storeTileV, Position::after);
  LOG_DEBUG("===== write V =======\n",module);

  // ===== n. matmul2 micro-kernel (loadFragP, loadFragV, cache_read for P/V) =====
  auto loadFragVMap = getSmQKVToRegQKVMap(builder, "V");
  llvm::SmallVector<mlir::Value> svoperands{tIdx[0], k2_inner.getInductionVar()};
  std::vector<int64_t> widthsV{cfg["BLOCK_SCATTER_WIDTH_V"], cfg["WARP_SCATTER_WIDTH_V"]};
  mlir::affine::AffineForOp loadFragP;
  if (useSplitKPV) {
    int64_t WLPX = cfg["WARP_LAYOUT_P_X"];
    int64_t WSWK = cfg["WARP_SCATTER_WIDTH_K"];
    int64_t WARP_SZ = cfg["WARP_SIZE"];
    int64_t stride = WLPX * WSWK;
    int64_t PTr_ = cfg["PTr"];

    builder.setInsertionPointToStart(k2_inner.getBody());
    auto loc = builder.getUnknownLoc();
    auto elemTy = mlir::dyn_cast<mlir::MemRefType>(tileP[0].getType()).getElementType();

    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto d2 = builder.getAffineDimExpr(2);
    auto kExpr = d0 + d1;
    auto ownerLx = (kExpr % stride).floorDiv(WSWK);
    auto myLx    = (d2 % WARP_SZ) % WLPX;

    auto ownerMap = mlir::AffineMap::get(3, 0, {ownerLx}, builder.getContext());
    auto myMap    = mlir::AffineMap::get(3, 0, {myLx}, builder.getContext());
    auto operands3 = mlir::ValueRange{k2_midder.getInductionVar(), k2_inner.getInductionVar(), tIdx[0]};
    auto ownerIdx = builder.create<mlir::affine::AffineApplyOp>(loc, ownerMap, operands3);
    auto myIdx    = builder.create<mlir::affine::AffineApplyOp>(loc, myMap, operands3);

    auto ownerI32 = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI32Type(), ownerIdx.getResult());
    auto myI32    = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI32Type(), myIdx.getResult());
    auto isMine   = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, ownerI32, myI32);

    auto colExpr = kExpr.floorDiv(stride) * WSWK + kExpr % WSWK;
    auto colMap  = mlir::AffineMap::get(2, 0, {colExpr}, builder.getContext());
    auto colIdx  = builder.create<mlir::affine::AffineApplyOp>(
        loc, colMap, mlir::ValueRange{k2_midder.getInductionVar(), k2_inner.getInductionVar()});

    auto zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemTy, 0.0));

    auto wrapBody = [&](mlir::OpBuilder &b, mlir::Location l,
                        mlir::Value /*iv*/, mlir::ValueRange) {
      for (int64_t r = 0; r < PTr_; ++r) {
        auto rConst = b.create<mlir::arith::ConstantIndexOp>(l, r);
        auto pVal = b.create<mlir::affine::AffineLoadOp>(
            l, tileP[0], mlir::ValueRange{rConst, colIdx.getResult()});
        auto selected = b.create<mlir::arith::SelectOp>(l, isMine, pVal.getResult(), zero.getResult());
        b.create<mlir::affine::AffineStoreOp>(l, selected, regP, mlir::ValueRange{rConst});
      }
      b.create<mlir::affine::AffineYieldOp>(l);
    };
    loadFragP = builder.create<mlir::affine::AffineForOp>(
        loc, 0, 1, 1, mlir::ValueRange{}, wrapBody);
  } else if (useShuffleP) {
    loadFragP = generateShufflePToRegP(builder, tileP[0], regP, tIdx[0],
        k2_midder.getInductionVar(), k2_inner.getInductionVar(), k2_inner);
  } else {
    auto loadFragPMap = getSmPToRegPMap(builder);
    llvm::SmallVector<mlir::Value> spoperands{tIdx[0], k2_midder.getInductionVar(), k2_inner.getInductionVar()};
    std::vector<int64_t> widthsP{cfg["BLOCK_SCATTER_WIDTH_P"], cfg["WARP_SCATTER_WIDTH_P"]};
    loadFragP = Rewriter::loadToRegisters(smP, regP, loadFragPMap, spoperands, widthsP, k2_inner, Position::begin, "");
  }
  auto loadFragV = Rewriter::loadToRegisters(smV, regV, loadFragVMap, svoperands, widthsV, loadFragP, Position::after, "");
  LOG_DEBUG("===== read sh_P/V =======\n",module);

  calMap = getCalculateMap(builder, "matmul");
  Rewriter::cache_read(xTileForOps[1], midBuf, regP, calMap, {yTileForOps[1].getInductionVar()});
  Rewriter::cache_read(xTileForOps[1], V, regV, calMap, {xTileForOps[1].getInductionVar()});
  LOG_DEBUG("===== load regV/P & cache_read =======\n",module);

  // ===== o. shuffle-P / split-K PV cleanup =====
  if (useShuffleP || useSplitKPV) {
    std::vector<mlir::Operation*> deadOps;
    qs_outer.walk([&](mlir::Operation* op) {
      if (mlir::isa<mlir::affine::AffineStoreOp, mlir::affine::AffineVectorStoreOp>(op))
        deadOps.push_back(op);
    });
    for (auto* op : llvm::reverse(deadOps)) op->erase();
    for (int pass = 0; pass < 4; ++pass) {
      deadOps.clear();
      qs_outer.walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::affine::AffineForOp, mlir::affine::AffineYieldOp>(op)
            && op->use_empty())
          deadOps.push_back(op);
      });
      if (deadOps.empty()) break;
      for (auto* op : llvm::reverse(deadOps)) op->erase();
    }
  }

  // ===== Store tileO to global O =====
  // For useSplitKPV: first reduce tileO across lane_x via warp shuffle,
  // then only lane_x==0 writes. Both must happen AFTER xBlockFor completes.
  {
    mlir::OpBuilder writeBuilder = getBuilder(xBlockFor, Position::after);
    auto loc = writeBuilder.getUnknownLoc();

    // Step 1: warp shuffle reduction for split-K PV (AFTER all blockx iterations)
    if (useSplitKPV) {
      int64_t OTr_ = cfg["OTr"], OTc_ = cfg["OTc"];
      int64_t WLPX = cfg["WARP_LAYOUT_P_X"];
      auto i32Ty = writeBuilder.getI32Type();
      auto widthI32 = writeBuilder.create<mlir::arith::ConstantOp>(loc, writeBuilder.getI32IntegerAttr(WLPX));

      for (int64_t dist = 1; dist < WLPX; dist *= 2) {
        auto distI32 = writeBuilder.create<mlir::arith::ConstantOp>(loc, writeBuilder.getI32IntegerAttr(dist));
        for (int64_t r = 0; r < OTr_; ++r) {
          for (int64_t c = 0; c < OTc_; ++c) {
            auto rIdx = writeBuilder.create<mlir::arith::ConstantIndexOp>(loc, r);
            auto cIdx = writeBuilder.create<mlir::arith::ConstantIndexOp>(loc, c);
            auto val = writeBuilder.create<mlir::affine::AffineLoadOp>(
                loc, tileO[0], mlir::ValueRange{rIdx, cIdx});
            auto shfl = writeBuilder.create<mlir::gpu::ShuffleOp>(
                loc, val.getResult(), distI32, widthI32, mlir::gpu::ShuffleMode::DOWN);
            auto sum = writeBuilder.create<mlir::arith::AddFOp>(loc, val.getResult(), shfl.getResult(0));
            writeBuilder.create<mlir::affine::AffineStoreOp>(
                loc, sum, tileO[0], mlir::ValueRange{rIdx, cIdx});
          }
        }
      }

      // Step 2: guard write-back — only lane_x==0 threads have correct reduced tileO
      mlir::Value tid = this->threadIdx.getIVs()[0];
      auto d0 = writeBuilder.getAffineDimExpr(0);
      auto modExpr = d0 % WLPX;
      auto intSet = mlir::IntegerSet::get(1, 0, {modExpr}, {true});
      auto ifOp = writeBuilder.create<mlir::affine::AffineIfOp>(
          loc, intSet, mlir::ValueRange{tid}, false);
      writeBuilder.setInsertionPointToStart(ifOp.getThenBlock());
    }

    llvm::SmallVector<int64_t> lbs{0, 0}, ubs{cfg["OTr"], cfg["OTc"]}, steps{1, 1};
    auto [oLoops, oIvs] = createNestedLoops(writeBuilder, lbs, ubs, steps);
    writeBuilder.setInsertionPointToStart(oLoops.back().getBody());

    auto oVal = writeBuilder.create<mlir::affine::AffineLoadOp>(loc, tileO[0], oIvs);

    // Global index for O store.
    // For normal path (SPLITK_PV=0), use the same O-layout decomposition as getTileOToGlobOMap:
    //   i/j are decomposed by block/warp scatter widths, then combined with warp/lane ids.
    // This avoids column mis-mapping when OTc is split across scatter lanes.
    // Keep the old compact mapping only for SPLITK_PV path.
    int batchCount = (int)Analyzer::getParallelIdx(this->blockIdx).size() - 1;
    int dimCount = batchCount + 4; // batch dims + by + tid + i + j
    auto dims = getExprs(writeBuilder, dimCount);
    auto byE = dims[batchCount];
    auto tidE = dims[batchCount + 1];
    auto iE = dims[batchCount + 2];
    auto jE = dims[batchCount + 3];
    mlir::AffineExpr rowE, colE;

    if (!useSplitKPV) {
      int64_t BLOY = cfg["BLOCK_LAYOUT_O_Y"];
      int64_t BLOX = cfg["BLOCK_LAYOUT_O_X"];
      int64_t WLOY = cfg["WARP_LAYOUT_O_Y"];
      int64_t WLOX = cfg["WARP_LAYOUT_O_X"];
      int64_t BSWP = cfg["BLOCK_SCATTER_WIDTH_P"];
      int64_t BSWV = cfg["BLOCK_SCATTER_WIDTH_V"];
      int64_t WSWP = cfg["WARP_SCATTER_WIDTH_P"];
      int64_t WSWV = cfg["WARP_SCATTER_WIDTH_V"];
      int64_t WARP_SZ = cfg["WARP_SIZE"];

      auto warp_y = tools::mapUtils::wapr_y(tidE, WARP_SZ, BLOX);
      auto warp_x = tools::mapUtils::wapr_x(tidE, WARP_SZ, BLOX);
      auto lane_y = tools::mapUtils::lane_y(tidE, WARP_SZ, WLOX);
      auto lane_x = tools::mapUtils::lane_x(tidE, WARP_SZ, WLOX);

      auto blockRepQ = iE.floorDiv(BSWP);
      auto warpRepQ = (iE % BSWP).floorDiv(WSWP);
      auto iterQ = iE % WSWP;

      auto blockRepK = jE.floorDiv(BSWV);
      auto warpRepK = (jE % BSWV).floorDiv(WSWV);
      auto iterK = jE % WSWV;

      auto ty = (blockRepQ * BLOY + warp_y) * WLOY * BSWP
              + (warpRepQ * WLOY + lane_y) * WSWP + iterQ;
      auto tx = (blockRepK * BLOX + warp_x) * WLOX * BSWV
              + (warpRepK * WLOX + lane_x) * WSWV + iterK;
      rowE = byE + ty;
      colE = tx;
    } else {
      rowE = byE + tidE.floorDiv(this->blockOX) * cfg["OTr"] + iE;
      colE = (tidE % this->blockOX) * cfg["OTc"] + jE;
    }

    llvm::SmallVector<mlir::AffineExpr> globExprs;
    for (int i = 0; i < batchCount; i++) globExprs.push_back(dims[i]);
    globExprs.push_back(rowE);
    globExprs.push_back(colE);

    auto globMap = mlir::AffineMap::get(dimCount, 0,
        llvm::ArrayRef<mlir::AffineExpr>(globExprs), writeBuilder.getContext());

    auto bivs = this->blockIdx.getIVs();
    llvm::SmallVector<mlir::Value> stOps(bivs.rbegin(), bivs.rend()-1);
    stOps.push_back(byIdx);
    stOps.push_back(this->threadIdx.getIVs()[0]);
    stOps.push_back(oIvs[0]);
    stOps.push_back(oIvs[1]);

    writeBuilder.create<mlir::affine::AffineStoreOp>(loc, oVal, O, globMap, stOps);

    // NOTE:
    // For the non-SPLITK path we now use a decomposed O-layout mapping where the
    // innermost j loop is no longer linear-contiguous in global memory (contains
    // floorDiv/mod decomposition). Vectorizing this loop can generate invalid
    // vector accesses and corrupt output.
    // Keep vectorize only on SPLITK path where the compact linear mapping is used.
    if (useSplitKPV) {
      Rewriter::vectorize(oLoops.back(), cfg["WARP_SCATTER_WIDTH_V"]);
    }

    // Erase matmul2's original ttileyDown write-back (from bufferize).
    // Our 6-level output write loop handles the output correctly.
    auto tydsFinal = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
    for (auto tyd : tydsFinal) {
      tyd.erase();
    }

    // Cleanup: erase midBuf (scores/P intermediate)
    if (this->midBuf.use_empty()) {
      this->midBuf.getDefiningOp()->erase();
    } else {
    }
  }
  LOG_DEBUG("===== store tileO to glob O =======\n",module);

  // ===== q. moveMemrefDefineAhead =====
  mlir::affine::AffineParallelOp threadParallelOp;
  funcOp.walk([&](mlir::affine::AffineParallelOp p) {
    auto attr = p.getOperation()->getAttr(AttrGPUIndex);
    auto stringattr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (std::string(stringattr.data()) == THREADIDX) {
      threadParallelOp = p;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  moveMemrefDefineAhead(threadParallelOp.getOperation());
  LOG_DEBUG("===== moveMemrefDefineAhead =======\n",module);

  // ===== r. SHARED_PREFETCH_P, REG_PREFETCH_P, REG_PREFETCH_O =====
  mlir::affine::AffineForOp regRearForOp, regRearForOp_;
  std::vector<mlir::affine::AffineForOp> pfLdRegForOps, pfLdSMForOps, pfLdRegForOps_, pfLdRegForOps__;
  if (cfg["SHARED_PREFETCH_P"]) {
    std::vector<mlir::affine::AffineForOp> LdRegForOps{loadTileK};
    std::vector<mlir::affine::AffineForOp> ldSMForOps{storeTileK};
    std::vector<mlir::Value> smBufs{smK};
    int64_t prefetchStep = cfg.at("Slice1");
    auto smResult = Rewriter::sharedMemroyPrefetch(k1_outer, LdRegForOps, ldSMForOps, k1_inner, smBufs);
    smK = smBufs[0];
    loadTileK = LdRegForOps[0];
    storeTileK = ldSMForOps[0];
    pfLdRegForOps = smResult.first; pfLdSMForOps = smResult.second;
    LOG_DEBUG("===== sharedMemroyPrefetch (K only, Q pre-loaded) =======\n",module);

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

  if (cfg["REG_PREFETCH_O"]) {
    std::vector<mlir::affine::AffineForOp> regLdRegForOps{loadFragP, loadFragV};
    std::vector<mlir::Value> regBufs{regP, regV};
    auto regResult = Rewriter::registersPrefetch(k2_inner, regLdRegForOps, yTileForOps[1], regBufs);
    regP = regBufs[0], regV = regBufs[1];
    loadFragP = regLdRegForOps[0], loadFragV = regLdRegForOps[1];
    pfLdRegForOps__ = regResult.first; regRearForOp_ = regResult.second;
    LOG_DEBUG("===== registersPrefetch =======\n",module);
  }

  // Causal tile-level guard: wrap fused xBlockFor body in affine.if (bx < by + Br)
  if (cfg.count("CAUSAL_MASK") && cfg.at("CAUSAL_MASK")) {
    auto loc = builder.getUnknownLoc();
    auto d0 = builder.getAffineDimExpr(0);
    auto d1 = builder.getAffineDimExpr(1);
    auto constraint = d0 - d1 + (int64_t)(cfg["Br"] - 1);
    auto intSet = mlir::IntegerSet::get(2, 0, {constraint}, {false});

    builder.setInsertionPointToStart(xBlockFor.getBody());
    auto ifOp = builder.create<mlir::affine::AffineIfOp>(
        loc, intSet,
        mlir::ValueRange{byIdx, xBlockFor.getInductionVar()},
        false);

    auto* loopBlock = xBlockFor.getBody();
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
    LOG_DEBUG("===== causal tile-level guard (fused loop) =======\n",module);
  }

  // ===== s. unrollAttribute =====
  Rewriter::unrollAttribute(module, cfg["UNROLL_NUM"]);
  LOG_DEBUG("===== unrollAttribute =======\n",module);
}

}
