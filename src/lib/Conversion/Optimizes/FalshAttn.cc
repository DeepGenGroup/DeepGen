#include "Conversion/Optimize.h"
#include <cmath>

namespace KernelCodeGen {

// ======================================= global to sm =========================================
std::array<int64_t, 7> FlashAttnOptimizer::getCfgDatas(const std::string& bufType) {
  // 有些属性相同，但是表示QKV不同矩阵的config数据将其统一返回

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

std::array<mlir::AffineExpr, 2> FlashAttnOptimizer::getGlobToSmExprs(const llvm::SmallVector<mlir::AffineExpr>& dims, 
                                                                     const std::array<int64_t, 7>& args) {
  // 因为glob到temp和temp到sm到map有一部分相同，所以可以共用一个函数
  mlir::AffineExpr tyIdx, txIdx;
  auto [blockTileY, blockTileX, isTran, globLoadWidth, globLoadAllWidth, globLoadRowWidth, loadContinuous] = args;
  // thread level
  if (loadContinuous) {
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

mlir::AffineMap FlashAttnOptimizer::getGlobQKToTempQKMap(mlir::OpBuilder& builder, const std::string& bufType) {
  // glob load data to temp reg
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by, bx, tid, k, iter}
  auto args = getCfgDatas(bufType);
  // block level
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
  // thread level
  auto [tyIdx, txIdx] = getGlobToSmExprs({dims[dimCount-3], dims[dimCount-1]}, args);  // tid, iter
  // create exprs
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<2; i++) {
    exprs.push_back(dims[i]);  // batch
  }
  exprs.push_back(row + tyIdx);
  exprs.push_back(col + txIdx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap FlashAttnOptimizer::getGlobVToTempVMap(mlir::OpBuilder& builder) {
  // glob V to temp V
  int dimCount = 6;
  auto dims = getExprs(builder, dimCount);  // {b1, b2, by(bx), tid, k, iter}
  auto args = getCfgDatas("V");
  // block level
  mlir::AffineExpr row = dims[dimCount-4] + dims[dimCount-2];
  // thread level
  auto [tyIdx, txIdx] = getGlobToSmExprs({dims[dimCount-3], dims[dimCount-1]}, args);  // tid, iter
  // create exprs
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<2; i++) {
    exprs.push_back(dims[i]);  // batch
  }
  exprs.push_back(row + tyIdx);
  exprs.push_back(txIdx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap FlashAttnOptimizer::getTempToSmMap(mlir::OpBuilder& builder, const std::string& bufType) {
  // 获取从temp reg 到sm的map
  int dimCount = 2;
  auto dims = getExprs(builder, dimCount); // {tid, iter}
  // init datas
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

mlir::AffineMap FlashAttnOptimizer::getTempToSmQPrologueMap(mlir::OpBuilder& builder) {
  // Q prologue: tempQ → expanded smQ[Hd×Br] with k_chunk offset
  // dims: {tid, k_chunk, iter}
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
std::array<int64_t, 8> FlashAttnOptimizer::getSmCfgDatas(const std::string& bufType) {
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

mlir::AffineMap FlashAttnOptimizer::getSmQKVToRegQKVMap(mlir::OpBuilder& builder, const std::string& bufType) {
  // sm to reg
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

mlir::AffineMap FlashAttnOptimizer::getSmQPrologueToRegQMap(mlir::OpBuilder& builder) {
  // reads from expanded smQ[Hd×Br] with k_outer offset
  // dims: {tid, k_outer, bk, blockRepIter, warpRepIter}
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

mlir::AffineMap FlashAttnOptimizer::getSmPToRegPMap(mlir::OpBuilder& builder) {
  // sm p to reg p（不能连续取，索引位置也是相反的）
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

mlir::affine::AffineForOp FlashAttnOptimizer::generateShufflePToRegP(
    mlir::OpBuilder& builder, mlir::Value tileP, mlir::Value regP,
    mlir::Value tid, mlir::Value k2_midder_iv, mlir::Value k2_inner_iv,
    mlir::affine::AffineForOp k2_inner) {
  // Broadcast P columns across lanes via warp shuffle instead of smP.
  // Fully unrolled at compile time — no runtime loop overhead.

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

  // Wrap in a single-trip loop [0,1) so callers that expect AffineForOp work.
  // Inside, emit PTr unrolled shuffle+store pairs — zero loop overhead at runtime.
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

mlir::AffineMap FlashAttnOptimizer::getSmToRegUpdateTtileOMap(mlir::OpBuilder& builder) {
  // sm factor to reg factor
  int dimCount = 3;
  auto dims = getExprs(builder, dimCount);  // {tid, blockRepIter, warpRepIter}
  auto args = getSmCfgDatas("P");
  int64_t blockLayout = args[0], warpLayout = args[2], blockScatter = args[4], warpScatter = args[6];
  mlir::AffineExpr widx = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], args[1]);
  mlir::AffineExpr lidx = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], args[3]);
  auto expr = (dims[1] * blockLayout + widx) * warpLayout * blockScatter + (dims[2] * warpLayout + lidx) * warpScatter;
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext()); 
}

mlir::AffineMap FlashAttnOptimizer::getRegUpdateTtileOMap(mlir::OpBuilder& builder) {
  // y离散化应该只对sm的load、store有影响，所以reg索引应该是不离散的
  int dimCount = 2;  // {blockrepeatp, warprepeatp}
  auto dims = getExprs(builder, dimCount);
  int64_t blockScatter = cfg["BLOCK_SCATTER_WIDTH_P"], warpScatter = cfg["WARP_SCATTER_WIDTH_P"];
  mlir::AffineExpr expr = dims[0] * blockScatter + dims[1] * warpScatter;
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext()); 
}

mlir::AffineMap FlashAttnOptimizer::getCalculateMap(mlir::OpBuilder& builder, std::string calculatetype) {
  // reg 内积
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

mlir::AffineMap FlashAttnOptimizer::getRegSumAndMaxMap(mlir::OpBuilder& builder) {
  // reg的max和sum的map
  auto iter = builder.getAffineDimExpr(0);
  llvm::SmallVector<mlir::AffineExpr> exprs{iter};
  return mlir::AffineMap::get(1, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

// ======================================= store tile ============================================
mlir::AffineMap FlashAttnOptimizer::getTilePToSmPMap(mlir::OpBuilder& builder) {
  // {tid, blockRepIterQ, blockRepIterK, warpRepIterQ, warpRepIterK, iterQ, iterK}
  int dimCount = 7;
  auto dims = getExprs(builder, dimCount); 
  auto warp_y = tools::mapUtils::wapr_y(dims[0], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
  auto warp_x = tools::mapUtils::wapr_x(dims[0], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_P_X"]);
  auto lane_y = tools::mapUtils::lane_y(dims[0], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);
  auto lane_x = tools::mapUtils::lane_x(dims[0], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_P_X"]);
  // create exprs
  auto ty = (dims[1] * cfg["BLOCK_LAYOUT_P_Y"] + warp_y * cfg["BLOCK_SCATTER_WIDTH_Q"]) * cfg["WARP_LAYOUT_P_Y"] + 
             dims[3] * cfg["WARP_LAYOUT_P_Y"] + lane_y * cfg["WARP_SCATTER_WIDTH_Q"] + dims[5];
  auto tx = (dims[2] * cfg["BLOCK_LAYOUT_P_X"] + warp_x * cfg["BLOCK_SCATTER_WIDTH_K"]) * cfg["WARP_LAYOUT_P_X"] + 
             dims[4] * cfg["WARP_LAYOUT_P_X"] + lane_x * cfg["WARP_SCATTER_WIDTH_K"] + dims[6];
  
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(ty);
  exprs.push_back(tx);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap FlashAttnOptimizer::getTileOToGlobOMap(mlir::OpBuilder& builder) {
  // store tileO to globO
  // {b1, b2, by, tid, blockRepIterQ, blockRepIterK, warpRepIterQ, warpRepIterK, iterQ, iterK}
  int dimCount = 10;
  auto dims = getExprs(builder, dimCount); 
  auto warp_y = tools::mapUtils::wapr_y(dims[3], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_O_X"]);
  auto warp_x = tools::mapUtils::wapr_x(dims[3], cfg["WARP_SIZE"], cfg["BLOCK_LAYOUT_O_X"]);
  auto lane_y = tools::mapUtils::lane_y(dims[3], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_O_X"]);
  auto lane_x = tools::mapUtils::lane_x(dims[3], cfg["WARP_SIZE"], cfg["WARP_LAYOUT_O_X"]);
  // create exprs
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

// ======================================= softmax block level ====================================
mlir::AffineMap FlashAttnOptimizer::getBlockLevelSmMap(mlir::OpBuilder& builder) {
  // softmax 在block level的for循环中，由于离散化的原因需要顺应其y离散
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

mlir::AffineMap FlashAttnOptimizer::getBlockLevelRegMap(mlir::OpBuilder& builder) {
  // y离散化应该只对sm的load、store有影响，所以reg索引应该是不离散的
  int dimCount = 3;  // {blockrepeatq, warprepeatq, width}
  auto dims = getExprs(builder, dimCount);
  int64_t blockScatter = cfg["BLOCK_SCATTER_WIDTH_Q"], warpScatter = cfg["WARP_SCATTER_WIDTH_Q"];
  mlir::AffineExpr expr = dims[0] + dims[1] + dims[2];
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>({expr}), builder.getContext()); 
}

// ==================================== 解析和计算必要的信息 =====================================

void FlashAttnOptimizer::computeTuneArgs() {
  // 线程的个数
  this->blockPY = cfg.at("Br") / cfg.at("PTr");
  this->blockPX = cfg.at("Bc") / cfg.at("PTc");
  this->blockOY = cfg.at("Br") / cfg.at("OTr");
  this->blockOX = cfg.at("Hd") / cfg.at("OTc");
  this->threadNum = blockPY * blockPX;
  // p = q * k  离散化重复的次数
  this->blockRepeatQ = cfg.at("PTr") / cfg.at("BLOCK_SCATTER_WIDTH_Q");
  this->blockRepeatK = cfg.at("PTc") / cfg.at("BLOCK_SCATTER_WIDTH_K");
  this->warpRepeatQ = cfg.at("BLOCK_SCATTER_WIDTH_Q") / cfg.at("WARP_SCATTER_WIDTH_Q");
  this->warpRepeatK = cfg.at("BLOCK_SCATTER_WIDTH_K") / cfg.at("WARP_SCATTER_WIDTH_K");
  // o = p * v  离散化重复的次数
  this->blockRepeatP = cfg.at("OTr") / cfg.at("BLOCK_SCATTER_WIDTH_P");
  this->blockRepeatV = cfg.at("OTc") / cfg.at("BLOCK_SCATTER_WIDTH_V");
  this->warpRepeatP = cfg.at("BLOCK_SCATTER_WIDTH_P") / cfg.at("WARP_SCATTER_WIDTH_P");
  this->warpRepeatV = cfg.at("BLOCK_SCATTER_WIDTH_V") / cfg.at("WARP_SCATTER_WIDTH_V");
  // 每个线程需要从glob加载的数据总量
  this->globLoadTotalWidthQ = cfg.at("Br") * cfg.at("Slice1") / this->threadNum;
  this->globLoadTotalWidthK = cfg.at("Bc") * cfg.at("Slice1") / this->threadNum;
  this->globLoadTotalWidthV = cfg.at("Hd") * cfg.at("Slice2") / this->threadNum;
  // glob非连续load时（未转置），block行加载的数据量（线程数量必须大于除数）
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
  // glob连续load时，一个block加载的数据总量
  this->globLoadAllWidthQ = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_Q");
  this->globLoadAllWidthK = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_K");
  this->globLoadAllWidthV = this->threadNum * cfg.at("GLOB_LOAD_WIDTH_V");

  // Warp-shuffle P: skip smP intermediate, broadcast P columns via shuffle.
  // Enabled only when config explicitly sets SHUFFLE_P=1 AND layout constraints
  // are satisfied (all K-column threads in one warp, matching P row assignment).
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

  // Split-K PV: each thread uses only its own PTc P columns for partial PV,
  // then warp-shuffle reduces tileO.  Eliminates smP store/barrier/load.
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
      this->useShuffleP = false;  // Split-K supersedes shuffle P
      llvm::errs() << "[opt] useSplitKPV=1: skipping smP, using split-K PV with tileO reduction\n";
    } else {
      llvm::errs() << "[opt] SPLITK_PV=1 requested but layout constraints not met, falling back to smP\n";
    }
  }
}

void FlashAttnOptimizer::parseFuncArgs(mlir::func::FuncOp funcOp) {
  // 解析kernel函数的参数基本信息
  typeQ = mlir::dyn_cast<mlir::MemRefType>(Q.getType());
  typeK = mlir::dyn_cast<mlir::MemRefType>(K.getType());
  typeV = mlir::dyn_cast<mlir::MemRefType>(V.getType());
  typeO = mlir::dyn_cast<mlir::MemRefType>(O.getType());
  typeMid = mlir::dyn_cast<mlir::MemRefType>(midBuf.getType());
  // get transpose args
  std::vector<bool> isTrans;
  auto transArr = funcOp->getAttr(ARGTRAN);
  auto transArrAttr = mlir::dyn_cast<mlir::ArrayAttr>(transArr);
  for (auto tran : transArrAttr) {
    auto tranAttr = mlir::dyn_cast<mlir::IntegerAttr>(tran);
    isTrans.push_back(tranAttr.getInt());
  }
  isTranQ = isTrans[0]; isTranK = isTrans[1]; isTranV = isTrans[2];
  // get mnk/batch
  auto shapeO = typeO.getShape();
  batchSize = shapeO[0]; headNum = shapeO[1];
  seqLen = shapeO[2]; headDim = shapeO[3];
}

bool FlashAttnOptimizer::applicable(mlir::func::FuncOp& funcOp, const std::map<std::string, int64_t>& config) {
  // 获取必要的信息
  this->cfg = config;
  mlir::ValueRange operands = funcOp.getArguments();
  this->Q = operands[0]; this->K = operands[1]; this->V = operands[2]; this->O = operands[3];
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

std::pair<std::array<mlir::Value, 5>, std::array<mlir::Value, 7>> FlashAttnOptimizer::createBasicBuffers() {
  // create all buffers
  auto dtypeQ = typeQ.getElementType();
  auto dtypeK = typeK.getElementType();
  auto dtypeV = typeV.getElementType();
  auto dtypeMid = typeMid.getElementType();

  // create shared memory buffers
  std::vector<std::vector<int64_t>> smShapes{
    {cfg.at("Slice1"), cfg.at("Br")}, {cfg.at("Slice1"), cfg.at("Bc")}, 
    {cfg.at("Slice2"), cfg.at("Hd")}, {cfg.at("Br"), cfg.at("Bc")}, {cfg.at("Br")}
  };
  std::vector<mlir::Type> smType{dtypeQ, dtypeK, dtypeV, dtypeMid, dtypeMid};
  std::vector<std::string> smDescs{"smQ", "smK", "smV", "smP", "smFactor"};
  auto sm = Rewriter::allocBuffers(smShapes, smType, MemorySpace::shared, smDescs, blockIdx);
  std::array<mlir::Value, 5> sm_;
  std::copy(sm.begin(), sm.end(), sm_.begin());

  // create registers buffers
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

//======================================== 有关flash的特定优化 =================================
std::vector<mlir::affine::AffineForOp> FlashAttnOptimizer::reduceAndBraodcast(mlir::Operation* localtionOp,
                                                                              const std::vector<mlir::Value>& regBufs,
                                                                              const std::vector<mlir::Value>& smBufs) {
  // 创建softmax相关的一些操作
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
  // create warp level ops
  auto warpLevelForOp = warpReduce(builder, ydim, width, regBufs, onlineSoftmax);
  // create block level ops
  auto blockLevelForOp = blockReduce(builder, ydim, width, tid, regBufs, smBufs, onlineSoftmax);
  // Broadcast both rowMax and rowSum inside each lane_x group.
  // The factor load map for tileO update is lane_x-collapsed (indexed by lane_y),
  // so regSum must also be uniform across lane_x.
  auto bcForOp = warpBroadcast(builder, ydim, width, /*buf*/{regBufs[0], regBufs[1]}, /*index*/0);
  return std::vector<mlir::affine::AffineForOp>{warpLevelForOp, blockLevelForOp, bcForOp};
}

void FlashAttnOptimizer::updateTileO(mlir::Operation* localtionOp, mlir::Value smBuf, mlir::Value tileO, std::string bufDesc) {
  // 加上对tileO进行迭代更新的
  mlir::OpBuilder builder = getBuilder(localtionOp->getParentOp(), Position::begin);
  // create regFactor buffer
  auto dtype = typeMid.getElementType();
  auto regBuf = createAllocOp(builder, {cfg["OTr"]}, dtype, MemorySpace::local, KCG_ALIGNBYTE, bufDesc);

  // create load and store smFactor to regFactor
  builder.setInsertionPointAfter(localtionOp);
  if (bufDesc != "regFactor") { builder.setInsertionPoint(localtionOp); }
  llvm::SmallVector<int64_t> lbs{0, 0}, ubs{this->blockRepeatP, this->warpRepeatP}, steps{1, 1};
  auto [newForOps, newIvs] = createNestedLoops(builder, lbs, ubs, steps);
  builder.setInsertionPointToStart(newForOps.back().getBody());
  // create load and store
  auto vectorType = mlir::VectorType::get(cfg["WARP_SCATTER_WIDTH_P"], dtype);
  auto ldmap = getSmToRegUpdateTtileOMap(builder);
  llvm::SmallVector<mlir::Value> ldOperands{threadIdx.getIVs()[0], newIvs[0], newIvs[1]};
  auto ld = builder.create<mlir::affine::AffineVectorLoadOp>(builder.getUnknownLoc(), vectorType, smBuf, ldmap, ldOperands);
  auto stmap = getRegUpdateTtileOMap(builder);
  builder.create<mlir::affine::AffineVectorStoreOp>(builder.getUnknownLoc(), ld, regBuf, stmap, newIvs);

  // create caculate forop
  builder.setInsertionPointAfter(newForOps.front());
  llvm::SmallVector<int64_t> ubs_{cfg["OTr"], cfg["OTc"]};
  auto [newForOps_, newIvs_] = createNestedLoops(builder, lbs, ubs_, steps);
  builder.setInsertionPointToStart(newForOps_.back().getBody());
  auto regld = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), regBuf, mlir::ValueRange({newIvs_[0]}));
  auto told = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), tileO, newIvs_);
  mlir::Value calculateVal;
  if (bufDesc != "regFactor") {
    calculateVal = builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(), told, regld);
  } else {
    calculateVal = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), told, regld);
  }
  builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), calculateVal, tileO, newIvs_);
  if (bufDesc != "regFactor") {
    // regsum单独处理
    mlir::Value tileOld;
    mlir::OpBuilder builder(localtionOp);
    std::vector<mlir::Operation*> delOps;
    std::vector<mlir::affine::AffineForOp> forOps;
    localtionOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
      if (auto loadOp = mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
        if (loadOp.getMemRef() == tileO) {
          tileOld = loadOp.getResult();
        } else {
          delOps.push_back(op);
        }
      } else if (auto storeOp = mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
        auto buf = storeOp.getMemRef();
        auto map = storeOp.getAffineMap();
        auto operands = storeOp.getMapOperands();
        builder.setInsertionPoint(storeOp);
        builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(), tileOld, buf, map, operands);
        storeOp.erase();
      } else if (!mlir::dyn_cast<mlir::affine::AffineYieldOp>(op)) {
        if (auto fop = mlir::dyn_cast<mlir::affine::AffineForOp>(op)) {
          forOps.push_back(fop);
        } else {
          delOps.push_back(op);
        }
      }
    });
    for (auto it=delOps.rbegin(); it!=delOps.rend(); it++) {
      mlir::Operation* delOp = *it;
      delOp->erase();
    }
    // split and reoder
    auto ps = Rewriter::split(forOps[0], {cfg["BLOCK_SCATTER_WIDTH_P"], cfg["WARP_SCATTER_WIDTH_P"]});
    auto vs = Rewriter::split(forOps[1], {cfg["BLOCK_SCATTER_WIDTH_V"], cfg["WARP_SCATTER_WIDTH_V"]});
    auto ps_outer = ps[0], ps_midder = ps[1], ps_inner = ps[2];
    auto vs_outer = vs[0], vs_midder = vs[1], vs_inner = vs[2];
    Rewriter::reorder({ps_outer, vs_outer, ps_midder, vs_midder, ps_inner, vs_inner});
    auto storeTileOMap = getTileOToGlobOMap(builder);
    auto bivs = this->blockIdx.getIVs();
    llvm::SmallVector<mlir::Value> rtsOperands(bivs.rbegin(), bivs.rend()-1);
    rtsOperands.push_back(byIdx);
    rtsOperands.push_back(this->threadIdx.getIVs()[0]);
    for (int i=0; i<3; i++) {
      rtsOperands.push_back(ps[i].getInductionVar());
      rtsOperands.push_back(vs[i].getInductionVar());
    }
    Rewriter::cache_write(vs_inner, O, O, storeTileOMap, rtsOperands);
    Rewriter::vectorize(vs_inner, cfg["WARP_SCATTER_WIDTH_V"]);
    // remove all glob op
    this->initBufFor.erase();
    this->midBuf.getDefiningOp()->erase();
    this->sumBuf.getDefiningOp()->erase();
    this->maxBuf.getDefiningOp()->erase();
  }
}

void FlashAttnOptimizer::moveMemrefDefineAhead(mlir::Operation* threadParallelOp){
  auto parallelop = mlir::dyn_cast<mlir::affine::AffineParallelOp>(threadParallelOp);
  assert(parallelop != nullptr);
  mlir::affine::AffineForOp firstForOp {};
  std::vector<mlir::Operation*> opsToMove {};
  for(auto& childop : parallelop->getRegion(0).getOps()) {
    firstForOp = mlir::dyn_cast<mlir::affine::AffineForOp>(childop);
    if(firstForOp != nullptr){
      break;  // find first affine.for in thread parallelop
    }
  }
  assert(firstForOp != nullptr);
  // collect alloca & allocop
  parallelop.walk([&](mlir::memref::AllocaOp op){
    opsToMove.push_back(op.getOperation());
  });
  parallelop.walk([&](mlir::memref::AllocOp op){
    opsToMove.push_back(op.getOperation());
  });
  // move
  for(auto op : opsToMove){
    op->moveBefore(firstForOp);
  }
}

// ================================== applyOptimzer ========================================

void FlashAttnOptimizer::applyOptimzer(mlir::func::FuncOp& funcOp) {
  // optimize
  mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(funcOp->getParentOp());
  mlir::OpBuilder builder(module);
  // matmul1 + matmul2 k bufferize
  std::vector<mlir::affine::AffineForOp> tilePLoops{yTileForOps[0], xTileForOps[0]};
  auto tileP = Rewriter::bufferizeLoopCarryVar(kForOps[0], tilePLoops, MemorySpace::local, {"tileP"});
  std::vector<mlir::affine::AffineForOp> tileOLoops{yTileForOps[2], xTileForOps[2]};
  auto tileO = Rewriter::bufferizeLoopCarryVar(kForOps[1], tileOLoops, MemorySpace::local, {"tileO"});
  LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);
  // matmul1 + matmul2 k split and reorder
  auto k1 = Rewriter::split(kForOps[0], {cfg.at("Slice1")});
  auto k1_outer = k1[0], k1_inner = k1[1];
  Rewriter::reorder({k1_outer, k1_inner, yTileForOps[0], xTileForOps[0]});
  auto k2 = Rewriter::split(kForOps[1], {cfg.at("Bc"), cfg.at("Slice2")});
  auto k2_outer = k2[0], k2_midder = k2[1], k2_inner = k2[2];
  Rewriter::reorder({k2_outer, k2_midder, k2_inner, yTileForOps[2], xTileForOps[2]});
  LOG_DEBUG("===== after split & reorder all K =======\n",module);
  // alloc shared  and regisiters
  auto [sm, reg] = createBasicBuffers();
  auto [smQ, smK, smV, smP, smFactor] = sm;
  auto [tempQ, tempK, tempV, regQ, regK, regP, regV] = reg;
  LOG_DEBUG("===== after alloc_buffer =======\n",module);
  mlir::Value smQFull;
  if (cfg["Slice1"] == cfg["Hd"]) {
    smQFull = smQ;
  } else {
    smQFull = Rewriter::allocBuffers({{cfg["Hd"], cfg["Br"]}},
                                      {typeQ.getElementType()},
                                      MemorySpace::shared, {"smQFull"}, blockIdx)[0];
  }
  LOG_DEBUG("===== after alloc smQFull =======\n",module);
  // splitu and fuse forop into parallelop
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
  // Calculate 
  auto calMap = getCalculateMap(builder, "matmul");  // {iter}
  Rewriter::cache_read(xTileForOps[0], Q, regQ, calMap, {yTileForOps[0].getInductionVar()});
  Rewriter::cache_read(xTileForOps[0], K, regK, calMap, {xTileForOps[0].getInductionVar()});
  LOG_DEBUG("===== load regQ & cache_read =======\n",module);

  // 1.将tileP存储到mid的过程从循环中分离出来
  auto tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
  auto txds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttilexDown");
  std::vector<mlir::affine::AffineForOp> forOps{tyds[0], txds[0]};
  Rewriter::separateNoOpRelyForOp(forOps);
  LOG_DEBUG("===== separateNoOpRelyForOp =======\n",module);

  // scale scores after mm1: tileP[i][j] *= 1/sqrt(headDim)
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

  // 2.定义smmax和smsum以及rowsum和rowmax，以及将它们进行初始化
  auto smMaxAndSum = Rewriter::createHierarchyInitBuf(initBufFor, {cfg["Br"]}, threadIdx, MemorySpace::shared);
  auto regMaxAndSum = Rewriter::createHierarchyInitBuf(initBufFor, {cfg["PTr"]}, xBlockFors[0], MemorySpace::local);
  // auto [smMaxAndSum, regMaxAndSum] = Rewriter::createSMAndRegInitBuf(initBufFor, blockIdx, xBlockFors[0], {cfg["Br"]}, {cfg["PTr"]});
  auto smMax = smMaxAndSum[0], smSum = smMaxAndSum[1], regMax = regMaxAndSum[0], regSum = regMaxAndSum[1];
  LOG_DEBUG("===== createSMAndRegInitBuf =======\n",module);

  // 3.修改softmax上半部分求max和sum的部分，将glob的max和sum替换成reg的
  auto sumAndMaxRegMap = getRegSumAndMaxMap(builder);
  tyds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttileyDown");
  txds = collectOpsInfuncOp<mlir::affine::AffineForOp>(funcOp, FORDESC, "ttilexDown");
  Rewriter::cache_read(txds[0], maxBuf, regMax, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  Rewriter::cache_read(txds[0], sumBuf, regSum, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  Rewriter::cache_write(txds[0], maxBuf, regMax, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  Rewriter::cache_write(txds[0], sumBuf, regSum, sumAndMaxRegMap, {tyds[0].getInductionVar()});
  LOG_DEBUG("===== amend thread level load and store of max and sum  =======\n",module);
  // *** 将warp level / block level / broadcast 操作加上 ***
  auto softmaxForOps = reduceAndBraodcast(tyds[0], {regMax, regSum}, {smMax, smSum, smFactor});
  LOG_DEBUG("===== add warp level and block level ops of max and sum =======\n",module);
  // 将block层面的for循环进行切分，并修改sm和reg buf的map
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

  // 4.切store tileP的循环，然后mid glob使用smP替换
  auto qs = Rewriter::split(tyds[1], widthsQ);
  auto ks = Rewriter::split(txds[1], widthsK);
  auto qs_outer = qs[0], qs_midder = qs[1], qs_inner = qs[2];
  auto ks_outer = ks[0], ks_midder = ks[1], ks_inner = ks[2];
  Rewriter::reorder({qs_outer, ks_outer, qs_midder, ks_midder, qs_inner, ks_inner});
  mlir::Operation* toSmP;
  if (!useShuffleP && !useSplitKPV) {
    auto tilePToSmPMap = getTilePToSmPMap(builder);
    llvm::SmallVector<mlir::Value> rtsOperands{tIdx[0]};
    for (int i=0; i<3; i++) {
      rtsOperands.push_back(qs[i].getInductionVar());
      rtsOperands.push_back(ks[i].getInductionVar());
    }
    Rewriter::cache_write(ks_inner, midBuf, smP, tilePToSmPMap, rtsOperands);
    Rewriter::vectorize(ks_inner, cfg["WARP_SCATTER_WIDTH_K"]);
    toSmP = Rewriter::barrier(qs_outer, Position::after);
  } else {
    // Shuffle-P / Split-K PV: skip smP scatter-store, vectorize, and barrier.
    // Store-back loops still reference midBuf; cleaned up before updateTileO("ORegSum").
    toSmP = qs_outer.getOperation();
  }
  LOG_DEBUG("===== split & reorder tileP to smP & amend globMid to smP =======\n",module);

  // *** 将softmax rear 和matmul2的计算部分 fuse到blockx循环中 ***
  llvm::errs() << "[dbg] about to fuseForOps(bfors)\n";
  std::vector<std::vector<mlir::affine::AffineForOp>> bfors{{xBlockFors[0]}, {xBlockFors[1]}, {k2_outer}};
  auto xBlockFor = fuseForOps(bfors)[0];
  llvm::errs() << "[dbg] fuseForOps done, moveBefore\n";
  yTileForOps[1]->moveBefore(qs_outer);  // move tilePToSmP before
  calMap = getCalculateMap(builder, "softmax");  // {itery, iterx}
  Rewriter::cache_read(xTileForOps[1], midBuf, tileP[0], calMap, {yTileForOps[1].getInductionVar(), xTileForOps[1].getInductionVar()});
  Rewriter::cache_read(xTileForOps[1], maxBuf, regMax, sumAndMaxRegMap, {yTileForOps[1].getInductionVar()});
  Rewriter::cache_write(xTileForOps[1], midBuf, tileP[0], calMap, {yTileForOps[1].getInductionVar(), xTileForOps[1].getInductionVar()});
  llvm::errs() << "[dbg] about to updateTileO(regFactor)\n";
  // create tileO update ops
  updateTileO(toSmP, smFactor, tileO[0], "regFactor");
  llvm::errs() << "[dbg] updateTileO(regFactor) done\n";
  LOG_DEBUG("===== fuse blockx for op and move & softmax rear amend load and store =======\n",module);

  llvm::errs() << "[dbg] matmul2 part start\n";
  // matmul2部分的sm和reg的load和store {k2_midder, k2_inner, yTileForOps[2], xTileForOps[2]}
  // glob load to temp reg
  auto loadTileVMap = getGlobVToTempVMap(builder);
  llvm::SmallVector<mlir::Value> gvoperands(bIdx.begin(), bIdx.end()-1);  // {b1, b2, bx, tid, k}
  gvoperands.push_back(xBlockFor.getInductionVar()); gvoperands.push_back(tIdx[0]); gvoperands.push_back(k2_midder.getInductionVar());
  auto loadTileV = Rewriter::loadToRegisters(V, tempV, loadTileVMap, gvoperands, {cfg["GLOB_LOAD_WIDTH_V"]}, k2_midder, Position::begin, "");
  LOG_DEBUG("===== after read V =======\n",module);
  // temp reg load to sm
  auto storeTileVMap = getTempToSmMap(builder, "V");   // {tid, iter}
  auto storeTileV = Rewriter::loadFromRegisters(tempV, smV, storeTileVMap, {tIdx[0]}, {cfg["GLOB_LOAD_WIDTH_V"]}, loadTileV, Position::after, "");
  auto prefix_ = Rewriter::barrier(loadTileV, Position::before);
  auto suffix_ = Rewriter::barrier(storeTileV, Position::after);
  LOG_DEBUG("===== write V =======\n",module);
  // sm load to cal reg
  auto loadFragVMap = getSmQKVToRegQKVMap(builder, "V");
  llvm::SmallVector<mlir::Value> svoperands{tIdx[0], k2_inner.getInductionVar()};  // {tid, bk, blockRepIter, warpRepIter}
  std::vector<int64_t> widthsV{cfg["BLOCK_SCATTER_WIDTH_V"], cfg["WARP_SCATTER_WIDTH_V"]};
  mlir::affine::AffineForOp loadFragP;
  if (useSplitKPV) {
    // Split-K PV: each thread loads from its own tileP when the current
    // global column k belongs to it, otherwise fills regP with 0.
    // After the full k2 loop, tileO holds a partial sum that is reduced
    // across warp X-threads.
    int64_t WLPX = cfg["WARP_LAYOUT_P_X"];
    int64_t WSWK = cfg["WARP_SCATTER_WIDTH_K"];
    int64_t WARP_SZ = cfg["WARP_SIZE"];
    int64_t stride = WLPX * WSWK;
    int64_t PTr_ = cfg["PTr"];

    builder.setInsertionPointToStart(k2_inner.getBody());
    auto loc = builder.getUnknownLoc();
    auto elemTy = mlir::dyn_cast<mlir::MemRefType>(tileP[0].getType()).getElementType();

    // k = k2_midder + k2_inner
    // my_col_start = lane_x * WSWK  (within each stride-group)
    // is_mine = ((k % stride) / WSWK) == lane_x
    //         ≡ ((k % stride) floordiv WSWK) == ((tid % WARP_SIZE) % WLPX)
    auto d0 = builder.getAffineDimExpr(0);  // k2_midder
    auto d1 = builder.getAffineDimExpr(1);  // k2_inner
    auto d2 = builder.getAffineDimExpr(2);  // tid
    auto kExpr = d0 + d1;
    auto ownerLx = (kExpr % stride).floorDiv(WSWK);
    auto myLx    = (d2 % WARP_SZ) % WLPX;

    // Compute owner_lane_x and my_lane_x as index values
    auto ownerMap = mlir::AffineMap::get(3, 0, {ownerLx}, builder.getContext());
    auto myMap    = mlir::AffineMap::get(3, 0, {myLx}, builder.getContext());
    auto operands3 = mlir::ValueRange{k2_midder.getInductionVar(), k2_inner.getInductionVar(), tIdx[0]};
    auto ownerIdx = builder.create<mlir::affine::AffineApplyOp>(loc, ownerMap, operands3);
    auto myIdx    = builder.create<mlir::affine::AffineApplyOp>(loc, myMap, operands3);

    // is_mine = (owner == my_lane_x)
    auto ownerI32 = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI32Type(), ownerIdx.getResult());
    auto myI32    = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI32Type(), myIdx.getResult());
    auto isMine   = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, ownerI32, myI32);

    // tileP_col = (k / stride) * WSWK + k % WSWK
    auto colExpr = kExpr.floorDiv(stride) * WSWK + kExpr % WSWK;
    auto colMap  = mlir::AffineMap::get(2, 0, {colExpr}, builder.getContext());
    auto colIdx  = builder.create<mlir::affine::AffineApplyOp>(
        loc, colMap, mlir::ValueRange{k2_midder.getInductionVar(), k2_inner.getInductionVar()});

    auto zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemTy, 0.0));

    // Unrolled: for each row r, regP[r] = is_mine ? tileP[r][col] : 0
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
    llvm::errs() << "[dbg] generateSplitKRegP done\n";
  } else if (useShuffleP) {
    loadFragP = generateShufflePToRegP(builder, tileP[0], regP, tIdx[0],
        k2_midder.getInductionVar(), k2_inner.getInductionVar(), k2_inner);
    llvm::errs() << "[dbg] generateShufflePToRegP done\n";
  } else {
    auto loadFragPMap = getSmPToRegPMap(builder);
    llvm::SmallVector<mlir::Value> spoperands{tIdx[0], k2_midder.getInductionVar(), k2_inner.getInductionVar()};
    std::vector<int64_t> widthsP{cfg["BLOCK_SCATTER_WIDTH_P"], cfg["WARP_SCATTER_WIDTH_P"]};
    loadFragP = Rewriter::loadToRegisters(smP, regP, loadFragPMap, spoperands, widthsP, k2_inner, Position::begin, "");
  }
  auto loadFragV = Rewriter::loadToRegisters(smV, regV, loadFragVMap, svoperands, widthsV, loadFragP, Position::after, "");
  LOG_DEBUG("===== read sh_P/V =======\n",module);
  // Calculate 
  calMap = getCalculateMap(builder, "matmul");  // {iter}
  Rewriter::cache_read(xTileForOps[2], midBuf, regP, calMap, {yTileForOps[2].getInductionVar()});
  Rewriter::cache_read(xTileForOps[2], V, regV, calMap, {xTileForOps[2].getInductionVar()});
  LOG_DEBUG("===== load regV/P & cache_read =======\n",module);

  // When shuffle-P or split-K PV is active, the store-back loops still
  // reference midBuf.  Erase all non-structural ops so midBuf can be erased.
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
    llvm::errs() << "[dbg] cleaned store-back loops\n";
  }

  // Split-K PV: tileO is a partial sum — reduce across warp X-threads.
  if (useSplitKPV) {
    int64_t OTr_ = cfg["OTr"], OTc_ = cfg["OTc"];
    int64_t WLPX = cfg["WARP_LAYOUT_P_X"];
    int64_t WARP_SZ = cfg["WARP_SIZE"];
    // Insert after the last PV-related operation (tyds.back() or the last cache_read)
    mlir::OpBuilder reduceBuilder = getBuilder(tyds.back(), Position::before);
    auto loc = reduceBuilder.getUnknownLoc();
    auto i32Ty = reduceBuilder.getI32Type();
    auto widthI32 = reduceBuilder.create<mlir::arith::ConstantOp>(loc, reduceBuilder.getI32IntegerAttr(WARP_SZ));

    // log2(WLPX) rounds of shuffle-down reduction on tileO[OTr][OTc]
    for (int64_t dist = 1; dist < WLPX; dist *= 2) {
      auto distI32 = reduceBuilder.create<mlir::arith::ConstantOp>(loc, reduceBuilder.getI32IntegerAttr(dist));
      for (int64_t r = 0; r < OTr_; ++r) {
        for (int64_t c = 0; c < OTc_; ++c) {
          auto rIdx = reduceBuilder.create<mlir::arith::ConstantIndexOp>(loc, r);
          auto cIdx = reduceBuilder.create<mlir::arith::ConstantIndexOp>(loc, c);
          auto val = reduceBuilder.create<mlir::affine::AffineLoadOp>(
              loc, tileO[0], mlir::ValueRange{rIdx, cIdx});
          auto shfl = reduceBuilder.create<mlir::gpu::ShuffleOp>(
              loc, val.getResult(), distI32, widthI32, mlir::gpu::ShuffleMode::DOWN);
          auto sum = reduceBuilder.create<mlir::arith::AddFOp>(loc, val.getResult(), shfl.getResult(0));
          reduceBuilder.create<mlir::affine::AffineStoreOp>(
              loc, sum, tileO[0], mlir::ValueRange{rIdx, cIdx});
        }
      }
    }
    llvm::errs() << "[dbg] splitK tileO warp reduction done (" << OTr_ << "x" << OTc_
                 << ", " << (int)std::log2(WLPX) << " rounds)\n";
  }

  llvm::errs() << "[dbg] about to updateTileO(ORegSum)\n";
  updateTileO(tyds.back(), smSum, tileO[0], "ORegSum");
  llvm::errs() << "[dbg] updateTileO(ORegSum) done\n";
  LOG_DEBUG("===== update last TileO =======\n",module);

  mlir::affine::AffineParallelOp threadParallelOp;
  funcOp.walk([&](mlir::affine::AffineParallelOp p){
    auto attr = p.getOperation()->getAttr(AttrGPUIndex);
    auto stringattr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if(std::string(stringattr.data()) == THREADIDX){
      threadParallelOp = p;
      return mlir::WalkResult::interrupt();
    } 
  });
  llvm::errs() << "[dbg] about to moveMemrefDefineAhead\n";
  moveMemrefDefineAhead(threadParallelOp.getOperation());
  llvm::errs() << "[dbg] moveMemrefDefineAhead done\n";
  LOG_DEBUG("===== moveMemrefDefineAhead =======\n",module);

  mlir::affine::AffineForOp regRearForOp, regRearForOp_;
  std::vector<mlir::affine::AffineForOp> pfLdRegForOps, pfLdSMForOps, pfLdRegForOps_, pfLdRegForOps__;
  if (cfg["SHARED_PREFETCH_P"]) {
    llvm::errs() << "[dbg] SHARED_PREFETCH_P start\n";
    std::vector<mlir::affine::AffineForOp> LdRegForOps{loadTileK}, ldSMForOps{storeTileK};
    std::vector<mlir::Value> smBufs{smK};
    int64_t prefetchStep = cfg.at("Slice1");
    auto smResult = Rewriter::sharedMemroyPrefetch(k1_outer, LdRegForOps, ldSMForOps, k1_inner, smBufs);
    smK = smBufs[0];
    loadTileK = LdRegForOps[0];
    storeTileK = ldSMForOps[0];
    pfLdRegForOps = smResult.first; pfLdSMForOps = smResult.second;
    llvm::errs() << "[dbg] sharedMemroyPrefetch done\n";
    LOG_DEBUG("===== sharedMemroyPrefetch (K only, Q pre-loaded) =======\n",module);

    // After sharedMemroyPrefetch the k_outer IV is shifted by +step (range
    // changes from [0,ub) to [step,ub+step)).  K loading accounts for this,
    // but Q reads from smQFull need the *original* unshifted k value.
    // Create an affine.apply to compute (new_k_outer_iv - step) and patch
    // every smQFull load inside loadFragQ to use it.
    llvm::errs() << "[dbg] fixing smQFull k_outer IV shift\n";
    mlir::Value newKOuterIv = k1_outer.getInductionVar();
    // Bake the -step offset directly into the affine map of each smQFull
    // load.  No new SSA value is created — purely a compile-time map edit,
    // so subsequent transforms (registersPrefetch / doubleBufferAdjust)
    // carry it through transparently with zero runtime overhead.
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
    llvm::errs() << "[dbg] smQFull map patched OK\n";
    LOG_DEBUG("===== fix smQFull k_outer IV shift =======\n",module);
  }

  llvm::errs() << "[dbg] REG_PREFETCH_P=" << cfg["REG_PREFETCH_P"] << "\n";
  if (cfg["REG_PREFETCH_P"]) {
    std::vector<mlir::affine::AffineForOp> regLdRegForOps{loadFragQ, loadFragK};
    std::vector<mlir::Value> regBufs{regQ, regK};
    auto regResult = Rewriter::registersPrefetch(k1_inner, regLdRegForOps, yTileForOps[0], regBufs);
    regQ = regBufs[0], regK = regBufs[1];
    loadFragQ = regLdRegForOps[0], loadFragK = regLdRegForOps[1];
    pfLdRegForOps_ = regResult.first; regRearForOp = regResult.second;
    llvm::errs() << "[dbg] registersPrefetch done\n";
    LOG_DEBUG("===== registersPrefetch =======\n",module);
  }

  if (cfg["SHARED_PREFETCH_P"] && cfg["REG_PREFETCH_P"]) {
    llvm::errs() << "[dbg] doubleBufferAdjust start\n";
    Rewriter::doubleBufferAdjust(pfLdSMForOps, pfLdRegForOps, pfLdRegForOps_, regRearForOp);
    llvm::errs() << "[dbg] doubleBufferAdjust done\n";
    LOG_DEBUG("===== doublePerfetchAdjust =======\n",module);
  }

  if (cfg["REG_PREFETCH_O"]) {
    std::vector<mlir::affine::AffineForOp> regLdRegForOps{loadFragP, loadFragV};
    std::vector<mlir::Value> regBufs{regP, regV};
    auto regResult = Rewriter::registersPrefetch(k2_inner, regLdRegForOps, yTileForOps[2], regBufs);
    regP = regBufs[0], regV = regBufs[1];
    loadFragP = regLdRegForOps[0], loadFragV = regLdRegForOps[1];
    pfLdRegForOps__ = regResult.first; regRearForOp_ = regResult.second;
    LOG_DEBUG("===== registersPrefetch =======\n",module);
  }

  Rewriter::unrollAttribute(module, cfg["UNROLL_NUM"]);
  LOG_DEBUG("===== unrollAttribute =======\n",module);
}

}