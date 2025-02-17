#include "Conversion/Optimizer.h"
#include <cfloat>
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include <filesystem>
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

namespace KernelCodeGen {

MatmulConfigUtils::MatmulConfigUtils(const std::map<std::string,int>& config){
  this->BM = config.at(KEY_BLOCK_SIZE_M); 
  this->BN = config.at(KEY_BLOCK_SIZE_N); 
  this->BK = config.at(KEY_BLOCK_SIZE_K);
  this->TM = config.at(KEY_THREAD_SIZE_M);
  this->TN = config.at(KEY_THREAD_SIZE_N);
  // block的shape
  this->BLOCK_Y = BM / TM;
  this->BLOCK_X = BN / TN;
  // 离散化重复次数参数
  this->THREAD_NUM = BLOCK_X * BLOCK_Y * config.at(KEY_LOCAL_SPLIT_U);
  this->BLOCK_REPEAT_A = TM / config.at(KEY_WARP_SCATTER_WIDTH_A);
  this->WARP_REPEAT_A = config.at(KEY_WARP_SCATTER_WIDTH_A) / config.at(KEY_THREAD_SCATTER_WIDTH_A);
  this->BLOCK_REPEAT_B = TN / config.at(KEY_WARP_SCATTER_WIDTH_B);
  this->WARP_REPEAT_B = config.at(KEY_WARP_SCATTER_WIDTH_B) / config.at(KEY_THREAD_SCATTER_WIDTH_B);
  //不连续访存所需参数
  this->GLOB_LOAD_ROW_WIDTH_A = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_A);
  this->GLOB_LOAD_ROW_WIDTH_B = THREAD_NUM / BK * config.at(KEY_GLOB_LOAD_WIDTH_B);
  this->GLOB_STORE_ROW_WIDTH = THREAD_NUM / BM * config.at(KEY_GLOB_STORE_WIDTH);
  // 连续访存所需参数
  this->GLOB_LOAD_COL_THREAD_NUM_A = BM / config.at(KEY_GLOB_LOAD_WIDTH_A);
  this->GLOB_LOAD_COL_THREAD_NUM_B = BN / config.at(KEY_GLOB_LOAD_WIDTH_B);
  this->GLOB_LOAD_ROW_THREAD_NUM_A = THREAD_NUM / GLOB_LOAD_COL_THREAD_NUM_A;
  this->GLOB_LOAD_ROW_THREAD_NUM_B = THREAD_NUM / GLOB_LOAD_COL_THREAD_NUM_B;
  this->GLOB_STORE_COL_THREAD_NUM = BN / config.at(KEY_GLOB_STORE_WIDTH);
  this->GLOB_STORE_ROW_THREAD_NUM = THREAD_NUM / GLOB_STORE_COL_THREAD_NUM;
}

bool MatmulOptimizer::applicable(mlir::ModuleOp& module) {
  clear();
  auto&& matmulFuncs = Analyzer::collectFunctions(module, "Matmul");
  if (!matmulFuncs.size()) {
    llvm::errs() << "Optimization failed: No find Matmul funcOp.\n";
    return false;
  }

  for (auto& matmulFunc : matmulFuncs) {
    if (matmuls.count(matmulFunc) != 0 || matmulLoops.count(matmulFunc) != 0
      || matmulBuffers.count(matmulFunc) != 0) {
      llvm::errs() << "Optimization failed: Duplicated Matmul in module\n";
      return false;
    }

    matmuls.insert(matmulFunc);
    auto&& loops = Analyzer::collectFuncLoops(matmulFunc);
    matmulLoops[matmulFunc] = std::move(loops);
    auto funcArgs = matmulFunc.front().getArguments();

    MemoryBuffer ABC;
    ABC.A = funcArgs[0];
    ABC.B = funcArgs[1];
    ABC.C = funcArgs[2];
    matmulBuffers[matmulFunc] = ABC;
  }
  return true;
}

void test(mlir::AffineMap map) {
  LOG_DEBUG("",map);
}

mlir::AffineMap MatmulOptimizer::getGlobToTempMapA(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto by = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto k = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  if (config.at(KEY_LOAD_CONTINUOUS)) {
    auto tid_ = tid + iter * cfg.THREAD_NUM;
    auto sh_load = tools::mapUtils::reshapeBlock(tid_, {cfg.GLOB_LOAD_ROW_THREAD_NUM_A, cfg.GLOB_LOAD_COL_THREAD_NUM_A});
    exprs.push_back(sh_load[0] + k);
    exprs.push_back(by * cfg.BM + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_A));
  } else {
    auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
    exprs.push_back(sh_load[0] + k);
    exprs.push_back(by * cfg.BM + iter * cfg.GLOB_LOAD_ROW_WIDTH_A + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_A));
  }
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap MatmulOptimizer::getGlobToTempMapB(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto bx = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto k = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  if (config.at(KEY_LOAD_CONTINUOUS)) {
    auto tid_ = tid + iter * cfg.THREAD_NUM;
    auto sh_load = tools::mapUtils::reshapeBlock(tid_, {cfg.GLOB_LOAD_ROW_THREAD_NUM_B, cfg.GLOB_LOAD_COL_THREAD_NUM_B});
    exprs.push_back(sh_load[0] + k);
    exprs.push_back(bx * cfg.BN + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_B));
  } else {
    auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
    exprs.push_back(sh_load[0] + k);
    exprs.push_back(bx * cfg.BN + iter * cfg.GLOB_LOAD_ROW_WIDTH_B + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_B));
  }
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext());
}

mlir::AffineMap MatmulOptimizer::getTempToSharedMapSMA(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  if (config.at(KEY_LOAD_CONTINUOUS)) {
    auto tid_ = tid + iter * cfg.THREAD_NUM;
    auto sh_load = tools::mapUtils::reshapeBlock(tid_, {cfg.GLOB_LOAD_ROW_THREAD_NUM_A, cfg.GLOB_LOAD_COL_THREAD_NUM_A});
    exprs.push_back(sh_load[0]);
    exprs.push_back(sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_A));
  } else {
    auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
    exprs.push_back(sh_load[0]);
    exprs.push_back(iter * cfg.GLOB_LOAD_ROW_WIDTH_A + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_A));
  }
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getTempToSharedMapSMB(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  if (config.at(KEY_LOAD_CONTINUOUS)) {
    auto tid_ = tid + iter * cfg.THREAD_NUM;
    auto sh_load = tools::mapUtils::reshapeBlock(tid_, {cfg.GLOB_LOAD_ROW_THREAD_NUM_B, cfg.GLOB_LOAD_COL_THREAD_NUM_B});
    exprs.push_back(sh_load[0]);
    exprs.push_back(sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_B));
  } else {
    auto sh_load = tools::mapUtils::reshapeBlock(tid, {cfg.BK, cfg.THREAD_NUM/cfg.BK});
    exprs.push_back(sh_load[0]);
    exprs.push_back(iter * cfg.GLOB_LOAD_ROW_WIDTH_B + sh_load[1] * config.at(KEY_GLOB_LOAD_WIDTH_B));
  }
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getSharedToRegMapSMA(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto bk = builder.getAffineDimExpr(dimCount++);
  auto blockRepIter = builder.getAffineDimExpr(dimCount++);
  auto warpRepIter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto reTh = tools::mapUtils::reshapeBlock(tid, {config.at(KEY_LOCAL_SPLIT_U), cfg.THREAD_NUM/config.at(KEY_LOCAL_SPLIT_U)});
  auto warp_y = tools::mapUtils::wapr_y(reTh[1], config);
  auto lane_y = tools::mapUtils::lane_y(reTh[1], config);
  exprs.push_back(bk + reTh[0]);
  exprs.push_back((blockRepIter * config.at(KEY_BLOCK_LAYOUT_M) + warp_y) * config.at(KEY_WARP_LAYOUT_M) * config.at(KEY_WARP_SCATTER_WIDTH_A) + 
                  (warpRepIter * config.at(KEY_WARP_LAYOUT_M) + lane_y) * config.at(KEY_THREAD_SCATTER_WIDTH_A));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getSharedToRegMapSMB(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto bk = builder.getAffineDimExpr(dimCount++);
  auto blockRepIter = builder.getAffineDimExpr(dimCount++);
  auto warpRepIter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto reTh = tools::mapUtils::reshapeBlock(tid, {config.at(KEY_LOCAL_SPLIT_U), cfg.THREAD_NUM/config.at(KEY_LOCAL_SPLIT_U)});
  auto warp_x = tools::mapUtils::wapr_x(reTh[1], config);
  auto lane_x = tools::mapUtils::lane_x(reTh[1], config);
  exprs.push_back(bk + reTh[0]);
  exprs.push_back((blockRepIter * config.at(KEY_BLOCK_LAYOUT_N) + warp_x) * config.at(KEY_WARP_LAYOUT_N) * config.at(KEY_WARP_SCATTER_WIDTH_B) + 
                  (warpRepIter * config.at(KEY_WARP_LAYOUT_N) + lane_x) * config.at(KEY_THREAD_SCATTER_WIDTH_B));
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getCalculateMapReg(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto iter = builder.getAffineDimExpr(dimCount++);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(iter);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getRegToGlobMapC(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto by = builder.getAffineDimExpr(dimCount++);
  auto bx = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterA = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterB = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterA = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterB = builder.getAffineDimExpr(dimCount++);
  auto iterA = builder.getAffineDimExpr(dimCount++);
  auto iterB = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto warp_y = tools::mapUtils::wapr_y(tid, config);
  auto warp_x = tools::mapUtils::wapr_x(tid, config);
  auto lane_y = tools::mapUtils::lane_y(tid, config);
  auto lane_x = tools::mapUtils::lane_x(tid, config);
  exprs.push_back(by * cfg.BM + (blockRepIterA * config.at(KEY_BLOCK_LAYOUT_M) + warp_y * config.at(KEY_WARP_SCATTER_WIDTH_A)) * config.at(KEY_WARP_LAYOUT_M) + 
                                 warpRepIterA * config.at(KEY_WARP_LAYOUT_M) + lane_y * config.at(KEY_THREAD_SCATTER_WIDTH_A) + iterA);
  exprs.push_back(bx * cfg.BN + (blockRepIterB * config.at(KEY_BLOCK_LAYOUT_N) + warp_x * config.at(KEY_WARP_SCATTER_WIDTH_B)) * config.at(KEY_WARP_LAYOUT_N) + 
                                 warpRepIterB * config.at(KEY_WARP_LAYOUT_N) + lane_x * config.at(KEY_THREAD_SCATTER_WIDTH_B) + iterB);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getRegToSharedMapSMC(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterA = builder.getAffineDimExpr(dimCount++);
  auto blockRepIterB = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterA = builder.getAffineDimExpr(dimCount++);
  auto warpRepIterB = builder.getAffineDimExpr(dimCount++);
  auto iterA = builder.getAffineDimExpr(dimCount++);
  auto iterB = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  auto reTh = tools::mapUtils::reshapeBlock(tid, {config.at(KEY_LOCAL_SPLIT_U), cfg.THREAD_NUM/config.at(KEY_LOCAL_SPLIT_U)});
  auto warp_y = tools::mapUtils::wapr_y(reTh[1], config);
  auto warp_x = tools::mapUtils::wapr_x(reTh[1], config);
  auto lane_y = tools::mapUtils::lane_y(reTh[1], config);
  auto lane_x = tools::mapUtils::lane_x(reTh[1], config);
  exprs.push_back(reTh[0]);
  exprs.push_back((blockRepIterA * config.at(KEY_BLOCK_LAYOUT_M) + warp_y * config.at(KEY_WARP_SCATTER_WIDTH_A)) * config.at(KEY_WARP_LAYOUT_M) + 
                   warpRepIterA * config.at(KEY_WARP_LAYOUT_M) + lane_y * config.at(KEY_THREAD_SCATTER_WIDTH_A) + iterA);
  exprs.push_back((blockRepIterB * config.at(KEY_BLOCK_LAYOUT_N) + warp_x * config.at(KEY_WARP_SCATTER_WIDTH_B)) * config.at(KEY_WARP_LAYOUT_N) + 
                   warpRepIterB * config.at(KEY_WARP_LAYOUT_N) + lane_x * config.at(KEY_THREAD_SCATTER_WIDTH_B) + iterB);
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getReduceMapSMC(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iterSplitU = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  exprs.push_back(iterSplitU);
  if (config.at(KEY_REDUCE_C_CONTINUOUS)) {
    auto tid_ = tid + iter * cfg.THREAD_NUM;
    auto sh_store = tools::mapUtils::reshapeBlock(tid_, {cfg.GLOB_STORE_ROW_THREAD_NUM, cfg.GLOB_STORE_COL_THREAD_NUM});
    // llvm::outs() << sh_store[0] << "   " << sh_store[1] << "\n";
    exprs.push_back(sh_store[0]);
    exprs.push_back(sh_store[1] * config.at(KEY_GLOB_STORE_WIDTH));
  } else {
    auto sh_store = tools::mapUtils::reshapeBlock(tid, {cfg.BM, cfg.THREAD_NUM/cfg.BM});
    exprs.push_back(sh_store[0]);
    exprs.push_back(iter * cfg.GLOB_STORE_ROW_WIDTH + sh_store[1] * config.at(KEY_GLOB_STORE_WIDTH));
  }
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}

mlir::AffineMap MatmulOptimizer::getReduceMapRegC(mlir::OpBuilder& builder, const std::map<std::string, int>& config) {
  int dimCount = 0;
  auto by = builder.getAffineDimExpr(dimCount++);
  auto bx = builder.getAffineDimExpr(dimCount++);
  auto tid = builder.getAffineDimExpr(dimCount++);
  auto iter = builder.getAffineDimExpr(dimCount++);
  MatmulConfigUtils cfg(config);
  llvm::SmallVector<mlir::AffineExpr> exprs;
  if (config.at(KEY_REDUCE_C_CONTINUOUS)) {
    auto tid_ = tid + iter * cfg.THREAD_NUM;
    auto sh_store = tools::mapUtils::reshapeBlock(tid_, {cfg.GLOB_STORE_ROW_THREAD_NUM, cfg.GLOB_STORE_COL_THREAD_NUM});
    exprs.push_back(by * cfg.BM + sh_store[0]);
    exprs.push_back(bx * cfg.BN + sh_store[1] * config.at(KEY_GLOB_STORE_WIDTH));
  } else {
    auto sh_store = tools::mapUtils::reshapeBlock(tid, {cfg.BM, cfg.THREAD_NUM/cfg.BM});
    exprs.push_back(by * cfg.BM + sh_store[0]);
    exprs.push_back(bx * cfg.BN + iter * cfg.GLOB_STORE_ROW_WIDTH + sh_store[1] * config.at(KEY_GLOB_STORE_WIDTH));
  }
  return mlir::AffineMap::get(dimCount, 0, llvm::ArrayRef<mlir::AffineExpr>(exprs), builder.getContext()); 
}


void MatmulOptimizer::applyOptimzer(mlir::ModuleOp& module, std::map<std::string, int> config) {
  mlir::OpBuilder builder(module);
  for (auto& matmul : matmuls) {
    matmul->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
    auto loops = matmulLoops[matmul];
    auto loopM = loops[0], loopN = loops[1], loopK = loops[2];
    auto buffers = matmulBuffers[matmul];
    auto A = buffers.A, B = buffers.B, C = buffers.C;
    LOG_DEBUG("===== original mlir =======\n",module);

    auto m_axes = Rewriter::split(loopM, 3, {config["THREAD_SIZE_M"], config["BLOCK_SIZE_M"]});
    auto n_axes = Rewriter::split(loopN, 3, {config["THREAD_SIZE_N"], config["BLOCK_SIZE_N"]});
    auto m_outer = m_axes[0], m_mider = m_axes[1], m_inner = m_axes[2];
    auto n_outer = n_axes[0], n_mider = n_axes[1], n_inner = n_axes[2];
    Rewriter::reorder({m_outer, n_outer, m_mider, n_mider, m_inner, n_inner});
    LOG_DEBUG("===== after split & reorder =======\n",module);

    auto gridLevel = Rewriter::parallel({m_outer, n_outer});
    auto blockLevel = Rewriter::parallel({m_mider, n_mider});
    LOG_DEBUG("===== after parallel =======\n",module);

    std::vector<mlir::affine::AffineForOp> tileCLoops{m_inner, n_inner};
    auto regC = Rewriter::bufferizeLoopCarryVar(loopK, tileCLoops, "regC");
    LOG_DEBUG("===== after bufferizeLoopCarryVar =======\n",module);

    auto k_axes = Rewriter::split(loopK, 3, {config["LOCAL_SPLIT_U"], config["BLOCK_SIZE_K"]});
    auto k_outer = k_axes[0], k_mider = k_axes[1], k_inner = k_axes[2];
    LOG_DEBUG("===== after split =======\n",module);

    Rewriter::loopToParallelZ(k_inner, blockLevel);
    LOG_DEBUG("===== after loopToParallelZ =======\n",module);

    Rewriter::reorder({k_outer, k_mider, m_inner, n_inner});
    LOG_DEBUG("===== after reorder =======\n",module);

    int64_t blockThreads = Analyzer::getThreadPerBlock(blockLevel);
    auto glob_load_total_width_a = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_M"] / blockThreads;
    auto glob_load_total_width_b = config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_N"] / blockThreads;
    auto elementA = A.getType().dyn_cast<mlir::MemRefType>().getElementType();
    auto elementB = B.getType().dyn_cast<mlir::MemRefType>().getElementType();

    auto regB = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {config["THREAD_SIZE_N"]}, elementB, "regB");
    auto regA = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {config["THREAD_SIZE_M"]}, elementA, "regA");
    auto tempB = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {glob_load_total_width_b}, elementB, "tempB");
    auto tempA = Rewriter::alloc_buffer(blockLevel, MemorySpace::local, {glob_load_total_width_a}, elementA, "tempA");
    auto smB = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_N"]}, elementB, "smB");
    auto smA = Rewriter::alloc_buffer(gridLevel, MemorySpace::shared, {config["BLOCK_SIZE_K"], config["BLOCK_SIZE_M"]}, elementA, "smA");
    LOG_DEBUG("===== before alloc_buffer =======\n",module);

    int blockDimX = 0;
    int gridDimX = 0;
    Rewriter::parallelToOneDim(blockLevel, &blockDimX);
    // tools::opSetAttr(module,AttrGridDim, gridDimX);
    // tools::opSetAttr(module,AttrBlockDim, blockDimX);
    LOG_DEBUG("===== before parallelToOneDim =======\n",module);
    
    auto blockIdx = Analyzer::getParallelIdx(gridLevel);
    auto threadIdx = Analyzer::getParallelIdx(blockLevel);
    auto loadTileAMap = getGlobToTempMapA(builder, config);
    // test(loadTileAMap);
    auto loadTileA = Rewriter::read(A, tempA, loadTileAMap, {blockIdx[0], threadIdx[0], k_outer.getInductionVar()}, 
                                    {config["GLOB_LOAD_WIDTH_A"]}, k_outer, Position::begin);
    auto loadTileBMap = getGlobToTempMapB(builder, config);
    // test(loadTileBMap);
    auto loadTileB = Rewriter::read(B, tempB, loadTileBMap, {blockIdx[1], threadIdx[0], k_outer.getInductionVar()}, 
                                    {config["GLOB_LOAD_WIDTH_B"]}, loadTileA, Position::after);
    LOG_DEBUG("===== before read A/B =======\n",module);

    auto storeTileAMap = getTempToSharedMapSMA(builder, config);
    // test(storeTileAMap);
    auto storeTileA = Rewriter::write(tempA, smA, storeTileAMap, {threadIdx[0]}, {config["GLOB_LOAD_WIDTH_A"]}, loadTileB, Position::after);
    auto storeTileBMap = getTempToSharedMapSMB(builder, config);
    // test(storeTileBMap);
    auto storeTileB = Rewriter::write(tempB, smB, storeTileBMap, {threadIdx[0]}, {config["GLOB_LOAD_WIDTH_B"]}, storeTileA, Position::after);
    auto gpuBarrierPrefix = Rewriter::barrier(loadTileA, Position::before);
    auto gpuBarrierSuffix = Rewriter::barrier(storeTileB, Position::after);
    LOG_DEBUG("===== write A/B =======\n",module);

    auto loadFragAMap = getSharedToRegMapSMA(builder, config);
    // test(loadFragAMap);
    auto loadFragA = Rewriter::read(smA, regA, loadFragAMap, {threadIdx[0], k_mider.getInductionVar()}, 
                                    {config["WARP_SCATTER_WIDTH_A"], config["THREAD_SCATTER_WIDTH_A"]}, k_mider, Position::begin);
    auto loadFragBMap = getSharedToRegMapSMB(builder, config);
    // test(loadFragBMap);
    auto loadFragB = Rewriter::read(smB, regB, loadFragBMap, {threadIdx[0], k_mider.getInductionVar()}, 
                                    {config["WARP_SCATTER_WIDTH_B"], config["THREAD_SCATTER_WIDTH_B"]}, loadFragA, Position::after);
    LOG_DEBUG("===== read sh_A/B =======\n",module);

    auto cacheRead = getCalculateMapReg(builder, config);
    // test(cacheRead);
    Rewriter::cache_read(n_inner, A, regA, cacheRead, {m_inner.getInductionVar()});
    Rewriter::cache_read(n_inner, B, regB, cacheRead, {n_inner.getInductionVar()});
    LOG_DEBUG("===== load regA & cache_read =======\n",module);

    auto writeCbody = Rewriter::get_write(blockLevel, C);
    assert(writeCbody.size() == 1);
    auto m_inner_axes = Rewriter::split(writeCbody[0][0], 3, {config["WARP_SCATTER_WIDTH_A"], config["THREAD_SCATTER_WIDTH_A"]});
    auto n_inner_axes = Rewriter::split(writeCbody[0][1], 3, {config["WARP_SCATTER_WIDTH_B"], config["THREAD_SCATTER_WIDTH_B"]});
    auto m_inner_0 = m_inner_axes[0], m_inner_1 = m_inner_axes[1], m_inner_2 = m_inner_axes[2];
    auto n_inner_0 = n_inner_axes[0], n_inner_1 = n_inner_axes[1], n_inner_2 = n_inner_axes[2];
    Rewriter::reorder({m_inner_0, n_inner_0, m_inner_1, n_inner_1, m_inner_2, n_inner_2});
    LOG_DEBUG("===== load split & reorder regC to C =======\n",module);

    mlir::Value smC, regC_;
    if (config["LOCAL_SPLIT_U"] > 1) {
      auto elementC = C.getType().dyn_cast<mlir::MemRefType>().getElementType();
      regC_ = Rewriter::alloc_buffer(/*parallelLevel*/blockLevel, MemorySpace::local, {config["THREAD_SIZE_M"] * config["THREAD_SIZE_N"]}, elementC, "regC");
      smC = Rewriter::alloc_buffer(/*parallelLevel*/gridLevel, MemorySpace::shared, {config["LOCAL_SPLIT_U"], config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]}, elementC, "smC");

      auto cacheWriteShCMap = getRegToSharedMapSMC(builder, config);
      // test(cacheWriteShCMap);
      Rewriter::cache_write(m_inner_0, C, smC, cacheWriteShCMap, 
                            {threadIdx[0], m_inner_0.getInductionVar(),n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), 
                            n_inner_1.getInductionVar(), m_inner_2.getInductionVar(), n_inner_2.getInductionVar()});
      LOG_DEBUG("===== load cache_write regC to C =======\n",module);

      auto reduceCMap = getReduceMapSMC(builder, config);
      // test(reduceCMap);
      auto reduceCLoop = Rewriter::splitUReduce(smC, regC_, reduceCMap, {threadIdx[0]}, config["LOCAL_SPLIT_U"], config["GLOB_STORE_WIDTH"], m_inner_0, Position::after);
      auto reduceLoop_0 = reduceCLoop.first, reduceLoop_1 = reduceCLoop.second;
      LOG_DEBUG("===== load splitUReduce =======\n",module);

      auto writeCMap = getReduceMapRegC(builder, config);
      // test(writeCMap);
      Rewriter::splitUWrite(regC_, C, writeCMap, {blockIdx[0], blockIdx[1], threadIdx[0]}, config["LOCAL_SPLIT_U"], config["GLOB_STORE_WIDTH"], reduceLoop_1, Position::after);
      auto StoreBarrier1 = Rewriter::barrier(m_inner_0, Position::after);
      LOG_DEBUG("===== load write to C =======\n",module);
    } else {
      auto cacheWriteCMap = getRegToGlobMapC(builder, config);
      // test(cacheWriteCMap);
      Rewriter::cache_write(m_inner_0, C, C, cacheWriteCMap, {blockIdx[0], blockIdx[1], threadIdx[0], 
                            m_inner_0.getInductionVar(),n_inner_0.getInductionVar(), m_inner_1.getInductionVar(), 
                            n_inner_1.getInductionVar(),m_inner_2.getInductionVar(), n_inner_2.getInductionVar()});
      LOG_DEBUG("===== load cache_write regC to C =======\n",module);
    }

    Rewriter::vectorize(n_inner_2, config["THREAD_SCATTER_WIDTH_B"]);
    LOG_DEBUG("===== vectorize =======\n",module);
    
    mlir::affine::AffineForOp regRearForOp;
    std::vector<mlir::affine::AffineForOp> shPerfetchRegForOp, perfetchSharedForOp, regPerfetchRegForOp;
    if (config["SHARED_PREFETCH"]) {
      std::vector<mlir::affine::AffineForOp> shLoadRegForOps{loadTileA, loadTileB}, loadSharedForOps{storeTileA, storeTileB};
      std::vector<mlir::Value> smBufs{smA, smB};
      auto shResult = Rewriter::sharedPrefetch(k_outer, shLoadRegForOps, loadSharedForOps, k_mider, smBufs);
      smA = shResult.first[smA], smB = shResult.first[smB];
      shPerfetchRegForOp = shResult.second.first;
      perfetchSharedForOp = shResult.second.second;
      loadTileA = shLoadRegForOps[0], loadTileB = shLoadRegForOps[1];
      storeTileA = loadSharedForOps[0], storeTileB = loadSharedForOps[1];
      LOG_DEBUG("===== sharedPrefetch =======\n",module);
    }
  
    if (config["REG_PREFETCH"]) {
      std::vector<mlir::affine::AffineForOp> regLoadRegForOps{loadFragA, loadFragB};
      std::vector<mlir::Value> regBufs{regA, regB};
      auto regResult = Rewriter::registersPrefetch(k_mider, regLoadRegForOps, m_inner, regBufs);
      regA = regResult.first[regA], regB = regResult.first[regB];
      regPerfetchRegForOp = regResult.second.first;
      regRearForOp = regResult.second.second;
      loadFragA = regLoadRegForOps[0], loadFragB = regLoadRegForOps[1];
      LOG_DEBUG("===== registersPrefetch =======\n",module);
    }

    if (config["SHARED_PREFETCH"] && config["REG_PREFETCH"]) {
      Rewriter::doublePerfetchAdjust(perfetchSharedForOp, shPerfetchRegForOp, regPerfetchRegForOp, regRearForOp, {smA, smB}, {regA, regB});
      LOG_DEBUG("===== doublePerfetchAdjust =======\n",module);
    }

    if (config["LOCAL_SPLIT_U"] > 1) {
      Rewriter::bufferCombine({{smA, smB}, {smC}}, "smABC");
      Rewriter::bufferCombine({{regC}, {regC_}}, "regC");
      LOG_DEBUG("===== bufferCombine =======\n",module);
    }

    // int gridDims = 0;
    // Rewriter::parallelToOneDim(gridLevel, &gridDims);
    // tools::opSetAttr(module,AttrGridDim,gridDims);
    Rewriter::BlockMapping(gridLevel, config["BLOCK_MAPPING"]);
    LOG_DEBUG("===== BlockMapping gridLevel =======\n",module);

    Rewriter::unrollAttribute(module, config["UNROLL_NUM"]);
    LOG_DEBUG("===== unrollAttribute =======\n",module);
  }
}

}

