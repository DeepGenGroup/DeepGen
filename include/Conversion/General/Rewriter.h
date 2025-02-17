#ifndef _Rewriter_h_
#define _Rewriter_h_

#include "Common/Utils.h"
#include "Conversion/General/GeneralFuncs.h"
#include "Analysis/Analyzer.h"
#include "mlir/Support/LLVM.h"
#include <vector>

namespace KernelCodeGen {
namespace Rewriter {


std::vector<mlir::affine::AffineForOp> split(mlir::affine::AffineForOp forOp, uint64_t num_output, std::vector<int64_t> &&factors);

mlir::Value bufferizeLoopCarryVar(mlir::affine::AffineForOp &carryVarLoop, std::vector<mlir::affine::AffineForOp> &loops, std::string bufDesc);

void reorder(const std::vector<mlir::affine::AffineForOp> &forOp);

// mlir::affine::AffineParallelOp parallel(const std::vector<mlir::affine::AffineForOp> &forOp);
mlir::affine::AffineParallelOp parallel(const std::vector<mlir::affine::AffineForOp>& forOps);

void loopToParallelZ(mlir::affine::AffineForOp loop, mlir::affine::AffineParallelOp &parallelOp);

llvm::SmallVector<mlir::Value> parallelToOneDim(mlir::affine::AffineParallelOp &parallelOp, int* outUpperBound = nullptr);

template <typename ParentOpType>
mlir::Value alloc_buffer(ParentOpType father, MemorySpace ms, const std::vector<int64_t> shape_, mlir::Type dtype, std::string bufDesc) {
  llvm::ArrayRef<int64_t> shape(shape_);
  int64_t flatSize = 1;
  for (auto dim : shape){
    flatSize *= dim;
  }
  llvm::ArrayRef<int64_t> flatShape{1, flatSize};
  mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));

  mlir::OpBuilder builder(father->getContext());
  builder.setInsertionPointToStart(father.getBody());
  if (ms == MemorySpace::local)
  {
    // register alloc
    auto op = builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape);
    op->setAttr(AttrBufDescription, builder.getStringAttr(bufDesc));
    op.setAlignment(KCG_ALIGNBYTE);
    return op->getResult(0);
  }
  else
  {
    // shm alloc
    auto op = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), tensorShape);
    op->setAttr(AttrBufDescription, builder.getStringAttr(bufDesc));
    op.setAlignment(KCG_ALIGNBYTE);
    return op->getResult(0);
  }
}

mlir::Value _inner_alloc_buffer(mlir::OpBuilder &builder, mlir::MemRefType &type);

template <typename ContextOp>
mlir::Value alloc_buffer(ContextOp contextOp, Position pos, MemorySpace ms, const std::vector<int64_t> shape_, mlir::Type dtype) {
  llvm::ArrayRef<int64_t> shape(shape_);
  mlir::MemRefType tensorShape = mlir::MemRefType::get(
      shape, dtype, {}, static_cast<int>(ms));

  switch (pos){
  case Position::before:
  {
    mlir::OpBuilder builder(contextOp);
    return _inner_alloc_buffer(builder, tensorShape);
    // return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
  }
  case Position::after:
  {
    mlir::OpBuilder builder(contextOp->getContext());
    builder.setInsertionPointAfter(contextOp);
    return _inner_alloc_buffer(builder, tensorShape);
    // return builder.create<mlir::memref::AllocaOp>(builder.getUnknownLoc(), tensorShape)->getResult(0);
  }
  default:
  {
    assert(false && "alloc_buffer error!");
  }
  }
}


mlir::affine::AffineForOp read(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                              std::vector<int64_t> widths, mlir::affine::AffineForOp compute_at, Position pos);

mlir::affine::AffineForOp write(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                                std::vector<int64_t> widths, mlir::affine::AffineForOp compute_at, Position pos);

mlir::affine::AffineForOp write(mlir::OpBuilder &builder, mlir::Value src, mlir::Value dst,
                                mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, int64_t width);

mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos);

mlir::gpu::BarrierOp barrier(mlir::OpBuilder builder);

mlir::affine::AffineForOp vectorize(mlir::affine::AffineForOp readOrWrite, int64_t width);

std::pair<mlir::affine::AffineForOp, mlir::affine::AffineForOp> 
splitUReduce(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands,
               int localSplitU, int64_t globStoreWidth, mlir::affine::AffineForOp compute_at, Position pos);

mlir::affine::AffineForOp splitUWrite(mlir::Value src, mlir::Value dst, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands, 
                                      int localSplitU, int64_t globStoreWidth, mlir::affine::AffineForOp compute_at, Position pos);

mlir::Value bufferCombine(std::vector<std::vector<mlir::Value>> buffers, std::string bufDesc);

void BlockMapping(mlir::affine::AffineParallelOp gridLevel, int64_t groupWidth, bool isCol=true);

void cache_read(mlir::affine::AffineForOp scope, 
    mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

void cache_write(mlir::affine::AffineForOp scope, 
    mlir::Value src, mlir::Value cached, mlir::AffineMap map, llvm::SmallVector<mlir::Value> operands);

std::vector<std::vector<mlir::affine::AffineForOp>> get_write(
  mlir::affine::AffineParallelOp parallelLevel, mlir::Value dst);

void unrollAttribute(mlir::ModuleOp module, int unrollNum);

std::pair<std::map<mlir::Value, mlir::Value, BufferCompare>, std::pair<std::vector<mlir::affine::AffineForOp>, std::vector<mlir::affine::AffineForOp>>>
sharedPrefetch(mlir::affine::AffineForOp &forOp, std::vector<mlir::affine::AffineForOp> &loadRegForOps, std::vector<mlir::affine::AffineForOp> &loadSharedForOps, 
               mlir::affine::AffineForOp &calculateForOp, std::vector<mlir::Value> buffers);

std::pair<std::map<mlir::Value, mlir::Value, BufferCompare>, std::pair<std::vector<mlir::affine::AffineForOp>, mlir::affine::AffineForOp>>
registersPrefetch(mlir::affine::AffineForOp &forOp, std::vector<mlir::affine::AffineForOp> &loadRegForOps, 
                       mlir::affine::AffineForOp &calculateForOp, std::vector<mlir::Value> buffers);

void doublePerfetchAdjust(std::vector<mlir::affine::AffineForOp> &shShPerfetchForOps, std::vector<mlir::affine::AffineForOp> &shRegPerfetchForOps, 
                          std::vector<mlir::affine::AffineForOp> &regPerfetchForOps, mlir::affine::AffineForOp &rearForOp, 
                          std::vector<mlir::Value> smBufs, std::vector<mlir::Value> regBufs);

}  // rewriter

}  // KernelCodeGen

#endif // _Rewriter_h_