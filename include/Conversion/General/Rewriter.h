#ifndef _Rewriter_h_
#define _Rewriter_h_

#include "Common/Utils.h"
#include "Conversion/General/GeneralFuncs.h"
#include "Analysis/Analyzer.h"
#include "mlir/Support/LLVM.h"
#include <vector>

namespace KernelCodeGen {
namespace Rewriter {


std::vector<mlir::affine::AffineForOp> split(mlir::affine::AffineForOp forOp, 
                                              const std::vector<int64_t>& tile, 
                                              const std::vector<std::string>& forDescs={});

llvm::SmallVector<mlir::Value> bufferizeLoopCarryVar(mlir::affine::AffineForOp &carryVarLoop, 
                                               std::vector<mlir::affine::AffineForOp> &loops, 
                                               MemorySpace ms,
                                               const std::vector<std::string>& bufDescs);

void reorder(const std::vector<mlir::affine::AffineForOp> &forOp);

mlir::affine::AffineParallelOp parallel(std::vector<mlir::affine::AffineForOp> forOps, 
                                        std::string GPUIndexDesc="", 
                                        bool useApply=false);

void addLoopsToParallel(std::vector<mlir::affine::AffineForOp> loops, 
                        std::vector<mlir::affine::AffineParallelOp> &parallelOps, 
                        bool fuse=false);

std::vector<mlir::Value> allocBuffers(const std::vector<std::vector<int64_t>>& shapes, 
                          const std::vector<mlir::Type>& dtypes,
                          MemorySpace ms, 
                          const std::vector<std::string>& bufDescs, 
                          mlir::Operation* op, 
                          int alignment=KCG_ALIGNBYTE,
                          Position pos=Position::begin);

mlir::affine::AffineForOp loadToRegisters(mlir::Value src, 
                                          mlir::Value dst, 
                                          mlir::AffineMap map, 
                                          llvm::SmallVector<mlir::Value> operands, 
                                          std::vector<int64_t> widths, 
                                          mlir::affine::AffineForOp compute_at, 
                                          Position pos,
                                          const std::string& forDesc);

mlir::affine::AffineForOp loadFromRegisters(mlir::Value src, 
                                            mlir::Value dst, 
                                            mlir::AffineMap map, 
                                            llvm::SmallVector<mlir::Value> operands, 
                                            std::vector<int64_t> widths, 
                                            mlir::affine::AffineForOp compute_at, 
                                            Position pos,
                                            const std::string& forDesc);

mlir::gpu::BarrierOp barrier(mlir::affine::AffineForOp compute_at, Position pos);

mlir::gpu::BarrierOp barrier(mlir::OpBuilder builder);

mlir::affine::AffineForOp vectorize(mlir::affine::AffineForOp readOrWrite, int64_t width);

std::pair<mlir::affine::AffineForOp, mlir::affine::AffineForOp> splitUReduce(mlir::Value src, 
                                                                             mlir::Value dst, 
                                                                             mlir::AffineMap map, 
                                                                             llvm::SmallVector<mlir::Value> operands,
                                                                             int localSplitU, 
                                                                             int64_t globStoreWidth, 
                                                                             mlir::affine::AffineForOp compute_at, 
                                                                             Position pos);

mlir::affine::AffineForOp splitUWrite(mlir::Value src, 
                                      mlir::Value dst, 
                                      mlir::AffineMap map, 
                                      llvm::SmallVector<mlir::Value> operands, 
                                      int localSplitU, 
                                      int64_t globStoreWidth, 
                                      mlir::affine::AffineForOp compute_at, 
                                      Position pos, 
                                      const std::string& forDesc);

mlir::Value bufferCombine(std::vector<mlir::Value> buf1, std::vector<mlir::Value> buf2, std::string bufDesc);

std::array<mlir::Value, 2> blockMapping(mlir::affine::AffineParallelOp gridLevel, 
                                        const std::vector<int64_t>& blockTiles,
                                        const std::vector<int64_t>& gridShape, 
                                        int64_t groupM);

void cache_read(mlir::affine::AffineForOp scope, 
                mlir::Value src, 
                mlir::Value cached,
                mlir::AffineMap map, 
                llvm::SmallVector<mlir::Value> operands);

void cache_write(mlir::affine::AffineForOp scope, 
                 mlir::Value src, 
                 mlir::Value cached, 
                 mlir::AffineMap map, 
                 llvm::SmallVector<mlir::Value> operands);

void separateNoOpRelyForOp(std::vector<mlir::affine::AffineForOp> forOps);

std::vector<mlir::Value> createHierarchyInitBuf(mlir::affine::AffineForOp initForOp,
                                                const std::vector<int64_t>& newShape, 
                                                mlir::Operation* pos,
                                                MemorySpace space);

std::vector<std::vector<mlir::affine::AffineForOp>> get_write(mlir::affine::AffineParallelOp parallelLevel, 
                                                              mlir::Value dst);

void unrollAttribute(mlir::ModuleOp module, int unrollNum=16);

std::pair<std::vector<mlir::affine::AffineForOp>, std::vector<mlir::affine::AffineForOp>>
  sharedMemroyPrefetch(mlir::affine::AffineForOp &forKOp, 
                       std::vector<mlir::affine::AffineForOp> &ldRegForOps, 
                       std::vector<mlir::affine::AffineForOp> &ldSMForOps, 
                       mlir::affine::AffineForOp &calForOp, 
                       std::vector<mlir::Value> &buffers);

std::pair<std::vector<mlir::affine::AffineForOp>, mlir::affine::AffineForOp>
  registersPrefetch(mlir::affine::AffineForOp &forBKOp,
                    std::vector<mlir::affine::AffineForOp> &ldRegForOps, 
                    mlir::affine::AffineForOp &calForOp, 
                    std::vector<mlir::Value> &buffers);

void doubleBufferAdjust(std::vector<mlir::affine::AffineForOp> &pfLdSMForOps, 
                          std::vector<mlir::affine::AffineForOp> &pfLdRegForOps, 
                          std::vector<mlir::affine::AffineForOp> &regPfLdRegForOps, 
                          mlir::affine::AffineForOp &rearForOp);

}  // rewriter

}  // KernelCodeGen

#endif // _Rewriter_h_