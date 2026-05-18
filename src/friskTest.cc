#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"

#include <cmath>
#include <cassert>
#include <math.h>
#include <stdio.h>

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
// #include "Dialect/Frisk/Transforms/Passes.h"

using namespace mlir;

ModuleOp createFriskInitBarrierTest(MLIRContext &ctx) {
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();

  auto mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  auto kernelType = builder.getFunctionType(TypeRange{}, TypeRange{});
  auto kernel = builder.create<frisk::KernelOp>(loc, "init_barrier_test", kernelType);
  Block *kernelEntry = kernel.addEntryBlock();
  builder.setInsertionPointToStart(kernelEntry);

  constexpr int64_t kBarrierNum = 3;
  auto initBarrier = builder.create<frisk::InitBarrierOp>(loc, kBarrierNum);

  assert(initBarrier.getNum() == kBarrierNum && "init_barrier num attr mismatch");
  assert(static_cast<int64_t>(initBarrier.getBarriers().size()) == kBarrierNum &&
         "init_barrier result size mismatch");
  for (Value barrier : initBarrier.getBarriers()) {
    assert(isa<frisk::BarrierType>(barrier.getType()) &&
           "init_barrier result must be frisk.barrier");
  }

  Value barrier0 = initBarrier.getBarriers().front();
  builder.create<frisk::BarrierArrive>(loc, barrier0);
  builder.create<frisk::BarrierArriveExpectTx>(loc, barrier0, builder.getI64IntegerAttr(4096));

  builder.create<frisk::ForOp>(loc, 0, 8, 1, [&](Value iv) {
    auto c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto c2 = builder.create<arith::ConstantIndexOp>(loc, 2);
    auto add = builder.create<arith::AddIOp>(loc, iv, c1);
    auto mul = builder.create<arith::MulIOp>(loc, add, c2);
    auto div = builder.create<arith::DivSIOp>(loc, mul, c2);
    auto phase = builder.create<arith::XOrIOp>(loc, div, c1);
    builder.create<frisk::BarrierWait>(loc, barrier0, phase);
  });

  return mod;
}

template<int64_t BM=128, int64_t BN=128, int64_t THREAD_NUM=128>
ModuleOp createFriskAttn(MLIRContext &ctx, int64_t b, int64_t h, int64_t s, int64_t d) {
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto f32 = builder.getF32Type();
  auto f16 = builder.getF16Type();
  // create module
  auto mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());
  // module -> frisk.kernel
  auto arg_type = MemRefType::get({b, h, s, d}, f16, {}, static_cast<unsigned>(1));
  auto kernelType = builder.getFunctionType(TypeRange{arg_type, arg_type, arg_type, arg_type}, TypeRange{});
  auto kernel = builder.create<frisk::KernelOp>(loc, "attn", kernelType);
  Block *kernelEntry = kernel.addEntryBlock();
  builder.setInsertionPointToStart(kernelEntry);
  auto Q = kernelEntry->getArgument(0);
  auto K = kernelEntry->getArgument(1);
  auto V = kernelEntry->getArgument(2);
  auto O = kernelEntry->getArgument(3);
  // kernel body -> frisk.parallel
  auto parallel = builder.create<frisk::ParallelOp>(loc, ArrayRef<int64_t>{1, 32, s/BM}, THREAD_NUM);
  Block *parallelEntry = parallel.addEntryBlock();
  builder.setInsertionPointToStart(parallelEntry);
  auto bz = parallelEntry->getArgument(0);
  auto by = parallelEntry->getArgument(1);
  auto bx = parallelEntry->getArgument(2);
  // shared memory
  auto smemQ = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, d}, f16, 128, 3);
  auto smemK = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BN, d}, f16, 128, 3);
  auto smemV = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BN, d}, f16, 128, 3);
  auto smemO = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, d}, f16, 128, 3);
  // regsiters memory
  auto acc_s = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, BN}, f32, 0, 0);
  auto acc_s_cast = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, BN}, f16, 0, 0);
  auto acc_o = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, d}, f32, 0, 0);
  auto scores_max = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto scores_max_prev = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto scores_scale = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto scores_sum = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  auto logsum = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM}, f32, 0, 0);
  // init
  auto zore = builder.getAffineConstantExpr(0);
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto d2 = builder.getAffineDimExpr(2);
  auto mapQO = AffineMap::get(3, 0, ArrayRef<AffineExpr>{d0, d1, d2 * BM, zore}, &ctx);
  auto mapKV = AffineMap::get(3, 0, ArrayRef<AffineExpr>{d0, d1, d2, zore}, &ctx);
  auto Q_view = builder.create<frisk::BufferViewOp>(loc, Q, ValueRange({bz, by, bx}), mapQO, ArrayRef<int64_t>{1, 1, BM, d});
  builder.create<frisk::CopyOp>(loc, Q_view, smemQ.getResult());
  builder.create<frisk::FillOp>(loc, acc_o.getResult(), builder.getF32FloatAttr(0.0f));
  builder.create<frisk::FillOp>(loc, logsum.getResult(), builder.getF32FloatAttr(0.0f));
  builder.create<frisk::FillOp>(loc, scores_max.getResult(), builder.getF32FloatAttr(-INFINITY));
  // for
  builder.create<frisk::ForOp>(loc, 0, s, BN, [&](Value iv) {
    auto K_view = builder.create<frisk::BufferViewOp>(loc, K, ValueRange({bz, by, iv}), mapKV, ArrayRef<int64_t>{1, 1, BN, d});
    builder.create<frisk::CopyOp>(loc, K_view, smemK.getResult());
    // gemm
    builder.create<frisk::GemmOp>(loc, smemQ, smemK, acc_s, false, true);
    // ====== softmax ======
    // copy perv_max to max
    builder.create<frisk::CopyOp>(loc, scores_max, scores_max_prev);
    // fill max
    builder.create<frisk::FillOp>(loc, scores_max, builder.getF32FloatAttr(-INFINITY));
    // reduce max
    builder.create<frisk::ReduceOp>(loc, acc_s, scores_max, "max", 1);
    // m = max(score_max, score_max_prev)
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, scores_max, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_max_prev, blockIvs);
      auto m_elem = builder.create<arith::MaxNumFOp>(loc, elem1, elem2);
      builder.create<affine::AffineStoreOp>(loc, m_elem, scores_max, blockIvs);
    });
    // scale = exp(scores_max_prev - scores_max)
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, scores_max_prev, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_max, blockIvs);
      auto sub_elem = builder.create<arith::SubFOp>(loc, elem1, elem2);
      auto exp1 = builder.create<math::ExpOp>(loc, sub_elem);
      builder.create<affine::AffineStoreOp>(loc, exp1, scores_max, blockIvs);
    });
    // scc_s = exp(scc_s - scores_max)
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM, BN}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, acc_s, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_max, ValueRange({blockIvs[0]}));
      auto sub_elem = builder.create<arith::SubFOp>(loc, elem1, elem2);
      auto exp2 = builder.create<math::ExpOp>(loc, sub_elem);
      builder.create<affine::AffineStoreOp>(loc, exp2, acc_s, blockIvs);
    });
    // reduce sum
    builder.create<frisk::ReduceOp>(loc, acc_s, scores_sum, "add", 1);
    // logsum = logsum * scores_scale + scores_sum
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, logsum, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_scale, blockIvs);
      auto elem3 = builder.create<affine::AffineLoadOp>(loc, scores_sum, blockIvs);
      auto mul_elem = builder.create<arith::MulFOp>(loc, elem1, elem2);
      auto add_elem = builder.create<arith::AddFOp>(loc, mul_elem, elem3);
      builder.create<affine::AffineStoreOp>(loc, add_elem, logsum, blockIvs);
    });
    // copy acc_s to acc_s_cast
    builder.create<frisk::CopyOp>(loc, acc_s, acc_s_cast);
    // ====== rescale ======
    // acc_o = acc_o * scale
    builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM, d}, [&](ValueRange blockIvs) {
      auto elem1 = builder.create<affine::AffineLoadOp>(loc, acc_o, blockIvs);
      auto elem2 = builder.create<affine::AffineLoadOp>(loc, scores_scale, ValueRange({blockIvs[0]}));
      auto mul_elem = builder.create<arith::MulFOp>(loc, elem1, elem2);
      builder.create<affine::AffineStoreOp>(loc, mul_elem, acc_o, blockIvs);
    });
    // copy v to smemv
    auto V_view = builder.create<frisk::BufferViewOp>(loc, V, ValueRange({bz, by, iv}), mapKV, ArrayRef<int64_t>{1, 1, BN, d});
    builder.create<frisk::CopyOp>(loc, V_view, smemV);
    // gemm
    builder.create<frisk::GemmOp>(loc, acc_s_cast, smemV, acc_o, false, false);
  });
  // acc_o = acc_o / logsum
  builder.create<frisk::BlockOp>(loc, ArrayRef<int64_t>{BM, d}, [&](ValueRange blockIvs) {
    auto elem1 = builder.create<affine::AffineLoadOp>(loc, acc_o, blockIvs);
    auto elem2 = builder.create<affine::AffineLoadOp>(loc, logsum, ValueRange({blockIvs[0]}));
    auto div_elem = builder.create<arith::DivFOp>(loc, elem1, elem2);
    builder.create<affine::AffineStoreOp>(loc, div_elem, acc_o, blockIvs);
  });
  // copy acc_o to smemo
  builder.create<frisk::CopyOp>(loc, acc_o, smemO);
  // copy smemo to O
  auto O_view = builder.create<frisk::BufferViewOp>(loc, O, ValueRange({bz, by, bx}), mapQO, ArrayRef<int64_t>{1, 1, BM, d});
  builder.create<frisk::CopyOp>(loc, smemO, O_view);
  return mod;
}

template<int64_t BM=128, int64_t BN=128, int64_t BK=16, int64_t THREAD_NUM=128>
ModuleOp createFriskMatmul(MLIRContext &ctx, int64_t M, int64_t N, int64_t K) {
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto f32 = builder.getF32Type();
  // create module
  auto mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());
  // module -> frisk.kernel
  auto Aarg_type = MemRefType::get({M, K}, f32, {}, static_cast<unsigned>(1));
  auto Barg_type = MemRefType::get({K, N}, f32, {}, static_cast<unsigned>(1));
  auto Carg_type = MemRefType::get({M, N}, f32, {}, static_cast<unsigned>(1));
  auto kernelType = builder.getFunctionType(TypeRange{Aarg_type, Barg_type, Carg_type}, TypeRange{});
  auto kernel = builder.create<frisk::KernelOp>(loc, "Matmul", kernelType);
  Block *kernelEntry = kernel.addEntryBlock();
  builder.setInsertionPointToStart(kernelEntry);
  auto A = kernelEntry->getArgument(0);
  auto B = kernelEntry->getArgument(1);
  auto C = kernelEntry->getArgument(2);
  // kernel body -> frisk.parallel
  auto parallel = builder.create<frisk::ParallelOp>(loc, ArrayRef<int64_t>{M/BM, N/BN}, THREAD_NUM);
  Block *parallelEntry = parallel.addEntryBlock();
  builder.setInsertionPointToStart(parallelEntry);
  auto by = parallelEntry->getArgument(0);
  auto bx = parallelEntry->getArgument(1);
  // shared memory
  auto smemA = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, BK}, f32, 128, 3);
  auto smemB = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BK, BN}, f32, 128, 3);
  // regsiters memory
  auto acc = builder.create<frisk::AllocBufferOp>(loc, ArrayRef<int64_t>{BM, BN}, f32, 0, 0);
  // affine map
  auto zore = builder.getAffineConstantExpr(0);
  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto mapA = AffineMap::get(2, 0, ArrayRef<AffineExpr>{d0 * BM, d1}, &ctx);
  auto mapB = AffineMap::get(2, 0, ArrayRef<AffineExpr>{d1, d0 * BN}, &ctx);
  auto mapC = AffineMap::get(2, 0, ArrayRef<AffineExpr>{d0 * BM, d1 * BN}, &ctx);
  builder.create<frisk::FillOp>(loc, acc.getResult(), builder.getF32FloatAttr(0.0f));
  // for
  builder.create<frisk::ForOp>(loc, 0, K, BK, [&](Value iv) {
    auto A_view = builder.create<frisk::BufferViewOp>(loc, A, ValueRange({by, iv}), mapA, ArrayRef<int64_t>{BM, BK});
    builder.create<frisk::CopyOp>(loc, A_view, smemA.getResult());
    auto B_view = builder.create<frisk::BufferViewOp>(loc, B, ValueRange({bx, iv}), mapB, ArrayRef<int64_t>{BK, BN});
    builder.create<frisk::CopyOp>(loc, B_view, smemB.getResult());
    // gemm
    builder.create<frisk::GemmOp>(loc, smemA, smemB, acc, false, false);
  });
  auto C_view = builder.create<frisk::BufferViewOp>(loc, C, ValueRange({by, bx}), mapC, ArrayRef<int64_t>{BM, BN});
  builder.create<frisk::CopyOp>(loc, acc, C_view);
  return mod;
}

int main(int argc, char** argv) {
  if(argc <= 1){
    llvm::errs() << "error. usage : matmul|attn \n"; 
    return 0;
  }

  DialectRegistry registry;
  registerAllExtensions(registry);
  registerAllDialects(registry);
  MLIRContext ctx(registry);
  ctx.loadDialect<frisk::FriskDialect,
                  arith::ArithDialect,
                  affine::AffineDialect,
                  math::MathDialect,
                  func::FuncDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect,
                  gpu::GPUDialect>();

  // auto init_barrier_mod = createFriskInitBarrierTest(ctx);
  mlir::ModuleOp mod;
  if(argv[1] == std::string("matmul")){
    mod = createFriskMatmul(ctx, 2048, 2048, 2048);
  }
  else if(argv[1] == std::string("attn")){
    mod = createFriskAttn(ctx, 1, 32, 2048, 128);
  }
  // print
  // init_barrier_mod->print(llvm::outs());
  llvm::outs() << "\n";
  mod->print(llvm::outs());
  llvm::outs() << "\n";

  // lowering pass
//   PassManager pm(&ctx);
//   pm.enableVerifier(true);
//   pm.addPass(frisk::createConvertFriskToBasePass());
//   if (failed(pm.run(attn_mod))) {
//     llvm::errs() << "failed to run LoweringPass\n";
//     return 1;
//   }
//   if (failed(pm.run(matmul_mod))) {
//     llvm::errs() << "failed to run LoweringPass\n";
//     return 1;
//   }

  // print
  // init_barrier_mod->print(llvm::outs());
  // llvm::outs() << "\n";
  // attn_mod->print(llvm::outs());
  // llvm::outs() << "\n";
  // matmul_mod->print(llvm::outs());
  // llvm::outs() << "\n";
  return 0;
}
