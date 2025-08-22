#include "KernelCodeGen.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "config.h"

namespace KernelCodeGen {

KernelCodeGenerator::KernelCodeGenerator(const KernelCodeGenerator& other):
  KernelCodeGenerator(other.target,other.arch)
{;}


mlir::ModuleOp KernelCodeGenerator::createModule() {
  // 在同一个context下生成moduleop
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  return module;
}


std::vector<mlir::ModuleOp> KernelCodeGenerator::splitModule(mlir::ModuleOp& mod) {
  // split module 
  std::vector<mlir::ModuleOp> mods;
  auto kernels = getAllKernels(mod);
  for (auto kernel : kernels) {
    auto newMod = createModule();
    mlir::Block& moduleBlock = newMod.getBodyRegion().front();
    kernel->moveBefore(&moduleBlock, moduleBlock.end());
    mods.push_back(newMod);
  }
  return mods;
}

std::vector<std::string> KernelCodeGenerator::createKernels(mlir::ModuleOp& mod, std::vector<KernelData> kernelList) {
  // create all kernels
  std::vector<std::string> noSupKernels;
  for (auto kernel : kernelList) {
    std::vector<std::vector<int64_t>> inputShape(kernel.shapes.begin(), kernel.shapes.end()-kernel.outputArgNum);
    std::vector<std::vector<int64_t>> outputShape(kernel.shapes.end()-kernel.outputArgNum, kernel.shapes.end());
    std::vector<std::string> inputDType(kernel.dtypes.begin(), kernel.dtypes.end()-kernel.outputArgNum);
    std::vector<std::string> outputDType(kernel.dtypes.end()-kernel.outputArgNum, kernel.dtypes.end());
    if (kernel.type == "Matmul") {
      create<Operators::Matmul>(mod, inputShape, outputShape, inputDType, outputDType, kernel.isTrans, kernel.name);
    } else if (kernel.type == "Softmax") {
      create<Operators::Softmax>(mod, inputShape, outputShape, inputDType, outputDType, kernel.isTrans, kernel.name);
    } else {
      noSupKernels.push_back(kernel.type);
    }
  }
  // add attr
  for (auto kernel : kernelList) {
    mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp funcOp) {
      auto name = funcOp.getName().str();
      if (name == kernel.name) {
        mlir::OpBuilder builder(funcOp);
        funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
          auto forDesc = getStrAttr(forOp, FORDESC);
          if (forDesc == "x" || forDesc == "y") {
            forOp->setAttr(FORINCFUNC, builder.getStringAttr(name));
          }
        });
        return;
      }
    });
  }

  return noSupKernels;
}

bool KernelCodeGenerator::fusing(mlir::ModuleOp& mod, std::vector<FuseKernelData> fkList) {
  // kernel fusing
  for (auto kernels : fkList) {  // kernels
    auto fks = getSpecifiedKernels(mod, kernels.fuseKernels);
    // get batch kernels
    auto funcBatchs = getBatchFors(fks);
    // create new func and get new func args and mid vars
    mlir::OpBuilder builder(mod);
    builder.setInsertionPointToStart(mod.getBody());
    auto [newFuncOp, funcArgs, midVars] = createFuseFuncAndMidMems(builder, kernels);
    // 获取【新函数参数】需要替换的【旧函数参数】，根据【fks】提供的索引信息
    // funcargs{newArg0, newArg1, newArg2, newArg3} -> {{oldArg0}, {oldArg1}, {oldArg5}, {oldArg6}}
    auto argToArgs = collectOldMems(kernels.funcArgIndex, fks);
    // 【中间变量】替换【旧函数参数】
    // midvars{midVar0} -> {{oldArg2}, {oldArg3}, {oldArg4}}
    auto midToArgs = collectOldMems(kernels.midVarIndex, fks);
    // move ops in old func and fuse kernel
    moveOperation(newFuncOp, fks, funcArgs, midVars, argToArgs, midToArgs);
    // fuse batch forops
    fuseForOps(funcBatchs);
    LOG_DEBUG("===== batch fuse =====\n", mod);

    auto yloops = collectOpsInfuncOp<mlir::affine::AffineForOp>(newFuncOp, FORDESC, std::string{"y"});
    auto xloops = collectOpsInfuncOp<mlir::affine::AffineForOp>(newFuncOp, FORDESC, std::string{"x"});
    // x bufferize 将含有迭代变量的forx循环进行bufferize，形成foryx完美嵌套
    for (int i=0; i<xloops.size(); i++) {
      if (xloops[i].getNumIterOperands() > 0) {
        auto iterDescs = getArrayStrAttr(xloops[i], ITERVARDESC);  // 获取含有迭代变量的for循环的属性
        std::vector<mlir::affine::AffineForOp> upperLoops{yloops[i]};
        Rewriter::bufferizeLoopCarryVar(xloops[i], upperLoops, MemorySpace::global, iterDescs);
      }
    }
    LOG_DEBUG("===== X bufferize =====\n", mod);
    // 针对softmax这个算子，decouple 将y中含有两个x循环进行拆分，保证所有的fory下含有一个forx
    normalizeParaForOp(yloops);
    LOG_DEBUG("===== decouple =====\n", mod);
    // 先将算子进行更加小粒度的切分，变成一些reduce、elem-wise、binary、matmul级别的算子
    separateParaForOps(newFuncOp);   // ！！！！=== 重点设计 === ！！！！
    LOG_DEBUG("===== splitParaForOps =====\n", mod);
    // 将kernel内部可以进行融合的循环操作进行融合
    // 这个函数将用于将在for层面进行分析，将可以进行融合的for进行深度的融合（可行/未实现）
    fuseParaForOps(newFuncOp);   // ！！！！=== 重点设计 === ！！！！
    LOG_DEBUG("===== fuseParaForOps =====\n", mod);
  }
  return true;
}


bool KernelCodeGenerator::mapping(mlir::ModuleOp& mod, const std::map<std::string, std::map<std::string, int64_t>>& tileConfig) {
  // 并行化映射
  auto kernels = getAllKernels(mod);
  for (auto kernel : kernels) {
    // collect datas
    auto paraDims = getArrayStrAttr(kernel, PARALLELDIMS);  // 获取可并行维度
    auto yloops = collectOpsInfuncOp<mlir::affine::AffineForOp>(kernel, FORDESC, std::string{"y"});
    auto xloops = collectOpsInfuncOp<mlir::affine::AffineForOp>(kernel, FORDESC, std::string{"x"});
    // split & reorder & parallel 上一步做完必然会有fory和forx一一对应，大小相等
    std::vector<mlir::affine::AffineParallelOp> blockIdxs;  // collect all block parallel ops
    for (int i=0; i<yloops.size(); i++) {
      // split tile for
      auto funcName = getStrAttr(yloops[i], FORINCFUNC);
      std::vector<int64_t> tiley{tileConfig.at(funcName).at("BLOCK_SIZE_Y"), tileConfig.at(funcName).at("THREAD_SIZE_Y")};
      std::vector<int64_t> tilex{tileConfig.at(funcName).at("BLOCK_SIZE_X"), tileConfig.at(funcName).at("THREAD_SIZE_X")};
      auto bytyfory = Rewriter::split(yloops[i], tiley, {"blocky", "thready", "ttiley"});
      auto bxtxforx = Rewriter::split(xloops[i], tilex, {"blockx", "threadx", "ttilex"});
      // reorder tile for
      Rewriter::reorder({bytyfory[0], bxtxforx[0], bytyfory[1], bxtxforx[1], bytyfory[2], bxtxforx[2]});
      // parallel tile for
      std::vector<mlir::affine::AffineForOp> blockForOps;
      for (auto paraDim : paraDims) {
        if (paraDim == "y") blockForOps.push_back(bytyfory[0]);
        if (paraDim == "x") blockForOps.push_back(bxtxforx[0]);
      }
      auto blockIdx = Rewriter::parallel(blockForOps, BLOCKIDX, true);
      auto threadIdx = Rewriter::parallel({bytyfory[1], bxtxforx[1]}, THREADIDX);
      blockIdxs.push_back(blockIdx);
    }
    LOG_DEBUG("===== split & reorder & parallel block/thread tile =====\n", mod);

    // addLoopsToParallel 将batch添加进入parallel
    auto batchloops = collectOpsInfuncOp<mlir::affine::AffineForOp>(kernel, FORDESC, std::string{"batch"});
    if (batchloops.size()) {
      Rewriter::addLoopsToParallel(batchloops, blockIdxs);
      LOG_DEBUG("===== addLoopsToParallel =====\n", mod);
    }

    // 若block.x的循环还存在的话，就应该将这个循环下移到parallel内部
    auto bxloops = collectOpsInfuncOp<mlir::affine::AffineForOp>(kernel, FORDESC, std::string{"blockx"});
    if (bxloops.size()){
      blockForOpShiftDown(bxloops);
      LOG_DEBUG("===== blockForOpShiftDown =====\n", mod);
    }

    // 将多个parallel合成一个
    mlir::OpBuilder builder(kernel);
    // fuse blockidx
    auto blockParaOps = collectOpsInfuncOp<mlir::affine::AffineParallelOp>(kernel, AttrGPUIndex, BLOCKIDX);
    if (blockParaOps.size() > 1) {
      builder.setInsertionPointAfter(blockParaOps.back());
      auto blockIdx = fuseParallelOp(builder, blockParaOps);
    }
    // fuse threadidx
    auto threadParaOps = collectOpsInfuncOp<mlir::affine::AffineParallelOp>(kernel, AttrGPUIndex, THREADIDX);
    if (threadParaOps.size() > 1) {
      builder.setInsertionPointAfter(threadParaOps.back());
      auto threadIdx = fuseParallelOp(builder, threadParaOps);
    }
    LOG_DEBUG("===== fuseParallelOp blockIdx & threadIdx =====\n", mod);
    // erase single iteration forop and amend map of loadop or storeop
    eraseSingleIterForOps(kernel);
    LOG_DEBUG("===== eraseSingleIterForOps =====\n", mod);
    kernel->setAttr(std::string("func.state"), builder.getStringAttr("gpu"));
  }
  return true;
}


bool KernelCodeGenerator::optimize(mlir::ModuleOp& mod, const std::map<std::string, std::map<std::string, int64_t>>& tuneConfig) {
  // optimize
  // create opt tool pools
  // std::cout << "[lib] ========= optimize start " << std::endl;
  auto KernelTypes = Analyzer::collectFuncTypes(mod);
  // std::cout << "[lib] ========= optimize 0 " << std::endl;
  std::map<std::string, std::unique_ptr<Optimizer>> opts;
  for (auto KernelType : KernelTypes) {
    if (KernelType == "Matmul") {
      opts[KernelType] = std::make_unique<MatmulOptimizer>();
    } else if (KernelType == "FlashAttn") {
      opts[KernelType] = std::make_unique<FlashAttnOptimizer>();
    } else {
      llvm::errs() << "Optimization failed: Create Optimizer Failed.\n";
      return false;
    }
  }
  // std::cout << "[lib] ========= optimize mid " << std::endl;
  // optimize all kernel
  auto ntMap = Analyzer::collectNameTypeMap(mod);
  for (auto nt : ntMap) {  // {matmul1 : Matmul}
    auto opt = std::move(opts[nt.second]);
    auto funcOps = getSpecifiedKernels(mod, {nt.first});
    if (!opt->applicable(funcOps[0], tuneConfig.at(nt.first))) {
      // std::cout << "[lib] ========= optimize false " << std::endl;
      return false;   // collect matmul datas
    }
    opt->applyOptimzer(funcOps[0]);
  }
  // std::cout << "[lib] ========= optimize ends " << std::endl;
  return true;
}

bool KernelCodeGenerator::transform(mlir::ModuleOp& mod) {
  // dialect optimize
  mlir::MLIRContext* context = &(this->context);
  mlir::PassManager pm(context);
  pm.addPass(createParallelToGPUPass());
  pm.addPass(createCombineMemrefPass());
  pm.addPass(ReplaceAllocToGetglobalPass());
  pm.addPass(createAmendAllocaOpAddrSpacePass(this->target));
  pm.addNestedPass<func::FuncOp>(affine::createAffineLoopInvariantCodeMotionPass());
  pm.addNestedPass<func::FuncOp>(affine::createAffineLoopNormalizePass(true));
  pm.addPass(createAffineUnrollPass());
  pm.addPass(mlir::createCSEPass());  // 冗余消除
  pm.addPass(mlir::createSymbolDCEPass());  // 死代码消除/化简
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;
}

bool KernelCodeGenerator::lowering_(mlir::ModuleOp& mod) {
  // lowering
  mlir::MLIRContext* context = &(this->context);
  // == lowering to other dialect ==
  mlir::PassManager pm1(context);
  // affine to scf/vector
  pm1.addPass(mlir::createLowerAffinePass());
  pm1.addPass(mlir::createConvertVectorToGPUPass());
  pm1.addNestedPass<func::FuncOp>(mlir::createLoopInvariantCodeMotionPass());
  pm1.addPass(mlir::createCanonicalizerPass());         // 代数简化、死代码消除、冗余操作合并
  pm1.addPass(mlir::createCSEPass());                   // 冗余消除
  pm1.addPass(mlir::createSymbolDCEPass());             // 死代码消除/化简
  // scf to cf
  pm1.addPass(mlir::createSCFToControlFlowPass());
  if (mlir::failed(pm1.run(mod)))
    return false;
  
  if(this->target == Target::CUDA || this->target == Target::ROCm){
    // == lowering to llvm  ==
    mlir::PassManager pm2(context);
    // cf to llvm
    ConvertControlFlowToLLVMPassOptions cfOptions;
    cfOptions.indexBitwidth = INDEX_BIT_WIDTH;
    pm2.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));
    // vector to llvm
    pm2.addPass(createVectorToLLVMPass(INDEX_BIT_WIDTH));
    // memref to llvm
    FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
    memrefOptions.indexBitwidth = INDEX_BIT_WIDTH;
    // memrefOptions.useAlignedAlloc = true;
    pm2.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));
    pm2.addPass(createGlobalShmSetZeroPass());
    // func to llvm
    ConvertFuncToLLVMPassOptions funcOptions;
    funcOptions.indexBitwidth = INDEX_BIT_WIDTH;
    funcOptions.useBarePtrCallConv = true;
    pm2.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));
    pm2.addPass(createLLVMFuncOpAddGPUAttrPass(target));  // llvmfuncOp add nvvm/rocdl.kernel or nvvm.maxnid
    // gpu to rocdl/nvvm
    if(this->target == Target::CUDA || this->target == Target::ROCm ){
      pm2.addPass(createGPUToROCDLOrNVVMPass(this->target, INDEX_BIT_WIDTH));
    }
    // math to llvm
    pm2.addPass(mlir::createConvertMathToLLVMPass());  // ConvertMathToLLVMPassOptions options.approximateLog1p 精度换性能(true)
    // arith to llvm
    ArithToLLVMConversionPassOptions arithOptions;
    arithOptions.indexBitwidth = INDEX_BIT_WIDTH;
    pm2.addPass(mlir::createArithToLLVMConversionPass(arithOptions));
    // simipfy
    pm2.addPass(mlir::createCanonicalizerPass());
    pm2.addPass(mlir::createCSEPass());
    pm2.addPass(mlir::createSymbolDCEPass());
    if (mlir::failed(pm2.run(mod))){
      return false;
    }
  }
  LOG_DEBUG("========== after lowering_ =======\n",mod);
  return true;
}

std::string KernelCodeGenerator::readMLIRAndLowering(const std::string& filePath, bool isLLVM) {
  // read mlir file
  mlir::OwningOpRef<mlir::ModuleOp> mod = parseSourceFile<mlir::ModuleOp>(filePath, &(this->context));
  mlir::ModuleOp module = *mod;
  if(!isLLVM){
    this->transform(module);
    this->lowering_(module);
  }
  LOG_DEBUG("===== llvm: =======\n", module);
  auto path = this->translate(module);
  return path;
}

bool transforms(mlir::ModuleOp& mod, mlir::MLIRContext* context, Target target, const std::string& arch) {
  #define FLAG 1
  mlir::PassManager pm(context);
  // pm.addPass(createAddDebugLogPass());
  pm.addPass(createAddExternalLibPass(target, arch));      // 给mlir module添加lib属性

  // pm.addPass(createExtractAffineParallelPass());         // affine.parallel 根据内外层，将loopIvs 替换为bid、tid
  pm.addPass(createParallelToGPUPass());                   // affine paralleOp to GPU indexOp
  #if FLAG
  pm.addPass(createCombineMemrefPass());
  // pm.addPass(createFlattenMemrefPass());
  #endif
  pm.addPass(ReplaceAllocToGetglobalPass());
  #if FLAG
  pm.addPass(createAffineUnrollPass());                      // 对打了unroll属性的affine loop进行循环展开，展开次数和性能有很大关系
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());   // if的简化
  #endif
  pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopInvariantCodeMotionPass());   // 循环不变量移动
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createSimplifyAffineStructuresPass());  // 加入后会导致shm conflict 增加
  pm.addPass(createAmendAllocaOpAddrSpacePass(target));    // 按照 target 给 allocaOp 修改地址空间
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCSEPass());
  // pm.addPass(createCanonicalizerPass());  // 加入后会导致性能大幅下降。conflict增加
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;  
}


bool firstLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLowerAffinePass());                     // affine -> scf/vector
  // pm.addPass(mlir::createParallelLoopToGpuPass());               // scf.parallelOp -> gpu...
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(pm.run(mod)))
    return false;
  return true;  
}


bool secondLowering(mlir::ModuleOp& mod, mlir::MLIRContext* context, Target target) {
  mlir::PassManager pm(context);
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addNestedPass<mlir::func::FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCFToControlFlowPass());                    // scf -> cf

  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = INDEX_BIT_WIDTH;
  
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));        // cf -> llvm
  // pm.addPass(createConvertArithIndexToI64Pass());                      // 自定义 将arith中的constantOp的result为index类型的Op全部转成result为i64的op

  pm.addPass(createVectorToLLVMPass(/*indexBitwidth*/INDEX_BIT_WIDTH));                    // 自定义 vector to llvm pass
  // pm.addPass(mlir::createConvertVectorToLLVMPass());                       // vector -> llvm

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = INDEX_BIT_WIDTH;                              // 这个32会将malloc func的参数也定义为i32，以及将ptrtointOp的返回也是i32，llvm malloc func不支持i32
  // memrefOptions.useAlignedAlloc = true;                                    // 这个如果不开启的话，且上为i32，则llir转换失败，解决使用pass - createMallocFuncOpArgTypeI32ToI64Pass
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));  // memref -> llvm

  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createCSEPass());
  // pm.addPass(mlir::createSymbolDCEPass());

  ConvertFuncToLLVMPassOptions funcOptions;                                 // passes.h.inc文件中有通过tablegen生成的pass base类型 以及createxxx()
  funcOptions.indexBitwidth = INDEX_BIT_WIDTH;                              // func loewring 到 llvm 时，其index转到llvm上是使用i32类型
  funcOptions.useBarePtrCallConv = true;                                    // 使用裸指针，而不使用结构体指针表示memref类型
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));               // func -> llvm

  pm.addPass(createLLVMFuncOpAddGPUAttrPass(target));                       // llvmfuncOp add nvvm/rocdl.kernel or nvvm.maxnid
  pm.addPass(createGPUToROCDLOrNVVMPass(target, INDEX_BIT_WIDTH));          // GPU indexOp to rocdl/nvvm indexOp

  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = INDEX_BIT_WIDTH;
  pm.addPass(mlir::createArithToLLVMConversionPass(arithOptions));            // arith -> llvm
  // pm.addPass(createEraseRedundantUnCCastPass());                         // 手动写的去除多余UnrealizedCast
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());                // 内置去除多余cast的pass
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(createMallocFuncOpArgTypeI32ToI64Pass());                      // 将malloc 的 func 的函数签名换成 i64，ptrtointOp/callOp跟着换（因为如果强制使用malloci32，后续llvmtranslation报错，llvm malloc只支持i64）
  pm.addPass(createGlobalShmSetZeroPass());
  // pm.addPass(mlir::createLowerGpuOpsToROCDLOpsPass());
  // pm.addPass(createConvertGPUPrintToLLVMPass());
  
  // pm.addPass(mlir::createGpuToLLVMConversionPass());
  if (mlir::failed(pm.run(mod))){
    return false;
  }

  return true;  
}


bool KernelCodeGenerator::lowering(mlir::ModuleOp& mod/*, OUT std::vector<int>& griddims, OUT std::vector<int>& blockdims, OUT int& shmBytes*/) {
  // mod.dump();
  mlir::MLIRContext* context = &(this->context);
  LOG_DEBUG(" === start mlir =====\n",mod) ;

  transforms(mod, context, target, arch);
  LOG_DEBUG(" === after transforms =====\n",mod) ;

  firstLowering(mod, context);
  LOG_DEBUG(" === after firstLowering =====\n",mod) ;

  /*
  bool findKernel = false;
  auto op = mod.getOperation();
  int shmElements = 0;
  int elementBytes = 0;
  op->walk([&](mlir::memref::GlobalOp global){
    auto type = global.getTypeAttr().getValue();
    if(auto memtype = mlir::dyn_cast<mlir::MemRefType>(type)){
      shmElements = memtype.getShape()[0];
      auto etype = memtype.getElementType();
      elementBytes = etype.getIntOrFloatBitWidth() / 8;
    }
  });
#ifdef KCG_DEBUG
  std::cout << "[D] globalInfo: "<< shmElements << ":" << elementBytes << std::endl;
#endif
  shmBytes = (shmElements * elementBytes);
  op->walk([&](mlir::func::FuncOp f){
    if(findKernel){
      return;
    }
    if(f.getOperation()->hasAttr(AttrGridDim) ||f.getOperation()->hasAttr(AttrBlockDim) ){
      findKernel = true;
      griddims = tools::getIntArrayAttr(f,AttrGridDim);
      blockdims = tools::getIntArrayAttr(f,AttrBlockDim);
    }
  });
*/
  secondLowering(mod, context, target);
  LOG_DEBUG(" === after secondLowering =====\n",mod);
  
  return true;
}


std::string KernelCodeGenerator::translate(mlir::ModuleOp& mod) {

#if 1
  if (target == Target::ROCm) {
    const int wavesPerEU = 0;
    std::string llvmIR = std::move(translateMLIRToLLVMIR(mod, target, wavesPerEU));
    // llvm::outs() << " =========== after LLVM IR ============\n";
    // llvm::outs() << llvmIR << "\n";
    const std::string gfx_triple{"amdgcn-amd-amdhsa"};
    const std::string gfx_features{""};
    // const std::string gfx_features{"+code-object-v4"};
    return generateAmdgcnAndHsacoFromLLIRFile(llvmIR, "gfx" + arch, gfx_triple, gfx_features);
  }
  if(target == Target::CUDA){
    std::string llvmIR = std::move(translateMLIRToLLVMIR(mod, target));
    // llvm::outs() << " =========== after LLVM IR ============\n";
    // llvm::outs() << llvmIR << "\n";
    // const int capability = CUDA_CAP;
    const int version = PTXAS_VERSION;
    auto paths = generatePTXAndCubinFromLLIRFile(llvmIR, std::stoi(arch), version);
    return paths.second;
  }
  return "-";
#endif

#if 0  // 外部导入 mlir llvm
  mlir::MLIRContext testContext;
  testContext.loadDialect<
    func::FuncDialect,memref::MemRefDialect,scf::SCFDialect,gpu::GPUDialect, NVVM::NVVMDialect, 
    arith::ArithDialect,cf::ControlFlowDialect,LLVM::LLVMDialect,ROCDL::ROCDLDialect
  >();
  const char* llvmdialectfileName = "/home/xiebaokang/projects/mymlir/DeepGen/_tmp/our.mlir";
  auto temp = mlir::parseSourceFile<ModuleOp>(llvmdialectfileName,&testContext);
  auto testmod = temp.get();
  std::string llvmIR = std::move(translateMLIRToLLVMIR(testmod, target, 0));
  llvm::outs() << "======================llvm ir\n" << llvmIR << "\n";
  // const int capability = 80;
  // const int version = 81;
  // auto paths = generatePTXAndCubinFromLLIRFile(llvmIR, capability, version);
  // return paths.second;
#endif

#if 0  // 外部导入 llvm ir
  std::ifstream ifs("/home/xiebaokang/projects/mymlir/DeepGen/_tmp/our.llvm");
  std::stringstream buffer;
  if(ifs.is_open()){
    buffer << ifs.rdbuf();
    ifs.close();
  }
  auto llvmIR = buffer.str();
  // llvm::outs() << "======================llvm ir\n" << llvmIR << "\n";
  const int capability = 80;
  const int version = 81;
  auto paths = generatePTXAndCubinFromLLIRFile(llvmIR, capability, version);
  return paths.second;
#endif

}

}