#include "KernelCodeGen.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "config.h"

namespace KernelCodeGen {

std::unique_ptr<Optimizer> createOptimizer(const std::string& opName) {
  if (opName == "Matmul") {
    return std::make_unique<MatmulOptimizer>();
  }
  return nullptr;
}

KernelCodeGenerator::KernelCodeGenerator(const KernelCodeGenerator& other):
  KernelCodeGenerator(other.target,other.arch)
{
  ;
}


bool KernelCodeGenerator::optimize(mlir::ModuleOp &mod, std::map<std::string, int> config) {
  auto opNames = Analyzer::collectFuncNames(mod);
  for (auto opName : opNames) {
    auto opt = createOptimizer(opName);
    if (opt == nullptr) {
      llvm::errs() << "Optimization failed: Create Optimizer Failed.\n";
      return false;
    }
    if (!opt->applicable(mod)) return false;   // collect matmul datas
    opt->applyOptimzer(mod, config);
  }
  return true;
}


bool transforms(mlir::ModuleOp& mod, mlir::MLIRContext& context, Target target, const std::string& arch) {
#define FLAG 1
  mlir::PassManager pm(&context);
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



bool firstLowering(mlir::ModuleOp& mod, mlir::MLIRContext& context) {
  mlir::PassManager pm(&context);
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


bool secondLowering(mlir::ModuleOp& mod, mlir::MLIRContext& context, Target target) {
  mlir::PassManager pm(&context);
  // pm.addPass(createROCDLIdOpModifyPass());                      // 自定义 rocdl idop加attr (弃用)
  pm.addNestedPass<mlir::func::FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createConvertSCFToCFPass());                    // scf -> cf

  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = INDEX_BIT_WIDTH;
  
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfOptions));        // cf -> llvm
  // pm.addPass(createConvertArithIndexToI64Pass());                      // 自定义 将arith中的constantOp的result为index类型的Op全部转成result为i64的op

  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = INDEX_BIT_WIDTH;
  pm.addPass(mlir::createArithToLLVMConversionPass(arithOptions));            // arith -> llvm

  pm.addPass(createVectorToLLVMPass(/*indexBitwidth*/INDEX_BIT_WIDTH));                    // 自定义 vector to llvm pass
  // pm.addPass(mlir::createConvertVectorToLLVMPass());                       // vector -> llvm

  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = INDEX_BIT_WIDTH;                              // 这个32会将malloc func的参数也定义为i32，以及将ptrtointOp的返回也是i32，llvm malloc func不支持i32
  // memrefOptions.useAlignedAlloc = true;                                    // 这个如果不开启的话，且上为i32，则llir转换失败，解决使用pass - createMallocFuncOpArgTypeI32ToI64Pass
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(memrefOptions));  // memref -> llvm

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  ConvertFuncToLLVMPassOptions funcOptions;                                 // passes.h.inc文件中有通过tablegen生成的pass base类型 以及createxxx()
  funcOptions.indexBitwidth = INDEX_BIT_WIDTH;                              // func loewring 到 llvm 时，其index转到llvm上是使用i32类型
  funcOptions.useBarePtrCallConv = true;                                    // 使用裸指针，而不使用结构体指针表示memref类型
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcOptions));               // func -> llvm

  pm.addPass(createLLVMFuncOpAddGPUAttrPass(target));                       // llvmfuncOp add nvvm/rocdl.kernel or nvvm.maxnid
  pm.addPass(createGPUToROCDLOrNVVMPass(target, INDEX_BIT_WIDTH));          // GPU indexOp to rocdl/nvvm indexOp
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


bool KernelCodeGenerator::lowering(mlir::ModuleOp& mod, OUT std::vector<int>& griddims, OUT std::vector<int>& blockdims, OUT int& shmBytes) {
  // mod.dump();
  LOG_DEBUG(" === start mlir =====\n",mod) ;

  transforms(mod, context, target, arch);
  LOG_DEBUG(" === after transforms =====\n",mod) ;

  firstLowering(mod, context);
  LOG_DEBUG(" === after firstLowering =====\n",mod) ;

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
  std::cout << "[D] globalInfo: "<< shmElements << ":" << elementBytes << std::endl;
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

  secondLowering(mod, context, target);
  LOG_DEBUG(" === after secondLowering =====\n",mod) ;
  
  return true;
}


std::string KernelCodeGenerator::translate(mlir::ModuleOp& mod) {

#if 1
  if (target == Target::ROCm) {
    const int wavesPerEU = 0;
    std::string llvmIR = std::move(translateMLIRToLLVMIR(mod, target, wavesPerEU));

    const std::string gfx_triple{"amdgcn-amd-amdhsa"};
    const std::string gfx_features{""};
    return generateAmdgcnAndHsacoFromLLIRFile(llvmIR, "gfx" + arch, gfx_triple, gfx_features);
  } else {
    std::string llvmIR = std::move(translateMLIRToLLVMIR(mod, target));

    const int capability = CUDA_CAP;
    const int version = PTXAS_VERSION;
    auto paths = generatePTXAndCubinFromLLIRFile(llvmIR, capability, version);
    return paths.second;
  }
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