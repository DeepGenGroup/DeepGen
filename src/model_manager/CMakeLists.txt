message("KernelCodeGen_BINARY_DIR = ${KernelCodeGen_BINARY_DIR}")
include_directories(${KernelCodeGen_BINARY_DIR}/rocm-stablehlo)
add_compile_options(-g)
add_mlir_library(model_manager
  ModelManager.cc
    
  DEPENDS
    StablehloLinalgTransformsPassIncGen
    StablehloBase
    StablehloOps
    StablehloBaseIncGen
    PassesIncGen
    ChloDecompositionPatternsIncGen
    StablehloAggressiveSimplificationPatternsIncGen
    StablehloCompatibilityExpanderPatternsIncGen
    StablehloComplexMathExpanderPatternsIncGen
    StablehloLegalizeDeprecatedOpsPatternsIncGen
    VhloToVersionPatterns
    InterpreterOpsIncGen
    ChloOpsIncGen
    StablehloOpsIncGen
    VhloOpsIncGen
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_executable(kcg_modelProcessor main.cc)
target_link_libraries(kcg_modelProcessor PUBLIC 
  model_manager
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRNVVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRTargetLLVMIRImport
  MLIRTargetLLVMIRExport
  MLIRFuncToLLVM
  libTorchMLIRCAPI.a
  libTorchMLIRConversionPasses.a
  libTorchMLIRConversionUtils.a
  libTorchMLIRInitAll.a
  libTorchMLIRRefBackend.a
  libTorchMLIRTMTensorDialect.a
  libTorchMLIRTMTensorPasses.a
  libTorchMLIRTorchConversionDialect.a
  libTorchMLIRTorchConversionPasses.a
  libTorchMLIRTorchConversionToMLProgram.a
  libTorchMLIRTorchDialect.a
  libTorchMLIRTorchOnnxToTorch.a
  libTorchMLIRTorchPasses.a
  libTorchMLIRTorchToArith.a
  libTorchMLIRTorchToLinalg.a
  libTorchMLIRTorchToSCF.a
  libTorchMLIRTorchToStablehlo.a
  libTorchMLIRTorchToTensor.a
  libTorchMLIRTorchToTMTensor.a
  libTorchMLIRTorchToTosa.a
  libTorchMLIRTorchUtils.a
)
