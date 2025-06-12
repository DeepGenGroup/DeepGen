file(GLOB_RECURSE analysis_src ./Analysis/*.cc)
file(GLOB_RECURSE conversion_src ./Conversion/*.cc)
file(GLOB_RECURSE operators_src ./Operators/*.cc)
file(GLOB_RECURSE target_src ./Target/*.cc)
file(GLOB_RECURSE common_src ./Common/*.cc)
file(GLOB_RECURSE modelmanager_src ./ModelManager/*.cc)
add_compile_options(-g)
add_definitions(-DTORCH_MLIR_ENABLE_STABLEHLO)

add_library(kcg_runtime 
            ${analysis_src}
            ${conversion_src}
            ${operators_src}
            ${target_src}
            ${common_src}
        #     ${modelmanager_src}
            KernelCodeGen.cc
            )


#target_compile_options(kcg_runtime PRIVATE -frtti)
# MLIR有自己的一套RTTI，需要关掉才能正确继承MLIR中类
target_compile_options(kcg_runtime PUBLIC -fno-rtti)

set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
        )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

target_link_libraries(kcg_runtime
        PUBLIC
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
        )

message("KernelCodeGen_BINARY_DIR = ${KernelCodeGen_BINARY_DIR}")
# include_directories(${KernelCodeGen_BINARY_DIR}/rocm-stablehlo)

# add_mlir_library(model_manager
#         ./ModelManager/ModelManager.cc
# DEPENDS
#         StablehloLinalgTransformsPassIncGen
#         StablehloBase
#         StablehloOps
#         StablehloBaseIncGen
#         PassesIncGen
#         ChloDecompositionPatternsIncGen
#         StablehloAggressiveSimplificationPatternsIncGen
#         StablehloCompatibilityExpanderPatternsIncGen
#         StablehloComplexMathExpanderPatternsIncGen
#         StablehloLegalizeDeprecatedOpsPatternsIncGen
#         VhloToVersionPatterns
#         InterpreterOpsIncGen
#         ChloOpsIncGen
#         StablehloOpsIncGen
#         VhloOpsIncGen
# )

# get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# add_executable(kcg_modelProcessor ./ModelManager/main.cc  ${common_src})
# target_link_libraries(kcg_modelProcessor PUBLIC 
#         model_manager
#         ${dialect_libs}
#         ${conversion_libs}
#         ${translation_libs}
#         MLIRAnalysis
#         MLIRCallInterfaces
#         MLIRCastInterfaces
#         MLIRExecutionEngine
#         MLIRIR
#         MLIRLLVMCommonConversion
#         MLIRLLVMToLLVMIRTranslation
#         MLIRMemRefDialect
#         MLIRParser
#         MLIRPass
#         MLIRSideEffectInterfaces
#         MLIRTargetLLVMIRExport
#         MLIRTransforms
#         MLIRNVVMToLLVMIRTranslation
#         MLIRToLLVMIRTranslationRegistration
#         MLIRTargetLLVMIRImport
#         MLIRTargetLLVMIRExport
#         MLIRFuncToLLVM
#         libTorchMLIRCAPI.a
#         libTorchMLIRConversionPasses.a
#         libTorchMLIRConversionUtils.a
#         libTorchMLIRInitAll.a
#         libTorchMLIRRefBackend.a
#         libTorchMLIRTMTensorDialect.a
#         libTorchMLIRTMTensorPasses.a
#         libTorchMLIRTorchConversionDialect.a
#         libTorchMLIRTorchConversionPasses.a
#         libTorchMLIRTorchConversionToMLProgram.a
#         libTorchMLIRTorchDialect.a
#         libTorchMLIRTorchOnnxToTorch.a
#         libTorchMLIRTorchPasses.a
#         libTorchMLIRTorchToArith.a
#         libTorchMLIRTorchToLinalg.a
#         libTorchMLIRTorchToSCF.a
#         libTorchMLIRTorchToStablehlo.a
#         libTorchMLIRTorchToTensor.a
#         libTorchMLIRTorchToTMTensor.a
#         libTorchMLIRTorchToTosa.a
#         libTorchMLIRTorchUtils.a
#         libMLIRFuncAllExtensions.a
#         libMLIRTensorMeshShardingExtensions.a
#         libMLIRTensorAllExtensions.a
#         libMLIRFuncMeshShardingExtensions.a
#         libLLVMExtensions.a
#         libMLIRTransformDialectIRDLExtension.a
#         libMLIRTransformDebugExtension.a
#         libMLIRTransformLoopExtension.a
#         libMLIRTransformPDLExtension.a
#         libMLIRFuncInlinerExtension.a
#         MLIRTensorMeshShardingExtensions
#         StablehloPasses
#         TorchMLIRConversionUtils
#         MLIRTensorInferTypeOpInterfaceImpl
# )
