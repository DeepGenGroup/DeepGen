file(GLOB_RECURSE analysis_src ./Analysis/*.cc)
file(GLOB_RECURSE conversion_src ./Conversion/*.cc)
file(GLOB_RECURSE operators_src ./Operators/*.cc)
file(GLOB_RECURSE target_src ./Target/*.cc)
file(GLOB_RECURSE common_src ./Common/*.cc)
file(GLOB_RECURSE modelmanager_src ./ModelManager/*.cc)

add_library(kcg_runtime 
            ${analysis_src}
            ${conversion_src}
            ${operators_src}
            ${target_src}
            ${common_src}
            ${modelmanager_src}
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