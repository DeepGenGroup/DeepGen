set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS DeepgengraphTritonOps.td)
mlir_tablegen(DeepgengraphTritonOps.h.inc -gen-op-decls)
mlir_tablegen(DeepgengraphTritonOps.cpp.inc -gen-op-defs)
add_mlir_doc(DeepgengraphTritonOps DeepgengraphTritonOps DeepgengraphTriton/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS DeepgengraphTritonDialect.td)
mlir_tablegen(DeepgengraphTritonDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DeepgengraphTritonDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(DeepgengraphTritonDialect DeepgengraphTritonDialect DeepgengraphTriton/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DeepgengraphTritonTypes.td)
mlir_tablegen(DeepgengraphTritonTypes.h.inc -gen-typedef-decls)
mlir_tablegen(DeepgengraphTritonTypes.cpp.inc -gen-typedef-defs)


add_public_tablegen_target(DeepgengraphTritonTableGen)