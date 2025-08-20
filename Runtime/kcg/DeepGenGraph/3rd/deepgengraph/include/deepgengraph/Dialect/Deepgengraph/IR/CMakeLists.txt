# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set(LLVM_TARGET_DEFINITIONS DeepgengraphDialect.td)
# mlir_tablegen(DeepgengraphDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(DeepgengraphDialect.cpp.inc -gen-dialect-defs)
# add_mlir_doc(DeepgengraphDialect DeepgengraphDialect Deepgengraph/ -gen-dialect-doc)


# add_mlir_dialect(DeepgengraphOps deepgengraph)
# add_mlir_doc(DeepgengraphDialect DeepgengraphDialect Deepgengraph/ -gen-dialect-doc)
# # add_mlir_doc(DeepgengraphOps DeepgengraphOps Deepgengraph/ -gen-op-doc)


set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS DeepgengraphOps.td)
mlir_tablegen(DeepgengraphOps.h.inc -gen-op-decls)
mlir_tablegen(DeepgengraphOps.cpp.inc -gen-op-defs)
add_mlir_doc(DeepgengraphOps DeepgengraphOps Deepgengraph/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS DeepgengraphDialect.td)
mlir_tablegen(DeepgengraphDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DeepgengraphDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(DeepgengraphDialect DeepgengraphDialect Deepgengraph/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DeepgengraphTypes.td)
mlir_tablegen(DeepgengraphTypes.h.inc -gen-typedef-decls)
mlir_tablegen(DeepgengraphTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS DeepgengraphEnums.td)
mlir_tablegen(DeepgengraphEnums.h.inc -gen-enum-decls)
mlir_tablegen(DeepgengraphEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(DeepgengraphAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(DeepgengraphAttrs.cpp.inc -gen-attrdef-defs)


set(LLVM_TARGET_DEFINITIONS DeepgengraphInterfaces.td)
mlir_tablegen(DeepgengraphInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DeepgengraphInterfaces.cpp.inc -gen-op-interface-defs)


add_public_tablegen_target(DeepgengraphTableGen)