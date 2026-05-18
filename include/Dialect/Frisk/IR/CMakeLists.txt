set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set(LLVM_TARGET_DEFINITIONS FriskEnums.td)
# mlir_tablegen(FriskEnums.h.inc -gen-enum-decls)
# mlir_tablegen(FriskEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS FriskOps.td)
mlir_tablegen(FriskOps.h.inc -gen-op-decls)
mlir_tablegen(FriskOps.cpp.inc -gen-op-defs)
add_mlir_doc(FriskOps FriskOps Frisk/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS FriskDialect.td)
mlir_tablegen(FriskDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FriskDialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(FriskDialect FriskDialect Frisk/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS FriskAttributes.td)
mlir_tablegen(FriskAttributes.h.inc -gen-attrdef-decls -dialect=frisk)
mlir_tablegen(FriskAttributes.cpp.inc -gen-attrdef-defs -dialect=frisk)

set(LLVM_TARGET_DEFINITIONS FriskTypes.td)
mlir_tablegen(FriskTypes.h.inc -gen-typedef-decls)
mlir_tablegen(FriskTypes.cpp.inc -gen-typedef-defs)

# mlir_tablegen(FriskInterfaces.h.inc -gen-op-interface-decls)
# mlir_tablegen(FriskInterfaces.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS FriskEnums.td)
mlir_tablegen(FriskEnums.h.inc -gen-enum-decls -dialect=frisk)
mlir_tablegen(FriskEnums.cpp.inc -gen-enum-defs -dialect=frisk)

add_public_tablegen_target(FriskTableGen)