set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Frisk)
add_public_tablegen_target(FriskTransformsIncGen)