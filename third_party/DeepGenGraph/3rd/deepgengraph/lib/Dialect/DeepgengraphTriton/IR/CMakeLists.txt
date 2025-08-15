add_deepgengraph_library(DeepgengraphTritonIR
  DeepgengraphTritonOps.cpp
  DeepgengraphTritonTypes.cpp

  DEPENDS
  DeepgengraphTritonTableGen

  LINK_LIBS PUBLIC
  DeepgengraphIR

  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)


# add_mlir_dialect_library(DeepgengraphIR
#   DeepgengraphDialect.cpp
#   DeepgengraphOps.cpp

#   DEPENDS
#   DeepgengraphTableGen

#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRArithDialect
#   MLIRMathDialect
#   MLIRSCFDialect
# )