add_deepgengraph_library(DeepgengraphIR
  DeepgengraphOps.cpp
  DeepgengraphTypes.cpp

  DEPENDS
  DeepgengraphTableGen

  LINK_LIBS PUBLIC
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