add_deepgengraph_library(DeepgengraphAnalysis
  Parallelism.cpp

  DEPENDS
  DeepgengraphTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  DeepgengraphIR 
)