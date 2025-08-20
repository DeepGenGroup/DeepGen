add_deepgengraph_library(DeepgengraphToDeepgengraphTriton
  DeepgengraphToDeepgengraphTritonPass.cpp

  DEPENDS
  DeepgengraphConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  DeepgengraphIR 
  DeepgengraphTritonIR
)