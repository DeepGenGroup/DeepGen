add_deepgengraph_library(DeepgengraphTranslate
  to_triton.cpp

  DEPENDS
  DeepgengraphTritonTableGen

  LINK_LIBS PUBLIC
  DeepgengraphIR
  MLIRIR
)