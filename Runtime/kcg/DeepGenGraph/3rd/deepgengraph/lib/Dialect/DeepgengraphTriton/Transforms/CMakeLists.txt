add_deepgengraph_library(DeepgengraphTritonTransforms
  SqueezeBlock.cpp
  # Blocking.cpp
  UserReplicate.cpp

  DEPENDS
  DeepgengraphTritonTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  DeepgengraphIR
  DeepgengraphTritonIR
)