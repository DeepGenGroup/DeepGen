add_deepgengraph_library(DeepgengraphTransforms
  Simplify.cpp
  ReplaceExpAndLog.cpp

  LowerComplexReduce.cpp

  EraseTypeInKernel.cpp

  MulScalarHoisting.cpp
  BroadcastTransform.cpp

  EquivalentTransform.cpp

  PermuteHoisting.cpp

  AnnotateParallelism.cpp

  ToMask.cpp
  
  Parallelize.cpp

  Tiling.cpp

  DynamicFor.cpp

  RecoverTypeInKernel.cpp



  UserReplicate.cpp

  DEPENDS
  DeepgengraphTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  DeepgengraphIR
)