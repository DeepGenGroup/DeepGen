#ifndef DEEPGENGRAPH_ENUMS
#define DEEPGENGRAPH_ENUMS


include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.td"

include "mlir/IR/EnumAttr.td" 

def DEEPGENGRAPH_RNG_DISTRIBUTION_UNIT_UNIFORM : I32EnumAttrCase<"UNIT_UNIFORM", 1>;
def DEEPGENGRAPH_RNG_DISTRIBUTION_UNIT_GUMBEL : I32EnumAttrCase<"UNIT_GUMBEL", 2>;
def Deepgengraph_RngDistribution : I32EnumAttr<"RngDistribution",
    "XLA PRNG distribution to be used.",
    [
      DEEPGENGRAPH_RNG_DISTRIBUTION_UNIT_UNIFORM,
      DEEPGENGRAPH_RNG_DISTRIBUTION_UNIT_GUMBEL
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::deepgengraph";
}

def Deepgengraph_RngDistributionAttr : EnumAttr<Deepgengraph_Dialect, Deepgengraph_RngDistribution, "rng_distribution">;



def DEEPGENGRAPH_REDUCE_ADD : I32EnumAttrCase<"ADD", 1>;
def DEEPGENGRAPH_REDUCE_MUL : I32EnumAttrCase<"MUL", 2>;
def DEEPGENGRAPH_REDUCE_ANY : I32EnumAttrCase<"ANY", 3>;
def Deepgengraph_ReduceType : I32EnumAttr<"ReduceType",
    "Reduce type",
    [
      DEEPGENGRAPH_REDUCE_ADD,
      DEEPGENGRAPH_REDUCE_MUL,
      DEEPGENGRAPH_REDUCE_ANY
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::deepgengraph";
}


def Deepgengraph_ReduceTypeAttr : EnumAttr<Deepgengraph_Dialect, Deepgengraph_ReduceType, "reduce_type">;



def DEEPGENGRAPH_CMP_GT : I32EnumAttrCase<"GT", 1>;
def DEEPGENGRAPH_CMP_GE : I32EnumAttrCase<"GE", 2>;
def Deepgengraph_CmpType: I32EnumAttr<"CmpType",
    "Compare Type",
    [
      DEEPGENGRAPH_CMP_GT,
      DEEPGENGRAPH_CMP_GE
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::deepgengraph";
}
def Deepgengraph_CmpTypeAttr : EnumAttr<Deepgengraph_Dialect, Deepgengraph_CmpType, "cmp_type">;

#endif // DEEPGENGRAPH_ENUMS