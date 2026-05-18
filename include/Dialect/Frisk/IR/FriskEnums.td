#ifndef FRISK_ENUMS_TD
#define FRISK_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def MemorySpace_Local : I32EnumAttrCase<"Local", 0>;
def MemorySpace_Global  : I32EnumAttrCase<"Global", 1>;
def MemorySpace_Shared  : I32EnumAttrCase<"Shared", 3>;

def MemorySpaceEnum : I32EnumAttr<"MemorySpace",
    "Memory space for buffer allocation",
    [MemorySpace_Local, MemorySpace_Global, MemorySpace_Shared]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::frisk::attr";
}

def GemmWarpPolicy_Square : I32EnumAttrCase<"Square", 0>;
def GemmWarpPolicy_FullRow : I32EnumAttrCase<"FullRow", 1>;
def GemmWarpPolicy_FullCol : I32EnumAttrCase<"FullCol", 2>;

def GemmWarpPolicyEnum : I32EnumAttr<"GemmWarpPolicy",
    "Policy for GEMM warp partitioning",
    [GemmWarpPolicy_Square, GemmWarpPolicy_FullRow, GemmWarpPolicy_FullCol]> {
  let cppNamespace = "::mlir::frisk::attr";
}

#endif // FRISK_ENUMS_TD
