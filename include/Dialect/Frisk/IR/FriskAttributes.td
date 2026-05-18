#ifndef FRISK_ATTRIBUTES_TD
#define FRISK_ATTRIBUTES_TD

include "Dialect/Frisk/IR/FriskDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/Frisk/IR/FriskEnums.td"

def MemorySpaceAttr : EnumAttr<Frisk_Dialect, MemorySpaceEnum, "memory_space"> {
    let cppNamespace = "::mlir::frisk";
}

def GemmWarpPolicyAttr : EnumAttr<Frisk_Dialect, GemmWarpPolicyEnum, "gemm_warp_policy"> {
    let cppNamespace = "::mlir::frisk";
    let summary = "GEMM warp partitioning policy";
}

def LayoutAttr : AttrDef<Frisk_Dialect, "Layout"> {
    let cppNamespace = "::mlir::frisk";
    let summary = "Describes a logical-to-physical mapping for tile layouts";
    let mnemonic = "layout";
    let assemblyFormat = "`<` $inputShape `,` $forwardIndex (`,` $forwardThread^)? (`,` $replicateSize^)? `>`";

    let parameters = (ins
        AttrParameter<"::mlir::DenseI64ArrayAttr", "input shape">:$inputShape,
        AttrParameter<"::mlir::AffineMapAttr", "forward index mapping">:$forwardIndex,
        OptionalParameter<"::mlir::AffineMapAttr", "thread mapping">:$forwardThread,
        OptionalParameter<"::mlir::IntegerAttr", "replication factor">:$replicateSize
    );

}

#endif // FRISK_ATTRIBUTES_TD
