#ifndef FRISK_INTERFACES
#define FRISK_INTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// def LayoutMap : CppType<"mlir::DenseMap<mlir::Value, ::mlir::Attribute>&">;

def LayoutInterface : OpInterface<"LayoutInterface"> {
  let description = [{}];
  let methods = [
    InterfaceMethod<
      /*description=*/[{}],
      /*retTy=*/"mlir::LogicalResult",
      /*methodName=*/"inferLayout",
      /*args=*/(ins "mlir::OpBuilder &": $builder, 
                    "mlir::DenseMap<mlir::Value, ::mlir::Attribute> &":$LayoutMap),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return mlir::success(false);
      }]
    >
  ];
}



#endif // FRISK_INTERFACES