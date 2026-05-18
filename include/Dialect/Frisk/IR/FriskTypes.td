#ifndef FRISK_TYPES
#define FRISK_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "Dialect/Frisk/IR/FriskDialect.td"

class Frisk_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Frisk_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Frisk_BarrierType : Frisk_Type<"Barrier", "barrier"> {
  let summary = "Barrier handle type";
  let description = [{
    Lightweight handle representing a synchronization barrier in Frisk.
  }];
}





#endif // FRISK_TYPES
