#ifndef DEEPGENGRAPHTRITON_TYPES
#define DEEPGENGRAPHTRITON_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.td"

class DeepgengraphTritonTypeDef<string name, string _mnemonic, list<Trait> traits = []>
  :  TypeDef<DeepgengraphTriton_Dialect, name, traits> {
  let  mnemonic = _mnemonic;
}

def ASTT_PtrType : DeepgengraphTritonTypeDef<"Pointer", "ptr"> {
  let summary = "Pointer type";
  let description = [{}];
  let parameters = (ins "RankedTensorType":$pointee_type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "RankedTensorType":$pointee_type), [{
      return $_get(pointee_type.getContext(), pointee_type);
    }]>
  ];

  let assemblyFormat = "`<` $pointee_type `>`";
}

def ASTT_BlockPtrType : DeepgengraphTritonTypeDef<"BlockPointer", "block_ptr"> {
  let summary = "Block Pointer type";
  let description = [{}];
  let parameters = (ins "RankedTensorType":$pointee_type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "RankedTensorType":$pointee_type), [{
      return $_get(pointee_type.getContext(), pointee_type);
    }]>
  ];

  let assemblyFormat = "`{` $pointee_type `}`";
}



#endif // DEEPGENGRAPHTRITON_TYPES