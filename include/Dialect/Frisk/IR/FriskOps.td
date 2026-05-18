#ifndef FRISK_OPS
#define FRISK_OPS

include "Dialect/Frisk/IR/FriskDialect.td"
include "Dialect/Frisk/IR/FriskEnums.td"
include "Dialect/Frisk/IR/FriskAttributes.td"
include "Dialect/Frisk/IR/FriskTypes.td"
// include "Dialect/Frisk/IR/FriskInterfaces.td"

include "mlir/IR/OpBase.td"
// include "mlir/IR/AttrTypeBase.td"
// include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultRank


// Op Base
class Frisk_Op<string mnemonic, list<Trait> traits = []> : 
  Op<Frisk_Dialect, mnemonic, traits>;

def ImplicitFriskTerminator : SingleBlockImplicitTerminator<"EndOp">;

// KernelOp
def Frisk_KernelOp : Frisk_Op<"kernel", [ImplicitFriskTerminator, IsolatedFromAbove]> {
    let summary = "Kernel operation";
  let description = [{
    A simple kernel operation that represents a compute kernel.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$function_type
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
  ];

  let extraClassDeclaration = [{
    unsigned getNumArguments() {
      return getBody(0)->getNumArguments();
    }

    BlockArgument getArgument(unsigned idx) {
      return getBody(0)->getArgument(idx);
    }

    Block* addEntryBlock() {
      auto& region = getRegion();
      if (!region.empty()) {
        llvm_unreachable("entry block already exists");
      }
      auto* entry = new Block;
      region.push_back(entry);
      // block add args
      mlir::Type funcType = getFunctionType();
      if (auto funcTy = mlir::dyn_cast<FunctionType>(funcType)) {
        for (auto type : funcTy.getInputs()) {
          entry->addArgument(type, getLoc());
        }
      }
      mlir::OpBuilder builder(entry, entry->begin());
      builder.create<EndOp>(getLoc());
      return entry;
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// parallelOp
def Frisk_ParallelOp : Frisk_Op<"parallel", [
  HasParent<"KernelOp">,
  ImplicitFriskTerminator, AffineScope, 
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
]> {
  let summary = "parallel opeartion";
  let description = [{
    parallel
  }];

  let arguments = (ins
    I64Attr:$threads,
    DenseI64ArrayAttr:$ranges
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$ranges, "int64_t":$thread_num)>
  ];

  let extraClassDeclaration = [{
    std::vector<int64_t> getGrid() {
      std::vector<int64_t> ranges;
      for (auto range : getRangesAttr().asArrayRef()) {
        ranges.push_back(range);
      }
      return ranges;
    }

    unsigned getGridDims() { 
      return getRangesAttr().size(); 
    }

    int64_t getThreadNum() {
      return getThreads();
    }

    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }

    Block* addEntryBlock() {
      auto& region = getRegion();
      if (!region.empty()) {
        llvm_unreachable("entry block already exists");
      }
      auto* entry = new Block;
      region.push_back(entry);
      for (unsigned i=0; i<getGridDims(); i++) {
        entry->addArgument(IndexType::get(getContext()), getLoc());
      }
      mlir::OpBuilder builder(entry, entry->begin());
      builder.create<EndOp>(getLoc());
      return entry;
    }

    ::mlir::LogicalResult inferLayout(::mlir::OpBuilder &builder,
                                      llvm::DenseMap<::mlir::Value, ::mlir::Attribute> &layoutMap);
  }];

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;
}

// blockOp
def Frisk_BlockOp : Frisk_Op<"block", [Pure, ImplicitFriskTerminator, AffineScope]> {
  let summary = "block opeartion";
  let description = [{
    block
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$ranges
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$ranges, 
               CArg<"function_ref<void(ValueRange)>", "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn = function_ref<void(ValueRange)>;

    std::vector<int64_t> getBlockRanges() {
      std::vector<int64_t> ranges;
      for (auto range : getRangesAttr().asArrayRef()) {
        ranges.push_back(range);
      }
      return ranges;
    }

    int64_t getIterationNum() {
      int64_t iter_num = 1;
      for (int64_t range : getBlockRanges()) {
        iter_num *= range;
      }
      return iter_num;
    }

    int64_t getBlockDim() {
      return getRangesAttr().size();
    }

    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// forOp
def Frisk_ForOp : Frisk_Op<"for", [Pure, ImplicitFriskTerminator]> {
  let summary = "for opeartion";
  let description = [{
    for
  }];

  let arguments = (ins I64Attr:$lower, I64Attr:$upper, I64Attr:$step);

  let regions = (region SizedRegion<1>:$region);

  let builders = [OpBuilder<(ins "int64_t":$lowerBound, 
                  "int64_t":$upperBound,
              CArg<"int64_t", "1">:$step, 
              CArg<"function_ref<void(Value)>", "nullptr">:$bodyBuilder)>];
  
  let extraClassDeclaration = [{
    using BodyBuilderFn = function_ref<void(Value)>;

    BlockArgument getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }

    // /// Returns information about the lower bound as a single object.
    // int64_t getLowerBound() { return getProperties().lower; }

    // /// Returns information about the upper bound as a single object.
    // int64_t getUpperBound() { return getProperties().upper; }

    // /// Returns loop step.
    // int64_t getStep() { return getProperties().step; }
  }];

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;
}

// EndOp
def Frisk_EndOp : Frisk_Op<"end", [Terminator, ReturnLike]> {
  let summary = "Invisible kernel termination operation";
  let description = [{
    An invisible terminator operation for frisk kernels that dose not appear in printed IR.
  }];

  let arguments = (ins);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  let hasVerifier = 0;
  let assemblyFormat = "attr-dict";
}

// gemmOp
def Frisk_GemmOp: Frisk_Op<"gemm", [MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>]> {
  let summary = "General matrix multiplication operation";
  let description = [{}];
  let arguments = (ins
    AnyMemRef:$A,  // shm
    AnyMemRef:$B,
    AnyMemRef:$C,

    BoolAttr:$transA,
    BoolAttr:$transB 
  );

  let results = (outs);

  let extraClassDeclaration = [{
    Value getMatrixA() { return getA(); }
    Value getMatrixB() { return getB(); }
    Value getMatrixC() { return getC(); }
    ValueRange getAllMatrix() {
      return ValueRange(getOperation()->getOperands());
    }
    ::mlir::LogicalResult inferLayout(::mlir::OpBuilder &builder,
                                      llvm::DenseMap<::mlir::Value, ::mlir::Attribute> &layoutMap);
  }];


  let hasVerifier = 1;
  let assemblyFormat = " `(` $A `,` $B `)` `to` $C attr-dict `:` type($A) `,` type($B) `,` type($C)";
}

// ifOp
def Frisk_IfOp : Frisk_Op<"if", [Pure, ImplicitFriskTerminator]> {
  let summary = "if branch operation";
  let description = [{
    if-else branch operation with IntegerSet condition.
  }];

  let arguments = (ins IntegerSetAttr:$condition, Variadic<Index>:$operands);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let extraClassDeclaration = [{
    bool hasElse() { return !getElseRegion().empty(); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// gemmwaitOp
def Frisk_GemmWaitOp : Frisk_Op<"gemm_wait", [MemoryEffects<[MemRead]>]> {
  let summary = "wait for gemm write completion";
  let description = [{
    Wait until the specified memref buffer is written back after gemm commit.
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$buffer);

  let extraClassDeclaration = [{
    Value getWaitBuffer() { return getBuffer(); }
  }];

  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
}

// bufferViewOp
def Frisk_BufferViewOp : Frisk_Op<"buffer_view", [Pure]> {
  let summary = "create a memref view from source buffer and slice";
  let description = [{
    Create a view memref from a source memref using dynamic indices, an affine
    map and per-dimension ranges.
    The result memref keeps source element type and memory space, and uses
    ranges as result shape.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$indices,
    AffineMapAttr:$indexMap,
    DenseI64ArrayAttr:$ranges
  );

  let results = (outs
    Res<AnyMemRef>:$view
  );

  let builders = [
    OpBuilder<(ins "Value":$source, "ValueRange":$indices, "AffineMap":$indexMap, "ArrayRef<int64_t>":$ranges)>,
    OpBuilder<(ins "Value":$source, "ValueRange":$indices, "AffineMapAttr":$indexMap, "DenseI64ArrayAttr":$ranges)>
  ];

  let extraClassDeclaration = [{
    MemRefType getSourceType() { return cast<MemRefType>(getSource().getType()); }
    MemRefType getViewType() { return cast<MemRefType>(getView().getType()); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// allocOp
// def Frisk_AllocBufferOp: Frisk_Op<"alloc_buffer", [Pure]> {
//   let summary = "allocate buffer operation";
//   let description = [{ 
//     Allocates a buffer with the specified shape, element type and optional attributes.
//   }];

//   let arguments = (ins 
//     DenseI64ArrayAttr:$shape,
//     TypeAttr:$elementType,
//     DefaultValuedAttr<I64Attr, "0">:$alignment,
//     // DefaultValuedAttr<I64Attr, "0">:$memorySpace
//     DefaultValuedAttr<MemorySpaceAttr, "::mlir::frisk::attr::MemorySpace::Local">:$memorySpace
//   );

//   let results = (outs 
//     Res<AnyMemRef>:$result
//   );

//   let builders = [
//     OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
//     // OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment)>,
//     // OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment, "int64_t":$memorySpace)>
//     OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "Attribute":$memorySpace)>
//   ];

//   let extraClassDeclaration = [{
//     MemRefType getMemRefType() { 
//       return cast<MemRefType>(getResult().getType()); 
//     }
//     // ArrayRef<int64_t> getShape() { return getShapeAttr().asArrayRef(); }
//     // Type getElementType() { return getElementTypeAttr().getValue(); }
//     // int64_t getAlignment() { return getAlignmentAttr().getInt(); }
//     // int64_t getMemorySpace() { return getMemorySpaceAttr().getInt(); }
    
//     // 根据 memorySpace 获取 scope
//     StringRef getScope() {
//       switch (getMemorySpace()) {
//         case ::mlir::frisk::attr::MemorySpace::Local:
//           return "local";
//         case ::mlir::frisk::attr::MemorySpace::Global:
//           return "global";
//         case ::mlir::frisk::attr::MemorySpace::Shared:
//           return "shared";
//         // case 0: return "local";
//         // case 1: return "global"; 
//         // case 3: return "shared";
//         // default: return "unknown";
//       }
//     }
//   }];

//   let hasVerifier = 1;
//   let hasCustomAssemblyFormat = 1;
// }
def Frisk_AllocBufferOp: Frisk_Op<"alloc_buffer", [Pure]> {
  let summary = "allocate buffer operation";
  let description = [{ 
    Allocates a buffer with the specified shape, element type and optional attributes.
  }];

  let arguments = (ins 
    DenseI64ArrayAttr:$shape,
    TypeAttr:$elementType,
    DefaultValuedAttr<I64Attr, "0">:$alignment,
    DefaultValuedAttr<I64Attr, "0">:$memorySpace
  );

  let results = (outs 
    Res<AnyMemRef>:$result
  );

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment)>,
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment, "int64_t":$memorySpace)>
  ];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() { 
      return cast<MemRefType>(getResult().getType()); 
    }
    
    // 根据 memorySpace 获取 scope
    StringRef getScope() {
      switch (getMemorySpace()) {
        case 0: return "local";
        case 1: return "global"; 
        case 3: return "shared";
        case 5: return "local";
        default: return "unknown";
      }
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// copyOp
def Frisk_CopyOp: Frisk_Op<"copy", [MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>]> {
  let summary = "copy";
  let description = [{ 
    copy
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemRead]>:$src,
    Arg<AnyMemRef, "", [MemWrite]>:$dst
  );

  let extraClassDeclaration = [{
    Value getSrcMemRef() { return getOperand(0); }
    Value getDstMemRef() { return getOperand(1); }

    MemRefType getSrcMemRefType() { 
      return cast<MemRefType>(getSrcMemRef().getType()); 
    }
    MemRefType getDstMemRefType() { 
      return cast<MemRefType>(getDstMemRef().getType()); 
    }

  }];

  let hasVerifier = 1;
  let assemblyFormat = " $src `to` $dst attr-dict `:` type($src) `,` type($dst)";
}

// fillOp
def Frisk_FillOp: Frisk_Op<"fill", [MemoryEffects<[MemWrite]>]> {
  let summary = "fill";
  let description = [{ 
    fill
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemWrite]>:$memref,
    AnyAttr:$value
  );

  let extraClassDeclaration = [{
    MemRefType getMemType() { return cast<MemRefType>(getMemref().getType()); }
  }];

  let hasVerifier = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

// reduceOp
def Frisk_ReduceOp: Frisk_Op<"reduce", [MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>]> {
  let summary = "reduce";
  let description = [{ 
    reduce
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemRead]>:$src,  // read
    Arg<AnyMemRef, "", [MemWrite]>:$dst,  // write
    StrAttr:$kind,
    I64Attr:$dim
  );

  let extraClassDeclaration = [{
    MemRefType getSrcType() { return cast<MemRefType>(getSrc().getType()); }
    MemRefType getDstType() { return cast<MemRefType>(getDst().getType()); }

    ::mlir::LogicalResult inferLayout(::mlir::OpBuilder &builder,
                                      llvm::DenseMap<::mlir::Value, ::mlir::Attribute> &layoutMap);
  }];

  let hasVerifier = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// init_barrier
def Frisk_InitBarrierOp: Frisk_Op<"init_barrier", [Pure]> {
  let summary = "Initialize barriers";
  let description = [{
    Create a list of barrier handles.
    The `num` attribute specifies how many barriers to create.
    The op returns `num` results, each of which is a `frisk.barrier` handle.
  }];

  let arguments = (ins
    I64Attr:$num
  );

  let results = (outs
    Variadic<Frisk_BarrierType>:$barriers
  );

  let builders = [
    OpBuilder<(ins "int64_t":$num)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict `:` type($barriers)";
}

// barrier_arrive
def Frisk_BarrierArrive: Frisk_Op<"barrier_arrive", []> {
  let summary = "Arrive at a barrier";
  let description = [{
    Signal arrival at a synchronization barrier.
  }];

  let arguments = (ins
    Frisk_BarrierType:$barrier
  );

  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

// barrier_wait
def Frisk_BarrierWait: Frisk_Op<"barrier_wait", []> {
  let summary = "Wait on a barrier";
  let description = [{
    Wait on a synchronization barrier with an index-typed phase value.
  }];

  let arguments = (ins
    Frisk_BarrierType:$barrier,
    Index:$phase
  );

  let assemblyFormat = "$barrier `,` $phase attr-dict `:` type($barrier) `,` type($phase)";
}

// barrier_arrive_expect_tx
def Frisk_BarrierArriveExpectTx: Frisk_Op<"barrier_arrive_expect_tx", []> {
  let summary = "Arrive at a barrier with expected transfer size";
  let description = [{
    Signal arrival at a synchronization barrier and provide expected transfer
    size in bytes.
  }];

  let arguments = (ins
    Frisk_BarrierType:$barrier,
    I64Attr:$expected_tx_bytes
  );

  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

def Frisk_ConvertOp: Frisk_Op<"convert", [Pure, InferTypeOpAdaptor]> {
  let summary = "convert";
  let description = [{}];
  let arguments = (ins
    AnyMemRef:$operand,
    TypeAttr:$dst_type
  );

  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$operand `,` `type` `=` $dst_type attr-dict `:` functional-type(operands, results)";
}


#endif // FRISK_OPS
