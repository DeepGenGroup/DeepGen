#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "Conversion/LoweringPasses.h"

using namespace mlir;

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v)
{
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty, IntegerAttr::get(i32ty, v));
}

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content)
{
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do
  {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(rewriter.getIntegerType(8), contentSize);

  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr));
  }
  auto i32ty = rewriter.getIntegerType(32);
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, i32ty, IntegerAttr::get(i32ty, 0));
  Type globalPtrType = LLVM::LLVMPointerType::get(globalType, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      rewriter.create<LLVM::GEPOp>(UnknownLoc::get(ctx), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)),
                                   globalPtr, SmallVector<Value>({zero, zero}));
  return stringStart;
};

class KCGToLLVMTypeConverter : public mlir::LLVMTypeConverter
{
public:
  using TypeConverter::convertType;

  KCGToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, option, analysis)
  {
    // Internally store float8 as int8
    addConversion([&](mlir::Float8E4M3B11FNUZType type) -> std::optional<Type>
                  { return IntegerType::get(type.getContext(), 8); });
    addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<Type>
                  { return IntegerType::get(type.getContext(), 8); });
    addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type>
                  { return IntegerType::get(type.getContext(), 8); });
    addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type>
                  { return IntegerType::get(type.getContext(), 8); });
    // Internally store bfloat16 as int16
    addConversion([&](BFloat16Type type) -> std::optional<Type>
                  { return IntegerType::get(type.getContext(), 16); });
  }

  SmallVector<Value> unpackLLElements(Location loc,
                                      Value llvmStruct, ConversionPatternRewriter &rewriter, Type type)
  {
    assert(bool(llvmStruct) && "can not unpack null values");
    if (llvmStruct.getType().isIntOrIndexOrFloat() ||
        llvmStruct.getType().isa<LLVM::LLVMPointerType>())
      return {llvmStruct};
    ArrayRef<Type> types =
        llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
    SmallVector<Value> results(types.size());
    for (unsigned i = 0; i < types.size(); ++i)
    {
      Type type = types[i];
      results[i] = rewriter.create<LLVM::ExtractValueOp>(loc, type, llvmStruct, i);
    }
    return results;
  }
};

class ConvertTritonGPUOpToLLVMPatternBase
{
public:
  explicit ConvertTritonGPUOpToLLVMPatternBase(
      KCGToLLVMTypeConverter &typeConverter)
      : converter(&typeConverter) {}

  KCGToLLVMTypeConverter *getTypeConverter() const { return converter; }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------

  struct SmallVectorKeyInfo
  {
    static unsigned getHashValue(const SmallVector<unsigned> &key)
    {
      return llvm::hash_combine_range(key.begin(), key.end());
    }
    static bool isEqual(const SmallVector<unsigned> &lhs,
                        const SmallVector<unsigned> &rhs)
    {
      return lhs == rhs;
    }
    static SmallVector<unsigned> getEmptyKey()
    {
      return SmallVector<unsigned>();
    }
    static SmallVector<unsigned> getTombstoneKey()
    {
      return {std::numeric_limits<unsigned>::max()};
    }
  };

private:
  void restoreInsertionPointIfSet(OpBuilder::InsertPoint *insertPt,
                                  ConversionPatternRewriter &rewriter) const
  {
    if (insertPt->isSet())
    {
      rewriter.restoreInsertionPoint(*insertPt);
    }
    else
    {
      auto func =
          rewriter.getInsertionPoint()->getParentOfType<LLVM::LLVMFuncOp>();
      rewriter.setInsertionPointToStart(&func.getBody().front());
    }
  }

private:
  static SmallString<16> getUniqueFormatGlobalName(mlir::ModuleOp moduleOp)
  {
    const char formatStringPrefix[] = "printfFormat_";
    // Get a unique global name.
    unsigned stringNumber = 0;
    SmallString<16> stringConstName;
    do
    {
      stringConstName.clear();
      (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
    } while (moduleOp.lookupSymbol(stringConstName));
    return stringConstName;
  }

  template <typename T>
  static LLVM::LLVMFuncOp
  getOrDefineFunction(T &moduleOp, const Location loc,
                      ConversionPatternRewriter &rewriter, StringRef name,
                      LLVM::LLVMFunctionType type)
  {
    LLVM::LLVMFuncOp ret;
    if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name)))
    {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                              LLVM::Linkage::External);
    }
    return ret;
  }

protected:
  // The code is borrowed from https://reviews.llvm.org/D110448
  // from GPUPrintfOpToHIPLowering::matchAndRewrite().
  void llPrintfHIP(mlir::Location loc, mlir::ModuleOp moduleOp, StringRef msg,
                   ValueRange args, ConversionPatternRewriter &rewriter,
                   bool stderr = false) const
  {

    auto typeConverter = getTypeConverter();
    mlir::Type llvmI8 = typeConverter->convertType(rewriter.getI8Type());
    mlir::Type i8Ptr = typeConverter->getPointerType(llvmI8);
    mlir::Type llvmI32 = typeConverter->convertType(rewriter.getI32Type());
    mlir::Type llvmI64 = typeConverter->convertType(rewriter.getI64Type());

    auto ocklBegin = getOrDefineFunction(
        moduleOp, loc, rewriter,
        (stderr ? "__ockl_fprintf_stderr_begin" : "__ockl_printf_begin"),
        (LLVM::LLVMFunctionType::get(llvmI64, stderr ? ArrayRef<mlir::Type>()
                                                     : llvmI64)));
    LLVM::LLVMFuncOp ocklAppendArgs;
    if (!args.empty())
    {
      ocklAppendArgs = getOrDefineFunction(
          moduleOp, loc, rewriter, "__ockl_printf_append_args",
          LLVM::LLVMFunctionType::get(llvmI64,
                                      {llvmI64, /*numArgs*/ llvmI32, llvmI64,
                                       llvmI64, llvmI64, llvmI64, llvmI64,
                                       llvmI64, llvmI64, /*isLast*/ llvmI32}));
    }
    auto ocklAppendStringN = getOrDefineFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
        LLVM::LLVMFunctionType::get(
            llvmI64,
            {llvmI64, i8Ptr, /*length (bytes)*/ llvmI64, /*isLast*/ llvmI32}));

    /// Start the printf hostcall
    Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, llvmI64, 0);
    auto printfBeginCall = rewriter.create<LLVM::CallOp>(
        loc, ocklBegin, stderr ? ValueRange() : zeroI64);
    Value printfDesc = printfBeginCall.getResult();

    // Get a unique global name for the format.
    SmallString<16> stringConstName = getUniqueFormatGlobalName(moduleOp);

    SmallString<32> formatString(msg);
    formatString.push_back('\n'); // Triton adds CR for each print.
    formatString.push_back('\0'); // Null terminate for C
    size_t formatStringSize = formatString.size_in_bytes();

    Value prefixString =
        addStringToModule(loc, rewriter, "printfFormat_", formatString);

    auto prefixPtrType = ocklAppendStringN.getArgumentTypes()[1];
    prefixString = rewriter.create<LLVM::BitcastOp>(loc, prefixPtrType, prefixString);

    Value stringLen =
        rewriter.create<LLVM::ConstantOp>(loc, llvmI64, formatStringSize);

    Value oneI32 = rewriter.create<LLVM::ConstantOp>(loc, llvmI32, 1);
    Value zeroI32 = rewriter.create<LLVM::ConstantOp>(loc, llvmI32, 0);

    auto appendFormatCall = rewriter.create<LLVM::CallOp>(
        loc, ocklAppendStringN,
        ValueRange{printfDesc, prefixString, stringLen,
                   args.empty() ? oneI32 : zeroI32});
    printfDesc = appendFormatCall.getResult();

    // __ockl_printf_append_args takes 7 values per append call
    constexpr size_t argsPerAppend = 7;
    size_t nArgs = args.size();
    for (size_t group = 0; group < nArgs; group += argsPerAppend)
    {
      size_t bound = std::min(group + argsPerAppend, nArgs);
      size_t numArgsThisCall = bound - group;

      SmallVector<mlir::Value, 2 + argsPerAppend + 1> arguments;
      arguments.push_back(printfDesc);
      arguments.push_back(
          rewriter.create<LLVM::ConstantOp>(loc, llvmI32, numArgsThisCall));
      for (size_t i = group; i < bound; ++i)
      {
        Value arg = args[i];
        if (auto floatType = arg.getType().dyn_cast<FloatType>())
        {
          if (!floatType.isF64())
            arg = rewriter.create<LLVM::FPExtOp>(
                loc, typeConverter->convertType(rewriter.getF64Type()), arg);
          arg = rewriter.create<LLVM::BitcastOp>(loc, llvmI64, arg);
        }
        if (arg.getType().getIntOrFloatBitWidth() != 64)
          arg = rewriter.create<LLVM::ZExtOp>(loc, llvmI64, arg);

        arguments.push_back(arg);
      }
      // Pad out to 7 arguments since the hostcall always needs 7
      for (size_t extra = numArgsThisCall; extra < argsPerAppend; ++extra)
      {
        arguments.push_back(zeroI64);
      }

      auto isLast = (bound == nArgs) ? oneI32 : zeroI32;
      arguments.push_back(isLast);
      auto call = rewriter.create<LLVM::CallOp>(loc, ocklAppendArgs, arguments);
      printfDesc = call.getResult();
    }
  }

  KCGToLLVMTypeConverter *converter;
};

template <typename SourceOp>
class ConvertGPUOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp>,
      public ConvertTritonGPUOpToLLVMPatternBase
{
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertGPUOpToLLVMPattern(
      KCGToLLVMTypeConverter &typeConverter, PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter) {}

protected:
  KCGToLLVMTypeConverter *getTypeConverter() const
  {
    LLVMTypeConverter *ret =
        ((ConvertTritonGPUOpToLLVMPatternBase *)this)->getTypeConverter();
    return (KCGToLLVMTypeConverter *)ret;
  }
};

struct PrintOpConversion
    : public ConvertGPUOpToLLVMPattern<mlir::gpu::PrintfOp>
{
  using ConvertGPUOpToLLVMPattern<mlir::gpu::PrintfOp>::ConvertGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::PrintfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto loc = op->getLoc();
    SmallVector<Value, 16> operands;
    for (size_t i = 0; i < op.getNumOperands(); i++)
    {
      auto sub_operands = getTypeConverter()->unpackLLElements(
          loc, adaptor.getOperands()[i], rewriter, op.getOperand(i).getType());
      for (auto elem : sub_operands)
      {
        operands.push_back(elem);
      }
    }
    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << "[D]";
    if (!operands.empty())
    {
      os << getFormatSubstr(operands[0]);
    }

    for (size_t i = 1; i < operands.size(); ++i)
    {
      os << ", " << getFormatSubstr(operands[i]);
    }
#if 1
    llPrintfHIP(loc, op->getParentOfType<mlir::ModuleOp>(), formatStr, operands,
                rewriter);
#else
    llPrintf(formatStr, operands, rewriter);
#endif
    rewriter.eraseOp(op);
    return success();
  }

  std::string getFormatSubstr(Value value) const
  {
    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>())
    {
      return "%p";
    }
    else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64())
    {
      return "%f";
    }
    else if (type.isSignedInteger())
    {
      if (type.getIntOrFloatBitWidth() == 64)
        return "%lli";
      else
        return "%i";
    }
    else if (type.isUnsignedInteger() || type.isSignlessInteger())
    {
      if (type.getIntOrFloatBitWidth() == 64)
        return "%llu";
      else
        return "%u";
    }
    assert(false && "not supported type");
    return "";
  }

  // declare vprintf(i8*, i8*) as external function
  static LLVM::LLVMFuncOp
  getVprintfDeclaration(ConversionPatternRewriter &rewriter)
  {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("vprintf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *context = rewriter.getContext();

    SmallVector<Type> argsType{LLVM::LLVMPointerType::get(IntegerType::get(context, 8)),
                               LLVM::LLVMPointerType::get(IntegerType::get(context, 8))};
    auto funcType = LLVM::LLVMFunctionType::get(rewriter.getIntegerType(32), argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                             funcType);
  }

  // extend integer to int32, extend float to float64
  // this comes from vprintf alignment requirements.
  static std::pair<Type, Value>
  promoteValue(ConversionPatternRewriter &rewriter, Value value)
  {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    Value newOp = value;
    Type newType = type;
    auto loc = UnknownLoc::get(context);

    bool bUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32)
    {
      if (bUnsigned)
      {
        newType = rewriter.getIntegerType(32, false);
        newOp = rewriter.create<LLVM::ZExtOp>(loc, newType, value);
      }
      else
      {
        newType = rewriter.getIntegerType(32);
        newOp = rewriter.create<LLVM::SExtOp>(loc, newType, value);
      }
    }
    else if (type.isBF16() || type.isF16() || type.isF32())
    {
      newType = rewriter.getF64Type();
      newOp = rewriter.create<LLVM::FPExtOp>(loc, newType, value);
    }

    return {newType, newOp};
  }

  static void llPrintf(StringRef msg, ValueRange args,
                       ConversionPatternRewriter &rewriter)
  {
    assert(!msg.empty() && "printf with empty string not support");
    Type int8Ptr = LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));

    auto *ctx = rewriter.getContext();
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto funcOp = getVprintfDeclaration(rewriter);
    auto loc = UnknownLoc::get(ctx);

    Value one = createConstantI32(loc, rewriter, 1);
    Value zero = createConstantI32(loc, rewriter, 0);

    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    Value prefixString =
        addStringToModule(loc, rewriter, "printfFormat_", msgNewline);
    Value bufferPtr = rewriter.create<LLVM::ZeroOp>(loc, int8Ptr);

    SmallVector<Value, 16> newArgs;
    if (args.size() >= 1)
    {
      SmallVector<Type> argTypes;
      for (auto arg : args)
      {
        Type newType;
        Value newArg;
        std::tie(newType, newArg) = promoteValue(rewriter, arg);
        argTypes.push_back(newType);
        newArgs.push_back(newArg);
      }

      Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(structTy), one,
                                          /*alignment=*/0);

      for (const auto &entry : llvm::enumerate(newArgs))
      {
        auto index = createConstantI32(loc, rewriter, entry.index());
        auto fieldPtr = rewriter.create<LLVM::GEPOp>(
            loc,
            LLVM::LLVMPointerType::get(argTypes[entry.index()]),
            allocated,
            ArrayRef<Value>{zero, index});
        rewriter.create<LLVM::StoreOp>(loc, entry.value(), fieldPtr);
      }
      bufferPtr = rewriter.create<LLVM::BitcastOp>(loc, int8Ptr, allocated);
    }

    SmallVector<Value> operands{prefixString, bufferPtr};
    rewriter.create<LLVM::CallOp>(loc, funcOp, operands);
  }
};

void populateTritonGPUToLLVMPatterns(
    KCGToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit)
{
  patterns.add<PrintOpConversion>(typeConverter, benefit);
}



template <typename DerivedT>
class ConvertTritonGPUToLLVMBase : public ::mlir::OperationPass<mlir::ModuleOp> {
public:
  using Base = ConvertTritonGPUToLLVMBase;

  ConvertTritonGPUToLLVMBase() : ::mlir::OperationPass<mlir::ModuleOp>(::mlir::TypeID::get<DerivedT>()) {}
  ConvertTritonGPUToLLVMBase(const ConvertTritonGPUToLLVMBase &other) : ::mlir::OperationPass<mlir::ModuleOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("convert-triton-gpu-to-llvm");
  }
  ::llvm::StringRef getArgument() const override { return "convert-triton-gpu-to-llvm"; }

  ::llvm::StringRef getDescription() const override { return "Convert TritonGPU to LLVM"; }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ConvertTritonGPUToLLVM");
  }
  ::llvm::StringRef getName() const override { return "ConvertTritonGPUToLLVM"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit private
  /// instantiation because Pass classes should only be visible by the current
  /// library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonGPUToLLVMBase<DerivedT>)

};


struct ConvertKCGToLLVM : public ConvertTritonGPUToLLVMBase<ConvertKCGToLLVM>
{
  using ConvertTritonGPUToLLVMBase<
      ConvertKCGToLLVM>::ConvertTritonGPUToLLVMBase;
  ::llvm::StringRef getName() const override { return "ConvertKCGToLLVM"; }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override
  {
    return std::make_unique<ConvertKCGToLLVM>(*static_cast<const ConvertKCGToLLVM *>(this));
  }

  void runOnOperation() override
  {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    KCGToLLVMTypeConverter typeConverter(context, option);
    RewritePatternSet patterns(context);
    populateTritonGPUToLLVMPatterns(typeConverter, patterns, /*benefit*/ 10);
  }
};

namespace KernelCodeGen {
std::unique_ptr<OperationPass<ModuleOp>> createConvertGPUPrintToLLVMPass() {
  return std::make_unique<ConvertKCGToLLVM>();
}
}