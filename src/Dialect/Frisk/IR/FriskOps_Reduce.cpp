#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/Visitors.h"

#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskEnums.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Utils/LayoutUtils.h"

#define GET_OP_CLASSES
// #include "deepgengraph/Dialect/Frisk/IR/FriskOps.cpp.inc"
// #include "deepgengraph/Dialect/Frisk/IR/FriskDialect.cpp.inc"

namespace mlir {

namespace frisk {

namespace {

static std::optional<int64_t> inferThreadBlockSize(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto attr = cur->getAttrOfType<IntegerAttr>("frisk.threads"))
      return attr.getInt();
    if (auto parallel = dyn_cast<ParallelOp>(cur))
      return parallel.getThreads();
    cur = cur->getParentOp();
  }
  return std::nullopt;
}

static AffineMapAttr remapReduceDimension(OpBuilder &builder,
                                          AffineMapAttr mapAttr,
                                          unsigned removeDim,
                                          int64_t reduceExtent,
                                          bool useFloorDiv) {
  if (!mapAttr)
    return AffineMapAttr();

  AffineMap map = mapAttr.getValue();
  unsigned srcDims = map.getNumDims();
  if (removeDim >= srcDims)
    return AffineMapAttr();

  unsigned dstDims = srcDims - 1;
  unsigned placeholderDim = dstDims;
  unsigned newDimCount = dstDims + 1;

  MLIRContext *ctx = builder.getContext();
  AffineExpr placeholder = builder.getAffineDimExpr(placeholderDim);

  SmallVector<AffineExpr> dimSubs;
  dimSubs.reserve(srcDims);
  for (unsigned i = 0; i < srcDims; ++i) {
    if (i < removeDim) {
      dimSubs.push_back(builder.getAffineDimExpr(i));
      continue;
    }
    if (i == removeDim) {
      if (ShapedType::isDynamic(reduceExtent)) {
        dimSubs.push_back(placeholder);
      } else if (reduceExtent == 1) {
        dimSubs.push_back(builder.getAffineConstantExpr(0));
      } else {
        AffineExpr constant = builder.getAffineConstantExpr(reduceExtent);
        dimSubs.push_back(useFloorDiv ? placeholder.floorDiv(constant)
                                      : placeholder % constant);
      }
      continue;
    }
    dimSubs.push_back(builder.getAffineDimExpr(i - 1));
  }

  SmallVector<AffineExpr> symSubs;
  symSubs.reserve(map.getNumSymbols());
  for (unsigned i = 0; i < map.getNumSymbols(); ++i)
    symSubs.push_back(builder.getAffineSymbolExpr(i));

  SmallVector<AffineExpr> newResults;
  newResults.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults())
    newResults.push_back(expr.replaceDimsAndSymbols(dimSubs, symSubs));

  AffineMap newMap =
      AffineMap::get(newDimCount, map.getNumSymbols(), newResults, ctx);
  return AffineMapAttr::get(newMap);
}

static bool exprUsesDim(AffineExpr expr, unsigned dimIdx) {
  if (auto dim = mlir::dyn_cast<AffineDimExpr>(expr))
    return dim.getPosition() == dimIdx;
  if (mlir::isa<AffineSymbolExpr>(expr) || mlir::isa<AffineConstantExpr>(expr))
    return false;
  if (auto bin = mlir::dyn_cast<AffineBinaryOpExpr>(expr))
    return exprUsesDim(bin.getLHS(), dimIdx) ||
           exprUsesDim(bin.getRHS(), dimIdx);
  return false;
}

static AffineMapAttr condensePlaceholderDimension(OpBuilder &builder,
                                                  AffineMapAttr mapAttr,
                                                  bool keepPlaceholderSymbol) {
  if (!mapAttr)
    return AffineMapAttr();
  AffineMap map = mapAttr.getValue();
  unsigned dimCount = map.getNumDims();
  if (dimCount == 0)
    return mapAttr;

  unsigned placeholder = dimCount - 1;
  bool placeholderUsed = llvm::any_of(map.getResults(), [&](AffineExpr expr) {
    return exprUsesDim(expr, placeholder);
  });

  unsigned newSymCount = map.getNumSymbols();
  AffineExpr placeholderValue = builder.getAffineConstantExpr(0);
  if (keepPlaceholderSymbol && placeholderUsed) {
    placeholderValue = builder.getAffineSymbolExpr(newSymCount);
    ++newSymCount;
  }

  SmallVector<AffineExpr> dimSubs;
  dimSubs.reserve(dimCount);
  for (unsigned i = 0; i < dimCount; ++i) {
    if (i < placeholder)
      dimSubs.push_back(builder.getAffineDimExpr(i));
    else if (i == placeholder)
      dimSubs.push_back(placeholderValue);
    else
      dimSubs.push_back(builder.getAffineDimExpr(i - 1));
  }

  SmallVector<AffineExpr> symSubs;
  symSubs.reserve(map.getNumSymbols());
  for (unsigned i = 0; i < map.getNumSymbols(); ++i)
    symSubs.push_back(builder.getAffineSymbolExpr(i));

  SmallVector<AffineExpr> newResults;
  newResults.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults())
    // newResults.push_back(expr.replaceDimsAndSymbols(
    //     dimSubs, symSubs, /*numResultDims=*/dimCount - 1,
    //     /*numResultSymbols=*/newSymCount));
    newResults.push_back(expr.replaceDimsAndSymbols(
        dimSubs, symSubs));
  AffineMap newMap =
      AffineMap::get(dimCount - 1, newSymCount, newResults, builder.getContext());
  return AffineMapAttr::get(newMap);
}

static DenseI64ArrayAttr buildShapeAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> shape) {
  SmallVector<int64_t> values(shape.begin(), shape.end());
  if (values.empty())
    values.push_back(1);
  return builder.getDenseI64ArrayAttr(values);
}

static bool affineMapAttrsEqual(AffineMapAttr lhs, AffineMapAttr rhs) {
  if (!lhs && !rhs)
    return true;
  if (!lhs || !rhs)
    return false;
  return lhs == rhs;
}

static LogicalResult checkLayoutCompatibility(ReduceOp op,
                                              LayoutAttr computed,
                                              LayoutAttr existing) {
  auto emitConflict = [&](StringRef detail) -> LogicalResult {
    op.emitOpError("destination layout conflicts with reduce inference: ")
        << detail << "\nexpected=" << layoutDebugString(computed)
        << "\nactual=" << layoutDebugString(existing);
    return failure();
  };

  if (!existing)
    return success();

  if (computed.getInputShape() != existing.getInputShape())
    return emitConflict("input shapes disagree");
  if (!affineMapAttrsEqual(computed.getForwardIndex(),
                           existing.getForwardIndex()))
    return emitConflict("forward index maps differ");
  if (!affineMapAttrsEqual(computed.getForwardThread(),
                           existing.getForwardThread()))
    return emitConflict("thread maps differ");
  auto lhsRep = computed.getReplicateSize();
  auto rhsRep = existing.getReplicateSize();
  if (lhsRep && rhsRep && lhsRep.getInt() > rhsRep.getInt())
    return emitConflict("existing layout replicates fewer lanes than inferred");
  return success();
}

} // namespace

LogicalResult ReduceOp::inferLayout(OpBuilder &builder,
                                    DenseMap<Value, Attribute> &layoutMap) {
#if 0 
  auto srcType = dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(getDst().getType());
  if (!srcType || !dstType)
    return emitOpError("layout inference requires memref operands");

  auto parseMemorySpace = [&](MemRefType type,
                              StringRef label) -> std::optional<attr::MemorySpace> {
    unsigned raw = type.getMemorySpaceAsInt();
    if (auto symbolic = attr::symbolizeMemorySpace(raw))
      return *symbolic;
    emitOpError() << "operand " << label
                  << " resides in unsupported memory space " << raw;
    return std::nullopt;
  };

  auto srcSpace = parseMemorySpace(srcType, "src");
  auto dstSpace = parseMemorySpace(dstType, "dst");
  if (!srcSpace || !dstSpace)
    return failure();

  if (*srcSpace != attr::MemorySpace::Local ||
      *dstSpace != attr::MemorySpace::Local)
    return success();

  auto srcIt = layoutMap.find(getSrc());
  if (srcIt == layoutMap.end())
    return success();

  auto srcLayout = dyn_cast<LayoutAttr>(srcIt->second);
  if (!srcLayout)
    return emitOpError("source layout entry must be a frisk.layout attribute");

  auto srcIndexMap = srcLayout.getForwardIndex();
  auto srcThreadMap = srcLayout.getForwardThread();
  if (!srcIndexMap || !srcThreadMap)
    return success();

  DenseI64ArrayAttr layoutShape = srcLayout.getInputShape();
  if (!layoutShape)
    return emitOpError("source layout missing input shape metadata");

  ArrayRef<int64_t> layoutDims = layoutShape.asArrayRef();
  if (layoutDims.size() != static_cast<size_t>(srcType.getRank()))
    return emitOpError("source layout rank does not match source memref rank");

  int64_t dim = getDim();
  if (dim < 0 || dim >= srcType.getRank())
    return emitOpError("invalid reduce dimension ") << dim;

  int64_t reduceExtent = srcType.getShape()[dim];
  if (ShapedType::isDynamic(reduceExtent))
    return emitOpError(
        "layout inference requires static extent along the reduce dimension");
  if (reduceExtent <= 0)
    return emitOpError("reduce extent must be positive for layout inference");

  auto remappedIndex = remapReduceDimension(
      builder, srcIndexMap, static_cast<unsigned>(dim), reduceExtent,
      /*useFloorDiv=*/false);
  auto remappedThread = remapReduceDimension(
      builder, srcThreadMap, static_cast<unsigned>(dim), reduceExtent,
      /*useFloorDiv=*/false);

  int64_t reduceFactor = 1;
  if (remappedThread) {
    AffineMap threadMap = remappedThread.getValue();
    if (threadMap.getNumDims() == 0)
      return emitOpError("thread layout missing placeholder dimension");
    unsigned placeholder = threadMap.getNumDims() - 1;
    auto used = computeUsedExtentForDim(remappedThread, placeholder,
                                        reduceExtent);
    if (!used)
      return emitOpError("unable to analyze reduce dimension usage in "
                         "thread map for layout inference");
    reduceFactor = std::max<int64_t>(int64_t(1), *used);
  }

  auto dstIndexMap =
      condensePlaceholderDimension(builder, remappedIndex, /*keepPlaceholderSymbol=*/false);
  auto dstThreadMap =
      condensePlaceholderDimension(builder, remappedThread, /*keepPlaceholderSymbol=*/true);
  if (!dstIndexMap || !dstThreadMap)
    return emitOpError("failed to synthesize destination layout for reduce op");

  std::optional<int64_t> replicateValue;
  int64_t baseReplicate = 1;
  if (auto replicateAttr = srcLayout.getReplicateSize()) {
    baseReplicate = replicateAttr.getInt();
    if (baseReplicate <= 0)
      return emitOpError("source layout replicate extent must be positive");
    replicateValue = baseReplicate;
  }

  int64_t factor = std::max<int64_t>(int64_t(1), reduceFactor);
  int64_t base = std::max<int64_t>(int64_t(1), baseReplicate);
  if (llvm::MulOverflow(base, factor, base))
    return emitOpError("replicate extent overflow while inferring layout");
  replicateValue = base;

  IntegerAttr replicateAttr;
  if (replicateValue) {
    if (*replicateValue <= 0)
      return emitOpError("replicate extent must be positive");
    replicateAttr = builder.getI64IntegerAttr(*replicateValue);
  }

  if (replicateAttr) {
    auto threads = inferThreadBlockSize(getOperation());
    if (threads && *threads > 0) {
      int64_t replicate = replicateAttr.getInt();
      if (replicate > 0 && (*threads % replicate != 0) &&
          (replicate % *threads != 0)) {
        return emitOpError()
               << "reduce layout inference requires thread count divisible "
                  "by replicate extent (threads="
               << *threads << ", replicate=" << replicate << ")";
      }
    }
  }

  auto dstShapeAttr = buildShapeAttr(builder, dstType.getShape());
  LayoutAttr dstLayout = LayoutAttr::get(
      builder.getContext(), dstShapeAttr, dstIndexMap, dstThreadMap,
      replicateAttr);

  auto dstIt = layoutMap.find(getDst());
  if (dstIt != layoutMap.end()) {
    auto existingLayout = dyn_cast<LayoutAttr>(dstIt->second);
    if (!existingLayout)
      dstIt->second = dstLayout;
    if (auto layout = dyn_cast<LayoutAttr>(dstIt->second)) {
      if (failed(checkLayoutCompatibility(*this, dstLayout, layout)))
        return failure();
      return success();
    }
    dstIt->second = dstLayout;
    return success(true);
  }

  layoutMap.try_emplace(getDst(), dstLayout);
  #endif
  return success(true);
} 


} // namespace frisk
} // namespace mlir
