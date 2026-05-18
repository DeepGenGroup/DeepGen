#if ENABLE_LAYOUT_INFER
#include "deepgengraph/Dialect/Frisk/Utils/LayoutUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;

namespace mlir::frisk {
namespace {

struct SplitInfo {
  int64_t lowerFactor = 1;
  int64_t extent = 1;
};

static bool exprUsesDim(AffineExpr expr, unsigned dim) {
  if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr))
    return dimExpr.getPosition() == dim;
  if (mlir::isa<AffineSymbolExpr>(expr) || mlir::isa<AffineConstantExpr>(expr))
    return false;

  if (auto bin = mlir::dyn_cast<AffineBinaryOpExpr>(expr))
    return exprUsesDim(bin.getLHS(), dim) || exprUsesDim(bin.getRHS(), dim);
  return false;
}

static std::optional<int64_t> computeSpan(AffineExpr expr, unsigned dim,
                                          int64_t baseSpan) {
  if (auto cst = mlir::dyn_cast<AffineConstantExpr>(expr))
    return int64_t(1);
  if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr))
    return dimExpr.getPosition() == dim ? baseSpan : int64_t(1);
  if (auto sym = mlir::dyn_cast<AffineSymbolExpr>(expr))
    return int64_t(1);

  auto bin = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return std::nullopt;

  auto maybeL = computeSpan(bin.getLHS(), dim, baseSpan);
  auto maybeR = computeSpan(bin.getRHS(), dim, baseSpan);
  if (!maybeL || !maybeR)
    return std::nullopt;

  int64_t lhs = *maybeL;
  int64_t rhs = *maybeR;
  int64_t span = 1;

  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    if (llvm::AddOverflow(lhs, rhs, span))
      return std::nullopt;
    span = std::max<int64_t>(int64_t(1), span - 1);
    return span;
  }
  case AffineExprKind::Mul: {
    if (auto rhsConst = mlir::dyn_cast<AffineConstantExpr>(bin.getRHS())) {
      int64_t constVal = rhsConst.getValue();
      if (constVal == std::numeric_limits<int64_t>::min())
        return std::nullopt;
      int64_t factor = std::abs(constVal);
      if (llvm::MulOverflow(lhs, factor, span))
        return std::nullopt;
      return span;
    }
    if (auto lhsConst = mlir::dyn_cast<AffineConstantExpr>(bin.getLHS())) {
      int64_t constVal = lhsConst.getValue();
      if (constVal == std::numeric_limits<int64_t>::min())
        return std::nullopt;
      int64_t factor = std::abs(constVal);
      if (llvm::MulOverflow(rhs, factor, span))
        return std::nullopt;
      return span;
    }
    return std::nullopt;
  }
  case AffineExprKind::FloorDiv: {
    auto rhsConst = mlir::dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!rhsConst)
      return std::nullopt;
    int64_t divisor = rhsConst.getValue();
    if (divisor <= 0)
      return std::nullopt;
    return llvm::divideCeil(lhs, divisor);
  }
  case AffineExprKind::Mod: {
    auto rhsConst = mlir::dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!rhsConst)
      return std::nullopt;
    int64_t modulus = rhsConst.getValue();
    if (modulus <= 0)
      return std::nullopt;
    return std::min<int64_t>(lhs, modulus);
  }
  default:
    break;
  }

  return std::nullopt;
}

static LogicalResult collectSplits(AffineExpr expr, unsigned dim,
                                   int64_t baseExtent, int64_t lowerFactor,
                                   SmallVectorImpl<SplitInfo> &splits) {
  if (!exprUsesDim(expr, dim))
    return success();

  if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() != dim)
      return success();
    splits.push_back({lowerFactor, baseExtent});
    return success();
  }

  if (mlir::isa<AffineConstantExpr>(expr) || mlir::isa<AffineSymbolExpr>(expr))
    return success();

  auto bin = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return failure();

  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    if (failed(collectSplits(bin.getLHS(), dim, baseExtent, lowerFactor,
                             splits)))
      return failure();
    return collectSplits(bin.getRHS(), dim, baseExtent, lowerFactor, splits);
  }
  
  case AffineExprKind::Mul: {
    if (mlir::isa<AffineConstantExpr>(bin.getLHS()))
      return collectSplits(bin.getRHS(), dim, baseExtent, lowerFactor, splits);
    if (mlir::isa<AffineConstantExpr>(bin.getRHS()))
      return collectSplits(bin.getLHS(), dim, baseExtent, lowerFactor, splits);
    return failure();
  }

  case AffineExprKind::FloorDiv: {
    auto divisor = mlir::dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!divisor || divisor.getValue() <= 0)
      return failure();
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return failure();
    int64_t newExtent = llvm::divideCeil(*span, divisor.getValue());
    int64_t newLower = 0;
    if (llvm::MulOverflow(lowerFactor, divisor.getValue(), newLower))
      return failure();
    return collectSplits(bin.getLHS(), dim, newExtent, newLower, splits);
  }

  case AffineExprKind::Mod: {
    auto modulus = mlir::dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!modulus || modulus.getValue() <= 0)
      return failure();
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return failure();
    int64_t newExtent = std::min<int64_t>(*span, modulus.getValue());
    return collectSplits(bin.getLHS(), dim, newExtent, lowerFactor, splits);
  }

  default:
    break;
  }

  return failure();
}

} // namespace

std::optional<int64_t>
computeUsedExtentForDim(AffineMapAttr mapAttr, unsigned placeholderPos,
                        int64_t placeholderExtent) {
  if (!mapAttr)
    return int64_t(1);
  if (placeholderExtent <= 0)
    return std::nullopt;

  AffineMap map = mapAttr.getValue();
  if (placeholderPos >= map.getNumDims())
    return std::nullopt;

  SmallVector<SplitInfo, 4> splits;
  for (AffineExpr result : map.getResults()) {
    if (!exprUsesDim(result, placeholderPos))
      continue;
    if (failed(collectSplits(result, placeholderPos, placeholderExtent,
                             /*lowerFactor=*/1, splits)))
      return std::nullopt;
  }

  if (splits.empty())
    return int64_t(1);

  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 8> uniqueSplits;
  int64_t usedExtent = 1;
  for (const auto &split : splits) {
    auto key = std::make_pair(split.lowerFactor, split.extent);
    if (!uniqueSplits.insert(key).second)
      continue;
    if (split.extent <= 0)
      return std::nullopt;
    if (llvm::MulOverflow(usedExtent, split.extent, usedExtent))
      return std::nullopt;
  }

  return usedExtent;
}

} // namespace mlir::frisk
#endif