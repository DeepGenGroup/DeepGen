#include "Dialect/Frisk/IR/FriskEnums.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Frisk/IR/FriskAttributes.cpp.inc"

namespace mlir::frisk {
namespace {
void printShape(DenseI64ArrayAttr shape, llvm::raw_ostream &os) {
  os << "[";
  if (shape) {
    llvm::ArrayRef<int64_t> values = shape.asArrayRef();
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        os << ", ";
      os << values[i];
    }
  }
  os << "]";
}

void printAffineMapAttr(AffineMapAttr mapAttr, llvm::raw_ostream &os) {
  if (!mapAttr) {
    os << "<none>";
    return;
  }
  std::string buffer;
  llvm::raw_string_ostream valueStream(buffer);
  mapAttr.print(valueStream);
  valueStream.flush();
  os << valueStream.str();
}
} // namespace

void printLayoutDebug(LayoutAttr layout, llvm::raw_ostream &os) {
  if (!layout) {
    os << "<null layout>";
    return;
  }

  bool hasThreadMap = static_cast<bool>(layout.getForwardThread());
  os << (hasThreadMap ? "Fragment" : "Layout") << "(shape=";
  printShape(layout.getInputShape(), os);
  os << ", index=";
  printAffineMapAttr(layout.getForwardIndex(), os);
  if (hasThreadMap) {
    os << ", thread=";
    printAffineMapAttr(layout.getForwardThread(), os);
  }
  if (auto replicate = layout.getReplicateSize())
    os << ", replicate=" << replicate.getInt();
  os << ")";
}

std::string layoutDebugString(LayoutAttr layout) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  printLayoutDebug(layout, os);
  os.flush();
  return buffer;
}

void FriskDialect::initialize() {
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Frisk/IR/FriskAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"
      >();
}
} // namespace mlir::frisk
