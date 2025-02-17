#ifndef _Utils_h_
#define _Utils_h_

#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/IR/Module.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"


// reused Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

// IR Infrastructure
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"

// LLVM ADT
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.h.inc"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "PassDetail.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
#include <memory>
#include <functional>
#include <deque>
#include <string>
#include <random>
#include <cstdio>
#include <sstream>



namespace KernelCodeGen {

/***********  enumerations & basic class  *****************/
enum class Target {
  CUDA = 0,
  ROCm = 1,
};

enum class MemorySpace {
  global = 1,
  shared = 3,
  // local = 0,
  local = 5,
  constant = 4,
  unallocated = 7,
  inplace = 6,
};

enum class Position {
  before = 0,
  after = 1,
  begin = 2,
  end = 3,
};

enum class Layout {
  rowMajor = 0,
  colMajor = 1,
};


enum class KcgDtype : int {
  float8 = 1,
  float16 = 2,
  float32 = 4,
  float64 = 8,
  float128 = 16,
  int8 = 11,
  int16 = 12,
  int32 = 14,
  int64 = 18
};

enum class KcgKernelType : int {
  matmul = 1,
  conv2d = 2,
  poolmax = 3
  // other operators ...
};

using Config = std::map<std::string, int>;

struct KernelInfo {
  std::string m_hsacoPath;
  std::string m_kernelName;
  std::vector<int> m_gridDims = {1,1,1};
  std::vector<int> m_blockDims = {1,1,1};
};

static std::ostream& operator<<(std::ostream& s, KcgDtype ty){
  switch(ty){
    case KcgDtype::float8: s << "f8" ; break;
    case KcgDtype::float16: s << "f16" ; break;
    case KcgDtype::float32: s << "f32" ; break;
    case KcgDtype::float64: s << "f64" ; break;
    case KcgDtype::float128: s << "f128" ; break;
    case KcgDtype::int8: s << "i8" ; break;
    case KcgDtype::int16: s << "i16" ; break;
    case KcgDtype::int32: s << "i32" ; break;
    case KcgDtype::int64: s << "i64" ; break;
    default : break;
  }
  return s;
}

struct NVVMMetadata {
  llvm::SmallVector<int, 3> maxntid;
  bool isKernel{};
};

/************* attribute names & other naming rules ***********/ 

#define AttrKernelFunc    "nvvm.kernel"
#define AttrMaxBlockThreads    "nvvm.maxntid"
#define AttrVisibility    "sym_visibility" 
#define AttrExternLib     "kcg.externlibs"
#define AttrRootFunc      "kcg.rootfunc"
#define AttrKernelType    "kcg.kerneltype"
#define AttrDescription   "kcg.desc"
#define AttrGridDim       "func.grid.dim"
#define AttrBlockDim      "func.block.dim"
#define AttrKernelFunc     "nvvm.kernel"
#define AttrVisibility     "sym_visibility" 
#define AttrExternLib      "kcg.externlibs"
#define AttrRootFunc       "kcg.rootfunc"
#define AttrKernelType     "kcg.kerneltype"
#define AttrDescription    "kcg.desc"
#define AttrGridDim        "func.grid.dim"
#define AttrBlockDim       "func.block.dim"
#define AttrBufDescription "kcg.bufDesc"

#define SHM_VAR_NAME(i) (std::string("kcg_shm")+std::to_string(i))



/********  kernel keywords for matmul operator & other operators for tuning ***********/  

#define  KEY_BLOCK_SIZE_M         "BLOCK_SIZE_M"
#define  KEY_BLOCK_SIZE_N         "BLOCK_SIZE_N"
#define  KEY_BLOCK_SIZE_K         "BLOCK_SIZE_K"
#define  KEY_THREAD_SIZE_M        "THREAD_SIZE_M"
#define  KEY_THREAD_SIZE_N        "THREAD_SIZE_N"
#define  KEY_WARP_SIZE            "WARP_SIZE"
#define  KEY_BLOCK_LAYOUT_M       "BLOCK_LAYOUT_M"
#define  KEY_BLOCK_LAYOUT_N       "BLOCK_LAYOUT_N"
#define  KEY_WARP_LAYOUT_M        "WARP_LAYOUT_M"
#define  KEY_WARP_LAYOUT_N        "WARP_LAYOUT_N"
#define  KEY_DTYPE_A              "DATATYPE_A"
#define  KEY_DTYPE_B              "DATATYPE_B"
#define  KEY_DTYPE_C              "DATATYPE_C"
#define  KEY_M                    "M_SIZE"
#define  KEY_N                    "N_SIZE"
#define  KEY_K                    "K_SIZE"
#define  KEY_IS_A_TRANSPOSE       "IS_ATRANS"
#define  KEY_GLOB_LOAD_WIDTH_A     "GLOB_LOAD_WIDTH_A"
#define  KEY_GLOB_LOAD_WIDTH_B     "GLOB_LOAD_WIDTH_B"
#define  KEY_WARP_SCATTER_WIDTH_A    "WARP_SCATTER_WIDTH_A"
#define  KEY_WARP_SCATTER_WIDTH_B    "WARP_SCATTER_WIDTH_B"
#define  KEY_THREAD_SCATTER_WIDTH_A    "THREAD_SCATTER_WIDTH_A"
#define  KEY_THREAD_SCATTER_WIDTH_B    "THREAD_SCATTER_WIDTH_B"
#define  KEY_LOCAL_SPLIT_U     "LOCAL_SPLIT_U"
#define  KEY_BLOCK_MAPPING     "BLOCK_MAPPING"
#define  KEY_GLOB_STORE_WIDTH    "GLOB_STORE_WIDTH"
// added
#define KEY_UNROLL_NUM            "UNROLL_NUM"
#define KEY_REG_PREFETCH          "REG_PREFETCH"
#define KEY_SHARED_PREFETCH       "SHARED_PREFETCH"
#define KEY_LOAD_CONTINUOUS       "LOAD_CONTINUOUS"
#define KEY_REDUCE_C_CONTINUOUS   "REDUCE_C_CONTINUOUS"

/****************** other macro ******************** */

#define INDEX_BIT_WIDTH     32
#define KCG_ALIGNBYTE       16
// #define LOG_DEBUG(message,mod)  \
// {\
//   llvm::outs() << message;llvm::outs().flush(); mod.dump();\
// }

#define LOG_DEBUG(message,mod)  0

/*******************  common tool functions ****************/

namespace tools {
  
  std::string getenv(const char *name);
  mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype);
  std::string typeToStr(mlir::Type type);
  std::string KcgDtypeToStr(KcgDtype type);
  std::string KcgKernelTypeToString(KcgKernelType type);
  void opSetAttr(mlir::Operation* op, const std::string& name, const std::string& val);
  void opSetAttr(mlir::Operation* op, const std::string& name, int val);
  bool isOpAttrEqualToString(mlir::Operation* op, const std::string& name, const std::string& expectedvalue);
  uint64_t getIntAttr(mlir::Operation* op, const std::string& name);
  std::vector<int> getIntArrayAttr(mlir::Operation* op, const std::string& name);
  /* ******** for debug use *************** */
  void _opSetDescription(mlir::Operation* op, const std::string& attrValue);

  bool parseJsonToConfigs(std::string filename, std::vector<Config>& res);

namespace mapUtils {
  
  // map utils functions
  mlir::AffineExpr waprId(mlir::AffineExpr tid, const std::map<std::string, int>& config);
  mlir::AffineExpr wapr_x(mlir::AffineExpr tid, const std::map<std::string, int>& config);
  mlir::AffineExpr wapr_y(mlir::AffineExpr tid, const std::map<std::string, int>& config);
  mlir::AffineExpr laneId(mlir::AffineExpr tid, const std::map<std::string, int>& config);
  mlir::AffineExpr lane_x(mlir::AffineExpr tid, const std::map<std::string, int>& config);
  mlir::AffineExpr lane_y(mlir::AffineExpr tid, const std::map<std::string, int>& config);
  mlir::AffineExpr bid_y(mlir::AffineExpr bid, const std::map<std::string, int>& config);
  mlir::AffineExpr bid_x(mlir::AffineExpr bid, const std::map<std::string, int>& config);
  mlir::AffineExpr bid(mlir::AffineExpr bx,mlir::AffineExpr by, const std::map<std::string, int>& config);
  mlir::AffineExpr tid(mlir::AffineExpr tx,mlir::AffineExpr ty, const std::map<std::string, int>& config);

  llvm::SmallVector<mlir::AffineExpr> reshapeBlock(mlir::AffineExpr tid, const std::vector<int> shape);

}

}

}  // KernelCodeGen
#endif  // _Utils_h_