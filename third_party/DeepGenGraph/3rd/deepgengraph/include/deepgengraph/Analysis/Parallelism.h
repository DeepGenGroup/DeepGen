#ifndef DEEPGENGRAPH_ANALYSIS_PARALLELISM_H
#define DEEPGENGRAPH_ANALYSIS_PARALLELISM_H

#include <variant>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // MLIR 稀疏数据流分析基类
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"    // Deepgengraph 方言算子定义
#include "dbg.h"                                    // 调试宏 / 函数（dbg() 等）

namespace mlir {

///----------------------------------------------
/// 批次并查集：用于把属于同一并行组(batch_id)的维度合并
///----------------------------------------------
struct BatchSet {
  std::vector<int> father;   // 并查集父节点数组，索引即 batch_id
  int total_batch = 0;       // 当前独立 batch 组数量

  /// 查找 batch_id 的代表节点，带路径压缩
  int find(int x) {
    assert(x < (int)father.size());
    assert(x >= 0);
    return father[x] == x ? x : father[x] = find(father[x]);
  }

  /// 新建一个 batch 组，返回新的 batch_id
  int alloc_batch() {
    int size = (int)father.size();
    father.push_back(size);  // 自己的父节点指向自己
    total_batch += 1;
    return size;
  }

  /// 合并两个 batch 组
  void merge(int x, int y) {
    int fx = find(x);
    int fy = find(y);
    if (fx != fy) {
      father[fx] = fy;       // 把 x 的根指向 y 的根
      total_batch -= 1;      // 独立 batch 组数量减 1
    }
  }
};

///----------------------------------------------
/// 单个维度的并行属性
///----------------------------------------------
struct ParaType {
  enum Kind {
    kInit,    // 未初始化
    kBatch,   // 完全可并行（互不依赖）
    kReUse,   // 可并行但存在数据复用
    kNonPara  // 不可并行（有依赖 / 归约）
  };

  Kind kind;      // 并行属性类型
  int  batch_id;  // 若 kind 为 kBatch/kReUse，则归属的 batch 组 ID；否则为 -1

  ParaType() : kind(kInit), batch_id(-1) {}
  ParaType(Kind kind, int batch_id = -1) : kind(kind), batch_id(batch_id) {
    if (kind == kBatch || kind == kReUse) {
      assert(batch_id >= 0); // Batch / ReUse 必须有合法 batch_id
    }
  }

  // ----------- 静态 / 成员工具函数 --------------

  /// join：合并两个 ParaType，取“更严格”并行级别并合并 batch_id
  static ParaType join(const ParaType &lhs, const ParaType &rhs, BatchSet &batch_set) {
    ParaType ret(std::max(lhs.kind, rhs.kind), std::max(lhs.batch_id, rhs.batch_id));
    if (ret.kind == kBatch || ret.kind == kReUse) {
      ret.batch_id = std::max(lhs.batch_id, rhs.batch_id);
      if (lhs.batch_id >= 0 && rhs.batch_id >= 0) {
        batch_set.merge(lhs.batch_id, rhs.batch_id); // 合并并查集
      }
    }
    return ret;
  }

  /// 成员版本：in-place join
  void join_(const ParaType &other, BatchSet &batch_set) {
    kind = std::max(kind, other.kind);
    if (kind == kBatch || kind == kReUse) {
      batch_id = std::max(batch_id, other.batch_id);
      if (batch_id >= 0 && other.batch_id >= 0)
        batch_set.merge(batch_id, other.batch_id);
    }
  }

  /// 判断两个 ParaType 是否等价（考虑 batch_id 合并后是否指向同一组）
  static bool equal(const ParaType &lhs, const ParaType &rhs, BatchSet &batch_set) {
    if (lhs.kind != rhs.kind) return false;
    if ((lhs.kind == kBatch || lhs.kind == kReUse) &&
        batch_set.find(lhs.batch_id) != batch_set.find(rhs.batch_id))
      return false;
    return true;
  }

  // ---------- 调试打印辅助 ----------------------
  void print(raw_ostream &os) const {               // 不展开 batch 并查集
    switch (kind) {
      case kInit:   os << "Init"; break;
      case kBatch:  os << "Batch("  << batch_id << ")"; break;
      case kReUse:  os << "ReUse("  << batch_id << ")"; break;
      case kNonPara:os << "NonPara"; break;
      default:      os << "Unknown"; break;
    }
  }
  void print(raw_ostream &os, BatchSet &batch_set) { // 展开并查集代表
    switch (kind) {
      case kInit:   os << "Init"; break;
      case kBatch:  os << "Batch("  << batch_set.find(batch_id) << ")"; break;
      case kReUse:  os << "ReUse("  << batch_set.find(batch_id) << ")"; break;
      case kNonPara:os << "NonPara"; break;
      default:      os << "Unknown"; break;
    }
  }
};

///----------------------------------------------
/// 一个 Value（张量）整体的并行信息：每个维度一个 ParaType
///----------------------------------------------
struct ParaInfo {
  SmallVector<ParaType> info;   // 长度==rank

  ParaInfo() = default;
  explicit ParaInfo(size_t rank) : info(rank, ParaType()) {}

  static ParaInfo from_val(Value val); // 根据张量 rank 构造

  size_t getRank() const { return info.size(); }

  // 按 dims 顺序 permute（输出 rank == 原 rank）
  ParaInfo permute_by(ArrayRef<int64_t> dims) const {
    ParaInfo ret(getRank());
    for (int i = 0; i < (int)dims.size(); ++i)
      ret.info[i] = info[dims[i]];
    return ret;
  }
  // 逆 permute：把当前 info 写回到原索引
  ParaInfo permute_from(ArrayRef<int64_t> dims) const {
    ParaInfo ret(getRank());
    for (int i = 0; i < (int)dims.size(); ++i)
      ret.info[dims[i]] = info[i];
    return ret;
  }

  // ---------- 调试打印 --------------
  void print(raw_ostream &os) const {
    if (info.empty()) { os << "[Uninit]"; return; }
    os << "[";
    for (auto t : info) { t.print(os); os << ","; }
    os << "]";
  }
  void print(raw_ostream &os, BatchSet &set) {
    if (info.empty()) { os << "[Uninit]"; return; }
    os << "[";
    for (auto t : info) { t.print(os, set); os << ","; }
    os << "]";
  }

  // 取后 other.rank 个维度（对齐）
  ParaInfo slice_like(const ParaInfo &other) {
    assert(other.getRank() <= getRank());
    ParaInfo ret(other.getRank());
    int off = getRank() - other.getRank();          // 右对齐
    for (int i = 0; i < (int)other.getRank(); ++i)
      ret.info[i] = info[i + off];
    return ret;
  }

  /// 就地 join（维度右对齐）
  void join_(const ParaInfo &other, BatchSet &set) {
    assert(!info.empty() && !other.info.empty());
    if (getRank() < other.getRank()) {
      int off = other.getRank() - getRank();
      for (size_t i = 0; i < getRank(); ++i)
        info[i].join_(other.info[i + off], set);
    } else {
      int off = getRank() - other.getRank();
      for (size_t i = off; i < getRank(); ++i)
        info[i].join_(other.info[i - off], set);
    }
  }

  /// 左侧 rank 为准创建 join 结果
  static ParaInfo join(const ParaInfo &lhs, const ParaInfo &rhs, BatchSet &set) {
    assert(lhs.getRank() || rhs.getRank());
    ParaInfo ret = lhs;
    ret.join_(rhs, set);  // 利用成员 join_ 完成右对齐合并
    return ret;
  }

  /// 判断两 ParaInfo 是否等价
  static bool equal(const ParaInfo &lhs, const ParaInfo &rhs, BatchSet &set) {
    assert(lhs.getRank() == rhs.getRank());
    for (auto [t0, t1] : llvm::zip(lhs.info, rhs.info))
      if (!ParaType::equal(t0, t1, set)) return false;
    return true;
  }

  /// 设置某一维 ParaType（支持负索引）
  void set(int idx, ParaType t) {
    if (idx < 0) idx += (int)getRank();
    assert(idx >= 0 && idx < (int)getRank());
    info[idx] = t;
  }
};

///----------------------------------------------
/// ParallelismAnalysis：遍历 KernelOp，推断每个 Value 的 ParaInfo
///----------------------------------------------
class ParallelismAnalysis {
public:
  void initialize(deepgengraph::KernelOp);          // 初始化 val_info
  void run(deepgengraph::KernelOp, bool verbose=false); // 固定点迭代传播
  void clear() { val_info.clear(); }

  /// 调试输出全部 Value 的并行信息
  void dump() {
    for (auto &p : val_info) {
      std::string s;
      llvm::raw_string_ostream os(s);
      p.second.print(os, batch_set);
      p.first.dump();
      llvm::errs() << "\t" << s << "\n";
    }
  }

  ParaInfo getInfo(Value v) { return val_info[v]; }

  BatchSet batch_set;        // 并查集，全局共享

private:
  DenseMap<Value, ParaInfo> val_info; // Value -> 并行信息映射表
};

} // namespace mlir

#endif // DEEPGENGRAPH_ANALYSIS_PARALLELISM_H
