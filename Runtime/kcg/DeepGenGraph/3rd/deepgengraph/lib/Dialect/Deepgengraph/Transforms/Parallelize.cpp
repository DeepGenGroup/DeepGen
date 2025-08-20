#include "mlir/IR/PatternMatch.h"          // 引入 MLIR 框架中的 PatternMatch （模式匹配）头文件
#include "mlir/Pass/Pass.h"                // 引入 MLIR 框架中的 Pass （编译优化步骤）头文件
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // 引入 MLIR GreedyPatternRewriteDriver，用于贪心模式重写

#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"         // 引入 Deepgengraph 方言定义
#include "deepgengraph/Dialect/Deepgengraph/Transforms/Passes.h"       // 引入 Deepgengraph 方言中 Pass 定义

#include "deepgengraph/Analysis/Parallelism.h"    // 引入并行性分析模块

#include "dbg.h"                           // 引入调试宏或函数（可能用于调试输出）

namespace mlir::deepgengraph {

#define GEN_PASS_DEF_DEEPGENGRAPHPARALLELIZE
#include "deepgengraph/Dialect/Deepgengraph/Transforms/Passes.h.inc"   // 包含由 TableGen 生成的 DeepgengraphParallelize Pass 定义

} // namespace mlir::deepgengraph

namespace mlir::deepgengraph {

namespace {

// 定义一个 ParallelizePass 类，继承自 DeepgengraphParallelizeBase，并实现平行化的优化 Pass
class ParallelizePass : public ::mlir::deepgengraph::impl::DeepgengraphParallelizeBase<ParallelizePass> {
public:
  using DeepgengraphParallelizeBase::DeepgengraphParallelizeBase;  // 继承基类构造函数

  // 该函数根据并行性分析结果，为 KernelOp 设置其ParallelMap映射信息
  void set_parallel_maps(KernelOp kernel_op, DenseSet<int> &allocated_batch_ids, ParallelismAnalysis &ana,
                         SmallVector<KernelOp::ParallelMap> &parallel_maps, DenseSet<int> &squeezed_batch_ids,
                         DenseMap<int, int> &batch_id_to_map_id) const {
    // 遍历每一个已分配的 batch 集合（father 表示一个批次等价类的代表 ID）
    for (auto father : allocated_batch_ids) {
      KernelOp::ParallelMap map;            // 为当前 father（批次组）创建一个 ParallelMap 实例，记录此并行维度的信息
      // 遍历 KernelOp 的每一个参数（输入张量）
      for (auto arg : kernel_op.getArguments()) {
        auto info = ana.getInfo(arg);       // 获取当前参数的并行性信息（每个维度的 ParaType 类型等）
        auto shape = cast<RankedTensorType>(arg.getType()).getShape();  // 获取参数张量的形状（每个维度大小）
        assert(shape.size() == info.getRank());  // 断言形状维度数与分析信息维度数匹配
        int arg_dim = -1;                   // 初始化当前参数中属于此 father 批次的维度索引为 -1（表示未找到）
        // 遍历该参数的每个维度，检查哪些维度属于当前 father 的并行批次或复用
        for (int i = 0; i < info.getRank(); ++i) {
          // 如果第 i 个维度被标记为 Batch 且其 batch_id 属于当前 father 批次组
          if (info.info[i].kind == ParaType::Kind::kBatch && ana.batch_set.find(info.info[i].batch_id) == father) {
            // 如果 map 尚未设置单位数量(unit_num)，则初始化
            if (map.unit_num <= 0) {
              map.unit_num = shape[i];      // 将 unit_num 设为该维度的大小（因为Batch意味着整个维度可以并行）
              map.size_per_unit = 1;        // size_per_unit 设为 1，表示每个并行单元大小为1（即Batch维度会被完全平分为 unit_num 个单元）
            } else {
              // 如果 map 已经有 unit_num，则断言当前维度大小等于已有的 unit_num（确保多个Batch维度大小一致）
              assert(map.unit_num == shape[i]);
              assert(map.size_per_unit == 1);
            }
            assert(arg_dim == -1);
            arg_dim = i;                    // 标记当前参数中 parallel map 对应的维度索引为 i
          }
          // 如果第 i 个维度被标记为 ReUse 且其 batch_id 属于当前 father 批次组
          if (info.info[i].kind == ParaType::Kind::kReUse && ana.batch_set.find(info.info[i].batch_id) == father) {
            // 如果 map 尚未初始化，则根据启发式设置 unit_num 和 size_per_unit
            if (map.unit_num <= 0) {
              // 启发式：如果此维度大小 > 128，则将其切分成大小为128的块
              if (shape[i] > 128) {
                int size_per_unit = 128;
                assert(shape[i] % size_per_unit == 0);   // 健全性检查：维度长度必须能被128整除
                map.unit_num = shape[i] / size_per_unit; // unit_num 为将维度按128划分后的块数
                map.size_per_unit = size_per_unit;       // 每个并行单元的大小设为128
              } else {
                // 如果维度太小（<=128），则不再进一步分块，以整个维度为并行单位
                map.size_per_unit = 1;
                map.unit_num = (int)shape[i];            // unit_num 等于该维度长度本身
              }
            } else {
              // 如果 map 已经初始化过（例如通过之前某个 Batch 维度），验证当前维度大小与已有并行划分是否匹配
              assert(shape[i] % map.size_per_unit == 0);
              assert(shape[i] / map.size_per_unit == map.unit_num);
            }
            assert(arg_dim == -1);
            arg_dim = i;                  // 标记当前参数中属于该 father 的维度索引
          }
        }
        map.arg_dims.push_back(arg_dim);  // 记录此参数对应的并行维度索引（如果没有对应并行维度则为 -1）
      }

      // 处理 KernelOp 的返回值（即其内部计算的结果张量）
      auto ret_op = cast<ReturnOp>(kernel_op.getCallableRegion()->front().getTerminator());
      for (auto res : ret_op.getOperands()) {
        auto info = ana.getInfo(res);    // 获取结果的并行性分析信息
        auto shape = cast<RankedTensorType>(res.getType()).getShape();  // 获取结果张量形状
        assert(shape.size() == info.getRank());
        int res_dim = -1;               // 初始化当前结果中属于该 father 的维度索引为 -1
        // 遍历结果的每个维度，检查哪些维度属于当前 father 的并行批次或复用
        for (int i = 0; i < info.getRank(); ++i) {
          if (info.info[i].kind == ParaType::Kind::kBatch && ana.batch_set.find(info.info[i].batch_id) == father) {
            // 如果结果的第 i 个维度是 Batch 且属于当前 father
            assert(map.unit_num == shape[i]);    // 结果的该维度大小应该等于之前计算的 unit_num
            assert(map.size_per_unit == 1);      // Batch 类型对应的 size_per_unit 应为1
            assert(res_dim == -1);
            res_dim = i;                        // 标记该结果中的并行维度索引
          }
          if (info.info[i].kind == ParaType::Kind::kReUse && ana.batch_set.find(info.info[i].batch_id) == father) {
            // 如果结果的第 i 个维度是 ReUse 且属于当前 father
            assert(map.unit_num == shape[i] / map.size_per_unit);  // 验证结果维度长度与unit_num和size_per_unit匹配
            assert(map.size_per_unit == shape[i] / map.unit_num);  // 进一步验证 size_per_unit 与 unit_num 的关系
            assert(res_dim == -1);
            res_dim = i;                        // 标记该结果中的并行维度索引
          }
        }
        map.res_dims.push_back(res_dim);        // 记录此结果对应的并行维度索引（如果没有则为 -1）
      }

      parallel_maps.push_back(map);             // 将计算好的并行映射信息添加到 parallel_maps 列表中
      if (map.size_per_unit == 1) {
        // 如果 size_per_unit 为1，意味着这个并行维度将被“压缩”（因为每个并行块大小为1，不需要额外维度表示）
        squeezed_batch_ids.insert(father);      // 记录该 father 批次ID需要在之后压缩（移除）的维度
      }
      batch_id_to_map_id[father] = (int)parallel_maps.size() - 1;  // 将 father 批次ID 映射到 parallel_maps 中对应 ParallelMap 的索引
      // map.show();  // （调试用途）可以输出 map 的信息
    }

    kernel_op.setParallelMaps(parallel_maps);    // 将计算得到的 ParallelMap 列表设置到 kernel_op 中，供后续使用
  }

  // 匹配并重写 KernelOp，实现将其转换为并行的形式
  LogicalResult matchAndRewrite(KernelOp kernel_op, OpBuilder &builder) const {
    if (kernel_op.hasParallelMap()) {
      // 如果 KernelOp 已经具有 ParallelMap（说明已经并行化过），则不再处理，返回 failure 表示无法重写
      return failure();
    }
    ParallelismAnalysis ana;
    ana.initialize(kernel_op);
    ana.run(kernel_op);
    // 上述两行对 kernel_op 进行并行性分析，收集每个张量各维度的并行类型信息（Batch、ReUse等）以及批次分组信息。

    DenseSet<int> allocated_batch_ids;
    // 遍历 KernelOp 的输入参数，确定哪些批次维度需要并行处理
    for (auto arg : kernel_op.getArguments()) {
      auto info = ana.getInfo(arg);
      // 检查参数的每个维度
      for (size_t i = 0; i < info.getRank(); ++i) {
        auto &type = info.info[i];
        if (type.kind == ParaType::Kind::kBatch) {
          // 如果这个维度是 Batch 类型
          // 使用并查集（或等价类）找到这个 batch_id 所属的代表元素，并将其加入 allocated_batch_ids 集合
          // （这里 ana.batch_set.find(type.batch_id) 返回该 batch_id 所在集合的代表 ID，类似Union-Find操作）
          if (!allocated_batch_ids.contains(ana.batch_set.find(type.batch_id))) {
            allocated_batch_ids.insert(ana.batch_set.find(type.batch_id));
          }
        } else if (type.kind == ParaType::Kind::kReUse) {
          // 如果这个维度是 ReUse 类型（可重用的并行维度）
          auto shape = cast<RankedTensorType>(arg.getType()).getShape();
          // 为了局部性考虑，我们不并行化最内层维度（通常最内层维度对应连续内存，避免对缓存不利）
          bool is_innermost_dim = i == (int)(info.getRank()) - 1;
          if (!is_innermost_dim && !allocated_batch_ids.contains(ana.batch_set.find(type.batch_id))) {
            // 如果不是最内层维度且尚未加入，则将该 batch_id 的代表加入并行维度集合
            allocated_batch_ids.insert(ana.batch_set.find(type.batch_id));
          }
        }
      }
    }

    SmallVector<KernelOp::ParallelMap> parallel_maps;
    DenseSet<int> squeezed_batch_ids;
    DenseMap<int, int> batch_id_to_map_id;
    // 基于刚才选定的并行批次维度集合，设置 KernelOp 的 ParallelMap 列表和辅助映射
    set_parallel_maps(kernel_op, allocated_batch_ids, ana, parallel_maps, squeezed_batch_ids, batch_id_to_map_id);

    // 如果未启用 partition（划分）选项，则仅设置ParallelMap元数据，不实际重构 IR
    if (!partition) {
      return success();
    }

    // 以下代码在 partition 模式启用时执行，对 KernelOp 的内部 IR 进行重构，将计算分解到 ParallelForOp 中

    // **1. 创建新的入口基本块 (build_block)**
    Block *build_block = builder.createBlock(kernel_op.getCallableRegion());
    builder.setInsertionPointToStart(build_block);
    // createBlock(kernel_op.getCallableRegion()) 会在 kernel_op 的region（函数体）最前面插入一个新的空基本块，并返回指向它的指针
    // setInsertionPointToStart 将后续插入操作定位到该新基本块的起始处

    auto arg_types = kernel_op.getArgumentTypes();  // 获取 KernelOp 输入参数的类型列表
    auto res_types = kernel_op.getResultTypes();    // 获取 KernelOp 返回值（结果）的类型列表
    // 将 KernelOp 原有的参数类型添加为 build_block 的参数（保持与原函数相同签名）
    for (auto arg_type : arg_types) {
      build_block->addArgument(arg_type, kernel_op.getLoc());
    }

    // **2. 创建 ParallelForOp 操作**
    // 在新的入口块中创建一个 ParallelForOp，其作用类似于在 KernelOp 内部包裹一个并行循环
    // ParallelForOp 的构造：给定位置loc，结果类型列表，与循环迭代相关的参数（这里传入了原参数作为ParallelForOp的迭代初始值）
    auto para_for_op = builder.create<ParallelForOp>(kernel_op.getLoc(), res_types, build_block->getArguments());
    // ParallelForOp 本身包含一个 Region 将承载循环体
    Block *entry = builder.createBlock(&para_for_op.getRegion());
    // 在 para_for_op 的内部Region中新建一个基本块（entry），作为并行循环的循环体起始块

    // **3. 计算并调整并行循环体中参数和结果的类型 (para_arg_types 和 para_res_types)**
    SmallVector<Type> para_arg_types(arg_types.begin(), arg_types.end());  // 先复制原参数类型列表
    SmallVector<Type> para_res_types(res_types.begin(), res_types.end());  // 复制原结果类型列表
    // 遍历每个 ParallelMap（每个并行的批次维度）
    for (auto &map : parallel_maps) {
      // 针对每个输入参数，调整其类型中对应并行维度的大小
      for (size_t i = 0; i < map.arg_dims.size(); ++i) {
        if (map.arg_dims[i] < 0) continue;  // 如果此参数没有并行维度，跳过
        // 获取当前参数的类型并将其转换为RankedTensorType以便修改shape
        auto type = cast<RankedTensorType>(para_arg_types[i]);
        SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
        // 将并行维度的大小设为 map.size_per_unit（每个并行任务处理的块大小）
        shape[map.arg_dims[i]] = map.size_per_unit;
        // 用修改后的 shape 重建类型（维持原元素类型不变）
        para_arg_types[i] = RankedTensorType::get(shape, type.getElementType());
      }
      // 针对每个输出结果，调整其类型中对应并行维度的大小
      for (size_t i = 0; i < map.res_dims.size(); ++i) {
        if (map.res_dims[i] < 0) continue;
        auto type = cast<RankedTensorType>(para_res_types[i]);
        SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
        shape[map.res_dims[i]] = map.size_per_unit;
        para_res_types[i] = RankedTensorType::get(shape, type.getElementType());
      }
    }
    // **4. 压缩(squeeze)并行维度大小为1的情况** 
    // 如果某些并行维度 size_per_unit == 1，我们可以将该维度从张量形状中移除（因为每个分块只有1，等价于不需要显式维度）
    for (int arg_i = 0; arg_i < kernel_op.getNumArguments(); ++arg_i) {
      auto arg = kernel_op.getArgument(arg_i);
      auto info = ana.getInfo(arg);
      auto type = cast<RankedTensorType>(arg.getType());
      auto para_shape = cast<RankedTensorType>(para_arg_types[arg_i]).getShape();
      SmallVector<int64_t> new_shape;
      // 遍历当前参数的每个维度，对比并行前后的 shape
      for (int i = 0; i < (int)para_shape.size(); ++i) {
        auto &para_type = info.info[i];
        // 判断该维度是否是 Batch 或 ReUse 且属于需要压缩的批次（即 size_per_unit == 1）
        if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
            squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
          assert(para_shape[i] == 1);
          // 如果该维度是被压缩的并行维度，跳过（不加入 new_shape，相当于移除该维度）
        } else {
          new_shape.push_back(para_shape[i]);  // 非压缩维度保持原大小加入新形状列表
        }
      }
      // 根据 new_shape 构造参数的新类型（元素类型与原类型相同）
      para_arg_types[arg_i] = RankedTensorType::get(new_shape, type.getElementType());
    }
    // 对结果类型执行相同的压缩逻辑
    for (int res_i = 0; res_i < kernel_op.getNumResults(); ++res_i) {
      auto res = kernel_op.getCallableRegion()->front().getTerminator()->getOperand(res_i);
      auto info = ana.getInfo(res);
      auto type = cast<RankedTensorType>(res.getType());
      auto para_shape = cast<RankedTensorType>(para_res_types[res_i]).getShape();
      SmallVector<int64_t> new_shape;
      for (int i = 0; i < (int)para_shape.size(); ++i) {
        auto &para_type = info.info[i];
        if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
            squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
          assert(para_shape[i] == 1);
          // 移除被压缩的并行维度
        } else {
          new_shape.push_back(para_shape[i]);
        }
      }
      para_res_types[res_i] = RankedTensorType::get(new_shape, type.getElementType());
    }

    // **5. 设置并行循环体 (entry 块) 的参数**
    SmallVector<Value> para_id_args;   // 用于保存并行迭代索引（IDs）的占位参数
    SmallVector<Value> para_args;      // 用于保存调整后每个参数的占位
    // 为每个 ParallelMap（每个并行批次维度）添加一个 Index 类型的 block 参数（即 parallel for 循环的迭代索引）
    for (int i = 0; i < (int)parallel_maps.size(); ++i) {
      auto index_type = builder.getIndexType();
      auto id_arg = entry->addArgument(index_type, kernel_op.getLoc());
      para_id_args.push_back(id_arg);
    }
    // 添加调整后类型的参数到并行循环体的基本块（entry）中
    for (auto arg_type : para_arg_types) {
      auto para_arg = entry->addArgument(arg_type, kernel_op.getLoc());
      para_args.push_back(para_arg);
    }

    // **6. 克隆原始计算到并行循环体中，并调整索引和类型**
    IRMapping val_map;  // 用于映射旧的值到新的值（克隆过程中的映射表）
    // 将 KernelOp 原入口块中的参数映射到我们新并行块的参数上
    for (auto [original_arg, arg] : llvm::zip(kernel_op.getCallableRegion()->front().getArguments(), para_args)) {
      val_map.map(original_arg, arg);
    }
    Block *original_block = &kernel_op.getCallableRegion()->front();
    // 遍历 KernelOp 原始主体块中的每一个操作 (PreOrder 确保我们先处理定义再处理使用)
    original_block->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto const_op = dyn_cast<arith::ConstantOp>(op)) {
        // 如果是常量操作
        auto res = const_op.getResult();
        bool parallelized = false;
        if (isa<RankedTensorType>(res.getType())) {
          // 检查该常量输出（tensor）是否含有要并行的维度被压缩
          auto info = ana.getInfo(res);
          for (int i = 0; i < (int)info.getRank(); ++i) {
            auto para_type = info.info[i];
            // 判断此结果的每个维度，如果被标记为 Batch/ReUse 且该 batch_id 在需要压缩的集合中，则视为并行化
            if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
                squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
              parallelized = true;
            }
          }
        }
        if (!parallelized) {
          // 如果这个常量没有并行维度需要特殊处理，直接在新的并行循环体中克隆它
          auto new_op = builder.clone(*op);
          // 将旧操作结果映射到新操作结果
          for (auto [res, mapped_res] : llvm::zip(op->getResults(), new_op->getResults())) {
            val_map.map(res, mapped_res);
          }
        } else {
          // 如果常量涉及并行维度压缩（比如常量张量具有被压缩的维度），当前实现不支持，标记为不可达
          llvm_unreachable("not support");
        }
      } else if (auto mask_op = dyn_cast<MaskOp>(op)) {
        // 如果是 MaskOp（掩码操作，可能涉及张量子区域的提取/填充）
        auto new_op = builder.clone(*op, val_map);    // 根据映射直接克隆 MaskOp 及其操作数
        auto new_mask_op = cast<MaskOp>(new_op);

        auto res = mask_op.getResult();
        SmallVector<int64_t> sizes(mask_op.getSizes());  // 复制原 MaskOp 的 sizes 属性（表示每个维度的大小）
        auto info = ana.getInfo(res);
        // 对 MaskOp 的结果各维度检查，如果该维度是需要并行划分的，则用并行索引替换 MaskOp 对应维度的操作数
        for (int i = 0; i < (int)info.getRank(); ++i) {
          auto para_type = info.info[i];
          auto batch_id = para_type.batch_id <= 0 ? -1 : ana.batch_set.find(para_type.batch_id);
          if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
              allocated_batch_ids.contains(batch_id)) {
            // 如果此维度是并行维度
            int map_id = batch_id_to_map_id[batch_id];
            new_mask_op->setOperand(i, para_id_args[map_id]);       // 将 MaskOp 对应维度的操作数替换为并行循环的索引（parallel id）
            sizes[i] = parallel_maps[map_id].size_per_unit;         // 将该维度的 size 修改为每个并行单元的大小
          }
        }
        new_mask_op.setSizes(sizes);  // 更新 MaskOp 的 sizes 属性为新尺寸
        // 通过 InferTypeOpInterface 重新推断新 MaskOp 的结果类型（因为 shape 变化了）
        auto type_infer = cast<InferTypeOpInterface>(new_op);
        llvm::SmallVector<::mlir::Type, 1> new_types;
        auto success = type_infer.inferReturnTypes(new_op->getContext(), new_op->getLoc(), new_op->getOperands(),
                                                   new_op->getAttrDictionary(), new_op->getPropertiesStorage(),
                                                   new_op->getRegions(), new_types);
        assert(succeeded(success));
        // 将推断出的新结果类型赋给新操作
        for (size_t i = 0; i < new_types.size(); ++i) {
          new_op->getResult(i).setType(new_types[i]);
        }
        // 映射 MaskOp 原结果值到新 MaskOp 的结果值
        for (auto [res, mapped_res] : llvm::zip(op->getResults(), new_op->getResults())) {
          val_map.map(res, mapped_res);
        }

        // 不进入 MaskOp 的内部区域继续遍历（如果 MaskOp 有内部 region 的话），跳过其内部操作
        return WalkResult::skip();
      } else if (auto permute_op = dyn_cast<PermuteOp>(op)) {
        // 如果是张量维度置换操作 PermuteOp
        auto arg = permute_op.getOperand();
        auto permute_dims = permute_op.getDims();  // 原 PermuteOp 的维度置换列表
        auto info = ana.getInfo(arg);
        DenseMap<int, int> dim_map;
        // 构建旧维度到新维度的映射（去除被压缩的维度）
        for (int i = 0, j = 0; i < (int)info.getRank(); ++i) {
          auto para_type = info.info[i];
          if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
              squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
            dim_map[i] = -1;   // 如果该维度被压缩，则标记为 -1（表示丢弃该维度）
          } else {
            dim_map[i] = j++;  // 否则映射到新的维度索引（压缩维度被跳过）
          }
        }

        SmallVector<int64_t> new_permute_dims;
        for (auto dim : permute_dims) {
          if (dim_map[dim] >= 0) {
            new_permute_dims.push_back(dim_map[dim]);  // 只保留未被压缩的维度映射
          }
        }

        // 使用映射后的维度顺序创建新的 PermuteOp
        auto new_op = builder.create<PermuteOp>(op->getLoc(), val_map.lookup(arg), new_permute_dims);
        // 将旧 PermuteOp 的结果值映射到新 PermuteOp 的结果值
        val_map.map(permute_op.getResult(), new_op.getResult());
      } else if (auto reduce_op = dyn_cast<ReduceOp>(op)) {
        // 如果是 ReduceOp（张量归约操作，如求和、均值等）
        auto arg = reduce_op.getOperand();
        int64_t dim = reduce_op.getReduceDimensionAttr().getInt();  // 获取要归约的维度索引
        if (dim < 0) {
          // 处理负索引（负数表示从后往前数维度）
          dim += cast<RankedTensorType>(reduce_op.getOperand().getType()).getRank();
        }
        auto reduce_type = reduce_op.getReduceType();
        bool keep_dim = reduce_op.getKeepDim();  // 是否保留归约后的维度（例如 PyTorch keepdim 参数）
        auto info = ana.getInfo(arg);
        DenseMap<int, int> dim_map;
        // 类似 Permute 的逻辑，建立旧维度到新维度的映射，跳过被压缩的维度
        for (int i = 0, j = 0; i < (int)info.getRank(); ++i) {
          auto para_type = info.info[i];
          if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
              squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
            dim_map[i] = -1;
          } else {
            dim_map[i] = j++;
          }
        }
        // 归约维度本身不应是并行压缩维度（假定不会并行化要归约的维度）
        assert(dim_map[dim] >= 0);

        // 创建新的 ReduceOp，使用映射后的归约维度索引，并保持其他参数（归约类型和是否保留维度）不变
        auto new_op = builder.create<ReduceOp>(op->getLoc(), val_map.lookup(arg), dim_map[dim], reduce_type, keep_dim);
        // 将旧 ReduceOp 的结果值映射到新 ReduceOp 的结果值
        val_map.map(reduce_op.getResult(), new_op.getResult());
      } else if (isa<InferTypeOpInterface>(op) && !isa<TriluOp>(op)) {
        // 如果操作实现了 InferTypeOpInterface（可推断输出类型），且不是特殊情况 TriluOp
        auto new_op = builder.clone(*op);
        // 更新新克隆操作的每一个操作数为映射后的值（即用我们并行循环体中的值替换原来的值）
        for (size_t i = 0; i < new_op->getNumOperands(); ++i) {
          new_op->setOperand(i, val_map.lookup(new_op->getOperand(i)));
        }
        // 通过 InferType 接口推断新操作的输出类型（因为输入的shape可能变了，需要重新计算输出shape）
        auto type_infer = cast<InferTypeOpInterface>(new_op);
        llvm::SmallVector<::mlir::Type, 1> new_types;
        auto success = type_infer.inferReturnTypes(new_op->getContext(), new_op->getLoc(), new_op->getOperands(),
                                                   new_op->getAttrDictionary(), new_op->getPropertiesStorage(),
                                                   new_op->getRegions(), new_types);
        assert(succeeded(success));
        // 用推断的类型更新新操作的结果类型
        for (size_t i = 0; i < new_types.size(); ++i) {
          new_op->getResult(i).setType(new_types[i]);
        }
        // 映射旧操作结果到新操作结果
        for (auto [res, mapped_res] : llvm::zip(op->getResults(), new_op->getResults())) {
          val_map.map(res, mapped_res);
        }
      } else if (auto return_op = dyn_cast<ReturnOp>(op)) {
        // 如果是函数的 ReturnOp（KernelOp 的返回终止操作）
        // 收集 ParallelForOp 循环体需要 yield（产出）的值
        SmallVector<Value> yield_args;
        for (size_t i = 0; i < return_op.getNumOperands(); ++i) {
          auto res = return_op.getOperand(i);           // 原返回值
          auto new_res = val_map.lookup(res);           // 获取映射后的新值（应该是在并行循环体内计算得到的）
          auto type = cast<RankedTensorType>(new_res.getType());
          auto para_type = cast<RankedTensorType>(para_res_types[i]);
          // 验证新结果的形状是否与预期的并行结果类型一致
          assert(type.getRank() == para_type.getRank());
          for (int j = 0; j < (int)type.getRank(); ++j) {
            if (type.getShape()[j] != para_type.getShape()[j]) {
              kernel_op->dump();
              llvm_unreachable("result shape not verified");
            }
          }
          yield_args.push_back(new_res);
        }
        // 在并行循环体内创建 ParallelYieldOp，将计算所得的新结果作为并行循环的产出
        builder.create<ParallelYieldOp>(op->getLoc(), yield_args);

        // **7. 结束并行循环，恢复外层返回**
        builder.setInsertionPointAfter(para_for_op);
        // 在 ParallelForOp 之后（外层）插入 ReturnOp，将 ParallelForOp 的输出作为整个 KernelOp 的输出返回
        builder.create<ReturnOp>(op->getLoc(), para_for_op.getResults());
      } else {
        // 其他未支持的操作类型，标记为不可达（开发过程中用来提示出现意料之外的操作）
        op->dump();
        llvm_unreachable("not supported");
      }
      return WalkResult::advance();
    });

    // **8. 清理原始基本块**
    // 原始块中的操作已被克隆并映射到新并行循环块，现在可以安全删除原块
    assert(original_block->use_empty());
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*original_block))) {
      assert(op.use_empty());
      op.erase();       // 移除原块中的每个操作
    }
    original_block->erase();  // 删除原基本块本身

    // kernel_op->dump();  // （调试用途）输出 KernelOp 当前状态
    return success();
  }

  // runOnOperation 是 Pass 的入口，在每个 KernelOp 上运行优化
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp kernel_op = getOperation();
    // dbg(partition);  // （调试用途）输出 partition 标志
    OpBuilder builder(context);
    // 调用 matchAndRewrite 对 KernelOp 进行模式匹配和重写转换
    if (failed(matchAndRewrite(kernel_op, builder))) {
      signalPassFailure();  // 如果重写失败，则标记整个 Pass 失败
    }
  }
};

} // namespace (匿名命名空间结束，使ParallelizePass仅在本文件可见)

// 提供创建 ParallelizePass 的工厂函数
std::unique_ptr<mlir::Pass> createParallelizePass() {
  return std::make_unique<ParallelizePass>();
}
std::unique_ptr<mlir::Pass> createParallelizePass(const DeepgengraphParallelizeOptions &options) {
  return std::make_unique<ParallelizePass>(options);
}

} // namespace mlir::deepgengraph
