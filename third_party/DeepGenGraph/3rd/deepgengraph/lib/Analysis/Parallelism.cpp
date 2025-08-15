#include "deepgengraph/Analysis/Parallelism.h"       // 包含Parallelism.h头文件，定义并行性分析所需的类和函数
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h" // 包含Deepgengraph方言IR的头文件（定义Deepgengraph算子）
#include "dbg.h"                              // 包含调试宏和函数（例如dbg宏）

namespace mlir {

ParaInfo ParaInfo::from_val(Value val) {
  auto rank = cast<RankedTensorType>(val.getType()).getRank(); // 获取值的类型并转换为RankedTensorType，然后获取其秩（维度数）
  return ParaInfo(rank);                                       // 返回基于该rank创建的ParaInfo对象
}

void ParallelismAnalysis::initialize(deepgengraph::KernelOp kernel_op) {
  this->clear(); // 清除之前的分析结果
  for (auto arg : kernel_op.getArguments()) { 
    val_info[arg] = ParaInfo::from_val(arg); // 为每个Kernel参数创建初始的ParaInfo（根据参数的张量维度）
  }

  kernel_op.getCallableRegion()->front().walk([&](Operation *op) {
    for (auto res : op->getResults()) {
      if (isa<RankedTensorType>(res.getType())) { 
        val_info[res] = ParaInfo::from_val(res); // 如果结果是RankedTensorType，则创建对应ParaInfo进行初始化
      }
    }
  });
}

void ParallelismAnalysis::run(deepgengraph::KernelOp kernel_op, bool verbose) {
  auto block = &kernel_op.getCallableRegion()->front(); // 获取KernelOp的第一个基本块（block）

  bool changed = true;
  int iter = 0;
  while (changed) { // 迭代传播并行信息，直到不再有变化或超过上限
    if (iter > 1000) { 
      dbg(iter);                                         // 调试输出当前迭代次数
      llvm_unreachable("too many iterations");           // 防止陷入死循环，超过1000次迭代则报错退出
    }
    changed = false; // 每轮迭代开始时重置change标志
    iter += 1;       // 增加迭代计数
    // 以前序遍历block中的操作，以避免深入mask_op的内部区域
    block->walk<WalkOrder::PreOrder>([&](Operation *op) {
      // llvm::errs() << "op: " << op->getName() << "\n"; // （调试）打印当前操作的名称
      if (auto convert_op = dyn_cast<deepgengraph::ConvertOp>(op)) { // 如果当前操作是类型转换（ConvertOp）
        auto arg = convert_op.getOperand();                // 获取ConvertOp的输入操作数
        auto res = convert_op.getResult();                 // 获取ConvertOp的输出结果

        auto arg_old_info = val_info[arg];                 // 取得输入当前的并行信息
        auto res_old_info = val_info[res];                 // 取得输出当前的并行信息

        auto arg_new_info = ParaInfo::join(arg_old_info, res_old_info, batch_set); // 合并输入和输出的并行信息
        auto res_new_info = ParaInfo::join(res_old_info, arg_old_info, batch_set); // 反向合并输出和输入的并行信息

        if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) { // 如果合并后输入的信息有变化
          changed = true; 
          val_info[arg] = arg_new_info;   // 更新输入的并行信息
        }
        if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) { // 如果合并后输出的信息有变化
          changed = true;
          val_info[res] = res_new_info;   // 更新输出的并行信息
        }
      } else if (auto permute_op = dyn_cast<deepgengraph::PermuteOp>(op)) { // 如果当前操作是PermuteOp（维度重排）
        auto arg = permute_op.getOperand();               // 获取PermuteOp的输入张量
        auto res = permute_op.getResult();                // 获取PermuteOp的输出张量
        auto dims = permute_op.getDims();                 // 获取维度置换顺序数组

        auto arg_old_info = val_info[arg];                // 输入张量原有的并行信息
        auto res_old_info = val_info[res];                // 输出张量原有的并行信息

        auto arg_new_info = ParaInfo::join(arg_old_info, res_old_info.permute_from(dims), batch_set); // 将输出信息按dims映射回输入维度顺序后，与输入信息合并
        auto res_new_info = ParaInfo::join(res_old_info, arg_old_info.permute_by(dims), batch_set);   // 将输入信息按dims映射到输出维度顺序后，与输出信息合并

        if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) { // 如果输入信息更新
          changed = true;
          val_info[arg] = arg_new_info;
        }
        if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) { // 如果输出信息更新
          changed = true;
          val_info[res] = res_new_info;
        }
      } else if (auto bin_op = dyn_cast<deepgengraph::BroadcastableBinaryOpInterface>(op)) { // 如果当前操作实现了BroadcastableBinaryOpInterface（广播二元操作）
        auto lhs = bin_op.getLhs();                          // 左侧操作数
        auto rhs = bin_op.getRhs();                          // 右侧操作数
        auto res = bin_op.getResult();                       // 运算结果

        auto lhs_broadcasted_shape = bin_op.getLhsBroadcastedShape(); // 左操作数广播到结果形状后的形状
        auto rhs_broadcasted_shape = bin_op.getRhsBroadcastedShape(); // 右操作数广播到结果形状后的形状
        auto res_shape = bin_op.getExpectedResultShape();             // 期望的结果张量形状

        ParaInfo res_new_info(res_shape.size());                     // 新的结果并行信息对象（长度为结果维度数）
        ParaInfo lhs_new_info(lhs_broadcasted_shape.size());         // 新的左操作数并行信息对象（长度为左操作数广播后维度数）
        ParaInfo rhs_new_info(rhs_broadcasted_shape.size());         // 新的右操作数并行信息对象

        auto lhs_old_info = val_info[lhs];                           // 左操作数原有并行信息
        auto rhs_old_info = val_info[rhs];                           // 右操作数原有并行信息
        auto res_old_info = val_info[res];                           // 结果原有并行信息

        for (int i = 0; i < (int)res_shape.size(); ++i) {            // 遍历结果的每个维度
          auto lhs_d = lhs_broadcasted_shape[i];                     // 左侧该维度长度（广播后）
          auto rhs_d = rhs_broadcasted_shape[i];                     // 右侧该维度长度（广播后）
          auto batch_id = batch_set.alloc_batch();                   // 分配一个新的批次ID用于该维度

          if (lhs_d == 1 && rhs_d > 1) {                             // 左侧长度为1且右侧长度大于1（左侧该维是广播得到）
            lhs_new_info.info[i] = ParaType(ParaType::kNonPara);         // 左侧该维标记为非并行（因为是广播，不随批次变化）
            rhs_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);  // 右侧该维标记为复用类型，使用batch_id
            res_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);  // 结果该维也标记为复用同一batch_id
          } else if (lhs_d > 1 && rhs_d == 1) {                      // 右侧长度为1且左侧大于1（右侧该维是广播得到）
            lhs_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);  // 左侧该维标记为复用类型
            rhs_new_info.info[i] = ParaType(ParaType::kNonPara);         // 右侧该维标记为非并行
            res_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);  // 结果该维标记为复用类型
          } else {                                                  // 否则，左右两侧此维长度相等（均>1，正常并行维度）
            assert(lhs_d == rhs_d);
            lhs_new_info.info[i] = ParaType(ParaType::kBatch, batch_id);  // 左侧该维标记为新的批次维度
            rhs_new_info.info[i] = ParaType(ParaType::kBatch, batch_id);  // 右侧该维标记为相同批次维度
            res_new_info.info[i] = ParaType(ParaType::kBatch, batch_id);  // 结果该维也标记为该批次维度
          }
        }
        lhs_new_info.join_(lhs_old_info, batch_set); // 合并新旧左操作数的信息
        rhs_new_info.join_(rhs_old_info, batch_set); // 合并新旧右操作数的信息
        res_new_info.join_(res_old_info, batch_set); // 合并新旧结果的信息

        // 在下一步同时进行正向和反向传播，整合各维度的并行信息
        for (int i = 0; i < (int)res_shape.size(); ++i) {
          auto lhs_d = lhs_broadcasted_shape[i];
          auto rhs_d = rhs_broadcasted_shape[i];
          // 正向传播：用有实际数据维度的一侧更新结果的该维信息
          if (lhs_d == 1 && rhs_d > 1) {
            res_new_info.info[i].join_(rhs_new_info.info[i], batch_set); // 左侧为广播，结果该维与右侧信息对齐
          } else if (lhs_d > 1 && rhs_d == 1) {
            res_new_info.info[i].join_(lhs_new_info.info[i], batch_set); // 右侧为广播，结果该维与左侧信息对齐
          } else {
            res_new_info.info[i].join_(lhs_new_info.info[i], batch_set); // 左右维度都有数据：结果该维与左侧对齐
            res_new_info.info[i].join_(rhs_new_info.info[i], batch_set); // 同时结果该维也与右侧对齐（两者本就应一致）
          }
          // 反向传播：将结果的该维信息反馈回左右两侧
          lhs_new_info.info[i].join_(res_new_info.info[i], batch_set); // 将结果该维的信息合并回左侧该维
          rhs_new_info.info[i].join_(res_new_info.info[i], batch_set); // 将结果该维的信息合并回右侧该维
        }

        lhs_new_info = lhs_new_info.slice_like(lhs_old_info); // 截断调整左侧信息，使其维度数与原左操作数一致
        rhs_new_info = rhs_new_info.slice_like(rhs_old_info); // 截断调整右侧信息，使其维度数与原右操作数一致

        if (!ParaInfo::equal(lhs_new_info, lhs_old_info, batch_set)) { // 若左操作数并行信息有更新
          changed = true;
          val_info[lhs] = lhs_new_info;
        }
        if (!ParaInfo::equal(rhs_new_info, rhs_old_info, batch_set)) { // 若右操作数并行信息有更新
          changed = true;
          val_info[rhs] = rhs_new_info;
        }
        if (!ParaInfo::equal(res_new_info, res_old_info, batch_set)) { // 若结果并行信息有更新
          changed = true;
          val_info[res] = res_new_info;
        }
      } else if (auto dot_op = dyn_cast<deepgengraph::DotOp>(op)) { // 如果当前操作是矩阵乘法（DotOp）
        auto lhs = dot_op.getLhs();                  // 左矩阵
        auto rhs = dot_op.getRhs();                  // 右矩阵
        auto res = dot_op.getResult();               // 结果矩阵

        auto lhs_old_info = val_info[lhs];           // 左矩阵原有并行信息
        auto rhs_old_info = val_info[rhs];           // 右矩阵原有并行信息
        auto res_old_info = val_info[res];           // 结果矩阵原有并行信息

        size_t rank = res_old_info.getRank();        // 结果矩阵的秩（维度数）
        ParaInfo res_new_info(rank);                 // 创建新的结果并行信息对象
        ParaInfo lhs_new_info(rank);                 // 创建新的左矩阵并行信息对象
        ParaInfo rhs_new_info(rank);                 // 创建新的右矩阵并行信息对象
        assert(rank >= 2);                           // 确认矩阵乘法结果至少是二维的

        // 为除最后两维以外的批次维度分配批次ID并标记
        for (int i = 0; i < (int)rank - 2; ++i) {
          int batch_id = batch_set.alloc_batch(); 
          lhs_new_info.set(i, ParaType(ParaType::kBatch, batch_id)); // 左矩阵第i维标记为批次维度
          rhs_new_info.set(i, ParaType(ParaType::kBatch, batch_id)); // 右矩阵第i维标记为批次维度
          res_new_info.set(i, ParaType(ParaType::kBatch, batch_id)); // 结果矩阵第i维标记为批次维度
        }
        int row_batch_id = batch_set.alloc_batch();  // 分配行维度复用ID
        int col_batch_id = batch_set.alloc_batch();  // 分配列维度复用ID
        lhs_new_info.set(-2, ParaType(ParaType::kReUse, row_batch_id)); // 左矩阵倒数第二维（行数m）标记为复用ID（row_batch_id）
        res_new_info.set(-2, ParaType(ParaType::kReUse, row_batch_id)); // 结果矩阵倒数第二维（行数m）与左矩阵共享row_batch_id
        rhs_new_info.set(-1, ParaType(ParaType::kReUse, col_batch_id)); // 右矩阵最后一维（列数n）标记为复用ID（col_batch_id）
        res_new_info.set(-1, ParaType(ParaType::kReUse, col_batch_id)); // 结果矩阵最后一维（列数n）与右矩阵共享col_batch_id
        lhs_new_info.set(-1, ParaType(ParaType::kNonPara));             // 左矩阵最后一维（k维度，即乘积维）标记为非并行
        rhs_new_info.set(-2, ParaType(ParaType::kNonPara));             // 右矩阵倒数第二维（k维度）标记为非并行

        lhs_new_info.join_(lhs_old_info, batch_set); // 合并左矩阵新旧信息
        rhs_new_info.join_(rhs_old_info, batch_set); // 合并右矩阵新旧信息
        res_new_info.join_(res_old_info, batch_set); // 合并结果矩阵新旧信息

        // 正向传播：将输入的批次/行/列维度信息传递给结果
        for (int i = 0; i < (int)rank - 2; ++i) {
          res_new_info.info[i].join_(lhs_new_info.info[i], batch_set); // 结果的第i个批次维度继承输入批次维度信息
          res_new_info.info[i].join_(lhs_new_info.info[i], batch_set); // （注意：这里重复join同一信息，可能是确保批次ID统一）
        }
        res_new_info.info[rank - 2].join_(lhs_new_info.info[rank - 2], batch_set); // 结果的行维度与左矩阵行维度对齐
        res_new_info.info[rank - 1].join_(rhs_new_info.info[rank - 1], batch_set); // 结果的列维度与右矩阵列维度对齐
        // 反向传播：将结果的并行信息反馈给输入矩阵
        for (int i = 0; i < (int)rank - 2; ++i) {
          lhs_new_info.info[i].join_(res_new_info.info[i], batch_set); // 左矩阵批次维度同步结果对应维度信息
          rhs_new_info.info[i].join_(res_new_info.info[i], batch_set); // 右矩阵批次维度同步结果对应维度信息
        }
        lhs_new_info.info[rank - 2].join_(res_new_info.info[rank - 2], batch_set); // 左矩阵行维度同步结果行维度信息
        rhs_new_info.info[rank - 1].join_(res_new_info.info[rank - 1], batch_set); // 右矩阵列维度同步结果列维度信息

        // 检查更新：如果有变化则记录并标记changed
        if (!ParaInfo::equal(res_new_info, res_old_info, batch_set)) {
          changed = true;
          val_info[res] = res_new_info;
        }
        if (!ParaInfo::equal(lhs_new_info, lhs_old_info, batch_set)) {
          changed = true;
          val_info[lhs] = lhs_new_info;
        }
        if (!ParaInfo::equal(rhs_new_info, rhs_old_info, batch_set)) {
          changed = true;
          val_info[rhs] = rhs_new_info;
        }
      } else if (isa<deepgengraph::Exp2Op, deepgengraph::ExpOp, deepgengraph::LogOp, deepgengraph::NegOp, deepgengraph::TanhOp>(op)) { // 如果当前操作是单目运算（Exp2/Exp/Log/Neg/Tanh等）
        assert(op->getNumOperands() == 1); // 确认该操作只有一个输入
        assert(op->getNumResults() == 1);  // 确认只有一个输出
        auto arg = op->getOperand(0);      // 获取输入操作数
        auto res = op->getResult(0);       // 获取输出结果

        auto arg_old_info = val_info[arg]; // 输入原有并行信息
        auto res_old_info = val_info[res]; // 输出原有并行信息

        // 单目运算：输入与输出形状一致，直接统一两者的并行信息
        auto new_info = ParaInfo::join(arg_old_info, res_old_info, batch_set); // 合并输入与输出的并行信息

        if (!ParaInfo::equal(arg_old_info, new_info, batch_set)) { // 如果输入信息有更新
          changed = true;
          val_info[arg] = new_info;
        }
        if (!ParaInfo::equal(res_old_info, new_info, batch_set)) { // 如果输出信息有更新
          changed = true;
          val_info[res] = new_info;
        }
      } else if (auto reduce_op = dyn_cast<deepgengraph::ReduceOp>(op)) { // 如果当前操作是ReduceOp（归约操作）
        if (reduce_op.getInit() == nullptr) {             // 仅处理无初始值的归约（如sum等）
          auto arg = reduce_op.getOperand();              // 被归约的输入张量
          auto res = reduce_op.getResult();               // 归约结果张量
          int64_t dim = reduce_op.getReduceDimensionAttr().getValue().getSExtValue(); // 获取要归约的维度索引
          bool keep_dim = reduce_op.getKeepDim();         // 是否在结果中保留该维度（大小为1）
          if (dim < 0) {
            // 如果dim为负索引，换算为正索引（加上输入rank）
            dim += (int64_t)cast<RankedTensorType>(arg.getType()).getRank();
          }

          auto arg_old_info = val_info[arg];              // 输入张量原有并行信息
          auto res_old_info = val_info[res];              // 输出张量原有并行信息

          ParaInfo arg_new_info(arg_old_info.getRank());  // 新的输入并行信息（长度与输入维度数相同）
          ParaInfo res_new_info(res_old_info.getRank());  // 新的输出并行信息

          arg_new_info.set(dim, ParaType(ParaType::kNonPara)); // 输入张量被归约的维度标记为非并行
          if (keep_dim) {
            res_new_info.set(dim, ParaType(ParaType::kNonPara)); // 如果结果保留该维度，则将结果对应维度标记为非并行
          }

          arg_new_info.join_(arg_old_info, batch_set); // 合并旧输入信息
          res_new_info.join_(res_old_info, batch_set); // 合并旧输出信息

          // 将并行信息在输入和输出间双向传播，使除了被归约维度外的其他维度信息对齐
          for (int arg_i = 0, res_i = 0; arg_i < arg_new_info.getRank(); ++arg_i) {
            if (arg_i == dim && !keep_dim)              // 如果是被归约的维度且输出不保留该维度
              continue;                                 // 跳过（输出在该维度没有对应）
            arg_new_info.info[arg_i].join_(res_new_info.info[res_i], batch_set); // 将输出的当前维度信息合并到输入对应维度
            res_new_info.info[res_i].join_(arg_new_info.info[arg_i], batch_set); // 将输入的当前维度信息合并回输出对应维度
            res_i += 1;                                 // 只有当输出存在此维度时才递增res_i
          }

          // 检查并更新val_info
          if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) { // 如果输入信息有更新
            changed = true;
            val_info[arg] = arg_new_info;
          }
          if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) { // 如果输出信息有更新
            changed = true;
            val_info[res] = res_new_info;
          }
        } else {
          llvm_unreachable("not supported"); // 有初始值的Reduce暂不支持，直接报错
        }
      } else if (auto reshape_op = dyn_cast<deepgengraph::ReshapeOp>(op)) { // 如果当前操作是ReshapeOp（重塑张量形状）
        auto arg = reshape_op.getOperand();                // 输入张量
        auto res = reshape_op.getResult();                 // 输出张量
        auto arg_shape = cast<RankedTensorType>(arg.getType()).getShape(); // 输入张量形状
        auto res_shape = cast<RankedTensorType>(res.getType()).getShape(); // 输出张量形状

        auto arg_old_info = val_info[arg];                 // 输入张量原有并行信息
        auto res_old_info = val_info[res];                 // 输出张量原有并行信息

        ParaInfo arg_new_info(arg_old_info.getRank());     // 新的输入并行信息对象
        ParaInfo res_new_info(res_old_info.getRank());     // 新的输出并行信息对象

        if (arg_shape.size() == res_shape.size() + 1) {    // 情况1：输入维度数比输出多1（两个维度合并成一个）
          // [..., x, y, ...] -> [..., z, ...]
          int arg_i = 0, res_i = 0;
          // 首先处理前面形状相同的部分
          while (arg_shape[arg_i] == res_shape[res_i]) {
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }
          assert(arg_shape[arg_i] * arg_shape[arg_i + 1] == res_shape[res_i]); // 验证输入的连续两维的乘积等于输出对应维度的大小
          auto b0 = batch_set.alloc_batch();
          auto b1 = batch_set.alloc_batch();
          auto b2 = batch_set.alloc_batch();
          assert(b0 >= 0 && b1 >= 0 && b2 >= 0);
          // 将要合并的两个输入维度分别赋予新的批次ID，将输出合并后的维度赋予另一个批次ID
          arg_new_info.info[arg_i] = ParaType(ParaType::kBatch, b0);
          arg_new_info.info[arg_i + 1] = ParaType(ParaType::kBatch, b1);
          res_new_info.info[res_i] = ParaType(ParaType::kBatch, b2);

          // 与旧信息合并，以保留之前推导的并行属性
          arg_new_info.info[arg_i].join_(arg_old_info.info[arg_i], batch_set);
          arg_new_info.info[arg_i + 1].join_(arg_old_info.info[arg_i + 1], batch_set);
          res_new_info.info[res_i].join_(res_old_info.info[res_i], batch_set);

          // 如果这三个相关维度中有任何一个原本被标记为非并行，则都设为非并行
          if (res_old_info.info[res_i].kind == ParaType::kNonPara ||
              arg_old_info.info[arg_i].kind == ParaType::kNonPara ||
              arg_old_info.info[arg_i + 1].kind == ParaType::kNonPara) {
            arg_new_info.info[arg_i].kind = ParaType::kNonPara;
            arg_new_info.info[arg_i + 1].kind = ParaType::kNonPara;
            res_new_info.info[res_i].kind = ParaType::kNonPara;
          }
          arg_i += 2; // 跳过已处理的两个输入维度
          res_i += 1; // 跳过已处理的一个输出维度
          // 继续处理剩余形状相同的部分
          while (arg_i < (int)arg_shape.size()) {
            assert(arg_shape[arg_i] == res_shape[res_i]);
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }
        } else if (arg_shape.size() + 1 == res_shape.size()) { // 情况2：输入维度数比输出少1（一个维度拆分成两个）
          // [..., x, ...] -> [..., y, z, ...]
          int arg_i = 0, res_i = 0;
          // 处理前面形状相同的部分，找到需要拆分的维度位置
          while (arg_shape[arg_i] == res_shape[res_i]) {
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }
          assert(arg_shape[arg_i] == res_shape[res_i + 1] * res_shape[res_i]); // 验证输出两个维度的乘积等于输入对应维度的大小
          auto b0 = batch_set.alloc_batch();
          auto b1 = batch_set.alloc_batch();
          auto b2 = batch_set.alloc_batch();
          assert(b0 >= 0 && b1 >= 0 && b2 >= 0);
          // 将输入要拆分的维度赋予一个批次ID，输出的两个新维度赋予各自新的批次ID
          arg_new_info.info[arg_i] = ParaType(ParaType::kBatch, b0);
          res_new_info.info[res_i] = ParaType(ParaType::kBatch, b1);
          res_new_info.info[res_i + 1] = ParaType(ParaType::kBatch, b2);

          // 合并旧信息，保持之前推断的属性
          arg_new_info.info[arg_i].join_(arg_old_info.info[arg_i], batch_set);
          res_new_info.info[res_i].join_(res_old_info.info[res_i], batch_set);
          res_new_info.info[res_i + 1].join_(res_old_info.info[res_i + 1], batch_set);

          // 如果拆分涉及的维度中有任何一个是非并行，则三个相关维度都标记为非并行
          if (res_old_info.info[res_i].kind == ParaType::kNonPara ||
              arg_old_info.info[arg_i].kind == ParaType::kNonPara ||
              res_old_info.info[res_i + 1].kind == ParaType::kNonPara) {
            arg_new_info.info[arg_i].kind = ParaType::kNonPara;
            res_new_info.info[res_i].kind = ParaType::kNonPara;
            res_new_info.info[res_i + 1].kind = ParaType::kNonPara;
          }
          arg_i += 1; // 跳过已处理的输入维度
          res_i += 2; // 跳过已处理的两个输出维度
          // 处理剩余形状相同的部分
          while (arg_i < (int)arg_shape.size()) {
            assert(arg_shape[arg_i] == res_shape[res_i]);
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }

          // 以下为已注释掉的旧实现代码片段
          // int arg_i = 0, res_i = 0;
          // while (arg_shape[arg_i] == res_shape[res_i]) {
          //   arg_new_info.info[arg_i].join_(res_old_info.info[res_i], batch_set);
          //   res_new_info.info[res_i].join_(arg_old_info.info[arg_i], batch_set);
          //   arg_i++;
          //   res_i++;
          // }
          // assert(arg_shape[arg_i] ==  res_shape[res_i] * res_shape[res_i + 1]);
          // if (res_new_info.info[res_i].kind == ParaType::kNonPara || res_new_info.info[res_i + 1].kind ==
          // ParaType::kNonPara || arg_new_info.info[arg_i].kind == ParaType::kNonPara) {
          //   arg_new_info.info[arg_i] = ParaType::kNonPara;
          //   res_new_info.info[res_i] = ParaType::kNonPara;
          //   res_new_info.info[res_i + 1] = ParaType::kNonPara;
          // } else {
          //   auto b0 = batch_set.alloc_batch();
          //   auto b1 = batch_set.alloc_batch();
          //   auto b2 = batch_set.alloc_batch();
          //   arg_new_info.info[arg_i].join_(ParaType(ParaType::kBatch, b0), batch_set);
          //   res_new_info.info[res_i].join_(ParaType(ParaType::kBatch, b2), batch_set);
          //   res_new_info.info[res_i + 1].join_(ParaType(ParaType::kBatch, b1), batch_set);
          // }
          // arg_i += 1;
          // res_i += 2;
          // while (arg_i < (int)arg_shape.size()) {
          //   assert(arg_shape[arg_i] == res_shape[res_i]);
          //   arg_new_info.info[arg_i].join_(res_old_info.info[res_i], batch_set);
          //   res_new_info.info[res_i].join_(arg_old_info.info[arg_i], batch_set);
          //   arg_i++;
          //   res_i++;
          // }
        } else {
          op->dump();
          llvm_unreachable("reshape not fully supported"); // 其他形状变换情况暂不支持，直接报错
        }
        if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) { // 如果输入并行信息有更新
          changed = true;
          val_info[arg] = arg_new_info;
        }
        if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) { // 如果输出并行信息有更新
          changed = true;
          val_info[res] = res_new_info;
        }
      } else if (isa<deepgengraph::MaskOp>(op)) { // 如果当前操作是MaskOp（掩码操作）
        // pass
        return WalkResult::skip();        // 跳过Mask操作内部的遍历（不深入分析其内部区域）
      } else if (op->getNumOperands() == 0 || isa<deepgengraph::ReturnOp>(op)) { // 如果操作没有操作数或是Return操作
        // pass
        return WalkResult::skip();        // 无需处理并行信息，跳过
      } else {
        op->dump();                                           // 打印未支持操作的调试信息
        llvm_unreachable("not supported");                    // 对于未知/不支持的操作，报错退出
      }
      return WalkResult::advance(); // 继续处理下一个操作
    });

    if (verbose) { // 如果需要输出调试信息
      dbg(iter);   // 打印当前迭代次数
      for (auto arg : kernel_op.getArguments()) {
        arg.dump(); // 打印参数值
        std::string str;
        llvm::raw_string_ostream os(str);
        val_info[arg].print(os, batch_set);           // 将参数的并行信息输出到字符串
        llvm::errs() << "\t" << str << "\n";          // 打印参数的并行信息
      }
      kernel_op.getCallableRegion()->front().walk([&](Operation *op) {
        op->dump();                                   // 打印操作
        for (auto res : op->getResults()) {
          llvm::errs() << "\t";
          std::string str;
          llvm::raw_string_ostream os(str);
          val_info[res].print(os, batch_set);         // 打印结果的并行信息
          llvm::errs() << "\t" << str << "\n";
        }
      });
    }
  }
  if (verbose) {
    dbg(iter); // 打印最终迭代次数
  }
}

/*
void ParallelismAnalysis::visitOperation(
  Operation *op,
  ArrayRef<const dataflow::Lattice<ParallelismInfo> *> operands,
  ArrayRef<dataflow::Lattice<ParallelismInfo> *> results) {
  dbg("into visit Operation");
  llvm::errs() << op->getName() << "\n";
  llvm::errs() << "arg_num: " << operands.size() << "\n";
  for (auto* arg: operands) {
    auto v = arg->getValue();
    std::string arg_type;
    llvm::raw_string_ostream os(arg_type);
    v.print(os);
    llvm::errs() << "\t" << arg_type << "\n";
  }
  llvm::errs() << "res_num: " << results.size() << "\n";
  for (auto* res: results) {
    auto v = res->getValue();
    std::string res_type;
    llvm::raw_string_ostream os(res_type);
    v.print(os);
    llvm::errs() << "\t" << res_type << "\n";
  }

  if (auto permute_op = dyn_cast<deepgengraph::PermuteOp>(op)) {
    assert(operands.size() == 1);
    assert(results.size() == 1);
    assert(operands[0]->getValue().isInitialized());

    auto dims = permute_op.getDims();
    ParallelismInfo new_para(operands[0]->getValue().getRank());
    for (auto en: llvm::enumerate(dims)) {
      new_para.setParallelismType(en.index(), operands[0]->getValue().getParallelismType(en.value()));
    }
    propagateIfChanged(results[0], results[0]->join(new_para));
  } else if (auto dot_op = dyn_cast<deepgengraph::DotOp>(op)) {
    assert(operands.size() == 2);
    assert(results.size() == 1);
    auto res_rank = cast<RankedTensorType>(dot_op.getResult()).getRank();

    ParallelismInfo common(res_rank, NoReuse);
    common.setParallelismType()

    // new_para.setParallelismType()

  } else if (operands.size() == 0) {
    for (auto [res_lattice, res]: llvm::zip(results, op->getResults())) {
      if (!res_lattice->getValue().isInitialized()) {
        propagateIfChanged(res_lattice, res_lattice->join(ParallelismInfo::getPessimisticValueState(res)));
      }
    }
  }
}
*/

} // namespace mlir
