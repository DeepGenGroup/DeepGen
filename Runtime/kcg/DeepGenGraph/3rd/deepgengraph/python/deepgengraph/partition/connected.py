from tqdm import tqdm
from itertools import combinations
import numpy as np

from .config import PartitionConfig

from deepgengraph.deepgengraph_ffi import ir

from deepgengraph.partition.kernel import Kernel
from deepgengraph.transform.common import annotate_parallelism
from deepgengraph.utils import get_sm_num


class Connected(PartitionConfig):

  def __init__(self, module: ir.module, func_name: str):
    self.opid_name_dict = {}
    partitions, output_ops = self._find_all_connected_subset(module, func_name)
    super().__init__(
      module=module,
      func_name=func_name,
      partitions=partitions,
      output_ops=output_ops,
      backends="triton",
    )

  # 遍历给定函数的 IR 计算图，找出图中所有彼此相连的算子子集
  def _find_all_connected_subset(self, module: ir.module, func_name: str):
    op2id = {}
    id2op = {}
    fn = module.get_function(func_name)

    def id_update(op):
      if op.get_name() != "func.return":
        i = len(op2id)
        op2id[op] = i
        id2op[i] = op

    fn.get_callable_region().front().walk(id_update)
    for op, idx in op2id.items():
      print(f"----- idx = {idx} {op.get_name()}")
      self.opid_name_dict[idx] = op.get_name()

    # 构建节点之间的邻接关系，获取邻居节点 ID 的函数。它根据 IR 图中的数据依赖关系，找出指定算子节点直接相连的所有邻接算子 ID。
    # 凡是该算子的输出流向的下游算子，以及提供输入给该算子的上游算子，都被视为邻居节点。
    def get_neighbor_ids(op_or_id):
      neighbor_ids = set()
      if isinstance(op_or_id, ir.operation):
        op = op_or_id
      else:
        assert isinstance(op_or_id, int)
        op = id2op[op_or_id]

      for i in range(op.get_num_operands()):
        v = op.get_operand(i)
        if v.get_defining_op() is not None:
          neighbor_ids.add(op2id[v.get_defining_op()])

      for i in range(op.get_num_results()):
        v = op.get_result(i)
        for user in v.get_users():
          if user in op2id:
            neighbor_ids.add(op2id[user])
      return neighbor_ids

    connected_partitions = set()

    # 从一个起始算子节点出发，dfs 会递归地沿邻接关系访问其所有可到达的算子，将这些节点都收集到同一子图分区中。
    def dfs(cur_op_ids, results, neighbors, excluded):
      results.add(frozenset(cur_op_ids))
      # print(f"{cur_op_ids=}")

      for op_id in neighbors - excluded:
        new_op_ids = cur_op_ids | {op_id}
        excluded = excluded | {op_id}
        new_neighbors = neighbors | get_neighbor_ids(op_id)
        dfs(new_op_ids, results, new_neighbors, excluded)

    excluded = set()
    for op, idx in op2id.items():
      excluded.add(idx)
      neighbor_ids = get_neighbor_ids(op)
      # print(f"{idx=} {neighbor_ids=}")
      dfs({idx}, connected_partitions, neighbor_ids, excluded)

    print(f"{len(connected_partitions)=}")
    print(connected_partitions)
    print("++++++++++++++++1")

    def pruning_non_op(results):
      non_op_ids = set()
      for op, idx in op2id.items():
        if op.get_name() == "deepgengraph.reshape":
          non_op_ids.add(idx)

      new_results = set()
      for kernel in results:
        # remove those kernel containing only non-op
        if len(kernel - non_op_ids) == 0:
          continue
        new_results.add(kernel)
      return new_results


    # 筛除不良分区。该函数检查每个候选子图中的所有算子类型，如果发现分区内存在 Triton 不支持的算子或其他不兼容模式，
    # 则将该分区标记为“bad”并予以移除。
    def pruning_bad_partition(results, provide_output=False):

      def partition_has_only_reshape(partition):
        for op_id in partition:
          op = id2op[op_id]
          if op.get_name() != "deepgengraph.reshape":
            return False
        return True

      # FIXME: we cannot generate them due to triton backend
      def partition_has_unsupported_op(partition):
        # FIXME: a global var/method for this?
        unsupported_ops = set([
          'deepgengraph.avg_pool',
        ])
        for op_id in partition:
          op = id2op[op_id]
          if op.get_name() in unsupported_ops:
            return True
        return False

      def inputs_has_constant_like(inputs):
        for v in inputs:
          op = v.get_defining_op()
          if op is not None and (op.get_name() == "deepgengraph.trilu" or op.get_name() == "arith.constant"):
            return True
        return False

      def inputs_has_reshape_user(inputs):
        for v in inputs:
          for user in v.get_users():
            if user.get_name() == "deepgengraph.reshape":
              return True
        return False

      def inputs_has_permute_def(inputs):
        for v in inputs:
          op = v.get_defining_op()
          if op is not None and op.get_name() == "deepgengraph.permute":
            return True
        return False

      def outputs_has_reshape_def(outputs):
        for v in outputs:
          op = v.get_defining_op()
          if op is not None and op.get_name() == "deepgengraph.reshape":
            return True
        return False

      def outputs_has_permute_user(outputs):
        for v in outputs:
          for user in v.get_users():
            if user.get_name() == "deepgengraph.permute":
              return True
        return False

      def outputs_has_constant_like_def(outputs):
        for v in outputs:
          op = v.get_defining_op()
          if op is not None and (op.get_name() == "deepgengraph.trilu" or op.get_name() == "arith.constant"):
            return True
        return False

      def outputs_has_no_compute_defs(outputs, ops):
        compute_op = set(op.get_name() for op in ops)
        compute_op.discard("deepgengraph.permute")
        compute_op.discard("deepgengraph.reshape")

        def has_no_compute_defs(cur_v):
          while True:
            op = cur_v.get_defining_op()
            if op is None:
              # argument of fn
              break
            if op.get_name() in compute_op:
              return False
            if op not in ops:
              # argument of this kernel
              break
            assert op.get_num_operands() == 1
            cur_v = op.get_operand(0)
          return True

        for v in outputs:
          if has_no_compute_defs(v):
            return True
        return False

      new_results = set()
      if not provide_output:
        for partition in tqdm(results, desc=f"pruning bad partition({provide_output=})"):
          op_ids = sorted(partition)
          mlir_ops = [id2op[i] for i in op_ids]
          mlir_inputs = Kernel.get_all_inputs(module, func_name, mlir_ops)
          mlir_output_ops = Kernel.get_emptyuse_ops(module, func_name, mlir_ops)
          mlir_outputs = [op.get_result(0) for op in mlir_output_ops]

          has_only_reshape = partition_has_only_reshape(op_ids)
          has_unsupported_ops = partition_has_unsupported_op(op_ids)
          if has_only_reshape or has_unsupported_ops:
            continue
          has_constant_input = inputs_has_constant_like(mlir_inputs)
          has_reshaped_user_of_input = inputs_has_reshape_user(mlir_inputs)
          has_reshaped_def_of_output = outputs_has_reshape_def(mlir_outputs)
          has_constant_def_of_output = outputs_has_constant_like_def(mlir_outputs)

          if has_constant_input or has_reshaped_user_of_input or has_reshaped_def_of_output or has_constant_def_of_output:
            continue

          # TODO: permute can be fused into kernel. Alwayse right?
          has_permute_user_of_output = outputs_has_permute_user(mlir_outputs)
          has_permute_def_of_input = inputs_has_permute_def(mlir_inputs)
          if has_permute_user_of_output or has_permute_def_of_input:
            continue

          new_results.add(partition)
      else:
        for partition, output_op_ids in tqdm(results, desc=f"pruning bad partition({provide_output=})"):
          op_ids = sorted(partition)
          mlir_ops = [id2op[i] for i in op_ids]
          mlir_inputs = Kernel.get_all_inputs(module, func_name, mlir_ops)
          mlir_outputs = [id2op[i].get_result(0) for i in output_op_ids]

          has_only_reshape = partition_has_only_reshape(op_ids)
          has_unsupported_ops = partition_has_unsupported_op(op_ids)
          if has_only_reshape or has_unsupported_ops:
            continue
          has_constant_input = inputs_has_constant_like(mlir_inputs)
          has_reshaped_user_of_input = inputs_has_reshape_user(mlir_inputs)
          has_reshaped_def_of_output = outputs_has_reshape_def(mlir_outputs)
          has_constant_def_of_output = outputs_has_constant_like_def(mlir_outputs)

          if has_constant_input or has_reshaped_user_of_input or has_reshaped_def_of_output or has_constant_def_of_output:
            continue

          # TODO: permute can be fused into kernel. Alwayse right?
          has_permute_user_of_output = outputs_has_permute_user(mlir_outputs)
          has_permute_def_of_input = inputs_has_permute_def(mlir_inputs)
          if has_permute_user_of_output or has_permute_def_of_input:
            continue

          has_no_compute_defs_of_output = outputs_has_no_compute_defs(mlir_outputs, mlir_ops)
          if has_no_compute_defs_of_output:
            continue

          new_results.add((partition, output_op_ids))

      return new_results

    connected_partitions = pruning_bad_partition(connected_partitions)
    print(f"after pruning bad partition: {len(connected_partitions)=}", flush=True)
    print(connected_partitions)
    print("++++++++++++++++2")
    

    # 筛除并行度差的分区。此步骤根据子图的规模和并行计算工作量来过滤掉在 GPU 上加速效果不佳的子图。
    # 如果一个分区涉及的张量规模很小、数据元素数量有限，导致并行计算量不足，那么认为该分区的 GPU 并行效率不高，将其剔除。
    def pruning_bad_para_partition(results):
      sm_num = get_sm_num()

      new_results = set()
      for pid, partition in enumerate(tqdm(results, desc="pruning bad para partition")):
        op_ids = sorted(partition)
        mlir_ops = [id2op[i] for i in op_ids]
        # output choices not affect parallelism
        mlir_outputs = [id2op[i].get_result(0) for i in op_ids]
        kernel_name = f"{func_name}_p{pid}"

        kernel = Kernel(
          module=module,
          func_name=func_name,
          kernel_name=kernel_name,
          ops=mlir_ops,
          outputs=mlir_outputs,
        )

        annotate_parallelism(kernel.kernel, module.context)
        para_num = kernel.get_parallelism()


        # para_num >= sm_num：意味着你为这个 kernel 划分出的并行单元不低于硬件上能同时跑的 SM 数，理论上能让每个 SM 都跑上至少一个 block，从而“饱和”全卡。
        if para_num >= sm_num:
          new_results.add(partition)

        kernel._erase()
      return new_results

    connected_partitions = pruning_bad_para_partition(connected_partitions)
    print(f"after pruning bad para partition: {len(connected_partitions)=}", flush=True)
    print(connected_partitions)
    print("++++++++++++++++3")


    # 扩展并筛除算术强度低的分区。该函数尝试将相邻的算子合并进当前子图，以扩大分区的范围，从而提升其算术强度。
    # 许多逐元素算子各自的算术强度不高，但若融合在一个 GPU 内核中执行，可以减少重复的内存读写，提高总体计算密度。
    def expand_and_pruning_bad_ai_partition(results, tall_and_thin_tensors):

      def get_tensor_size(v):
        ty = v.get_type().to_ranked_tensor_ty()
        if ty is not None:
          shape = ty.get_shape()
          size = 1
          for dim in shape:
            size *= dim
          return size
        else:
          return 0

      tensor_size = {}
      op_out_size = {}
      for op_id, op in id2op.items():
        op_out_size[op_id] = 0
        for i in range(op.get_num_results()):
          res = op.get_result(i)
          op_out_size[op_id] += get_tensor_size(res)
      for i in range(fn.get_callable_region().front().get_num_arguments()):
        v = fn.get_callable_region().front().get_argument(i)
        tensor_size[v] = get_tensor_size(v)
      for op_id, op in id2op.items():
        for i in range(op.get_num_results()):
          v = op.get_result(i)
          tensor_size[v] = get_tensor_size(v)

      new_results = set()
      for pid, partition in enumerate(tqdm(results, desc="expand and pruning bad ai partition")):
        op_ids = sorted(partition)
        mlir_ops = [id2op[i] for i in op_ids]
        mlir_inputs = Kernel.get_all_inputs(module, func_name, mlir_ops)
        mem_in_size = sum(tensor_size[v] for v in mlir_inputs)
        flops = sum(op.get_flops() for op in mlir_ops)

        emptyuse_ops = Kernel.get_emptyuse_ops(module, func_name, mlir_ops)
        emptyuse_op_ids = frozenset(op2id[op] for op in emptyuse_ops)
        for r in range(1, len(op_ids) + 1):
          for output_op_ids in combinations(op_ids, r):
            final_output_op_ids = frozenset(output_op_ids) | emptyuse_op_ids
            mem_size = sum(op_out_size[i] for i in final_output_op_ids) + mem_in_size
            ai = flops / mem_size
            # print(f"{frozenset(output_op_ids)} {flops=} {mem_size=} {ai=}")
            # if ai >= 4000:
            if ai < 200:
              continue
            mlir_inputs_set = set(mlir_inputs)
            mlir_outputs_set = set(id2op[i].get_result(0) for i in final_output_op_ids)
            if tall_and_thin_tensors.issuperset(mlir_inputs_set) and tall_and_thin_tensors.issuperset(mlir_outputs_set):
              new_results.add((partition, final_output_op_ids))
      return new_results


    # 识别“瘦高型”张量模式的分区。该函数检查候选分区中张量的形状特征，
    # 如果存在某些张量呈“高而窄”（一维特别长而另一维特别短）的瘦高型形状，则标记出这一情况。
    def identify_tall_and_thin_tensor():
      val_set = set()
      val_metrics = {}

      def get_metric(v):
        ty = v.get_type().to_ranked_tensor_ty()
        if ty is not None:
          shape = ty.get_shape()
          shape = np.array(shape)
          return float(np.prod(shape))
        else:
          # const?
          return float('-inf')

      for i in range(fn.get_callable_region().front().get_num_arguments()):
        arg = fn.get_callable_region().front().get_argument(i)
        val_metrics[arg] = get_metric(arg)
      for op in id2op.values():
        for i in range(op.get_num_results()):
          res = op.get_result(i)
          val_metrics[res] = get_metric(res)

      # normalize
      max_metric = max(val_metrics.values())
      val_metrics = {k: max_metric / v for k, v in val_metrics.items()}
      print("max_metric: ", max_metric)

      for val, metric in val_metrics.items():
        # FIXME: better metric
        def_op = val.get_defining_op()
        # print(f"[debug] {val} {def_op} {op2id.get(def_op, None)} {metric=}")
        if metric >= 128:
          val_set.add(val)
      # input and output should be kept
      for i in range(fn.get_callable_region().front().get_num_arguments()):
        val_set.add(fn.get_callable_region().front().get_argument(i))
      return_op = fn.get_terminator()
      assert return_op.get_name() == "func.return"
      for i in range(return_op.get_num_operands()):
        val_set.add(return_op.get_operand(i))
      # show
      # for val in val_set:
      #   def_op = val.get_defining_op()
      #   print(f"{val} {def_op} {op2id.get(def_op, None)}")
      return val_set

    val_set = identify_tall_and_thin_tensor()
    print(f"tall_and_thin tensors: {len(val_set)}", flush=True)
    connected_partitions = expand_and_pruning_bad_ai_partition(connected_partitions, val_set)
    print(f"after expand and pruning bad ai partition: {len(connected_partitions)=}", flush=True)
    print(connected_partitions)
    print("++++++++++++++++4")
    connected_partitions = pruning_bad_partition(connected_partitions, provide_output=True)
    print(f"after pruning bad partition(provide_output=True): {len(connected_partitions)=}", flush=True)
    print(connected_partitions)
    print("++++++++++++++++5")

    # for pid, (partition, output_op_ids) in enumerate(connected_partitions):
    #   op_ids = sorted(partition)
    #   mlir_ops = [id2op[i] for i in op_ids]
    #   mlir_outputs = [id2op[i].get_result(0) for i in output_op_ids]

    #   kernel_name = f"{func_name}_p{pid}"
    #   kernel = Kernel(
    #     module=module,
    #     func_name=func_name,
    #     kernel_name=kernel_name,
    #     ops=mlir_ops,
    #     outputs=mlir_outputs,
    #   )
    #   kernel.dump()

    partitions = [sorted(e[0]) for e in connected_partitions]
    output_ops = [sorted(e[1]) for e in connected_partitions]

    return partitions, output_ops
