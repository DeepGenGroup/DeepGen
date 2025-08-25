from deepgengraph.deepgengraph_ffi import passes, ir
from .utils import get_pass_manager


def simplify(op, context=None):
  top_pm, pm = get_pass_manager(op, context)
  passes.add_deepgengraph_simplify(pm)
  top_pm.run(op)


def annotate_parallelism(op, context=None):
  top_pm, pm = get_pass_manager(op, context)
  passes.add_deepgengraph_annotate_parallelism(pm)
  top_pm.run(op)


def optimize(op, context=None):

  print("opopopopopopop_naive")
  print(op)
  top_pm, pm = get_pass_manager(op, context)

  def simplify(_pm):
    passes.add_deepgengraph_simplify(pm)
    passes.add_cse(pm)

  simplify(pm)
  # 把 KernelOp 内部所有浮点类型张量的 元素类型 临时统一成 f64，
  # 并把 原来的元素类型 记录到属性 deepgengraph.erased_type
  # 这样后续一系列 pattern-rewriting/算子融合 Pass 就只需要处理一种浮点类型，
  # 而真正生成代码前再由后续的 RecoverTypeInKernelPass 把原类型恢复回来。
  passes.add_deepgengraph_erase_type_in_kernel(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_erase_type")
  # print(op)
  simplify(pm)
  # 将所有的 ExpOp（自然指数）替换成以 2 为底的指数 Exp2Op，前者等价于二底指数与乘常量 log₂(e) 的组合。
  # %r = "deepgengraph.ExpOp"(%x) : (tensor<…xf32>) -> tensor<…xf32>
  # ----- 》
  # %c = arith.constant dense<log2(e)> : tensor<1xf32>
  # %m = "arith.MulOp"(%x, %c) : (tensor<…xf32>, tensor<1xf32>) -> tensor<…xf32>
  # %r = "deepgengraph.Exp2Op"(%m) : tensor<…xf32> -> tensor<…xf32>
  # 因为处理器一般对2^x这种比较友好，计算更快
  passes.add_deepgengraph_replace_exp_and_log(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_replace_exp_and_log")
  # print(op)
  simplify(pm)
  # 在不改变数值语义的前提下，利用一系列代数等价变换（distribution、commutation、常量折叠等），
  # 自动化地重写整个 kernel（或并行循环体）里的算子次序和结构，以期减少张量的内存访问量，从而提升性能。
  # ！！！！核心pass
  # 所有变换严格等价
  passes.add_deepgengraph_equivalent_transform(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_equivalent_transform")
  # print(op)
  simplify(pm)
  # 把 Deepgengraph Dialect 中的三角矩阵截断操作 TriluOp（仅支持上三角，assert: is_upper == true）
  # 转换成更通用的 MaskOp + MaskYieldOp 组合，从而利用统一的 Mask 机制来表示三角截断。
  # 统一三角截断为 Mask 语义，有利于后续通用优化（例如多 Mask 合并、下沉到 Kernel、生成 GPU/SIMD 代码时统一处理等）。
  passes.add_deepgengraph_to_mask(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_to_mask")
  # print(op)

  simplify(pm)
  # 将一个已由分析（ParallelismAnalysis）标记了哪些维度可并行 的 KernelOp，重写成一个嵌套在 ParallelForOp 中的形式，从而在这些批量（batch）或复用（reuse）维度上并行执行。
  # 以在指定的「批次维度」（batch）或「复用维度」（reuse）上并行化计算。
  passes.add_deepgengraph_parallelize(pm, True)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_parallelize")
  # print(op)

  simplify(pm)
  # 在进行一次reorder
  passes.add_deepgengraph_equivalent_transform(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_equivalent_transform")
  # print(op)
  simplify(pm)
  # 本 Pass 首先对大规模 DotOp 做分块（Tile），再在这个块级循环上智能地前融、后融、同层融，
  # 乃至多次使用合并，最大程度地将相关算子集中到同一层循环里。
  passes.add_deepgengraph_tiling(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_deepgengraph_tiling")
  # print(op)
  simplify(pm)
  # 没看懂
  passes.add_deepgengraph_dynamic_for(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_dynamic_for")
  # print(op)
  simplify(pm)
  # 前面吧 KernelOp 的输入从 f16、bf16 都改成 f64，
  # 并在参数属性里记录了原来的 deepgengraph.erased_type = f16 / bf16，这个pass根据这些进行了还原包括调用使用递归地进行还原
  passes.add_deepgengraph_recover_type_in_kernel(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_recover_type_in_kernel")
  # print(op)
  simplify(pm)
  # 把 Deepgengraph Dialect 中的并行循环和块操作，转换为 Triton Dialect 
  # (triton::DeviceKernelOp、BlockPointerOfOp、BlockLoadOp、BlockStoreOp) 
  # 以及标准 SCF ForOp，从而把高层并行抽象编译到底层 GPU kernel。
  passes.add_deepgengraph_convert_deepgengraph_to_deepgengraphtriton(pm)
  # top_pm.run(op)
  # top_pm, pm = get_pass_manager(op, context)
  # print("opopopopopopop_recover_deepgengraph_to_deepgengraphtriton")
  # print(op)
  simplify(pm)
  # 去掉那些在 BlockPointer 上“多余”的维度（全是 1），简化后端生成的 block 操作。
  passes.add_deepgengraphtriton_squeeze_block(pm)
  simplify(pm)

  top_pm.run(op)
  # print("opopopopopopop_squeeze_block")
  # print(op)


def module_to_py(module):
  py_str = passes.translate_module_to_py(module, True)
  return py_str


def kernel_to_py(kernel, add_import=True, add_benchmark=True):
  py_str = passes.translate_kernel_to_py(kernel, add_import, add_benchmark)
  return py_str
