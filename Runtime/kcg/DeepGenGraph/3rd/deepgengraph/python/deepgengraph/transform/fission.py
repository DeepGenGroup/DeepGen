from deepgengraph.deepgengraph_ffi import passes, ir

from .utils import get_pass_manager


def fission(op, context=None):
  top_pm, pm = get_pass_manager(op, context)
  passes.add_cse(pm) # 标准化pass，公共子表达式消除
  passes.add_deepgengraph_simplify(pm) # 图化简pass（做的很激进）
  passes.add_deepgengraph_lower_complex_reduce(pm) # 展开 softmax、normalize 算子成基础op
  passes.add_cse(pm) # 标准化pass，公共子表达式消除
  top_pm.run(op)
