import torch
import numpy as np
import onnx
from deepgengraph_exp.utils import torch_module_to_onnx

def compile(model, input_names, inputs, output_names, system):
  if system == 'torch':
    f = model
  elif system == 'dynamo':
    torch._dynamo.reset()
    f = torch.compile(model) 
  elif system == 'tensorrt':
    from deepgengraph_exp.trtllm_utils import trt_build_engine_from_onnx, trt_build_independent_runtime
    onnx_model = torch_module_to_onnx(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
    )
    engine = trt_build_engine_from_onnx(onnx_model)
    f = trt_build_independent_runtime(engine)
  elif system == 'xla':
    import torch_xla.core.xla_model as xm
    def _f(*args):
      o = model(*args)
      xm.mark_step()
      xm.wait_device_ops()
      return o
    f = _f
  elif system == 'tvm':
    from deepgengraph_exp.tvm_utils import meta_scheduler_tune, tvm_build_independent_runtime
    lib = meta_scheduler_tune(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
      # num_trials_per_iter=64,
      # max_trials_per_task=1000,
      num_trials_per_iter=4,
      max_trials_per_task=128,
      exported_lib_path=None,
    )
    f = tvm_build_independent_runtime(lib, input_names, output_names)
  elif system == 'our':
    from deepgengraph.translate import deepgengraph_from_onnx
    from deepgengraph.transform import fission
    from deepgengraph.transform.common import simplify
    from deepgengraph.partition.connected import Connected
    onnx_model = torch_module_to_onnx(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
      simplify=False,
    )
    # print(onnx.helper.printable_graph(onnx_model.graph), flush=True)
    func_name = model.__class__.__name__

    import time
    tik = time.time()
    module = deepgengraph_from_onnx(onnx_model, func_name)
    print("--------step: aft deepgengraph_from_onnx----------")
    # module.dump()
    fission(module)
    # 3rd/deepgengraph/python/deepgengraph/transform/fission.py
    ### SimplifyPass，图化简pass，3rd/deepgengraph/lib/Dialect/Deepgengraph/Transforms/Simplify.cpp
    # 去除冗余：冗余的 permute、reshape、convert 一律删掉；
    # 语义映射：把“加和再比较”模式改成更快的“any”操作；
    # 常量折叠：把分步的数值操作合并并预先计算常量；
    # 掩码优化：在掩码后乘全非零常数时省去一次乘法。
    ### LowerComplexReducePass，3rd/deepgengraph/lib/Dialect/Deepgengraph/Transforms/LowerComplexReduce.cpp
    # 展开 softmax、normalize 算子成基础op
    print("--------step: aft fission----------")
    # module.dump()
    simplify(module)
    # 3rd/deepgengraph/python/deepgengraph/transform/common.py - def simplify()
    # 在执行一遍SimplifyPass
    print("--------step: aft simplify----------")
    # module.dump()
    partition = Connected(module, func_name)
    # 3rd/deepgengraph/python/deepgengraph/partition/connected.py -- __init__()
    # _find_all_connected_subset 返回两个重要结果：partitions 和 output_ops。其中，partitions 是一个列表，每个元素是一组算子节点组成的子图（分区）。
    # 这些算子在原始计算图中通过数据依赖彼此连接，构成一个可独立执行的连通子图。output_ops 则表示各分区的输出算子集合——即每个分区中哪些算子的输出需提供给分区外部使用
    # （例如作为整个模型的输出，或作为其他分区的输入）。通过 output_ops 可以明确每个分区对外暴露的接口，从而在执行该分区生成结果后，能够将这些结果正确地传递回主计算图中后续的计算。
    # 其实相当于在这里将各种合法的kernel组合都输出出来（剔除包含不支持算子的“坏”分区；然后剔除并行度过低（规模太小，GPU加速收益不明显）以及计算密集度过低的分区），
    # 删除并行度过低的使用了后续的并行性分析pass AnnotateParallelismPass，通过比较
    # 比如 a->b->c 起点 a 那一次 DFS 会把 {a}、{a, b}、{a, b, c} 这 3 个不同规模的连通子集都塞进 results

    print("--------step: aft Connected----------")
    # print(partition)
    # print(partition.module)
    # partition.module.dump()
    partition.optimize()
    # 3rd/deepgengraph/python/deepgengraph/partition/config.py -- optimize(）
    # 在这个pass进行优化以及Codegen
    print("--------step: aft optimize----------")
    print(partition.module)
    analyze_ir_and_gen_code(str(partition.module))
    # partition.module.dump()
    tok = time.time()
    tuning_s = tok - tik
    print(f"tuning time: {tuning_s} sec", flush=True)
    # perf = partition.profile()
    # py_str = partition.codegen(perf)

    our = {}
    import tempfile
    import importlib
    import sys
    # with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
    #   f.write(py_str)
    #   path = f.name
    # print(f"write code to {path}", flush=True)
    # spec = importlib.util.spec_from_file_location('our', path)
    # pymod = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(pymod)
    f=None

    # f = getattr(pymod, func_name)
  elif system == 'flashinfer':
    import flashinfer
    model_name = model.__class__.__name__
    if model_name == 'Attn':
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        q_len = q.shape[1]
        head_num = q.shape[2]
        head_dim = q.shape[3]
        kv_len = k.shape[1]
        kv_head_num = k.shape[2]
        batch_size = q.shape[0]
        assert batch_size == 1
 
        q = q.view(q_len, head_num, head_dim)
        k = k.view(kv_len, kv_head_num, head_dim)
        v = v.view(kv_len, kv_head_num, head_dim)
        out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        return out.view(batch_size, q_len, head_num, head_dim)
    else:
      assert model_name == 'Gemma2'
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        q_len = q.shape[1]
        head_num = q.shape[2]
        head_dim = q.shape[3]
        kv_len = k.shape[1]
        kv_head_num = k.shape[2]
        batch_size = q.shape[0]
        assert batch_size == 1
 
        q = q.view(q_len, head_num, head_dim)
        k = k.view(kv_len, kv_head_num, head_dim)
        v = v.view(kv_len, kv_head_num, head_dim)
        out = flashinfer.single_prefill_with_kv_cache(q, k, v, logits_soft_cap=50.0, causal=True)
        return out.view(batch_size, q_len, head_num, head_dim)
    f = _f
  elif system == 'flashattn':
    from flash_attn.flash_attn_interface import flash_attn_func
    model_name = model.__class__.__name__
    if model_name == 'Attn':
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        out = flash_attn_func(q, k, v, causal=True)
        return out
    else:
      assert model_name == 'Gemma2'
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        out = flash_attn_func(q, k, v, softcap=50.0, causal=True)
        return out
    f = _f
  else:
    raise NotImplementedError(f"system {system} not implemented")
  
  return f

from typing import Dict,List

class VarInfo :
  index = 0
  @staticmethod
  def get_name() :
    VarInfo.index += 1
    return f"var_{VarInfo.index}"
  
  def __init__(self, shape, val = 0):
    self.m_tensorShape = shape
    self.m_value = val
    self.m_name = VarInfo.get_name()
    

g_varMap : Dict[str,VarInfo] = {}

def parse_func_def(ir : str, varMap : Dict[str,VarInfo]) :
  #   func.func @Attn(%arg0: tensor<1x4096x32x128xf16> loc(unknown), %arg1: tensor<1x4096x32x128xf16> loc(unknown), %arg2: tensor<1x4096x32x128xf16> loc(unknown)) -> tensor<1x4096x32x128xf16> {
  global g_varMap
  funcName = ir[ir.find('@')+1 : ir.find('(')]
  argst = ir.find('(')
  arget = ir.find('->')
  argstr = ir[ir.find('(')+1 : ir.find('->')]
  args = argstr.split(',')
  argnames = []
  code = f"def {funcName}("
  for arg in args :
    arg = arg.strip()
    sep = arg.find(':')
    argName = arg[0:sep]
    argType = arg[arg.find('<')+1 : arg.find('>')]
    argshape = [int(i) for i in argType.split('x')[0:-1]] 
    info = VarInfo(argshape)
    g_varMap[argName] = info  # {%0 : {varinfo{name = "var_1", [5,5], val = 0}}}
    code += (g_varMap[argName].m_name + ',')
  code = code[0:-1]
  code += ") : "
  return code
  
    
def get_varname(argnameInIR : str) -> str :
  global g_varMap 
  if argnameInIR not in g_varMap.keys() :
    g_varMap[argnameInIR] = VarInfo([])
  return g_varMap[argnameInIR].m_name


def parse_op(ir : str, varMap : Dict[str,VarInfo]) -> str :
  [op_def, op_sig] = ir.split(':')
  op_def = op_def.strip()
  op_sig = op_sig.strip()
  
  def __split_opname_and_after(ir : str, opname : str) -> List[str] : # [lval, opname, after-words, signature]
    st = ir.find(opname)
    lval = ir[0:ir.find('=')].strip()
    after = ir[st+len(opname) : ].strip()
    [afterwords, sig] = after.split(':')
    afterwords = afterwords.replace(' ','')
    return [lval,afterwords,sig]
  
  def __parse_constant(ir : str) :
  #     %cst = arith.constant dense<1.131250e+01> : tensor<1xf16> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"arith.constant")
    isTensor = False
    if afters.find('dense') != -1 :
      isTensor = True; v = afters[afters.find('dense<') + len('dense<') : -1]
    else:
      v = afters
    code = f"{get_varname(lval)} = {v}"
    return code
  
  def __parse_deepgengraph_trilu(ir : str) :
    #     %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16 loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.trilu")
    shapestr = afters[afters.find('shape=[') + len('shape=[') : afters.find('],val')]
    valstr = afters[afters.find(',val=') + len(',val=') : ]
    diagonal = afters[afters.find('diagonal=') + len('diagonal=') : afters.find(',is_upper')]
    if valstr == '0xFC00' :
      valstr = "'-inf'"
    code = f"{get_varname(lval)} = torch.tril(torch.full(({shapestr}), float({valstr})) , diagonal = {diagonal}).to(var_1.device)"
    return code
  
  def __parse_deepgengraph_permute(ir : str) :
    #     %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.permute")
    src = afters[0:afters.find(',')]
    dims = afters[afters.find('dims=[') + len('dims=[') : afters.find(']')]
    code = f"{get_varname(lval)} = {get_varname(src)}.permute({dims})"
    return code
  def __parse_deepgengraph_dot(ir : str) :
    #     %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.dot")
    a = afters[0:afters.find(',')]
    b = afters[afters.find(',')+1 : ]
    code = f"{get_varname(lval)} = torch.matmul({get_varname(a)},{get_varname(b)})"
    return code

  def __parse_deepgengraph_div(ir : str) :
    #     %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.div")
    a = afters[0:afters.find(',')]
    b = afters[afters.find(',')+1 : ]
    code = f"{get_varname(lval)} = {get_varname(a)} / {get_varname(b)}"
    return code

  def __parse_deepgengraph_add(ir : str) :
    #     %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.add")
    a = afters[0:afters.find(',')]
    b = afters[afters.find(',')+1 : ]
    code = f"{get_varname(lval)} = {get_varname(a)} + {get_varname(b)}"
    return code
  
  def __parse_deepgengraph_convert(ir : str) :
    #     %7 = deepgengraph.convert %6, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.convert")
    rhs = afters[0:afters.find(',')]
    get_varname(lval)
    g_varMap[lval].m_name = g_varMap[rhs].m_name
    return ""
  
  def __parse_deepgengraph_exp(ir : str) :
    #     %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.exp")
    i = afters
    code = f"{get_varname(lval)} = torch.exp({get_varname(i)})"
    return code
  
  def __parse_deepgengraph_reduce(ir:str) :
    #     %9 = deepgengraph.reduce(%8), dim = -1, op =  ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32> loc(#loc)
    [lval, afters, sig] = __split_opname_and_after(ir,"deepgengraph.reduce")
    src = afters[1:afters.find(')')]
    dim = afters[afters.find('dim=')+len('dim=') : afters.find(',op=')]
    op = afters[afters.find('op=') + len("op=") : afters.find(",keep_dim")]
    keepdim = afters[afters.find('keep_dim=') + len('keep_dim=') : ]
    if keepdim == 'true' :
      keepdim = "True"
    else:
      keepdim = "False"
    if op == 'ADD':
      code = f"{get_varname(lval)} = torch.sum({get_varname(src)}, dim = {dim}, keepdim={keepdim})"
      return code
    else :
      assert False, f'unsupport reduceop {op}'
#   func.func @Attn(%arg0: tensor<1x4096x32x128xf16> loc(unknown), %arg1: tensor<1x4096x32x128xf16> loc(unknown), %arg2: tensor<1x4096x32x128xf16> loc(unknown)) -> tensor<1x4096x32x128xf16> {
#     %cst = arith.constant dense<1.131250e+01> : tensor<1xf16> loc(#loc)
#     %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16 loc(#loc)
#     %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
#     %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
#     %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16> loc(#loc)
#     %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
#     %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf16>, tensor<1xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
#     %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
#     %7 = deepgengraph.convert %6, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32> loc(#loc)
#     %8 = deepgengraph.exp %7 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32> loc(#loc)
#     %9 = deepgengraph.reduce(%8), dim = -1, op =  ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32> loc(#loc)
#     %10 = deepgengraph.div %8, %9 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32> loc(#loc)
#     %11 = deepgengraph.convert %10, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16> loc(#loc)
#     %12 = deepgengraph.dot %11, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
#     %13 = deepgengraph.permute %12, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16> loc(#loc)
#     return %13 : tensor<1x4096x32x128xf16> loc(#loc)
#   } loc(#loc)

  if ir.find("arith.constant") != -1 :
    return __parse_constant(ir)
  elif ir.find('deepgengraph.trilu') != -1 :
    return __parse_deepgengraph_trilu(ir)
  elif ir.find('deepgengraph.permute') != -1 :
    return __parse_deepgengraph_permute(ir)
  elif ir.find( 'deepgengraph.dot' ) != -1 :
    return __parse_deepgengraph_dot(ir)
  elif ir.find( 'deepgengraph.div' ) != -1 :
    return __parse_deepgengraph_div(ir)
  elif ir.find( 'deepgengraph.add' ) != -1 :
    return __parse_deepgengraph_add(ir)
  elif ir.find( 'deepgengraph.convert' ) != -1 :
    __parse_deepgengraph_convert(ir)
  elif ir.find( 'deepgengraph.exp' ) != -1 :
    return __parse_deepgengraph_exp(ir)
  elif ir.find('deepgengraph.reduce') != -1 :
    return __parse_deepgengraph_reduce(ir)
  else:
    assert False, f"invalid ir!"
  return ""
  
def parse_return_op(ir : str) :
  ret = ir.strip().split(' ')[1]
  code = f"return {get_varname(ret)}"
  return code

def parse_ir_line(ir : str, out_codes : List) :
  LINE_KIND_FUNC_DEF = 0
  LINE_KIND_FUNC_RETURN = 1
  LINE_KIND_OP = 2
  varMap = {}
  
  if ir.find('func.func') != -1 :
    c = parse_func_def(ir,varMap)
    out_codes.append(c)
  elif ir.find('return') != -1 :
    c = parse_return_op(ir)
    out_codes.append('\t' + c)
  else:
    code = parse_op(ir,varMap)
    out_codes.append('\t' + code)
  return
  

def analyze_ir_and_gen_code(ir : str) :
    lines = ir.splitlines()


    irlines = []
    out_codes = []
    start = False
    for line in lines :
      if not start and line.find('func.func') != -1 :
        start = True
      if start :
        irlines.append(line)
      if line.find('return') != -1 :
        start = False; break
    
    for line in irlines :
      parse_ir_line(line, out_codes)
    print("---- analyze ok!")
    
    codestr = ""
    for c in out_codes :
      codestr += (c + "\n")
    print(codestr)
    