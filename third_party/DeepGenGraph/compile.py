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
    # partition.module.dump()
    tok = time.time()
    tuning_s = tok - tik
    print(f"tuning time: {tuning_s} sec", flush=True)
    perf = partition.profile()
    py_str = partition.codegen(perf)

    our = {}
    import tempfile
    import importlib
    import sys
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
      f.write(py_str)
      path = f.name
    print(f"write code to {path}", flush=True)
    spec = importlib.util.spec_from_file_location('our', path)
    pymod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pymod)

    f = getattr(pymod, func_name)
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
