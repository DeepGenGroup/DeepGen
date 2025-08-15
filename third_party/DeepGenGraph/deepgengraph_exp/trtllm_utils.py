import time
import math
import ctypes
from collections.abc import Iterable

import torch
import numpy as np

import tensorrt as trt
try:
  import tensorrt_llm as trtllm
  from tensorrt_llm.network import Network, net_guard
  from tensorrt_llm._common import default_net
  from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_str, trt_dtype_to_torch
  from tensorrt_llm.logger import set_level
  from tensorrt_llm.plugin.plugin import ContextFMHAType
except ImportError:
  print("tensorrt_llm not found")
  # copy from: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.11.0/tensorrt_llm/_utils.py#L263
  _trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int64: torch.int64,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
    trt.bfloat16: torch.bfloat16,
    trt.fp8: torch.float8_e4m3fn,
  }
  def trt_dtype_to_torch(dtype):
    ret = _trt_to_torch_dtype_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret

def torch_dtype_to_str(dtype):
  return trt_dtype_to_str(torch_dtype_to_trt(dtype))

def trtllm_build_engine(module, prepare_input_kwargs, plugin_config={}, strongly_typed=True, use_legacy_plugin_config=False, save_engine=False, engine_name=None, save_graph=False, verbose=False, disable=False):
  if engine_name is None:
    engine_name = module.__class__.__name__
  if disable:
    print(f"trtllm build {engine_name} disabled", flush=True)
    return None
  if verbose:
    set_level("verbose")
  builder = trtllm.Builder()
  builder.strongly_typed = strongly_typed
  assert strongly_typed == True
  builder_config = builder.create_builder_config(
    name=engine_name,
    precision="float32",
    profiling_verbosity='detailed',
  )
  net = builder.create_network()
  if use_legacy_plugin_config:
    net.plugin_config.to_legacy_setting()
  net.plugin_config.update_from_dict(plugin_config)
  print(f"trtllm build {engine_name} Config: {net.plugin_config}", flush=True)

  with net_guard(net):
    net.set_named_parameters(module.named_parameters())
    inputs = module.prepare_inputs(**prepare_input_kwargs)
    outputs = module(**inputs)
    if not isinstance(outputs, Iterable):
      outputs = (outputs,)
    module.mark_outputs(*outputs)

  # FIXME: bug for large model?
  if save_graph:
    import graphviz
    path = f"{engine_name}.pdf"
    dot_src = net.to_dot()
    dot = graphviz.Source(dot_src)
    dot.render(outfile=path)
  
  plan = builder.build_engine(net, builder_config)

  if save_engine:
    path = f"{engine_name}.engine"
    with open(path, "wb") as f:
      f.write(plan)

  return plan


def trtllm_build_independent_runtime(engine_or_path, verbose=False):
  print(f"trtllm independent runtime building...", flush=True)
  if isinstance(engine_or_path, str):
    with open(f"{engine_or_path}.engine", "rb") as f:
      engine = f.read()
  else:
    engine = engine_or_path
  assert engine is not None
 
  trt_logger = trt.Logger(trt.Logger.INFO)
  if verbose:
    trt_logger.min_severity = trt.Logger.Severity.VERBOSE
  runtime = trt.Runtime(trt_logger)
  engine = runtime.deserialize_cuda_engine(engine)

  input_specs = []
  output_specs = []
  for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    is_input = False
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
      is_input = True
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    binding = {
      "index": i,
      "name": name,
      "dtype": trt_dtype_to_torch(dtype),
      "shape": tuple(shape),
    }
    if is_input:
      input_specs.append(binding)
    else:
      output_specs.append(binding)

  context = engine.create_execution_context()
  stream = torch.cuda.Stream()
  # stream = torch.cuda.current_stream()
  if len(output_specs) > 1:
    def f(*args):
      device = args[0].device
      for i, arg in enumerate(args):
        spec = input_specs[i]
        ptr = arg.data_ptr()
        context.set_tensor_address(spec["name"], ptr)

      outputs = []
      for i, spec in enumerate(output_specs):
        res = torch.empty(spec["shape"], dtype=spec["dtype"], device=device)
        ptr = res.data_ptr()
        context.set_tensor_address(spec["name"], ptr)
        outputs.append(res)
      ok = context.execute_async_v3(stream.cuda_stream)
      stream.synchronize()
      assert ok
      return tuple(outputs)
    return f
  else:
    def f(*args):
      device = args[0].device
      for i, arg in enumerate(args):
        spec = input_specs[i]
        ptr = arg.data_ptr()
        context.set_tensor_address(spec["name"], ptr)

      outputs = []
      for i, spec in enumerate(output_specs):
        res = torch.empty(spec["shape"], dtype=spec["dtype"], device=device)
        ptr = res.data_ptr()
        context.set_tensor_address(spec["name"], ptr)
        outputs.append(res)
      ok = context.execute_async_v3(stream.cuda_stream)
      stream.synchronize()
      assert ok
      return outputs[0]
    return f

def trtllm_build_runtime(engine_or_path, inputs, outputs):
  print(f"trtllm runtime building...", flush=True)
  if isinstance(engine_or_path, str):
    with open(f"{engine_or_path}.engine", "rb") as f:
      engine = f.read()
  else:
    engine = engine_or_path
  assert engine is not None
  session = trtllm.runtime.Session.from_serialized_engine(engine)
  inputs_info = [
    trtllm.runtime.TensorInfo(name, torch_dtype_to_trt(tensor.dtype), tensor.shape)
    for name, tensor in inputs.items()
  ]
  session.infer_shapes(inputs_info)
  # stream = torch.cuda.current_stream()
  stream = torch.cuda.Stream()
  if len(outputs) == 1:
    def f(*args):
      session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
      return tuple(outputs.values())[0]
    return f
  else:
    def f(*args):
      session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
      return tuple(outputs.values())
    return f

def update_parameters(trt_module, torch_module, extra_mapping_func=None):
  # ref: https://github.com/NVIDIA/TensorRT-LLM/issues/1524
  # ref: https://github.com/pytorch/pytorch/issues/110285
  # ref: https://github.com/pytorch/pytorch/issues/109873
  from tensorrt_llm._utils import np_bfloat16
  m = {k: v for k, v in trt_module.named_parameters()}
  tm = {k: v for k, v in torch_module.named_parameters()}
  for k, v in trt_module.named_parameters():
    src_v = None
    if extra_mapping_func is not None:
      src_v = extra_mapping_func(k, tm)
    else:
      src_v = tm[k]
    if src_v.dtype == torch.bfloat16:
      v.value = src_v.detach().cpu().view(dtype=torch.float16).numpy().view(np_bfloat16)
    else:
      v.value = src_v.detach().cpu().numpy()



def trt_build_engine_from_onnx(onnx_model, engine_fn=None, verbose=False):
  trt_logger = trt.Logger(trt.Logger.INFO)
  if verbose:
    trt_logger.min_severity = trt.Logger.Severity.VERBOSE

  onnx_model_buf = onnx_model.SerializeToString()
  trt.init_libnvinfer_plugins(trt_logger, namespace="")
  builder = trt.Builder(trt_logger)
  config = builder.create_builder_config()
  # config.set_memory_pool_limit(
  #   trt.MemoryPoolType.WORKSPACE, 8 * (2**30)
  # ) # 8 GB
  config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)

  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
  parser = trt.OnnxParser(network, trt_logger)

  if not parser.parse(onnx_model_buf):
    print(f"failed to parse onnx")
    print(onnx.helper.printable_graph(model.graph))
    exit(-1)

  engine_bytes = builder.build_serialized_network(network, config)
  if engine_bytes is None:
    print("failed to create engine")
    exit(-1)

  if engine_fn is not None:
    with open(engine_fn, "wb") as f:
      print(f"serializing engine to file: {engine_fn}")
      f.write(engine_bytes)

  return engine_bytes

def trt_build_independent_runtime(engine_bytes, verbose=False):
  trt_logger = trt.Logger(trt.Logger.INFO)
  if verbose:
    trt_logger.min_severity = trt.Logger.Severity.VERBOSE

  runtime = trt.Runtime(trt_logger)
  engine = runtime.deserialize_cuda_engine(engine_bytes)

  input_specs = []
  output_specs = []
  for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    is_input = False
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
      is_input = True
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    binding = {
      "index": i,
      "name": name,
      "dtype": trt_dtype_to_torch(dtype),
      "shape": tuple(shape),
    }
    if is_input:
      input_specs.append(binding)
    else:
      output_specs.append(binding)

  context = engine.create_execution_context()
  # stream = torch.cuda.Stream()
  stream = torch.cuda.current_stream()
  def f(*args):
    device = args[0].device
    for i, arg in enumerate(args):
      spec = input_specs[i]
      ptr = arg.data_ptr()
      context.set_tensor_address(spec["name"], ptr)

    outputs = []
    for i, spec in enumerate(output_specs):
      res = torch.empty(spec["shape"], dtype=spec["dtype"], device=device)
      ptr = res.data_ptr()
      context.set_tensor_address(spec["name"], ptr)
      outputs.append(res)
    ok = context.execute_async_v3(stream.cuda_stream)
    assert ok
    if len(outputs) == 1:
      return outputs[0]
    else:
      return tuple(outputs)

  return f
