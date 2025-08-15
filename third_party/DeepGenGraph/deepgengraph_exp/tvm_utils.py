import io
import os
import time

import torch
import onnx


import tvm
import tvm.relay as relay
from tvm.support import describe
from tvm.relay.frontend import from_onnx
from tvm import meta_schedule as ms
from tvm.contrib import graph_executor
from tvm import auto_scheduler
from tvm.meta_schedule.runner import Runner

def get_tvm_target(use_cublas=True):
  if torch.cuda.get_device_capability() == (8, 0):
    # a100
    target = tvm.target.Target("nvidia/nvidia-a100")
  else:
    assert torch.cuda.get_device_capability() == (9, 0)
    # h100
    target = tvm.target.Target("nvidia/nvidia-h100")
  if use_cublas:
    target = tvm.target.Target(str(target) + " -libs=cublas")
  return target

def torch_module_to_tvm(module, input_names, inputs, output_names):
  onnx_bytes = io.BytesIO()
  torch.onnx.export(
    module,
    args=tuple(inputs),
    f=onnx_bytes,
    input_names=input_names,
    output_names=output_names,
    verbose=False,
  )
  onnx_model = onnx.load_model_from_string(onnx_bytes.getvalue())
  # print(onnx.helper.printable_graph(onnx_model.graph))
  shape_dict = {input_name: list(input_.shape) for input_name, input_ in zip(input_names, inputs)}
  mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)
  return mod, params

def auto_scheduler_tune(module, input_names, inputs, output_names, num_trials=1000, exported_lib_path=None):
  describe()
  mod, params = torch_module_to_tvm(module, input_names, inputs, output_names)
  target = get_tvm_target()
  print(f"[tvm auto_scheduler]: {target=}", flush=True)

  log_file = f"{module.__class__.__name__}.json"
  print(f"[tvm auto_scheduler]: {log_file=}", flush=True)

  hardware_params = auto_scheduler.HardwareParams(
    num_cores=-1,
    vector_unit_bytes=16,
    cache_line_bytes=64,
    max_shared_memory_per_block=int(target.attrs["max_shared_memory_per_block"]),
    max_threads_per_block=int(target.attrs["max_threads_per_block"]),
    # The value `max_local_memory_per_block` is not used in AutoScheduler,
    # but is required by the API.
    max_local_memory_per_block=12345678,
    max_vthread_extent=8,
    warp_size=32,
  )

  with ms.Profiler() as profiler:
    tasks, task_weights = auto_scheduler.extract_tasks(
      mod["main"], params,
      target=target,
      hardware_params=hardware_params,
    )
    for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
      print(
        f"==== Task {idx}: {task.desc} "
        f"(weight {task_weight} key: {task.workload_key}) ====="
      )
      print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(
      auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        runner="local",
        measure_callbacks=[
          auto_scheduler.RecordToFile(log_file),
        ],
      ),
      adaptive_training=True,
    )

    with auto_scheduler.ApplyHistoryBest(log_file):
      with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True},):
        lib = relay.build(mod, target=target, params=params)

  print("[tvm auto_scheduler] tuing time:")
  print(profiler.table())

  # create graph executor
  dev = tvm.cuda()
  module = graph_executor.GraphModule(lib["default"](dev))
  data_tvm = {name: tvm.nd.array(t.cpu().numpy(), dev) for name, t in zip(input_names, inputs)}
  module.set_input(**data_tvm)

  # evaluate
  print(f"[tvm auto_scheduler] eval inference")
  print(module.benchmark(dev, repeat=1, number=100))

  if exported_lib_path is not None:
    lib.export_library(exported_lib_path)
  
  return lib

def meta_scheduler_tune(module, input_names, inputs, output_names, num_trials_per_iter=64, max_trials_per_task=1000, max_trials_global=-1, exported_lib_path=None):
  # describe()
  mod, params = torch_module_to_tvm(module, input_names, inputs, output_names)
  target = get_tvm_target()
  print(f"[tvm ms]: {target=}", flush=True)

  log_file = f"{module.__class__.__name__}.json"
  print(f"[tvm ms]: {log_file=}", flush=True)

  extracted_tasks = ms.relay_integration.extract_tasks(mod, target=target, params=params)
  print(f"[tvm ms]: kernel partitions: {len(extracted_tasks)=}", flush=True)

  max_trials_global = max_trials_global if max_trials_global > 0 else len(extracted_tasks) * max_trials_per_task
  print(f"[tvm ms]: {num_trials_per_iter=}", flush=True)
  print(f"[tvm ms]: {max_trials_per_task=}", flush=True)
  print(f"[tvm ms]: {max_trials_global=}", flush=True)

  pid = os.getpid()
  with ms.Profiler() as profiler:
    # runner = 'local'
    runner = Runner.create('local', timeout_sec=120.0, max_workers="physical")
    database = ms.relay_integration.tune_relay(
      mod=mod,
      params=params,
      target=target,
      strategy="evolutionary",
      num_trials_per_iter=num_trials_per_iter,
      max_trials_per_task=max_trials_per_task,
      max_trials_global=max_trials_global,
      seed=0,
      runner=runner,
      work_dir=f"./tvm_tuned_{pid}",
    )
    print("[tvm ms] profile done, compiling", flush=True)
    tik = time.time()
    lib = ms.relay_integration.compile_relay(
      database=database,
      mod=mod,
      params=params,
      target=target,
      backend="graph",
    )
    tok = time.time()
    compile_s = tok - tik
  print("[tvm ms]: tuning time", flush=True)
  print(profiler.table())

  # create graph executor
  dev = tvm.cuda()
  module = graph_executor.GraphModule(lib["default"](dev))
  data_tvm = {name: tvm.nd.array(t.cpu().numpy(), dev) for name, t in zip(input_names, inputs)}
  module.set_input(**data_tvm)

  # evaluate
  print(f"[tvm ms] eval inference", flush=True)
  print(module.benchmark(dev, repeat=1, number=100))


  if exported_lib_path is not None:
    lib.export_library(exported_lib_path)
  
  return lib


def tvm_build_runtime(lib, input_names, inputs, output_names):
  for t in inputs:
    assert isinstance(t, torch.Tensor)

  dev = tvm.cuda()
  module = graph_executor.GraphModule(lib["default"](dev))
  data_tvm = {name: tvm.nd.array(t.cpu().numpy(), dev) for name, t in zip(input_names, inputs)}
  module.set_input(**data_tvm)

  def f(*args):
    module.run()
    torch.cuda.synchronize()
    if len(output_names) == 1:
      return module.get_output(0)
    else:
      return [module.get_output(i) for i in range(len(output_names))]
  return f

def tvm_build_independent_runtime(lib, input_names, output_names):
  dev = tvm.cuda()
  module = graph_executor.GraphModule(lib["default"](dev))

  def f(*args):
    # data = {name: tvm.nd.from_dlpack(arg.__dlpack__()) for name, arg in zip(input_names, args)}
    for i, arg in enumerate(args):
      data_tvm = tvm.nd.from_dlpack(arg.detach().__dlpack__())
      module.set_input(input_names[i], data_tvm)
    module.run()
    torch.cuda.synchronize()
    if len(output_names) == 1:
      return torch.utils.dlpack.from_dlpack(module.get_output(0).to_dlpack())
    else:
      return [torch.utils.dlpack.from_dlpack(module.get_output(i).to_dlpack()) for i in range(len(output_names))]
  return f
