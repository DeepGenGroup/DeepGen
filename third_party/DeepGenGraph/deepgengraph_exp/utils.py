import ctypes
import time
import torch
import math
import numpy as np
import io

def torch_module_to_onnx(module, input_names, inputs, output_names, simplify=True):
  import onnx
  import onnxsim
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
  if simplify:
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check
  return onnx_model

_cudart = ctypes.CDLL('libcudart.so')

def profile_start():
  ret = _cudart.cudaProfilerStart()
  if ret != 0:
    raise Exception("cudaProfilerStart() returned %d" % ret)

# FIXME: not stop profiling
def profile_stop():
  ret = _cudart.cudaProfilerStop()
  if ret != 0:
    raise Exception("cudaProfilerStop() returned %d" % ret)


def perf(label, f, args, kwargs={}, gflops=None, mem_gb=None, run=4, warmup=4, profile=False):
  print(f"warmup start", flush=True)
  torch.cuda.synchronize()
  for _ in range(warmup):
    torch.cuda.synchronize()
    o = f(*args, **kwargs)
    torch.cuda.synchronize()
  print(f"warmup done", flush=True)

  if profile:
    profile_start()
  ms = []
  for _ in range(run):
    torch.cuda.synchronize()
    tik = time.time()
    o = f(*args, **kwargs)
    torch.cuda.synchronize()
    tok = time.time()
    ms.append((tok - tik) * 1000.0)
  if profile:
    profile_stop()

  min_ms = np.min(ms)
  max_ms = np.max(ms)
  avg_ms = np.mean(ms)
  msg = f'[{label}] avg {avg_ms:.4f} ms, min {min_ms:.4f} ms, max {max_ms:.4f} ms'
  if gflops is not None:
    msg += f', {gflops / (avg_ms / 1000.0)} gflops/s'
  if mem_gb is not None:
    msg += f', {mem_gb / (avg_ms / 1000.0)} gb/s'
  
  msg += f' ({run} runs, {warmup} warmups)' if not profile else f' ({run} runs, {warmup} warmups, profiled)'
  print(msg, flush=True)


def loss(out, ref):
  assert isinstance(out, torch.Tensor), f"{type(out)=}"
  assert isinstance(ref, torch.Tensor), f"{type(ref)=}"
  assert out.dtype == ref.dtype, f"{out.dtype=} {ref.dtype=}"
  assert out.shape == ref.shape, f"{out.shape=} {ref.shape=}"
  out = out.cpu()
  ref = ref.cpu()
  if out.dtype in {torch.int32, torch.int64, torch.bool}:
    err_num = torch.sum(out != ref).item()
    total_num = out.numel()
    return {"err_num": err_num, "err": err_num / total_num}
  else:
    abs_max_loss = (out.double() - ref.double()).abs().max().item()
    abs_mean_loss = (out.double() - ref.double()).abs().mean().item()
    mask = ref != 0
    nz_out = out[mask]
    nz_ref = ref[mask]
    rel_max_loss = ((out.double() - ref.double()) / ref).abs().max().item()
    rel_mean_loss = ((out.double() - ref.double()) / ref).abs().mean().item()
    nz_rel_max_loss = ((nz_out.double() - nz_ref.double()) / nz_ref).abs().max().item()
    nz_rel_mean_loss = ((nz_out.double() - nz_ref.double()) / nz_ref).abs().mean().item()
    mse_loss_f = torch.nn.MSELoss(reduction='mean')
    mse_loss = mse_loss_f(out.double(), ref.double()).item()
    rmse_loss = math.sqrt(mse_loss)
    return {
      'abs_max': abs_max_loss,
      'abs_mean': abs_mean_loss,
      'rel_max': rel_max_loss,
      'rel_mean': rel_mean_loss,
      'nz_rel_max': nz_rel_max_loss,
      'nz_rel_mean': nz_rel_mean_loss,
      'mse': mse_loss,
      'rmse': rmse_loss,
    }


def compare(outs, refs, names):
  if not isinstance(outs, list) and not isinstance(outs, tuple):
    assert len(names) == 1, f"{names=}, {type(outs)=}"
    print(f"{names[0]} loss: {loss(outs, refs)}", flush=True)
  else:
    assert len(outs) == len(refs), f"{len(outs)=} {len(refs)=}"
    assert len(outs) == len(names), f"{len(outs)=} {len(names)=}"
    for out, ref, name in zip(outs, refs, names):
      print(f"{name} loss: {loss(out, ref)}", flush=True)

def display(outs, refs, names):
  if not isinstance(outs, list):
    assert len(names) == 1
    print(f"{names[0]} out: {outs}", flush=True)
    print(f"{names[0]} ref: {refs}", flush=True)
  else:
    assert len(outs) == len(refs), f"{len(outs)=} {len(refs)=}"
    assert len(outs) == len(names), f"{len(outs)=} {len(names)=}"
    for out, ref, name in zip(outs, refs, names):
      print(f"{name} out: {out}", flush=True)
      print(f"{name} ref: {ref}", flush=True)
