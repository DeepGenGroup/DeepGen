import click
import torch
import numpy as np
import random

from deepgengraph_exp.cases.kernels import *

from deepgengraph_exp.utils import perf, loss, compare, display, AttnInfo

from compile import compile

def gflops_and_mib(seqlen, f, *args):
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.empty_cache()
  start_peak_mem_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
  print(f"{start_peak_mem_mib=}", flush=True)
  f(*args)
  end_peak_mem_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
  print(f"{end_peak_mem_mib=}", flush=True)

  batch_size = 1
  head_num = 32
  head_dim = 128
  gflops = 4 * batch_size * head_num * seqlen * seqlen * head_dim * 1e-9 / 2
  return gflops, end_peak_mem_mib



# @click.command()
# @click.option('--model', '-m', default='attn', help='Model name')
# @click.option('--system', '-s', default='torch', help='System name')
# @click.option('--seqlen', default=4096, help='seqlen')
# @click.option('--show_result', is_flag=True, help='show result')
# @click.option('--check/--no-check', default=True, help='check result with torch')
def main(model, system, seqlen, show_result, check, extra_args = []):
  print(f"{model=} {system=} {seqlen=}")
  assert model in KERNEL_ZOO, f"model {model} not found in KERNEL_ZOO {KERNEL_ZOO.keys()}"

  seed = 0
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  cls = KERNEL_ZOO[model]
  if cls is Attn and len(extra_args) > 0 :
    [batch_size, head_num, seq_len, hd] = extra_args 
    model = Attn(head_num,head_num,hd)
    seqlen = seq_len
  else:
    model = cls()
  model = model.eval().cuda()
  specs = model.prepare(q_len=seqlen, kv_len=seqlen)
  input_names = list(specs['input'].keys())
  inputs = [specs['input'][name] for name in input_names]
  output_names = specs['output']

  print(f"{input_names=}")
  print(f"{output_names=}")
  print(f"{check=}")


  assert system in ['torch', 'tensorrt', 'tvm', 'xla', 'korch', 'einnet', 'our', 'flashinfer', 'flashattn', 'dynamo']
  if system == 'flashinfer' or system == 'flashattn':
    # flashinfer: https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.0/flashinfer-0.1.0+cu121torch2.2-cp310-cp310-linux_x86_64.whl
    # flashattn: pip install packaging flash-attn==2.6.1 --no-build-isolation
    # need rebuild triton
    model_name = model.__class__.__name__
    assert model_name == 'Attn' or 'Gemma2', f'{model_name=}'
  run = 10
  warmup = 100

  f = compile(
    model=model,
    input_names=input_names,
    inputs=inputs,
    output_names=output_names,
    system=system,
  )
  if f is not None:
    gflops, mib = gflops_and_mib(seqlen, f, *inputs)
    print(f"{gflops=}", flush=True)
    print(f"{mib=}", flush=True)

    if check:
      print(f"checking {system}...")
      outs_ref = model(*inputs)
      outs = f(*inputs)
      torch.cuda.synchronize()
      compare(outs, outs_ref, output_names)
      if show_result:
        display(outs, outs_ref, output_names)
      
    # perf(
    #   label=system,
    #   f=f,
    #   args=inputs,
    #   run=run,
    #   warmup=warmup,
    #   profile=True,
    #   gflops=gflops,
    # )

def optimize_graph(bathch, head_num, seq_len, hd) :
  AttnInfo.Batch = bathch
  AttnInfo.HeadNum = head_num
  AttnInfo.SeqLen = seq_len
  AttnInfo.Hd = hd
  return main('attn', 'our', seq_len, True, False, extra_args = [bathch, head_num, seq_len, hd])

if __name__ == '__main__':
  main()