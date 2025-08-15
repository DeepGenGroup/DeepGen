import json
import click
import torch
import numpy as np
import random
import os

from deepgengraph_exp.cases.kernels import KERNEL_ZOO
from deepgengraph_exp.cases.models import MODEL_ZOO

from deepgengraph_exp.utils import perf

from compile import compile

from transformers import AutoConfig, AutoTokenizer


def llm_setup(weight_path, seqlen, layer_num):
  device = torch.cuda.current_device()

  hf_config = AutoConfig.from_pretrained(weight_path)
  cache_budget = 512
  assert cache_budget < seqlen
  hf_config.cache_budget = cache_budget
  hf_config.roco_recent = 256
  hf_config.tau = 1.5
  corm_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device)
  for i in range(seqlen):
    corm_mask[i] /= i + 1
  hf_config.corm_mask = corm_mask


  if layer_num is not None:
    hf_config.num_hidden_layers = layer_num
  hf_config.rotary_base = getattr(hf_config, 'rope_theta', 10000.0)
  print(f"{hf_config.num_hidden_layers=}")

  assert hf_config.max_position_embeddings >= seqlen
  hf_config.max_position_embeddings = seqlen

  batch_size = 1
  data_path = os.path.dirname(os.path.abspath(__file__)) + "/vcsum.jsonl"
  print(f"{data_path=}")
  with open(data_path, "r", encoding='utf-8') as f:
    data = json.loads(f.readline())
    data = data['context']
  token_ids = AutoTokenizer.from_pretrained(weight_path)(data).input_ids[:seqlen]
  assert len(token_ids) == seqlen
  token_ids = torch.tensor(token_ids, dtype=torch.int64, device=device).reshape(batch_size, seqlen).contiguous()
  print(f"{token_ids.shape=}")
  print(f"{token_ids.grad=}")

  return hf_config, token_ids


@click.command()
@click.option('--model', '-m', default='attn', help='Model name')
@click.option('--system', '-s', default='torch', help='System name')
@click.option('--seqlen', type=int, default=4096, help='seqlen')
@click.option('--layer_num', type=int, default=None, help='layer_num')
@click.option('--platform', '-p', default='yes', help='platform(yes, qiyuan, fuse0)')
def main(model, system, seqlen, layer_num, platform):
  print(f"{model=} {system=} {seqlen=} {layer_num=}")
  assert model in KERNEL_ZOO, f"model {model} not found in KERNEL_ZOO {KERNEL_ZOO.keys}"
  seed = 0
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  kernel_cls = KERNEL_ZOO[model]
  model_cls = MODEL_ZOO[model]

  kernel = kernel_cls().eval().cuda()
  specs = kernel.prepare(q_len=seqlen, kv_len=seqlen)
  input_names = list(specs['input'].keys())
  inputs = [specs['input'][name] for name in input_names]
  output_names = specs['output']

  print(f"{input_names=}")
  print(f"{output_names=}")

  assert system in ['torch', 'tensorrt', 'tvm', 'xla', 'korch', 'einnet', 'our', 'dynamo']
  kernel_f = compile(
    model=kernel,
    input_names=input_names,
    inputs=inputs,
    output_names=output_names,
    system=system,
  )

  weight_zoo_path = os.path.dirname(os.path.abspath(__file__)) + "/weight_zoo.json"
  print(f"{weight_zoo_path=}")
  with open(weight_zoo_path, "r") as f:
    weight_zoo = json.load(f)

  weight_path = weight_zoo[platform]
  hf_config, token_ids = llm_setup(weight_path, seqlen, layer_num)

  # build model
  model = model_cls(
    hf_config=hf_config,
    attn_f=kernel_f,
  )
  for param in model.parameters():
    param.requires_grad = False
  model = model.eval().cuda()

  run = 50
  warmup = 50
  perf(
    label=system,
    f=model,
    args=(token_ids,),
    run=run,
    warmup=warmup,
    profile=True,
  )

if __name__ == '__main__':
  main()