import os
import safetensors
import safetensors.torch
import json
import deepgengraph_exp._csrc as native_ops


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def collect_hf_weight(hf_model_path, names=None, use_safe_tensor=True, device='cpu'):
  if use_safe_tensor:
    weight_index_fn = 'model.safetensors.index.json'
    default_weight_fn = 'model.safetensors'
  else:
    weight_index_fn = 'pytorch_model.bin.index.json'
    default_weight_fn = 'unimplemeted_error'
  
  if os.path.exists(os.path.join(hf_model_path, weight_index_fn)):
    with open(os.path.join(hf_model_path, weight_index_fn)) as f:
      mapping = json.load(f)
      mapping = mapping['weight_map']
      files = [mapping[name] for name in names] if names is not None else mapping.values()
      files = set(files)
  else:
    assert os.path.exists(os.path.join(hf_model_path, default_weight_fn)), f"{default_weight_fn}"
    files = set([default_weight_fn])

  weight_dict = {}
  if use_safe_tensor:
    print(f"using safe tensor: {files=}")
    for file_name in tqdm(files):
      file = os.path.join(hf_model_path, file_name)
      weight = safetensors.torch.load_file(file, device=device)
      for name in weight.keys():
        param = weight[name]
        weight_dict[name] = param
  else:
    print(f"using torch bin: {files=}")
    for file_name in files:
      file = os.path.join(hf_model_path, file_name)
      weight = torch.load(file, map_location=torch.device(device))
      for name in weight.keys():
        param = weight[name]
        weight_dict[name] = param

  return weight_dict

def collect_weight_dict(hf_config):
  weight_dict = collect_hf_weight(hf_config.name_or_path)
  for layer_idx in tqdm(range(hf_config.num_hidden_layers)):
    input_layernorm_weight = weight_dict.pop(f"model.layers.{layer_idx}.input_layernorm.weight")
    attn_q_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
    attn_k_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
    attn_v_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
    attn_qkv_proj_weight = torch.cat([attn_q_proj_weight, attn_k_proj_weight, attn_v_proj_weight], dim=0)
    attn_o_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
    mlp_gate_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.mlp.gate_proj.weight")
    mlp_up_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.mlp.up_proj.weight")
    mlp_gate_up_proj_weight = torch.cat([mlp_gate_proj_weight, mlp_up_proj_weight], dim=0)
    mlp_down_proj_weight = weight_dict.pop(f"model.layers.{layer_idx}.mlp.down_proj.weight")
    post_layernorm_weight = weight_dict.pop(f"model.layers.{layer_idx}.post_attention_layernorm.weight")

    weight_dict[f"layers.{layer_idx}.input_layernorm.weight"] = input_layernorm_weight
    weight_dict[f"layers.{layer_idx}.attention.qkv_proj.weight"] = attn_qkv_proj_weight
    weight_dict[f"layers.{layer_idx}.attention.out_proj.weight"] = attn_o_proj_weight
    # weight_dict[f"layers.{layer_idx}.mlp.gate_proj.weight"] = mlp_gate_proj_weight
    # weight_dict[f"layers.{layer_idx}.mlp.up_proj.weight"] = mlp_up_proj_weight
    weight_dict[f"layers.{layer_idx}.mlp.gate_up_proj.weight"] = mlp_gate_up_proj_weight
    weight_dict[f"layers.{layer_idx}.mlp.down_proj.weight"] = mlp_down_proj_weight
    weight_dict[f"layers.{layer_idx}.post_layernorm.weight"] = post_layernorm_weight
    
  embed_tokens_weight = weight_dict.pop(f"model.embed_tokens.weight")
  rms_norm_weight = weight_dict.pop(f"model.norm.weight")
  weight_dict[f"embed_tokens.weight"] = embed_tokens_weight
  weight_dict[f"rms_norm.weight"] = rms_norm_weight

  return weight_dict

  with torch.no_grad():
    for name, param in self.named_parameters():
      if 'embed_positions' not in name:
        param.copy_(weight_dict[name])
  del weight_dict


def init_cos_sin_cache(theta, dim, max_position):
  inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
  sinusoid_inp = torch.einsum("i,j -> ij", torch.arange(max_position, dtype=torch.float32), inv_freq).to(torch.float32)
  concat = torch.concat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
  concat = concat.view(1, max_position, dim).contiguous()
  return concat


def rotary_embedding_online_cuda(pos, q, k, head_dim, rope_theta):
  assert q.dim() == 3 and q.shape[0] == 1
  assert k.dim() == 3 and k.shape[0] == 1
  native_ops.rotary_embedding_online(pos, q.view(q.shape[1], q.shape[2]), k.view(k.shape[1], k.shape[2]), head_dim, rope_theta)

def silu_and_mul_cuda(x):
  assert x.dim() == 3 and x.shape[0] == 1
  num_token = x.shape[1]
  dim = x.shape[2] // 2
  out = torch.empty(x.shape[0], num_token, dim, dtype=x.dtype, device=x.device)
  native_ops.silu_and_mul(out.view(num_token, dim), x.view(num_token, x.shape[2]))
  return out

def rms_norm_cuda(x, weight, eps):
  out = torch.empty_like(x)
  native_ops.rms_norm(out, x, weight, eps)
  return out


class RmsNorm(nn.Module):
  def __init__(self, dim, eps, dtype):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
  
  def norm_torch(self, x):
    x_f32 = x.float()
    output = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
    return output.type_as(x) * self.weight
  
  def forward(self, x):
    # return self.norm_torch(x)
    return rms_norm_cuda(x, self.weight, self.eps)



class MLP(nn.Module):
  def __init__(self, hidden_size, intermediate_size, hidden_act, bias=False, dtype=None):
    super().__init__()
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.hidden_act = hidden_act
    assert self.hidden_act == 'silu'
    # self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)
    # self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)
    self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias, dtype=dtype)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype)
  
  def forward(self, x):
    # gate = self.gate_proj(x)
    # gate = F.silu(gate)
    # up = self.up_proj(x)
    # out = gate * up
    # out = self.down_proj(out)
    gate_up = self.gate_up_proj(x)
    out = silu_and_mul_cuda(gate_up)
    out = self.down_proj(out)
    return out