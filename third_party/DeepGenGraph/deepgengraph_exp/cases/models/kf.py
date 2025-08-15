import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llama_base import collect_weight_dict, init_cos_sin_cache, RmsNorm, MLP, rotary_embedding_online_cuda


class Attention(nn.Module):
  def __init__(self, hf_config, attn_f, layer_idx):
    super().__init__()
    self.layer_idx = layer_idx
    self.hidden_size = hf_config.hidden_size
    self.head_num = hf_config.num_attention_heads
    self.kv_head_num = hf_config.num_key_value_heads
    self.head_dim = self.hidden_size // self.head_num
    self.kv_hidden_size = self.kv_head_num * self.head_dim
    self.group_size = self.head_num // self.kv_head_num
    self.cache_budget = hf_config.cache_budget
    self.dtype = hf_config.torch_dtype

    self.attn_f = attn_f
    self.tau = hf_config.tau

    self.norm_factor = math.sqrt(self.head_dim)
    self.max_position = hf_config.max_position_embeddings
    self.rotary_base = hf_config.rotary_base
    self.rotary_dim = self.head_dim

    self.embed_positions = nn.Parameter(init_cos_sin_cache(theta=self.rotary_base, dim=self.rotary_dim, max_position=self.max_position))
    self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size + 2 * self.kv_hidden_size, bias=False, dtype=self.dtype)
    self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=self.dtype)
  
  def forward(self, x, kv_caches):
    qkv = self.qkv_proj(x)
    q, k, v = qkv.split([self.hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=-1)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    pos = torch.arange(0, q.shape[1], dtype=torch.int32, device=x.device)
    rotary_embedding_online_cuda(pos, q, k, self.head_dim, self.rotary_base)
    torch.cuda.synchronize()
    exp_rand = torch.empty(q.shape[0], self.head_num, q.shape[1], k.shape[1], dtype=torch.float32, device=q.device).uniform_(1e-8, 1.0)

    out, kf_score = self.attn_f(
      q.view(q.shape[0], q.shape[1], self.head_num, self.head_dim),
      k.view(k.shape[0], k.shape[1], self.kv_head_num, self.head_dim),
      v.view(v.shape[0], v.shape[1], self.kv_head_num, self.head_dim),
      exp_rand,
    )
    out = out.view(out.shape[0], out.shape[1], -1)

    _, selected_indices = torch.topk(kf_score, k=self.cache_budget, dim=-1, sorted=True)
    selected_indices = selected_indices.sort(dim=-1)[0]
    selected_indices = selected_indices.transpose(1, 2)[:, :, :, None].expand(q.shape[0], self.cache_budget, self.kv_head_num, self.head_dim)
    k_cache = torch.gather(k.view(k.shape[0], -1, self.kv_head_num, self.head_dim), dim=1, index=selected_indices).view(k.shape[0], 1, -1, self.kv_head_num, self.head_dim)
    v_cache = torch.gather(v.view(v.shape[0], -1, self.kv_head_num, self.head_dim), dim=1, index=selected_indices).view(v.shape[0], 1, -1, self.kv_head_num, self.head_dim)
    kv_cache = torch.cat([k_cache, v_cache], dim=1)
    kv_caches.append((kf_score, kv_cache))
    out = self.out_proj(out)
    return out, kv_caches

class LlamaLayer(nn.Module):
  def __init__(self, hf_config, attn_f, layer_idx):
    super().__init__()
    self.layer_idx = layer_idx
    self.hf_config = hf_config
    self.input_layernorm = RmsNorm(dim=hf_config.hidden_size, eps=hf_config.rms_norm_eps, dtype=hf_config.torch_dtype)
    self.attention = Attention(hf_config, attn_f, layer_idx)
    self.mlp = MLP(
      hidden_size=hf_config.hidden_size,
      intermediate_size=hf_config.intermediate_size,
      hidden_act=hf_config.hidden_act,
      dtype=hf_config.torch_dtype,
      bias=False,
    )
    self.post_layernorm = RmsNorm(dim=hf_config.hidden_size, eps=hf_config.rms_norm_eps, dtype=hf_config.torch_dtype)
  
  def forward(self, x, kv_caches):
    res = x
    x = self.input_layernorm(x)
    attn_out, kv_caches = self.attention(x, kv_caches)
    x = res + attn_out
    res = x
    x = self.post_layernorm(x)
    mlp_out = self.mlp(x)
    x = res + mlp_out
    return x, kv_caches

class Llama(nn.Module):
  def __init__(self, hf_config, attn_f):
    super().__init__()
    self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size, dtype=hf_config.torch_dtype)
    self.layers = nn.ModuleList()
    for layer_idx in range(hf_config.num_hidden_layers):
      self.layers.append(LlamaLayer(hf_config, attn_f, layer_idx))
    self.rms_norm = RmsNorm(dim=hf_config.hidden_size, eps=hf_config.rms_norm_eps, dtype=hf_config.torch_dtype)
    self.hf_config = hf_config

    self.load_param(hf_config)
  
  def load_param(self, hf_config):
    weight_dict = collect_weight_dict(hf_config)
    with torch.no_grad():
      for name, param in self.named_parameters():
        if 'embed_positions' not in name:
          param.copy_(weight_dict[name])
    del weight_dict

  def forward(self, token_ids):
    kv_caches = []
    x = self.embed_tokens(token_ids)
    for i in range(len(self.layers)):
      x, kv_caches = self.layers[i](x, kv_caches)
    x = self.rms_norm(x)
    return x, kv_caches


