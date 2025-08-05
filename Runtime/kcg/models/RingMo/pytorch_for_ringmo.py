from types import NoneType
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn.init import trunc_normal_
from typing import Any, List, Union
from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.distributed as dist

from kcg.TorchInjector import *
from kcg.ModelUtils import *


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 定义专家网络
class Expert(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, mlp_drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(p=mlp_drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 定义MoE层
class MoELayer(nn.Module):
    def __init__(self, gate_type, model_dim, experts, act_layer=nn.GELU, mlp_drop=0.0,
                 scan_expert_func=None, seeds=None, batch_prioritized_routing=False,
                 normalize_gate=False, is_gshard_loss=False):
        super().__init__()
        self.gate_type = gate_type
        self.model_dim = model_dim
        self.num_experts = experts['count_per_node']
        self.hidden_size = experts['hidden_size_per_expert']
        self.top_k = gate_type['k']
        self.capacity_factor = gate_type['capacity_factor']
        self.gate_noise = gate_type['gate_noise']
        self.fp32_gate = gate_type['fp32_gate']
        if 'proj_dim' in gate_type:
            self.proj_dim = gate_type['proj_dim']
        else:
            self.proj_dim = model_dim
        if 'init_t' in gate_type:
            self.init_t = gate_type['init_t']
        else:
            self.init_t = 1.0
        
        # 门控网络
        if self.gate_type['type'] == 'top':
            self.gate = nn.Linear(model_dim, self.num_experts)
        elif self.gate_type['type'] == 'cosine_top':
            self.gate = nn.Linear(model_dim, self.proj_dim)
            self.expert_centroids = nn.Parameter(torch.randn(self.num_experts, self.proj_dim))
        
        # 专家网络
        self.experts = nn.ModuleList([Expert(model_dim, self.hidden_size, act_layer, mlp_drop) for _ in range(self.num_experts)])
        
        # 其他参数
        self.batch_prioritized_routing = batch_prioritized_routing
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss
    
    def forward(self, x):
        # x: [batch_size, seq_len, model_dim]
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.model_dim)  # [batch_size * seq_len, model_dim]
        
        # 门控网络
        if self.gate_type['type'] == 'top':
            gate_scores = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
            if self.normalize_gate:
                gate_scores = F.softmax(gate_scores, dim=-1)
            # 添加噪声
            if self.gate_noise > 0:
                noise = Normal(0, self.gate_noise).sample(gate_scores.shape).to(gate_scores.device)
                gate_scores = gate_scores + noise
            # 选择前k个专家
            topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        elif self.gate_type['type'] == 'cosine_top':
            proj_x = self.gate(x_flat)  # [batch_size * seq_len, proj_dim]
            proj_x = F.normalize(proj_x, dim=-1)
            centroids = F.normalize(self.expert_centroids, dim=-1)
            cosine_sim = torch.matmul(proj_x, centroids.t())  # [batch_size * seq_len, num_experts]
            gate_scores = cosine_sim / self.init_t
            if self.normalize_gate:
                gate_scores = F.softmax(gate_scores, dim=-1)
            # 选择前k个专家
            topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # 简化的路由：为每个token选择一个专家（top-1）
        selected_expert_idx = topk_indices[:, 0]  # 只选top-1专家
        outputs = []
        for i in range(batch_size * seq_len):
            expert_idx = selected_expert_idx[i]
            expert_output = self.experts[expert_idx](x_flat[i].unsqueeze(0))
            outputs.append(expert_output)
        outputs = torch.cat(outputs, dim=0).view(batch_size, seq_len, self.model_dim)
        
        # 辅助损失（暂时设为0，实际需要根据需求计算）
        l_aux = torch.tensor(0.0, device=x.device)
        
        return outputs, l_aux

class MoEMlp(nn.Module):
    def __init__(self, in_features, hidden_features, num_local_experts, top_value, act_layer=nn.GELU,
                 capacity_factor=1.25, cosine_router=False, normalize_gate=False, use_bpr=True, 
                 is_gshard_loss=True,gate_noise=1.0, cosine_router_dim=256, cosine_router_init_t=0.5, 
                 mlp_drop=0.0, init_std=0.02, mlp_fc2_bias=True):
        
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_local_experts = num_local_experts
        self.top_value = top_value
        self.capacity_factor = capacity_factor
        self.cosine_router = cosine_router
        self.normalize_gate = normalize_gate
        self.use_bpr = use_bpr
        self.init_std = init_std
        self.mlp_fc2_bias = mlp_fc2_bias

        self.dist_rank = torch.distributed.get_rank()

        _gate_type = {'type': 'cosine_top' if cosine_router else 'top',
                      'k': top_value, 'capacity_factor': capacity_factor,
                      'gate_noise': gate_noise, 'fp32_gate': True}
        if cosine_router:
            _gate_type['proj_dim'] = cosine_router_dim
            _gate_type['init_t'] = cosine_router_init_t
            
        self._moe_layer = MoELayer(
            gate_type=_gate_type,
            model_dim=in_features,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            experts={'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_features},
            scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),  # 传入但不处理
            seeds=(1, self.dist_rank + 1, self.dist_rank + 1),  # 传入但不处理
            batch_prioritized_routing=use_bpr,
            normalize_gate=normalize_gate,
            is_gshard_loss=is_gshard_loss,
        )
        if not self.mlp_fc2_bias:
            for expert in self._moe_layer.experts:
                expert.fc2.bias.requires_grad = False

    def forward(self, x):
        x, l_aux = self._moe_layer(x)
        return x, l_aux

    def extra_repr(self) -> str:
        return f'[Statistics-{self.dist_rank}] param count for MoE, ' \
               f'in_features = {self.in_features}, hidden_features = {self.hidden_features}, ' \
               f'num_local_experts = {self.num_local_experts}, top_value = {self.top_value}, ' \
               f'cosine_router={self.cosine_router} normalize_gate={self.normalize_gate}, use_bpr = {self.use_bpr}'

    def _init_weights(self):
        if hasattr(self._moe_layer, "experts"):
            trunc_normal_(self._moe_layer.experts.batched_fc1_w, std=self.init_std)
            trunc_normal_(self._moe_layer.experts.batched_fc2_w, std=self.init_std)
            nn.init.constant_(self._moe_layer.experts.batched_fc1_bias, 0)
            nn.init.constant_(self._moe_layer.experts.batched_fc2_bias, 0)

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,
                 use_moe=False, num_local_experts=1, top_value=1, capacity_factor=1.25, cosine_router=False,
                 normalize_gate=False, use_bpr=True, is_gshard_loss=True, gate_noise=1.0,
                 cosine_router_dim=256, cosine_router_init_t=0.5, mlp_fc2_bias=True, init_std=0.02, blk_idx = 0):
        
        super().__init__()
        self.idx = blk_idx
        self.dim = dim
        self.use_moe = use_moe
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
            proj_drop=drop)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.use_moe:
            self.mlp = MoEMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, 
                             num_local_experts=num_local_experts, top_value=top_value, 
                             capacity_factor=capacity_factor, cosine_router=cosine_router,
                             normalize_gate=normalize_gate, use_bpr=use_bpr, is_gshard_loss=is_gshard_loss,
                             gate_noise=gate_noise, cosine_router_dim=cosine_router_dim,
                             cosine_router_init_t=cosine_router_init_t, mlp_drop=drop,
                             mlp_fc2_bias=mlp_fc2_bias, init_std=init_std)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)  
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            # self.attn_mask = nn.Parameter(attn_mask, requires_grad=False)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        def manual_roll(input_tensor, shifts, dims=None):
            """
            Manually implements torch.roll functionality, shifting elements along specified dimensions.
            
            Args:
                input_tensor (torch.Tensor): Input tensor to be rolled.
                shifts (int or tuple of ints): Number of places to shift elements. Positive means right/down, negative means left/up.
                dims (int or tuple of ints, optional): Dimensions along which to roll. If None, flattens tensor and rolls.
            
            Returns:
                torch.Tensor: Rolled tensor.
            """
            # Input validation
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor")
            
            # Handle shifts and dims as single values or tuples
            if isinstance(shifts, int):
                shifts = (shifts,)
            if dims is None:
                dims = tuple(range(input_tensor.dim()))
            elif isinstance(dims, int):
                dims = (dims,)
            
            if len(shifts) != len(dims):
                raise ValueError("shifts and dims must have the same length")
            
            # Normalize dims to handle negative indices
            dims = tuple(d % input_tensor.dim() for d in dims)
            # Clone input to avoid modifying it
            result = input_tensor.clone()
            # Perform roll for each dimension
            for shift, dim in zip(shifts, dims):
                if shift == 0:
                    continue
                # Normalize shift to be within dimension size
                dim_size = input_tensor.size(dim)
                shift = shift % dim_size
                # Create index tensor for slicing
                indices = torch.arange(dim_size, device=input_tensor.device)
                rolled_indices = (indices - shift) % dim_size
                # Prepare index for advanced indexing
                index = [slice(None)] * input_tensor.dim()
                index[dim] = rolled_indices
                # Apply roll using advanced indexing
                result = result.index_select(dim, rolled_indices).view_as(result)
            return result

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = manual_roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows, nW*B, window_size, window_size, C
            x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  
        # W-MSA/SW-MSA, nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = manual_roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # FFN
        shortcut = x
        x = self.norm2(x)
        if self.use_moe:
            x, l_aux = self.mlp(x)
            x = shortcut + self.drop_path(x)
        else:
            x = shortcut + self.drop_path(self.mlp(x))
            
        # pp_rank = mpu.get_pipeline_model_parallel_rank()
        # rank = dist.get_rank()
        # print(f"\033[31m LOG: \033[0m rank {rank}, block {self.idx} forward over")

        if self.use_moe:
            return x, l_aux
        else:
            return x
        
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 192.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=192, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformer(MegatronModule):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 192
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, pre_process, post_process, config:TransformerConfig, img_size=192, patch_size=4, 
                 in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], 
                 blocks_pp_stages=[], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, total_layers=4, 
                 
                 mlp_fc2_bias=True, init_std=0.02, moe_blocks=[[0,1], [0,1], [0], [0]], 
                 num_local_experts=[4, 4, 4, 4], top_value=1, capacity_factor=1.25, 
                 cosine_router=False, normalize_gate=False, use_bpr=True, is_gshard_loss=True, 
                 gate_noise=1.0, cosine_router_dim=256, cosine_router_init_t=0.5,
                 aux_loss_weight=0.01, **kwargs):

        super().__init__(config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.num_classes = num_classes
        
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (4- 1))
        self.mlp_ratio = mlp_ratio
        self.in_chans = in_chans
        self.patch_size = patch_size
        # for moe
        self.aux_loss_weight = aux_loss_weight

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size
        
        if not pre_process: 
            self.patch_embed = None

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 计算此级流水线的 input_resolution
        if len(blocks_pp_stages) == 0:
            assert 1 == pp_world_size
            blocks_pp_stages = [sum(depths)]
        
        assert sum(blocks_pp_stages) == sum(depths)
        pp_world_size = mpu.get_pipeline_model_parallel_world_size()
        print()
        assert len(blocks_pp_stages) == pp_world_size or 1 == pp_world_size

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        blocks_before = sum(blocks_pp_stages[:pp_rank])
        layer_idx = 1
        
        while blocks_before >= sum(depths[:layer_idx]):
            layer_idx += 1
        downsamples_before = layer_idx - 1
        assert downsamples_before <= 3
        
        moe_blocks_idx_overall = []
        for i, blk_idxes_i in enumerate(moe_blocks):
            for blk_idx in blk_idxes_i:
                moe_blocks_idx_overall.append(sum(depths[:i]) + blk_idx)

        self.layers = nn.ModuleList()
        num_heads_i = num_heads[downsamples_before]
        # build layers
        for i in range(blocks_pp_stages[pp_rank]):
            # downsampled input size
            input_resolution = [hw // 2 ** downsamples_before for hw in grid_size]
            dim = int(embed_dim * 2 ** downsamples_before)
            # use moe
            use_moe = (i + blocks_before) in moe_blocks_idx_overall
            # append a block of num_heads_i
            blk = SwinTransformerBlock(dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads_i, window_size=window_size,
                                 shift_size=0 if ((i + blocks_before) % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process, 
                                 
                                 use_moe=use_moe,
                                 num_local_experts=num_local_experts[downsamples_before],
                                 top_value=top_value,
                                 capacity_factor=capacity_factor,
                                 cosine_router=cosine_router,
                                 normalize_gate=normalize_gate,
                                 use_bpr=use_bpr,
                                 is_gshard_loss=is_gshard_loss,
                                 gate_noise=gate_noise,
                                 cosine_router_dim=cosine_router_dim,
                                 cosine_router_init_t=cosine_router_init_t,
                                 mlp_fc2_bias=mlp_fc2_bias,
                                 init_std=init_std,
                                 blk_idx=i+blocks_before)

            self.layers.append(blk)
            assert i + blocks_before < sum(depths[:downsamples_before+1])
            # when the sum of this stages' blocks and before blocks reached specific depth,
            # append a downsample layer
            if i + blocks_before + 1 == sum(depths[:downsamples_before+1]):
                if downsamples_before+1 == len(depths):
                    break   # no downsample at the end
                downsamples_before += 1
                num_heads_i = num_heads[downsamples_before]
                self.layers.append(PatchMerging(input_resolution, dim=dim, norm_layer=norm_layer))
        
        # downsampled seq_length
        self.final_seq = num_patches // patch_size ** downsamples_before

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class SwinTransformerForRingMo(SwinTransformer):
    """swim transformer for ringmo"""

    def __init__(self, 
                 pre_process,
                 post_process,
                 use_checkpoint=False,
                 **kwargs):
        
        self.config =  TransformerConfig(
            pipeline_model_parallel_size=kwargs['pipeline_model_parallel_size'],
            pipeline_dtype = kwargs['pipeline_dtype'],
            num_attention_heads=1,
            num_layers=1,
            hidden_size=1,
            variable_seq_lengths= True, # therefore three params above is useless
            batch_p2p_sync=False ,
            deallocate_pipeline_outputs=True,
            batch_p2p_comm=False,
            moe_token_dispatcher_type='alltoall',
            fp16=True,
            #deallocate_pipeline_outputs=False,
        )
        
        self.use_checkpoint = use_checkpoint

        super(SwinTransformerForRingMo, self).__init__(pre_process, post_process, self.config, **kwargs)
        assert self.num_classes == 0

        self.use_lbp = None
        self.pre_process = pre_process
        self.post_process = post_process
        self.hw = int(self.final_seq ** 0.5)
        self.encoder_stride = 32
        self.input_tensor = None
        self.x_in = None
        self.share_embeddings_and_output_weights = False

    def set_input_tensor(self, input_tensor : torch.Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def _check_input(self, inputs):
        if not self.use_lbp:
            return inputs[0], None, inputs[1]

        return inputs[0], inputs[1], inputs[2]

    def forward(self, inputs:Union[tuple , NoneType]):
        """construct of SwinTransformerForRingMo"""
        # 从流水线前级获取的隐藏状态
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        
        
        x_rec = None 
        x_in = None
        if self.pre_process:
            assert not isinstance(inputs,NoneType) 
            x_in, lbp_in, mask_in = self._check_input(inputs)

            x = torch.mul(x_in, torch.sub(1, mask_in))
            x = self.patch_embed(x)

            if self.ape:
                x = torch.add(x, self.absolute_pos_embed)
            x = self.pos_drop(x)

        if not self.pre_process and self.input_tensor is not None:
            x = self.input_tensor[0] if isinstance(self.input_tensor,List) else self.input_tensor

        # 这里应该串行地把每个block的l_aux加起来，但因为moelayer中把l_aux固定为0了，
        # 所以只是写了单个stage累计l_aux的逻辑
        l_aux = 0.0
        
        # log = f"rank:{dist.get_rank()}, stage:{pp_rank}, device:{x.device}, forward start"
        # print(log)

        for layer_id, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            if(layer_id == len(self.layers)-1):
                rank = dist.get_rank()
                print(f"\033[31m LOG: \033[0m rank {rank}, forward over")

            if isinstance(x, tuple):
                cur_l_aux = x[1]
                x = x[0]
                l_aux = cur_l_aux + l_aux
        
        l_aux = l_aux * self.aux_loss_weight

        if self.post_process:
            x = self.norm(x)
            x = x.permute(0, 2, 1)
            z = torch.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))
            x = self.decoder(z)

        # log = f"rank:{dist.get_rank()}, stage:{pp_rank}, x.device:{x.device}, forward over"
        # print(log)
        
        # not sure if this is right
        # return x_rec , l_aux
        return x 
        
