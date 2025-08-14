import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from kcg.TorchInjector import *
from kcg.ModelUtils import *


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 16
    max_seq_len: int = 2048


f_linear = CustomLinear
# f_linear = nn.Linear
# f_matmul = torch.mm
f_matmul = OpProxy.f_matmul

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
        self.qkv = f_linear(dim, dim * 3, bias=qkv_bias, f_mm=f_matmul)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = f_linear(dim, dim,f_mm=f_matmul)
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
    # 计算需要划分的窗口数量
    H_windows = H // window_size
    W_windows = W // window_size
    
    # 确保输入尺寸能被窗口大小整除
    if H % window_size != 0 or W % window_size != 0:
        # 计算需要填充的尺寸
        H_padded = (H_windows + 1) * window_size
        W_padded = (W_windows + 1) * window_size
        # 对输入进行填充
        x = F.pad(x, (0, 0, 0, W_padded - W, 0, H_padded - H))
        # 更新高度和宽度
        H = H_padded
        W = W_padded
        # 重新计算窗口数量
        H_windows = H // window_size
        W_windows = W // window_size
    
    x = x.view(B, H_windows, window_size, W_windows, window_size, C)
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
    # 计算实际窗口数量
    num_windows = windows.shape[0]
    B = num_windows // ((H * W) // (window_size * window_size))
    
    # 计算填充后的尺寸
    H_padded = (H + window_size - 1) // window_size * window_size
    W_padded = (W + window_size - 1) // window_size * window_size
    
    # 计算窗口网格
    H_windows = H_padded // window_size
    W_windows = W_padded // window_size
    
    # 重塑张量
    x = windows.view(B, H_windows, W_windows, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
    
    # 裁剪回原始尺寸
    x = x[:, :H, :W, :].contiguous()
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = f_linear(in_features, hidden_features , f_mm=f_matmul)
        self.act = act_layer()
        self.fc2 = f_linear(hidden_features, out_features , f_mm=f_matmul)
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
        self.fc1 = f_linear(in_features, hidden_features, f_mm= f_matmul)
        self.act = act_layer()
        self.fc2 = f_linear(hidden_features, in_features, f_mm= f_matmul)
        self.dropout = nn.Dropout(p=mlp_drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 简化的MoE层实现
class MoELayer(nn.Module):
    def __init__(self, gate_type, model_dim, experts, act_layer=nn.GELU, mlp_drop=0.0,
                 normalize_gate=False, **kwargs):  # 移除分布式相关参数
        super().__init__()
        self.gate_type = gate_type
        self.model_dim = model_dim
        self.num_experts = experts['count_per_node']
        self.hidden_size = experts['hidden_size_per_expert']
        self.top_k = gate_type['k']
        self.capacity_factor = gate_type['capacity_factor']
        self.gate_noise = gate_type['gate_noise']
        self.fp32_gate = gate_type.get('fp32_gate', False)
        self.proj_dim = gate_type.get('proj_dim', model_dim)
        self.init_t = gate_type.get('init_t', 1.0)
        
        # 门控网络
        if self.gate_type['type'] == 'top':
            self.gate = f_linear(model_dim, self.num_experts, f_mm= f_matmul)
        elif self.gate_type['type'] == 'cosine_top':
            self.gate = f_linear(model_dim, self.proj_dim, f_mm= f_matmul)
            self.expert_centroids = nn.Parameter(torch.randn(self.num_experts, self.proj_dim))
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(model_dim, self.hidden_size, act_layer, mlp_drop) 
            for _ in range(self.num_experts)
        ])
        
        # 其他参数
        self.normalize_gate = normalize_gate
    
    def forward(self, x):
        # x: [batch_size, seq_len, model_dim]
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.model_dim)  # [batch_size * seq_len, model_dim]
        
        # 门控网络
        if self.gate_type['type'] == 'top':
            gate_scores = self.gate(x_flat)
            if self.normalize_gate:
                gate_scores = F.softmax(gate_scores, dim=-1)
            # 添加噪声
            if self.gate_noise > 0:
                noise = torch.randn_like(gate_scores) * self.gate_noise
                gate_scores = gate_scores + noise
            # 选择前k个专家
            topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        elif self.gate_type['type'] == 'cosine_top':
            proj_x = self.gate(x_flat)
            proj_x = F.normalize(proj_x, dim=-1)
            centroids = F.normalize(self.expert_centroids, dim=-1)
            cosine_sim = f_matmul(proj_x, centroids.t())
            gate_scores = cosine_sim / self.init_t
            if self.normalize_gate:
                gate_scores = F.softmax(gate_scores, dim=-1)
            # 选择前k个专家
            topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # 路由：为每个token选择专家
        selected_expert_idx = topk_indices[:, 0]  # 只选top-1专家
        outputs = []
        for i in range(batch_size * seq_len):
            expert_idx = selected_expert_idx[i]
            expert_output = self.experts[expert_idx](x_flat[i].unsqueeze(0))
            outputs.append(expert_output)
        outputs = torch.cat(outputs, dim=0).view(batch_size, seq_len, self.model_dim)
        
        # 辅助损失（推理时不需要）
        l_aux = torch.tensor(0.0, device=x.device)
        
        return outputs, l_aux

class MoEMlp(nn.Module):
    def __init__(self, in_features, hidden_features, num_local_experts, top_value, act_layer=nn.GELU,
                 capacity_factor=1.25, cosine_router=False, normalize_gate=False, 
                 gate_noise=1.0, cosine_router_dim=256, cosine_router_init_t=0.5, 
                 mlp_drop=0.0, init_std=0.02, mlp_fc2_bias=True, **kwargs):  # 移除分布式相关参数
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_local_experts = num_local_experts
        self.top_value = top_value
        self.capacity_factor = capacity_factor
        self.cosine_router = cosine_router
        self.normalize_gate = normalize_gate
        self.init_std = init_std
        self.mlp_fc2_bias = mlp_fc2_bias

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
            experts={'type': 'ffn', 'count_per_node': num_local_experts, 
                     'hidden_size_per_expert': hidden_features},
            normalize_gate=normalize_gate,
        )
        if not self.mlp_fc2_bias:
            for expert in self._moe_layer.experts:
                expert.fc2.bias.requires_grad = False

    def forward(self, x):
        x, l_aux = self._moe_layer(x)
        return x, l_aux

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, hidden_features={self.hidden_features}, ' \
               f'num_local_experts={self.num_local_experts}, top_value={self.top_value}, ' \
               f'cosine_router={self.cosine_router} normalize_gate={self.normalize_gate}'

# Swin Transformer块
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,
                 use_moe=False, num_local_experts=1, top_value=1, capacity_factor=1.25, cosine_router=False,
                 normalize_gate=False, gate_noise=1.0,
                 cosine_router_dim=256, cosine_router_init_t=0.5, mlp_fc2_bias=True, 
                 init_std=0.02, blk_idx=0, **kwargs):  # 移除分布式相关参数
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
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, 
            window_size=(self.window_size, self.window_size), 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if self.use_moe:
            self.mlp = MoEMlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                act_layer=act_layer, 
                num_local_experts=num_local_experts, 
                top_value=top_value, 
                capacity_factor=capacity_factor, 
                cosine_router=cosine_router,
                normalize_gate=normalize_gate, 
                gate_noise=gate_noise, 
                cosine_router_dim=cosine_router_dim,
                cosine_router_init_t=cosine_router_init_t, 
                mlp_drop=drop,
                mlp_fc2_bias=mlp_fc2_bias, 
                init_std=init_std
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                act_layer=act_layer, 
                drop=drop
            )

        # 创建注意力掩码
        if self.shift_size > 0:
            H, W = self.input_resolution
            # 确保尺寸能被窗口整除
            H_padded = (H + self.window_size - 1) // self.window_size * self.window_size
            W_padded = (W + self.window_size - 1) // self.window_size * self.window_size
            
            img_mask = torch.zeros((1, H_padded, W_padded, 1))
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
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # 逆循环移位
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
            
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # FFN
        shortcut = x
        x = self.norm2(x)
        if self.use_moe:
            x, l_aux = self.mlp(x)
            x = shortcut + self.drop_path(x)
            return x, l_aux
        else:
            x = self.mlp(x)
            x = shortcut + self.drop_path(x)
            return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = f_linear(4 * dim, 2 * dim, bias=False, f_mm=f_matmul)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchEmbed(nn.Module):
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
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保输入尺寸能被patch_size整除
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            # 计算需要填充的尺寸
            H_padded = (H + self.patch_size[0] - 1) // self.patch_size[0] * self.patch_size[0]
            W_padded = (W + self.patch_size[1] - 1) // self.patch_size[1] * self.patch_size[1]
            # 对输入进行填充
            x = F.pad(x, (0, W_padded - W, 0, H_padded - H))
            # 更新高度和宽度
            H = H_padded
            W = W_padded
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x

# 完整的Swin Transformer模型
class SwinTransformer(nn.Module):
    def __init__(self, img_size=192, patch_size=4, in_chans=3, num_classes=1000, 
                 embed_dim=96, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], 
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 moe_blocks=[[0,1], [0,1], [0], [0]], num_local_experts=[4, 4, 4, 4], 
                 top_value=1, capacity_factor=1.25, cosine_router=False, 
                 normalize_gate=False, gate_noise=1.0, cosine_router_dim=256, 
                 cosine_router_init_t=0.5, mlp_fc2_bias=True, init_std=0.02, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.mlp_ratio = mlp_ratio
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        # 图像分块嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size
        self.grid_size = grid_size
        
        # 绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 计算MoE块索引
        moe_blocks_idx_overall = []
        for i, blk_idxes_i in enumerate(moe_blocks):
            for blk_idx in blk_idxes_i:
                moe_blocks_idx_overall.append(sum(depths[:i]) + blk_idx)

        # 构建所有层
        self.layers = nn.ModuleList()
        downsamples = 0
        layer_idx = 0
        
        for i_stage in range(len(depths)):
            # 当前阶段的输入分辨率
            input_resolution = [hw // (2 ** downsamples) for hw in grid_size]
            dim = int(embed_dim * 2 ** downsamples)
            
            # 构建当前阶段的块
            for i_block in range(depths[i_stage]):
                # 确定是否使用MoE
                use_moe = (layer_idx) in moe_blocks_idx_overall
                
                blk = SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads[downsamples], 
                    window_size=window_size,
                    shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_stage]) + i_block],
                    norm_layer=norm_layer,
                    fused_window_process=fused_window_process,
                    use_moe=use_moe,
                    num_local_experts=num_local_experts[downsamples],
                    top_value=top_value,
                    capacity_factor=capacity_factor,
                    cosine_router=cosine_router,
                    normalize_gate=normalize_gate,
                    gate_noise=gate_noise,
                    cosine_router_dim=cosine_router_dim,
                    cosine_router_init_t=cosine_router_init_t,
                    mlp_fc2_bias=mlp_fc2_bias,
                    init_std=init_std,
                    blk_idx=layer_idx
                )
                self.layers.append(blk)
                layer_idx += 1
            
            # 添加下采样层（最后一个阶段除外）
            if i_stage < len(depths) - 1:
                downsample = PatchMerging(
                    input_resolution=input_resolution,
                    dim=dim,
                    norm_layer=norm_layer
                )
                self.layers.append(downsample)
                downsamples += 1
                input_resolution = [hw // 2 for hw in input_resolution]
                dim *= 2

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = f_linear(self.num_features, num_classes, f_mm=f_matmul) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, f_linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # 图像分块嵌入
        x = self.patch_embed(x)
        
        # 添加绝对位置编码
        if self.ape:
            x = x + self.absolute_pos_embed
            
        x = self.pos_drop(x)
        
        # 通过所有层
        for layer in self.layers:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                if isinstance(layer, SwinTransformerBlock):
                    x = layer(x)
                    if isinstance(x, tuple):
                        x = x[0]  # 忽略MoE的辅助损失
                else:
                    x = layer(x)
        
        # 最终归一化
        x = self.norm(x)
        return x

    def forward(self, x):
        # 输入: [B, C, H, W]
        x = self.forward_features(x)  # [B, L, C]
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.avgpool(x)    # [B, C, 1]
        x = torch.flatten(x, 1)  # [B, C]
        
        # 分类头
        x = self.head(x)
        return x

# 使用示例
def create_swin_model():
    return SwinTransformer(
        img_size=192,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[4,8,16,32],  # 注意需要能被embed_dim整除
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        ape=False,
        moe_blocks=[[0,1], [0,1], [0, 2, 4, 6, 8, 10, 12, 14, 16], [0]],
        num_local_experts=[4, 4, 4, 4],
        top_value=1,
        capacity_factor=1.25
    )

# # 测试推理
# if __name__ == "__main__":
#     model = create_swin_model()
#     print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
#     # 创建随机输入图像
#     input_tensor = torch.randn(1, 3, 192, 192)
    
#     # 推理
#     with torch.no_grad():
#         output = model(input_tensor)
#         print(f"输出形状: {output.shape}, 输出值: {output[0, :5]}")