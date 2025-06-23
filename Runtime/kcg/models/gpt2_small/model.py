import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from kcg.TorchInjector import *
from kcg.ModelUtils import *

# F_baseline_mm = triton_matmul.bmm
F_baseline_mm = torch.matmul

@dataclass
class ModelArgs:
    n_layers = 12
    hidden_dim = 768
    head_num = 12
    ffn_hidden_dim = 3072
    embedding_dim = 768
    vocab_size = 32000  # 词汇表大小
    max_position_embeddings = 1024
    type_vocab_size = 32000


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_embeddings, type_vocab_size):
        super(Embedding, self).__init__()
        # self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.position_embedding = nn.Embedding(max_position_embeddings, embedding_dim)
        # self.type_embedding = nn.Embedding(type_vocab_size, embedding_dim)
        
        # 初始化所有嵌入层为常数
        self.token_embedding = self._create_fixed_embedding(vocab_size, embedding_dim)
        self.position_embedding = self._create_fixed_embedding(max_position_embeddings, embedding_dim)
        self.type_embedding = self._create_fixed_embedding(type_vocab_size, embedding_dim)
        self.layer_norm = LayerNorm(embedding_dim)
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)  # batch_size, seq_len
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        type_embeddings = self.type_embedding(token_type_ids)

        embeddings = token_embeddings + position_embeddings + type_embeddings
        embeddings = self.layer_norm(embeddings)
        return embeddings
    
    def _create_fixed_embedding(self, num_embeddings, embedding_dim):
        emb = nn.Embedding(num_embeddings, embedding_dim)
        # 初始化为固定值 (这里用全0，可修改为其他值)
        nn.init.constant_(emb.weight, 1.5)
        # 冻结参数 (可选)
        emb.weight.requires_grad = False
        return emb

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_hidden_dim, isBaseline):
        super(FeedForward, self).__init__()
        # self.start_linear = nn.Linear(dim, ffn_hidden_dim, bias=False)
        if isBaseline :
            # f_mm = triton_matmul.bmm
            f_mm = F_baseline_mm
        else:
            f_mm = OpProxy.f_matmul
        self.start_linear = CustomLinear(dim, ffn_hidden_dim, bias=False, f_mm=f_mm)
        self.gelu = nn.GELU()
        self.end_linear = CustomLinear(ffn_hidden_dim, dim, bias=False, f_mm=f_mm)
    
    def forward(self, x):
        start = self.start_linear(x)
        gelu = self.gelu(start)
        end = self.end_linear(gelu)
        return end



class Attention(nn.Module):
    def __init__(self, dim, head_num, isBaseline = True):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num
        if isBaseline :
            f_mm = F_baseline_mm
        else:
            f_mm = OpProxy.f_matmul
        self.wq = CustomLinear(dim, head_num * self.head_dim, bias=False, f_mm = f_mm)
        self.wk = CustomLinear(dim, head_num * self.head_dim, bias=False, f_mm = f_mm)
        self.wv = CustomLinear(dim, head_num * self.head_dim, bias=False, f_mm = f_mm)
        self.wo = CustomLinear(head_num * self.head_dim, dim, bias=False, f_mm = f_mm)
        # self.wq = nn.Linear(dim, head_num * self.head_dim, bias=False)
        # self.wk = nn.Linear(dim, head_num * self.head_dim, bias=False)
        # self.wv = nn.Linear(dim, head_num * self.head_dim, bias=False)
        # self.wo = nn.Linear(head_num * self.head_dim, dim, bias=False)
        self.f_matmul = f_mm
    
    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape  # [batch_size, seq_len, hidden_dim]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, self.head_num, self.head_dim)  # [batch_size, seq_len, head_num, head_dim]
        xk = xk.view(batch_size, seq_len, self.head_num, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.head_num, self.head_dim)

        query = xq.transpose(1, 2)  # [batch_size, seq_len, head_dim, head_num]
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        scores = self.f_matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(query)
        output = self.f_matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, dim, head_num, ffn_hidden_dim, isBaseline = True):
        super(TransformerBlock, self).__init__()
        self.attn_layernorm = LayerNorm(dim)
        self.attn = Attention(dim, head_num, isBaseline)
        self.ffn_layernorm = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_hidden_dim, isBaseline)
    
    def forward(self, x, mask):
         a = self.attn(self.attn_layernorm(x), mask)
         x = a + x
         m = self.ffn(self.ffn_layernorm(x))
         x = x + m
         return x



class GPT2(nn.Module):
    def __init__(self,isBaseline = True):
        super(GPT2, self).__init__()
        args = ModelArgs()
        self.embeddings = Embedding(args.vocab_size, args.embedding_dim, args.max_position_embeddings, args.type_vocab_size)
        self.encoders = nn.ModuleList()
        for _ in range(args.n_layers):
            self.encoders.append(TransformerBlock(args.hidden_dim, args.head_num, args.ffn_hidden_dim, isBaseline))
        if isBaseline :
            self.last_linear = CustomLinear(args.hidden_dim, args.vocab_size,bias=True,f_mm= F_baseline_mm)
        else:
            self.last_linear = CustomLinear(args.hidden_dim, args.vocab_size,bias=True,f_mm= OpProxy.f_matmul)
            # self.last_linear = nn.Linear(args.hidden_dim, args.vocab_size)
    
    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        h = self.embeddings(tokens)

        mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        for encoder in self.encoders:
            h = encoder(h, mask)
        output = self.last_linear(h)
        return output
    

