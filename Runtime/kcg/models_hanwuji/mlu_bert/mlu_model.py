import torch, math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from kcg.TorchInjector import *
from kcg.ModelUtils import *

@dataclass
class ModelArgs:
    n_layers = 16
    hidden_dim = 1024
    head_num = 16
    ffn_hidden_dim = 4096
    embedding_dim = 1024
    vocab_size = 32000  # 词汇表大小
    max_position_embeddings = 32000
    type_vocab_size = 32000

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_embeddings, type_vocab_size):
        super(BertEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)  # 
        self.position_embedding = nn.Embedding(max_position_embeddings, embedding_dim)
        self.type_embedding = nn.Embedding(type_vocab_size, embedding_dim)
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


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_hidden_dim, base="our"):
        super(FeedForward, self).__init__()
        self.device = "cpu" if base != "our" else "mlu"
        self.start_linear = nn.Linear(dim, ffn_hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.end_linear = nn.Linear(ffn_hidden_dim, dim, bias=False)
    
    def forward(self, x):
        start = self.start_linear(x).to(self.device)
        gelu = self.gelu(start).to("mlu")
        end = self.end_linear(gelu)
        return end


class Attention(nn.Module):
    def __init__(self, dim, head_num):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.wq = nn.Linear(dim, head_num * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, head_num * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, head_num * self.head_dim, bias=False)
        self.wo = nn.Linear(head_num * self.head_dim, dim, bias=False)
    
    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.head_num, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.head_num, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.head_num, self.head_dim)

        query = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(query)
        output = torch.matmul(scores, values)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class EncodeBlock(nn.Module):
    def __init__(self, dim, head_num, ffn_hidden_dim, base="our"):
        super(EncodeBlock, self).__init__()
        self.attn = Attention(dim, head_num)
        self.attn_laylernorm = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_hidden_dim, base)
        self.ffn_layernorm = LayerNorm(dim)
    
    def forward(self, x, mask):
        attn_x = self.attn(x, mask)
        attn_ln_x = self.attn_laylernorm(x + attn_x)
        ffn_x = self.ffn(attn_ln_x)
        ffn_x = self.ffn_layernorm(attn_ln_x + ffn_x)
        return ffn_x


class BERT(nn.Module):
    def __init__(self, base="our"):
        super(BERT, self).__init__()
        args = ModelArgs()
        self.embeddings = BertEmbedding(args.vocab_size, args.embedding_dim, args.max_position_embeddings, args.type_vocab_size)
        self.encoders = nn.ModuleList()
        for i in range(args.n_layers):
            self.encoders.append(EncodeBlock(args.hidden_dim, args.head_num, args.ffn_hidden_dim, base))
    
    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        h = self.embeddings(tokens)

        mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        for encoder in self.encoders:
            h = encoder(h, mask)
        return h
