import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
from kcg.TorchInjector import *
from kcg.ModelUtils import *

# 定义Llama模型的参数
# class ModelArgs:
#     dim = 4096
#     n_layers = 8
#     n_heads = 8
#     vocab_size = 16384  # 只能运行 2^ 的尺度
#     # vocab_size = 50000  # 只能运行 2^ 的尺度
#     max_seq_len = 4096
#     device = 'cuda' if torch_ns.is_available() else 'cpu'
class ModelArgs:
    dim = 4096
    n_layers = 8
    n_heads = 8
    vocab_size = 8192  # 只能运行 2^ 的尺度
    # vocab_size = 50000  # 只能运行 2^ 的尺度
    max_seq_len = 2048
    device = 'cuda' if torch_ns.is_available() else 'cpu'

# 定义RMSNorm层
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, args, f_matmul : Callable = torch.matmul):
        super().__init__()
        self.args = args
        self.wq = nn.Linear(args.dim, args.n_heads * (args.dim // args.n_heads), bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * (args.dim // args.n_heads), bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * (args.dim // args.n_heads), bias=False)
        self.wo = nn.Linear(args.n_heads * (args.dim // args.n_heads), args.dim, bias=False)
        self.F_matmul = f_matmul
        
    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        F_matmul = self.F_matmul
        scores = F_matmul(q, k.transpose(-2, -1)) / (self.args.dim ** 0.5)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / (self.args.dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = F_matmul(attn_weights, v)
        # context = torch.matmul(attn_weights, v)
        return self.wo(context)

# 定义前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

# 定义Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, args, f_matmul = torch.matmul):
        super().__init__()
        self.attention_norm = RMSNorm(args.dim)
        self.attention = Attention(args,f_matmul)
        self.ffn_norm = RMSNorm(args.dim)
        self.feedforward = FeedForward(args.dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feedforward(self.ffn_norm(h))

# 定义Llama模型
class Llama(nn.Module):
    def __init__(self, args, opProxy : OpProxy ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if opProxy is None :
            self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        else:
            self.layers = nn.ModuleList([TransformerBlock(args, opProxy.f_matmul) for _ in range(args.n_layers)])
            
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x):
        h = self.tok_embeddings(x)
        for layer in self.layers:
            h = layer(h)
            h = self.norm(h)
        return self.output(h)


# 定义Llama模型
class LlamaOld(nn.Module):
    def __init__(self, args ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x):
        h = self.tok_embeddings(x)
        for layer in self.layers:
            h = layer(h)
            h = self.norm(h)
        return self.output(h)

    
# 如何运行模型
def run_model(model, args : ModelArgs, input_ids : torch.Tensor) :
    input_ids = torch.randint(0, args.vocab_size, (1, args.max_seq_len)).to(args.device)
    def _f() :
        out = model(input_ids)
        return out
    return _f


if __name__ == "__main__":
    PathManager.init(clearPkl=True, clearCache=True, clearTmp=True, clearDump=True)
    # 测试Llama模型
    args = ModelArgs()
    # build model
    DeviceInfo.init_cuda(7)
    model = Llama(args, OpProxy()).to(args.device)
    # model = LlamaOld(args).to(args.device)
    input_ids = torch.randint(0, args.vocab_size, (1, args.max_seq_len)).to(args.device)
    
    optimizedModel = model
    # optimizedModel = get_op_optimized_model(model).to(7)
    compile_model(7, run_model(optimizedModel,args,input_ids))
    
    def f_benchmark():
        return optimizedModel(input_ids)
    def f_base():
        return model(input_ids)
    
    out0,t0 = evaluate_model_time(f_base)
    out1,t1 = evaluate_model_time(f_benchmark)
    
    print(f"=== model run time : ours ={t1}, base = {t0}, speedup : {(t0-t1)/t0}")
    opCallCounter = OpProxy.GetOpCallCounts()
    print("==== call ops :",opCallCounter)
    mmCallCount = opCallCounter[matmul.MatmulOp.__name__]
    
    if torch.allclose(out0,out1,atol=1e-1,rtol=1e-1):
        print("===== model test correct ")
    else:
        diff, maxerr = compare_with_error(out0,out1)
        print(f"===== model test error ! diff, maxerr = {diff, maxerr}")
        print("baseline = ",out0)
        print("user = ", out1)