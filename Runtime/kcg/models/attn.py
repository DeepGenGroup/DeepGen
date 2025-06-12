import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
from kcg.TorchInjector import *
from kcg.ModelUtils import *

# 定义Llama模型的参数
class ModelArgs:
    dim = 512
    n_layers = 8
    n_heads = 8
    vocab_size = 50000
    max_seq_len = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        scores = self.F_matmul(q, k.transpose(-2, -1)) / (self.args.dim ** 0.5)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / (self.args.dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = self.F_matmul(attn_weights, v)
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

    
def get_model(args : ModelArgs, opProxy = OpProxy() ):
    DeviceInfo.init_cuda(7)
    # proxy = None
    model = LlamaOld(args).to(args.device)
    # model = Llama(args, opProxy).to(args.device)
    return model

def run_model(model, args : ModelArgs, input_ids : torch.Tensor = None) :
    if input_ids is None :
        input_ids = torch.randint(0, args.vocab_size, (1, args.max_seq_len)).to(args.device)
    output = model(input_ids)
    return output



def compile_model( model, args : ModelArgs, devId : int) :
    output = run_model(model,args)
    print("=== e2e ends : ", output.shape) # 输出形状应为 (1, max_seq_len, vocab_size)
    from kcg.KernelTuneUtils import kernel_tuning
    for (Ty , args ) in OpProxy.GetCollectedKernelArgs() :
        assert issubclass(Ty,OpInterface) , f"Ty must be inherited from OpInterface : invalid {Ty.__name__}"
        assert isinstance(args,List)
        assert isinstance(args[-1],torch.dtype)
        ts = None
        if Ty is matmul.MatmulOp :
            import kcg.tuning.NewCfgTest as tune_mm
            ts = tune_mm.getTuneSpaceWithBaseargs('/home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json',args)
        elif Ty is attention.AttentionOp :
            import kcg.tuning.attn_FP32_test as tune_att
            ts = tune_att.getTuneSpace([1,1,1,1],[])
        else:
            assert False, f"invalid ty : {Ty.__name__}"
        kernel_tuning(Ty,'/home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json',devId,ts)
        
        
        # TODO: compile kernel using val
        # TODO: Regist compiled kernel info into OpProxy()


# 旧的实现方式：侵入式，对模型代码做修改替换算子
# if __name__ == "__main__":
#     # 测试Llama模型
#     args = ModelArgs()
#     opProxy = OpProxy()
#     # build model
#     model = get_model(args,opProxy)
    
    
#     # first run, then compile and tuning kernels
#     compile_model(opProxy,model,args)
#     # benchmark
#     run_model(model,args)
    
#     info = DeviceInfo.get_gpu_info()
#     print('info.compute_units = ',info.compute_units)
#     print('info.shared_mem_per_block = ',info.shared_mem_per_block)
#     print('info.regs_per_block = ',info.regs_per_block)
#     print('info.warp_size = ',info.warp_size)
#     print('info.global_mem = ',info.global_mem)
#     print('info.max_thread_per_block = ',info.max_thread_per_block)
#     print('info.clock_rate_khz = ',info.clock_rate_khz)
#     print('info.mem_clock_rate_khz = ',info.mem_clock_rate_khz)
#     print('info.mem_bus_width = ',info.mem_bus_width)
#     print('info.l2_cache_size = ',info.l2_cache_size )
#     print( DeviceInfo.get_device_count() ) 


if __name__ == "__main__":
    # 测试Llama模型
    args = ModelArgs()
    # build model
    model = get_model(args)
    input_ids = torch.randint(0, args.vocab_size, (1, args.max_seq_len)).to(args.device)
    out0 = run_model(model,args, input_ids)
    
    optimizedModel = get_op_optimized_model(model).to(7)
    compile_model(optimizedModel,args, 7)
    out1 = run_model(optimizedModel,args,input_ids)
    if torch.allclose(out0,out1,atol=1e-5,rtol=1e-5):
        print("===== test correct ")
    else:
        print("===== test fail ")
    