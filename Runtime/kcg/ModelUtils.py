import torch
import torch.nn as nn
import functools
import contextlib
import sys
import types
import math
from kcg.TorchInjector import *
from kcg.KernelTuneUtils import kernel_compile_tuning
import numpy as np

# ===================== 1. 自定义实现 =====================
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, f_mm = torch.matmul):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.f_mm = f_mm
        # 初始化权重参数 (shape: [out_features, in_features])
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # 初始化偏置参数 (shape: [out_features])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化参数
        self.reset_parameters(isRandom=False)
    
    def reset_parameters(self, isRandom = True):
        if isRandom :
        # 使用与 torch.nn.Linear 相同的初始化方法
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            # 固定权重值（例如全0）
            nn.init.constant_(self.weight, 1.1)
            # 固定偏置值（例如全1）
            if self.bias is not None:
                nn.init.constant_(self.bias, 1.0)
        
    def forward(self, x):
        # 使用 torch.matmul 进行矩阵乘法
        wt = self.weight.t()
        output = self.f_mm(x, wt)
        
        # 添加偏置
        if self.bias is not None:
            output += self.bias
        
        return output
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


# ===================== 2. 模块替换工具 =====================
def replace_LinearToCustomLlinear(module, target_class, replacement_class):
    """递归替换模型中的所有目标模块"""
    for name, child in module.named_children():
        if isinstance(child, target_class):
            # 创建替换模块并复制参数
            replacement = replacement_class(
                child.in_features, 
                child.out_features, 
                child.bias is not None
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            
            # 替换模块
            setattr(module, name, replacement)
        else:
            # 递归处理子模块
            replace_LinearToCustomLlinear(child, target_class, replacement_class)

# ===================== 3. 函数替换工具 =====================
class MatmulReplacer:
    """上下文管理器用于临时替换 torch.matmul"""
    def __init__(self, custom_fn):
        self.custom_fn = custom_fn
        self.original_fn = None
    
    def __enter__(self):
        # 保存原始函数
        self.original_fn = torch.matmul
        
        # 替换为自定义函数
        torch.matmul = self.custom_fn
        
        # 替换特殊方法（用于操作符重载）
        torch.Tensor.__matmul__ = lambda self, other: self.custom_fn(self, other)
        torch.Tensor.__rmatmul__ = lambda self, other: self.custom_fn(other, self)
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复原始函数
        torch.matmul = self.original_fn
        
        # 恢复特殊方法
        torch.Tensor.__matmul__ = self.original_fn
        torch.Tensor.__rmatmul__ = self.original_fn

# # ===================== 4. 模块方法包装器 =====================
# def wrap_forward_method(module, pre_hook=None, post_hook=None):
#     """包装模块的 forward 方法"""
#     original_forward = module.forward
    
#     @functools.wraps(original_forward)
#     def wrapped_forward(*args, **kwargs):
#         if pre_hook:
#             args, kwargs = pre_hook(*args, **kwargs)
        
#         result = original_forward(*args, **kwargs)
        
#         if post_hook:
#             result = post_hook(result)
        
#         return result
    
#     module.forward = wrapped_forward
#     return module

# ===================== 5. 使用示例 =====================
# if __name__ == "__main__":
#     # 示例模型
#     class SampleModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear1 = nn.Linear(10, 20)
#             self.linear2 = nn.Linear(20, 5)
#             self.weight = torch.randn(20, 20)
            
#         def forward(self, x):
#             x = self.linear1(x)
#             x = torch.matmul(x, self.weight)
#             x = self.linear2(x)
#             return x
    
#     # 创建模型实例
#     input_data = torch.randn(3, 10)
#     model = SampleModel()
#     print("原始模型结构:")
#     print(model)
#     out0 = model(input_data)
    
#     # 1. 替换所有 nn.Linear 模块
#     replace_module(model, nn.Linear, CustomLinear)
#     print("\n替换 Linear 后的模型结构:")
#     print(model)
    
#     # 2. 创建包装器模型，在 forward 中应用 matmul 替换
#     class WrappedModel(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model
            
#         def forward(self, *args, **kwargs):
#             with MatmulReplacer(custom_matmul):
#                 return self.model(*args, **kwargs)
    
#     wrapped_model = WrappedModel(model)
    
#     # 3. 测试
#     print("\n测试模型:")
#     output = wrapped_model(input_data)
#     print("输出形状:", output.shape, output)
#     print("输出形状0:", out0.shape, out0)
    
#     if torch.allclose(out0,output,atol=1e-5, rtol=1e-5) :
#         print("out0 and out are same")
#     else:
#         print("out0 and out are diff")
    
#     # # 4. 另一种方法：直接修改 forward 方法
#     # def pre_forward(x):
#     #     print("前向传播开始")
#     #     return (x,), {}
    
#     # def post_forward(result):
#     #     print("前向传播结束")
#     #     return result
    
#     # wrap_forward_method(model, pre_forward, post_forward)
    
#     # print("\n使用直接修改 forward 方法:")
#     # output = model(input_data)
#     # print("输出形状:", output.shape)


def get_op_optimized_model(original_model : nn.Module) -> nn.Module :
    # 创建模型实例
    # input_data = torch.randn(3, 10)
    
    # 1. 替换所有 nn.Linear 模块
    replace_LinearToCustomLlinear(original_model, nn.Linear, CustomLinear)
    
    # 2. 创建包装器模型，在 forward 中应用 matmul 替换
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, *args, **kwargs):
            with MatmulReplacer(OpProxy.f_matmul):
                return self.model(*args, **kwargs)
    
    wrapped_model = WrappedModel(original_model)
    return wrapped_model

def get_baseline_model(original_model : nn.Module) -> nn.Module :
    # 创建模型实例
    # input_data = torch.randn(3, 10)
    
    # 1. 替换所有 nn.Linear 模块
    replace_LinearToCustomLlinear(original_model, nn.Linear, CustomLinear)
    
    # 2. 创建包装器模型，在 forward 中应用 matmul 替换
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, *args, **kwargs):
            import kcg.Operators.triton_matmul as trtion_mm
            with MatmulReplacer(trtion_mm.bmm):
                return self.model(*args, **kwargs)
    
    wrapped_model = WrappedModel(original_model)
    return wrapped_model


def compile_model(devId : int, f_run_model : Callable, collectInfoOnly = False) :
    output = f_run_model()
    print("=== e2e ends : ", output.shape) # 输出形状应为 (1, max_seq_len, vocab_size)
    mmTemplateJson = f'{PathManager.project_dir()}/TuningConfigs/GEMM_cfg_32.json'
    attnJson = f'{PathManager.project_dir()}/TuningConfigs/attn_llama2.json'
    for (Ty , args ) in OpProxy.GetCollectedKernelArgs() :
        assert issubclass(Ty,OpInterface) , f"Ty must be inherited from OpInterface : invalid {Ty.__name__}"
        assert isinstance(args,List)
        assert isinstance(args[-1],torch.dtype)
        ts = None
        if Ty is matmul.MatmulOp :
            import kcg.tuning.NewCfgTest as tune_mm
            ts = tune_mm.getTuneSpaceWithBaseargs(mmTemplateJson,args)
            print("collected mm args = ",args)
        elif Ty is attention.AttentionOp :
            print("----------- attention detected. graph optimizing ... ------")
            import kcg.tuning.attn_FP32_test as tune_att
            print("collected attn args = ",args)
            [batch, head_num, seq_len, headdim  ]= args[0:-1]
            ts = tune_att.getTuneSpace([batch, head_num, seq_len, headdim], attnJson, [])
        else:
            assert False, f"invalid ty : {Ty.__name__}"
        if not collectInfoOnly:
            tuneRes = kernel_compile_tuning(Ty, mmTemplateJson ,devId ,ts)
            OpProxy.registKernel(tuneRes)

def evaluate_model_time(f : Callable) -> Tuple[torch.Tensor, float]:
    ret = []
    for i in range(5):
        ev_st = torch_ns.Event(enable_timing=True)
        ev_et = torch_ns.Event(enable_timing=True)
        ev_st.record()
        out1 = f()
        ev_et.record()
        torch_ns.synchronize()
        eps = ev_st.elapsed_time(ev_et)
        ret.append(eps)
    return (out1, np.median(ret))


def registerPreCompiledKernelByJson(jsonpath : str,devid : int) :
    if not os.path.exists(jsonpath) :
        return False
    info = None
    with open(jsonpath) as f:
        info = json.load(f)
    for k in info['kernels']:
        opty = None
        if k['type'] == 'matmul':
            opty = matmul.MatmulOp
        elif k['type'] == 'attention' :
            opty = attention.AttentionOp
        else:
            assert False, f"try to regist invalid opty {k['type']}"
        registerPreCompiledKernel(opty,k['kernelName'],11,devid)
    return True

def registerPreCompiledKernel(opTy : Type[OpInterface] ,kernlName : str, speedup : float, devId : int) -> str :
    # kernlName = 'kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN2WSWM1WSWN2LSU1Map4GSW0UN8RP0SP0LC1RC0'
    if opTy is matmul.MatmulOp :
        ta = matmul.MatmulTuningArgs()
        op = matmul.MatmulOp()
    elif opTy is attention.AttentionOp :
        ta = attention.AttentionTuningArgs()
        op = attention.AttentionOp()
        
    ta.assignWithKernelName(kernlName)
    info = ta.getCompileNeededInfo()
    p = get_platform_type()
    backendType = EnumBackendType.CUDA
    arch = "80"
    if p == 'dcu' :
        if is_hip() :
            arch = "906"
            backendType = EnumBackendType.HIP
    if p == 'mlu' :
        arch = "370"
        backendType = EnumBackendType.MLU
    if p == 'npu' :
        arch = "370"
        backendType = EnumBackendType.NPU
        
    ba, kernelCfg,  packedKernl = op.Compile(devId,backendType,arch,info)    
    
    tr = TuneResult()
    tr.OpTy = opTy
    tr.bestSpeedup = speedup
    tr.bestKernelConfig = kernelCfg
    tr.bestKernelBaseArg = info.baseArgs
    tr.bestConfigPkl = tr.saveToPkl()
    OpProxy.registKernel(tr)
    return tr.bestConfigPkl


# 3. 释放模型内存的完整步骤
def release_model(model):
    """安全释放模型及其相关资源"""
    # 3.1 删除模型引用
    model.zero_grad()  # 清除梯度缓存
    del model  # 删除模型引用
    
    # 3.2 强制垃圾回收
    gc.collect()  # 回收Python对象
    
    # 3.3 清空CUDA缓存
    if torch_ns.is_available():
        torch_ns.empty_cache()  # 释放未使用的GPU显存
    
    # 3.4 验证内存释放
    if torch_ns.is_available():
        print(f"释放后GPU内存: {torch_ns.memory_allocated()/1e6:.2f} MB")




def create_fixed_embedding(num_embeddings, embedding_dim):
    emb = nn.Embedding(num_embeddings, embedding_dim)
    # 初始化为固定值 (这里用全0，可修改为其他值)
    nn.init.constant_(emb.weight, 1.5)
    # 冻结参数 (可选)
    emb.weight.requires_grad = False
    return emb

        
# def f_att(f_mm : Callable, query : torch.Tensor ,keys : torch.Tensor, values : torch.Tensor, mask : torch.Tensor = None) :
#     scores = f_mm(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # q*k
#     if mask is not None:
#         scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
#     scores = F.softmax(scores.float(), dim=-1).type_as(query)
#     output = f_mm(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
#     return output
