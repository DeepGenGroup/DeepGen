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
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.f_mm = torch.matmul
        # 初始化权重参数 (shape: [out_features, in_features])
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # 初始化偏置参数 (shape: [out_features])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        # 使用与 torch.nn.Linear 相同的初始化方法
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
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


def compile_model(devId : int, f_run_model : Callable) :
    output = f_run_model()
    print("=== e2e ends : ", output.shape) # 输出形状应为 (1, max_seq_len, vocab_size)
    mmTemplateJson = f'{PathManager.project_dir()}/TuningConfigs/GEMM_cfg_32.json'
    for (Ty , args ) in OpProxy.GetCollectedKernelArgs() :
        assert issubclass(Ty,OpInterface) , f"Ty must be inherited from OpInterface : invalid {Ty.__name__}"
        assert isinstance(args,List)
        assert isinstance(args[-1],torch.dtype)
        ts = None
        if Ty is matmul.MatmulOp :
            import kcg.tuning.NewCfgTest as tune_mm
            ts = tune_mm.getTuneSpaceWithBaseargs(mmTemplateJson,args)
            print("collected args = ",args)
        elif Ty is attention.AttentionOp :
            import kcg.tuning.attn_FP32_test as tune_att
            ts = tune_att.getTuneSpace([1,1,1,1],[])
        else:
            assert False, f"invalid ty : {Ty.__name__}"
        tuneRes = kernel_compile_tuning(Ty, mmTemplateJson ,devId ,ts)
        OpProxy.registKernel(tuneRes)

def evaluate_model_time(f : Callable) -> Tuple[torch.Tensor, float]:
    ret = []
    for i in range(5):
        ev_st = torch.cuda.Event(enable_timing=True)
        ev_et = torch.cuda.Event(enable_timing=True)
        ev_st.record()
        out1 = f()
        ev_et.record()
        torch.cuda.synchronize()
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
    backendType = EnumBackendType.CUDA
    arch = "80"
    if is_hip() :
        arch = "906"
        backendType = EnumBackendType.HIP
    ba, kernelCfg,  packedKernl = op.Compile(devId,backendType,arch,info)    
    
    tr = TuneResult()
    tr.OpTy = opTy
    tr.bestSpeedup = speedup
    tr.bestKernelConfig = kernelCfg
    tr.bestKernelBaseArg = info.baseArgs
    tr.bestConfigPkl = tr.saveToPkl()
    OpProxy.registKernel(tr)
    return tr.bestConfigPkl