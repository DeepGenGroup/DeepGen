import multiprocessing
import os
import torch
from typing import List

envname = 'TEST_ENV'

class TestData :
    def __init__(self):
        self.data = 111

def test_proc(o : TestData, index : int) :
    print(f"subpproc env = {os.environ.get(envname)}")
    o.data += index
    print("data = ",o.data)
    
def main() :
    ctx = multiprocessing.get_context('spawn')
    os.environ[envname] = "111"
    print(f"==== main proc : {os.environ.get(envname)}")
    obj = TestData()
    procs = []
    for i in range(0,5):
        p = ctx.Process(target=test_proc,args=[obj,i,])
        p.start()
        procs.append(p)
    for p in procs :
        p.join()
    print('====== main stopped ')
    return

def is_hip():
    import torch
    return torch.version.hip is not None

class DeviceInfo :
    @staticmethod
    def get_cuda_stream(idx=None):
        if idx is None:
            idx = DeviceInfo.get_current_device()
        try:
            # print(f"[D]--------- DeviceInfo.get_current_device() is {idx}")
            from torch._C import _cuda_getCurrentRawStream
            return _cuda_getCurrentRawStream(idx)
        except ImportError:
            import torch
            return torch.cuda.current_stream(idx).cuda_stream

    @staticmethod
    def get_current_device():
        import torch
        return torch.cuda.current_device()

    @staticmethod
    def set_current_device(idx):
        import torch
        torch.cuda.set_device(idx)

    @staticmethod
    def set_visible_devices(devids : List):
        import torch
        import os
        envname = 'CUDA_VISIBLE_DEVICES'
        if is_hip() :
            envname = 'HIP_VISIBLE_DEVICES'
        # if DeviceInfo.get_visible_devices() is None:
        expr = ''
        for id in devids:
            expr += str(id) + ','
        os.environ[envname] = expr[0:-1]
        print(f"==== set {envname}={os.environ[envname]}  =====",flush=True)

    
    @staticmethod
    def get_visible_devices():
        import os
        envname = 'CUDA_VISIBLE_DEVICES'
        if is_hip() :
            envname = 'HIP_VISIBLE_DEVICES'
        return os.environ.get(envname) 
    
    @staticmethod
    def get_device_capability(idx):
        import torch
        return torch.cuda.get_device_capability(idx)
    
    @staticmethod
    def get_warp_size():
        if is_hip():
            return 64
        else:
            return 32

def init_cuda(_devId) :
    DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
    print("init_cuda devid=",_devId)
    DeviceInfo.set_visible_devices([_devId])
    DeviceInfo.set_current_device(_devId)  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
    if not torch.cuda.is_available() :
        torch.cuda.init()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dev = 7
    init_cuda(dev)
    a = torch.randn(1024,1024,dtype=torch.float32, device=f"cuda:{dev}")
    b = torch.randn(1024,1024,dtype=torch.float32, device=f"cuda:{dev}")
    c = torch.empty(1024,1024,dtype=torch.float32, device=f"cuda:{dev}")
    st = torch.Event(enable_timing=True)
    et = torch.Event(enable_timing=True)
    d = torch.matmul(a,b)
    st.record()
    c = torch.matmul(a,b)
    torch.cuda.synchronize()
    et.record()
    if torch.allclose(c,d):
        print("test correct")
    else:
        print("test error")




# import torch
# import torch.nn.functional as F

# def test_attention_operator(batch, seq_len, head_num, head_dim):
#     # 设置随机种子以保证可重复性
#     torch.manual_seed(42)
    
#     # 生成随机输入Q, K, V（形状为batch, head_num, seq_len, head_dim）
#     q = torch.randn(batch, head_num, seq_len, head_dim)
#     k = torch.randn(batch, head_num, seq_len, head_dim)
#     v = torch.randn(batch, head_num, seq_len, head_dim)
    
#     # 自定义Attention算子实现
#     def custom_attention(q, k, v):
#         scale = (q.size(-1)) ** 0.5
#         attn = torch.matmul(q, k.transpose(-2, -1)) / scale
#         attn = F.softmax(attn, dim=-1)
#         output = torch.matmul(attn, v)
#         return output
    
#     # 计算自定义结果
#     custom_output = custom_attention(q, k, v)
    
#     # 使用PyTorch的scaled_dot_product_attention作为参考
#     reference_output = F.scaled_dot_product_attention(q, k, v)
    
#     # 检查前向输出是否接近
#     assert torch.allclose(custom_output, reference_output, atol=1e-6), "前向输出不一致！"
    
#     # 反向传播测试
#     q.requires_grad_(True)
#     k.requires_grad_(True)
#     v.requires_grad_(True)
    
#     # 自定义结果并反向
#     custom_output = custom_attention(q, k, v)
#     loss = custom_output.sum()
#     loss.backward()
    
#     # 检查梯度是否存在
#     assert q.grad is not None, "q的梯度未计算！"
#     assert k.grad is not None, "k的梯度未计算！"
#     assert v.grad is not None, "v的梯度未计算！"
    
#     print("所有测试通过！前向和反向传播正确。")

# # 参数设置
# batch = 2
# seq_len = 10
# head_num = 4
# head_dim = 8

# # 运行测试
# test_attention_operator(batch, seq_len, head_num, head_dim)




