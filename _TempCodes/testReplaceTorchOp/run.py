# import time
# import numpy as np
# import torch

# from torch.utils.cpp_extension import load

# cuda_module = load(name="add",
#                    sources=["add2.cpp", "add2.cu"],
#                    verbose=True)

# # c = a + b (shape: [n])
# n = 1024 * 1024
# a = torch.rand(n, device="cuda:0")
# b = torch.rand(n, device="cuda:0")
# cuda_c = torch.rand(n, device="cuda:0")

# def run_cuda():
#     cuda_module.torch_launch_add2(cuda_c, a, b, n)
#     return cuda_c

# def run_torch():
#     # return None to avoid intermediate GPU memory application
#     # for accurate time statistics
#     a + b
#     torch.mm()
#     return None

# run_cuda() #一个是跑cuda算子
# run_torch() #一个是直接跑torch


import torch
import add  # 导入编译好的扩展模块

# 设置张量大小
n = 1024  # 元素数量

# 创建输入张量 (放在GPU上)
a = torch.rand(n, device='cuda')
b = torch.rand(n, device='cuda')

# 创建输出张量 (必须与输入大小相同，放在GPU上)
c = torch.empty_like(a)

# 调用CUDA扩展函数
add.torch_launch_add2(c, a, b, n)

# 验证结果
expected = a + b
if torch.allclose(c, expected, atol=1e-5):
    print("✅ 结果正确! CUDA扩展工作正常")
    print(f"前5个结果: {c[:5].cpu().numpy()}")
    print(f"前5个参考: {expected[:5].cpu().numpy()}")
else:
    print("❌ 结果错误! 请检查CUDA扩展")
    diff = torch.abs(c - expected)
    print(f"最大差异: {diff.max().item()}")
    print(f"差异位置: {torch.argmax(diff)}")