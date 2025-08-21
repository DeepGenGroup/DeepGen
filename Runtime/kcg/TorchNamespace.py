import torch
torch_ns = torch.cuda

try:
    import torch_mlu
    torch_ns = torch.mlu
except ImportError :
    pass
try:
    import torch_npu
    torch_ns = torch.npu
except ImportError :
    pass

