import torch
torch_ns = torch.cuda
__plat_kind = 'dcu'
try:
    import torch_mlu
    torch_ns = torch.mlu
    __plat_kind = 'mlu'
except ImportError :
    pass
try:
    import torch_npu
    torch_ns = torch.npu
    __plat_kind = 'npu'
except ImportError :
    pass

def get_platform_type() :
    return __plat_kind

def dev_name(devid) :
    if get_platform_type() == 'npu' :
        return f"npu:{devid}"
    if get_platform_type() == 'mlu' :
        return f"mlu:{devid}"
    return f"cuda:{devid}"