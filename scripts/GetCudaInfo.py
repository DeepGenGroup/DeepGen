
import os
import re
import subprocess
import torch

def get_current_device():
    import torch
    return torch.cuda.current_device()

def get_device_capability(idx):
    import torch
    return torch.cuda.get_device_capability(idx)

def get_cuda_capability(capability):
    if capability is None:
        device = get_current_device()
        capability = get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
    return capability

def ptx_get_version() -> int:
    result = subprocess.check_output(["ptxas", "--version"], stderr=subprocess.STDOUT)
    if result is not None:
        version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
        if version is not None:
            cuda_version = version.group(1)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Deepgen only support CUDA 10.0 or higher")

print("cuda capability = ",get_cuda_capability(None))
print("ptxas version = ", ptx_get_version()) 