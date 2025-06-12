from kcg.Utils import *

ret= build("testCudaInfo","CudaGPUInfo.cc","/home/xushilong/DeepGen/Runtime/kcg/gpuInfoCode")
mod_name='gpu_info'

print(ret)

import importlib.util

spec = importlib.util.spec_from_file_location("gpu_info", ret)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
get_gpu_info = getattr(mod, "get_gpu_info")
get_device_count = getattr(mod, "get_device_count")
print( get_gpu_info() )
print( get_device_count() )