import importlib.util

path = "/home/xiebaokang/projects/mlir/DeepGen/python/utils/lib/gpu_info_cuda.cpython-39-x86_64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("gpu_info", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

getInfo = mod.get_gpu_info

print(getInfo().arch)