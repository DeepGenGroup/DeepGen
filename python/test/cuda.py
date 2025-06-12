import importlib.util

path = "/home/xiebaokang/projects/evaluate/DeepGen/python/utils/lib/gpu_info_rocm.cpython-38-x86_64-linux-gnu.so"
spec = importlib.util.spec_from_file_location("gpu_info", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

getInfo = mod.get_gpu_info

print(getInfo().compute_units)

"""
    name;
    compute_units;
    shared_mem_per_block;
    regs_per_block;
    warp_size;
    arch;
    max_thread_per_block;
    clock_rate_khz;
    mem_clock_rate_khz;
    mem_bus_width;
    l2_cache_size;
    global_mem;
"""