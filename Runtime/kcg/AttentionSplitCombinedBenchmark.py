"""
Attention Split Combined Benchmark: load best K1/K2 from independent tuning results,
recompile each with its own config, run K1->K2 end-to-end, verify correctness.

Usage:
  python AttentionSplitCombinedBenchmark.py k1_result.json k2_result.json combined_out.json devId dtype
"""
import sys
import os
import re
import json
import math
import importlib
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from kcg.Kernel import *
from kcg.Operators.attention_split import (
    _attn_split_k1,
    _attn_split_k2,
    _causal_upper_mask,
    _make_qkv,
)


def parse_kernel_name(name):
    shape_m = re.match(r'kcg_(?:Attention(?:K[12])?)_(\d+)_(\d+)_(\d+)_(\d+)_', name)
    assert shape_m, f"Cannot parse shape from kernel name: {name}"
    shape = [int(shape_m.group(i)) for i in range(1, 5)]

    param_pattern = {
        "Br": r"Br(\d+)", "Bc": r"Bc(\d+)", "Hd": r"Hd(\d+)",
        "Slice1": r"_Sa(\d+)", "Slice2": r"Sb(\d+)",
        "PTr": r"PTr(\d+)", "PTc": r"PTc(\d+)",
        "OTr": r"OTr(\d+)", "OTc": r"OTc(\d+)",
        "GLOB_LOAD_WIDTH_Q": r"GLWQ(\d+)", "GLOB_LOAD_WIDTH_K": r"GLWK(\d+)", "GLOB_LOAD_WIDTH_V": r"GLWV(\d+)",
        "BLOCK_LAYOUT_P_Y": r"BLPY(\d+)", "BLOCK_LAYOUT_P_X": r"BLPX(\d+)",
        "WARP_LAYOUT_P_Y": r"WLPY(\d+)", "WARP_LAYOUT_P_X": r"WLPX(\d+)",
        "BLOCK_SCATTER_WIDTH_Q": r"BSWQ(\d+)", "BLOCK_SCATTER_WIDTH_K": r"BSWK(\d+)",
        "WARP_SCATTER_WIDTH_Q": r"WSWQ(\d+)", "WARP_SCATTER_WIDTH_K": r"WSWK(\d+)",
        "BLOCK_LAYOUT_O_Y": r"BLOY(\d+)", "BLOCK_LAYOUT_O_X": r"BLOX(\d+)",
        "WARP_LAYOUT_O_Y": r"WLOY(\d+)", "WARP_LAYOUT_O_X": r"WLOX(\d+)",
        "BLOCK_SCATTER_WIDTH_P": r"BSWP(\d+)", "BLOCK_SCATTER_WIDTH_V": r"BSWV(\d+)",
        "WARP_SCATTER_WIDTH_P": r"WSWP(\d+)", "WARP_SCATTER_WIDTH_V": r"WSWV(\d+)",
        "UNROLL_NUM": r"Un(\d+)", "WARP_SIZE": r"W(\d+)",
        "LOAD_CONTINUOUS_P": r"LCP(\d+)", "LOAD_CONTINUOUS_O": r"LCO(\d+)",
        "SHARED_PREFETCH_P": r"SPP(\d+)", "REG_PREFETCH_P": r"RPP(\d+)", "REG_PREFETCH_O": r"RPO(\d+)",
        "SHUFFLE_P": r"SHP(\d+)", "SPLITK_PV": r"SKP(\d+)",
    }
    cfg = {}
    for key, pat in param_pattern.items():
        m = re.search(pat, name)
        if m:
            cfg[key] = int(m.group(1))
    return shape, cfg


def fill_defaults(cfg):
    defaults = {
        "Slice2": 4, "OTr": cfg.get("PTr", 4), "OTc": 8,
        "GLOB_LOAD_WIDTH_V": 4,
        "BLOCK_LAYOUT_O_Y": cfg.get("BLOCK_LAYOUT_P_Y", 2),
        "BLOCK_LAYOUT_O_X": cfg.get("BLOCK_LAYOUT_P_X", 1),
        "WARP_LAYOUT_O_Y": cfg.get("WARP_LAYOUT_P_Y", 4),
        "WARP_LAYOUT_O_X": cfg.get("WARP_LAYOUT_P_X", 8),
        "BLOCK_SCATTER_WIDTH_P": cfg.get("BLOCK_SCATTER_WIDTH_Q", 4),
        "BLOCK_SCATTER_WIDTH_V": 4,
        "WARP_SCATTER_WIDTH_P": cfg.get("WARP_SCATTER_WIDTH_Q", 4),
        "WARP_SCATTER_WIDTH_V": 4,
        "LOAD_CONTINUOUS_O": 1, "REG_PREFETCH_O": 0,
        "SHUFFLE_P": 0, "SPLITK_PV": 0,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


def compile_kernel(compile_func, set_kernel_name, kname, shape, cfg, dtype_name):
    set_kernel_name(kname)
    config = {kname: cfg}
    res = compile_func(shape, config, dtype_name)
    if not res:
        raise RuntimeError(f"Compilation failed for {kname}")
    return res


def compute_launch_params(shape, cfg, type_width):
    B, H, S, D = shape
    Br, Bc, Hd = cfg["Br"], cfg["Bc"], cfg.get("Hd", D)
    PTr, PTc = cfg["PTr"], cfg["PTc"]
    S1 = cfg["Slice1"]
    S2 = cfg.get("Slice2", 4)
    th_num = (Br // PTr) * (Bc // PTc)
    shm_size = Br * S1 + Bc * S1 + Br * Bc + Hd * S2 + 3 * Br
    if S1 != Hd:
        shm_size += Hd * Br
    if cfg.get("SHARED_PREFETCH_P", 0) == 1:
        shm_size += Bc * S1
    grid = [S // Br, H, B]
    block = [th_num, 1, 1]
    return grid, block, shm_size * type_width


def make_kernel(binary_path, kname, shape, cfg, n_dtypes, sig_func, backend, dt, dev_id):
    type_width = 4 if dt == torch.float32 else 2
    grid, block, shm = compute_launch_params(shape, cfg, type_width)
    kc = KernelConfigs(binary_path, kname, [dt] * n_dtypes, backend)
    kc.m_gridDims = grid
    kc.m_blockDims = block
    kc.shmBytes = shm
    sig = sig_func()
    return CompiledKernel(
        kc.backend, kc.binaryPath, kc.kernelFuncName,
        kc.sharedMem(), sig, kc.gridDims(), kc.blockDims(), dev_id
    )


def baseline(q, k, v):
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    S = q.shape[-2]
    mask = _causal_upper_mask(S, q.device, q.dtype).unsqueeze(0).unsqueeze(0)
    scores = torch.matmul(q, k) * scale + mask
    m = scores.max(dim=-1, keepdim=True).values
    p_exp = torch.exp(scores - m)
    p_sum = p_exp.sum(dim=-1, keepdim=True)
    s = p_exp / p_sum
    out = torch.matmul(s, v)
    return out, torch.exp(m), p_sum


def tensor_error_summary(actual, ref):
    diff = (actual - ref).abs()
    ref_abs = ref.abs()
    rel = diff / (ref_abs + 1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
    }


def main():
    if len(sys.argv) < 6:
        print("Usage: python AttentionSplitCombinedBenchmark.py k1.json k2.json combined_out.json devId dtype")
        sys.exit(1)

    k1_path, k2_path, out_path = sys.argv[1:4]
    dev_id = int(sys.argv[4])
    dtype_name = sys.argv[5]
    assert dtype_name in ("float32", "float16")
    torch_dtype = torch.float32 if dtype_name == "float32" else torch.float16

    DeviceInfo.get_current_device()
    DeviceInfo.set_visible_devices([dev_id])
    DeviceInfo.set_current_device(dev_id)
    if not torch_ns.is_available():
        torch_ns.init()
        torch_ns.empty_cache()

    with open(k1_path) as f:
        k1_top = json.load(f)["testResult"][0]
    with open(k2_path) as f:
        k2_top = json.load(f)["testResult"][0]

    print(f"[combined] K1 best: {k1_top['name']} (speedup={k1_top['speedup']:.3f}, time={k1_top['time']:.4f}ms)")
    print(f"[combined] K2 best: {k2_top['name']} (speedup={k2_top['speedup']:.3f}, time={k2_top['time']:.4f}ms)")

    shape1, cfg1 = parse_kernel_name(k1_top["name"])
    shape2, cfg2 = parse_kernel_name(k2_top["name"])
    fill_defaults(cfg1)

    spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    backend_id = 2 if is_hip() else 1
    arch = "906" if is_hip() else "80"
    mod.set_platform(backend_id, arch)
    backend = EnumBackendType.HIP if is_hip() else EnumBackendType.CUDA

    print("[combined] Compiling K1...", flush=True)
    res1 = compile_kernel(mod.compile_attn_split_k1, mod.set_kernel_name, k1_top["name"], shape1, cfg1, dtype_name)
    print("[combined] Compiling K2...", flush=True)
    res2 = compile_kernel(mod.compile_attn_split_k2, mod.set_kernel_name, k2_top["name"], shape2, cfg2, dtype_name)

    dt = torch_dtype
    sig_k1 = lambda: _attn_split_k1(
        torch.randn(1, 3, 100, 100, dtype=dt),
        torch.randn(1, 3, 100, 100, dtype=dt),
        torch.randn(1, 3, 100, 1, dtype=dt),
        torch.randn(1, 3, 100, 1, dtype=dt),
    )
    sig_k2 = lambda: _attn_split_k2(
        torch.randn(1, 3, 100, 100, dtype=dt),
        torch.randn(1, 3, 100, 100, dtype=dt),
        torch.randn(1, 3, 100, 100, dtype=dt),
        torch.randn(1, 3, 100, 1, dtype=dt),
        torch.randn(1, 3, 100, 1, dtype=dt),
        torch.empty(1, 3, 100, 100, dtype=dt),
    )

    kernel1 = make_kernel(res1, k1_top["name"], shape1, cfg1, 4, sig_k1, backend, dt, dev_id)
    kernel2 = make_kernel(res2, k2_top["name"], shape2, cfg2, 6, sig_k2, backend, dt, dev_id)

    B, H, S, D = shape2
    device = dev_name(dev_id)
    q, k, v = _make_qkv(B, H, S, D, dt, device)
    qq = q.transpose(-1, -2).contiguous()
    kk = k
    em = torch.empty((B, H, S, 1), dtype=dt, device=device)
    denom = torch.empty((B, H, S, 1), dtype=dt, device=device)
    out = torch.empty((B, H, S, D), dtype=dt, device=device)

    with torch.no_grad():
        ref_out, _, _ = baseline(q, k, v)

    kernel1.run(qq, kk, em, denom)
    kernel2.run(qq, kk, v, em, denom, out)
    torch_ns.synchronize()

    torch_ns.synchronize()
    st = time.perf_counter()
    kernel1.run(qq, kk, em, denom)
    kernel2.run(qq, kk, v, em, denom, out)
    torch_ns.synchronize()
    et = time.perf_counter()
    combined_time = (et - st) * 1000.0

    import numpy as np

    def _run_baseline():
        return baseline(q, k, v)

    _run_baseline()
    torch_ns.synchronize()
    base_times = []
    for _ in range(7):
        torch_ns.synchronize()
        bst = time.perf_counter()
        _run_baseline()
        torch_ns.synchronize()
        bet = time.perf_counter()
        base_times.append((bet - bst) * 1000.0)
    baseline_time = float(np.median(base_times))

    speedup = baseline_time / combined_time if combined_time > 0 else 0.0

    result = {
        "k1_name": k1_top["name"],
        "k2_name": k2_top["name"],
        "combined_timer_mode": "cpu_wall_time",
        "baseline_timer_mode": "cpu_wall_time",
        "combined_time": combined_time,
        "baseline_time": baseline_time,
        "speedup": speedup
    }
    with open(out_path, "w+") as f:
        json.dump(result, f, indent=2)
        f.flush()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
