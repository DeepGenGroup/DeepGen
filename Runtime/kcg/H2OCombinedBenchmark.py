"""
H2O Combined Benchmark: load best K1/K2/K3 from independent tuning results,
recompile each with its own config, run K1->K2->K3 end-to-end, verify correctness.

Usage:
  python H2OCombinedBenchmark.py k1_result.json k2_result.json k3_result.json combined_out.json devId dtype
"""
import sys
import os
import re
import json
import math
import importlib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from kcg.Kernel import *
from kcg.Operators.attention_h2o import (
    _h2o_split_k1, _h2o_split_k2, _h2o_split_k3, _causal_upper_mask,
)


def parse_kernel_name(name):
    """Parse tiling config from kernel name like kcg_H2OK1_1_32_4096_64_Br32Bc64..."""
    shape_m = re.match(r'kcg_(?:Attention|H2OK[123])_(\d+)_(\d+)_(\d+)_(\d+)_', name)
    assert shape_m, f"Cannot parse shape from kernel name: {name}"
    shape = [int(shape_m.group(i)) for i in range(1, 5)]

    param_pattern = {
        'Br': r'Br(\d+)', 'Bc': r'Bc(\d+)', 'Hd': r'Hd(\d+)',
        'Slice1': r'_Sa(\d+)', 'Slice2': r'Sb(\d+)',
        'PTr': r'PTr(\d+)', 'PTc': r'PTc(\d+)',
        'OTr': r'OTr(\d+)', 'OTc': r'OTc(\d+)',
        'GLOB_LOAD_WIDTH_Q': r'GLWQ(\d+)', 'GLOB_LOAD_WIDTH_K': r'GLWK(\d+)', 'GLOB_LOAD_WIDTH_V': r'GLWV(\d+)',
        'BLOCK_LAYOUT_P_Y': r'BLPY(\d+)', 'BLOCK_LAYOUT_P_X': r'BLPX(\d+)',
        'WARP_LAYOUT_P_Y': r'WLPY(\d+)', 'WARP_LAYOUT_P_X': r'WLPX(\d+)',
        'BLOCK_SCATTER_WIDTH_Q': r'BSWQ(\d+)', 'BLOCK_SCATTER_WIDTH_K': r'BSWK(\d+)',
        'WARP_SCATTER_WIDTH_Q': r'WSWQ(\d+)', 'WARP_SCATTER_WIDTH_K': r'WSWK(\d+)',
        'BLOCK_LAYOUT_O_Y': r'BLOY(\d+)', 'BLOCK_LAYOUT_O_X': r'BLOX(\d+)',
        'WARP_LAYOUT_O_Y': r'WLOY(\d+)', 'WARP_LAYOUT_O_X': r'WLOX(\d+)',
        'BLOCK_SCATTER_WIDTH_P': r'BSWP(\d+)', 'BLOCK_SCATTER_WIDTH_V': r'BSWV(\d+)',
        'WARP_SCATTER_WIDTH_P': r'WSWP(\d+)', 'WARP_SCATTER_WIDTH_V': r'WSWV(\d+)',
        'UNROLL_NUM': r'Un(\d+)', 'WARP_SIZE': r'W(\d+)',
        'LOAD_CONTINUOUS_P': r'LCP(\d+)', 'LOAD_CONTINUOUS_O': r'LCO(\d+)',
        'SHARED_PREFETCH_P': r'SPP(\d+)', 'REG_PREFETCH_P': r'RPP(\d+)', 'REG_PREFETCH_O': r'RPO(\d+)',
        'SHUFFLE_P': r'SHP(\d+)', 'SPLITK_PV': r'SKP(\d+)',
    }
    cfg = {}
    for key, pat in param_pattern.items():
        m = re.search(pat, name)
        if m:
            cfg[key] = int(m.group(1))
    return shape, cfg


def fill_defaults(cfg):
    """Fill missing O-related params with safe defaults (for K1/K2 configs)."""
    defaults = {
        'Slice2': 4, 'OTr': cfg.get('PTr', 4), 'OTc': 8,
        'GLOB_LOAD_WIDTH_V': 4,
        'BLOCK_LAYOUT_O_Y': cfg.get('BLOCK_LAYOUT_P_Y', 2),
        'BLOCK_LAYOUT_O_X': cfg.get('BLOCK_LAYOUT_P_X', 1),
        'WARP_LAYOUT_O_Y': cfg.get('WARP_LAYOUT_P_Y', 4),
        'WARP_LAYOUT_O_X': cfg.get('WARP_LAYOUT_P_X', 8),
        'BLOCK_SCATTER_WIDTH_P': cfg.get('BLOCK_SCATTER_WIDTH_Q', 4),
        'BLOCK_SCATTER_WIDTH_V': 4,
        'WARP_SCATTER_WIDTH_P': cfg.get('WARP_SCATTER_WIDTH_Q', 4),
        'WARP_SCATTER_WIDTH_V': 4,
        'LOAD_CONTINUOUS_O': 1, 'REG_PREFETCH_O': 0,
        'SHUFFLE_P': 0, 'SPLITK_PV': 0,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    br, ptc = cfg['Br'], cfg['PTc']
    cfg.setdefault('blockDim.x', br // cfg['PTr'] * (cfg['Bc'] // ptc))
    cfg.setdefault('blockDim.y', 1)
    cfg.setdefault('blockDim.z', 1)
    return cfg


def compile_kernel(compile_func, set_kernel_name, kname, shape, cfg):
    """Compile a single kernel and return (binaryPath, funcName, gridDims, blockDims, shmBytes)."""
    set_kernel_name(kname)
    config = {kname: cfg}
    dtype_str = "float32"
    res = compile_func(shape, config, dtype_str)
    if not res:
        raise RuntimeError(f"Compilation failed for {kname}")
    return res


def main():
    if len(sys.argv) < 7:
        print("Usage: python H2OCombinedBenchmark.py k1.json k2.json k3.json combined_out.json devId dtype")
        sys.exit(1)

    k1_path, k2_path, k3_path, out_path = sys.argv[1:5]
    dev_id = int(sys.argv[5])
    dtype_name = sys.argv[6]
    assert dtype_name in ("float32", "float16")
    torch_dtype = torch.float32 if dtype_name == "float32" else torch.float16

    DeviceInfo.get_current_device()
    DeviceInfo.set_visible_devices([dev_id])
    DeviceInfo.set_current_device(dev_id)
    if not torch_ns.is_available():
        torch_ns.init()
        torch_ns.empty_cache()

    with open(k1_path) as f: k1_top = json.load(f)["testResult"][0]
    with open(k2_path) as f: k2_top = json.load(f)["testResult"][0]
    with open(k3_path) as f: k3_top = json.load(f)["testResult"][0]

    print(f"[combined] K1 best: {k1_top['name']} (speedup={k1_top['speedup']:.3f}, time={k1_top['time']:.4f}ms)")
    print(f"[combined] K2 best: {k2_top['name']} (speedup={k2_top['speedup']:.3f}, time={k2_top['time']:.4f}ms)")
    print(f"[combined] K3 best: {k3_top['name']} (speedup={k3_top['speedup']:.3f}, time={k3_top['time']:.4f}ms)")

    shape1, cfg1 = parse_kernel_name(k1_top['name'])
    shape2, cfg2 = parse_kernel_name(k2_top['name'])
    shape3, cfg3 = parse_kernel_name(k3_top['name'])

    fill_defaults(cfg1)
    fill_defaults(cfg2)

    spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    compile_k1 = mod.compile_h2o_split_k1
    compile_k2 = mod.compile_h2o_split_k2
    compile_k3 = mod.compile_h2o_split_k3
    set_kernel_name = mod.set_kernel_name

    backend_id = 2 if is_hip() else 1
    arch = "906" if is_hip() else "80"
    mod.set_platform(backend_id, arch)
    backend = EnumBackendType.HIP if is_hip() else EnumBackendType.CUDA

    k1_name = k1_top['name']
    k2_name = k2_top['name']
    k3_name = k3_top['name']

    print("[combined] Compiling K1...", flush=True)
    res1 = compile_kernel(compile_k1, set_kernel_name, k1_name, shape1, cfg1)
    print("[combined] Compiling K2...", flush=True)
    res2 = compile_kernel(compile_k2, set_kernel_name, k2_name, shape2, cfg2)
    print("[combined] Compiling K3...", flush=True)
    res3 = compile_kernel(compile_k3, set_kernel_name, k3_name, shape3, cfg3)

    dt = torch_dtype
    type_width = 4 if dt == torch.float32 else 2

    def compute_launch_params(shape, cfg):
        """Compute gridDims, blockDims, shmBytes from config."""
        B, H, S, D = shape
        Br, Bc, Hd = cfg['Br'], cfg['Bc'], cfg.get('Hd', D)
        PTr, PTc = cfg['PTr'], cfg['PTc']
        S1 = cfg['Slice1']
        S2 = cfg.get('Slice2', 4)
        th_num = (Br // PTr) * (Bc // PTc)
        shm_size = Br * S1 + Bc * S1 + Br * Bc + Hd * S2 + 3 * Br
        if S1 != Hd:
            shm_size += Hd * Br
        if cfg.get('SHARED_PREFETCH_P', 0) == 1:
            shm_size += Bc * S1
        grid = [S // Br, H, B]
        block = [th_num, 1, 1]
        return grid, block, shm_size * type_width

    def make_kernel(binary_path, kname, shape, cfg, n_dtypes, sig_func, target_dev=None):
        if target_dev is None:
            target_dev = dev_id
        grid, block, shm = compute_launch_params(shape, cfg)
        kc = KernelConfigs(binary_path, kname, [dt] * n_dtypes, backend)
        kc.m_gridDims = grid
        kc.m_blockDims = block
        kc.shmBytes = shm
        sig = sig_func()
        return CompiledKernel(kc.backend, kc.binaryPath, kc.kernelFuncName,
                              kc.sharedMem(), sig, kc.gridDims(), kc.blockDims(), target_dev)

    sig_k1 = lambda: _h2o_split_k1(torch.randn(1,3,100,100), torch.randn(1,3,100,100),
                                     torch.randn(1,3,100,1), torch.randn(1,3,100,1))
    sig_k2 = lambda: _h2o_split_k2(torch.randn(1,3,100,100), torch.randn(1,3,100,100),
                                     torch.randn(1,3,100,1), torch.randn(1,3,100,1), torch.randn(1,3,100))
    sig_k3 = lambda: _h2o_split_k3(torch.randn(1,3,100,100), torch.randn(1,3,100,100), torch.randn(1,3,100,100),
                                     torch.randn(1,3,100,1), torch.randn(1,3,100,1), torch.empty(1,3,100,100))

    kernel1 = make_kernel(res1, k1_name, shape1, cfg1, 4, sig_k1)
    kernel2 = make_kernel(res2, k2_name, shape2, cfg2, 5, sig_k2)
    kernel3 = make_kernel(res3, k3_name, shape3, cfg3, 6, sig_k3)

    B, H, S, D = shape3
    device = dev_name(dev_id)

    q_base = torch.ones((B, H, S, D), dtype=dt, device=device)
    k_base = torch.ones((B, H, D, S), dtype=dt, device=device)
    v_base = torch.ones((B, H, S, D), dtype=dt, device=device)
    qq = q_base.transpose(-1, -2).contiguous()
    kk = k_base
    em = torch.empty((B, H, S, 1), dtype=dt, device=device)
    denom = torch.empty((B, H, S, 1), dtype=dt, device=device)
    row_sum = torch.empty((B, H, S), dtype=dt, device=device)
    out = torch.empty((B, H, S, D), dtype=dt, device=device)

    # === PyTorch baseline ===
    scale = 1.0 / math.sqrt(float(D))
    mask = _causal_upper_mask(S, device, dt).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_s = q_base * scale
        scores = torch.matmul(q_s, k_base) + mask
        m_val = scores.max(dim=-1, keepdim=True).values
        ref_em = torch.exp(m_val)
        sum_ex = torch.exp(scores).sum(dim=-1, keepdim=True)
        ref_denom = sum_ex / ref_em
        ref_p = torch.exp(scores) / (ref_em * ref_denom)
        ref_row_sum = ref_p.sum(dim=2, keepdim=False)
        ref_out = torch.matmul(ref_p, v_base)

    # === Warmup ===
    kernel1.run(qq, kk, em, denom)
    kernel2.run(kk, qq, em, denom, row_sum)
    kernel3.run(qq, kk, v_base, em, denom, out)
    torch_ns.synchronize()

    # === Timed: K1 → K2 → K3 (serial) ===
    st = torch_ns.Event(enable_timing=True)
    et = torch_ns.Event(enable_timing=True)
    st.record()
    kernel1.run(qq, kk, em, denom)
    kernel2.run(kk, qq, em, denom, row_sum)
    kernel3.run(qq, kk, v_base, em, denom, out)
    et.record()
    torch_ns.synchronize()
    combined_time = st.elapsed_time(et)

    # === Baseline timing (warmup + 7 runs, take median) ===
    import numpy as np
    def _run_baseline():
        scores = torch.matmul(q_base, k_base) / math.sqrt(float(D))
        scores = scores + mask
        m_val = scores.max(dim=-1, keepdim=True).values
        p_exp = torch.exp(scores - m_val)
        p_sum = p_exp.sum(dim=-1, keepdim=True)
        s = p_exp / p_sum
        out_bl = torch.matmul(s, v_base)
        rs_bl = s.sum(dim=2, keepdim=False)
        return out_bl, rs_bl
    _run_baseline()
    torch_ns.synchronize()
    base_times = []
    for _ in range(7):
        bst = torch_ns.Event(enable_timing=True)
        bet = torch_ns.Event(enable_timing=True)
        bst.record()
        _run_baseline()
        bet.record()
        torch_ns.synchronize()
        base_times.append(bst.elapsed_time(bet))
    baseline_time = float(np.median(base_times))

    # === Correctness check ===
    k1_ok = torch.allclose(denom, ref_denom, rtol=1e-3, atol=1e-3)
    k2_ok = torch.allclose(row_sum, ref_row_sum, rtol=1e-3, atol=1e-3)
    k3_ok = torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3)

    speedup = baseline_time / combined_time if combined_time > 0 else 0

    print(f"[combined] K1 correct: {k1_ok}")
    print(f"[combined] K2 correct: {k2_ok}")
    print(f"[combined] K3 correct: {k3_ok}")
    print(f"[combined] Combined time: {combined_time:.4f} ms")
    print(f"[combined] Baseline time: {baseline_time:.4f} ms")
    print(f"[combined] Speedup: {speedup:.4f}")

    result = {
        "k1": {"name": k1_name, "time": k1_top["time"], "individual_speedup": k1_top["speedup"]},
        "k2": {"name": k2_name, "time": k2_top["time"], "individual_speedup": k2_top["speedup"]},
        "k3": {"name": k3_name, "time": k3_top["time"], "individual_speedup": k3_top["speedup"]},
        "combined_time": combined_time,
        "baseline_time": baseline_time,
        "combined_speedup": speedup,
        "k1_correct": k1_ok,
        "k2_correct": k2_ok,
        "k3_correct": k3_ok,
        "all_correct": k1_ok and k2_ok and k3_ok,
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[combined] Results saved to {out_path}")


if __name__ == '__main__':
    main()
