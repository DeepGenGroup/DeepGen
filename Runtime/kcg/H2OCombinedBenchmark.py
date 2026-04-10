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
import statistics
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from kcg.Kernel import *
from kcg.Operators.attention_h2o import (
    _h2o_split_k1, _h2o_split_k2, _h2o_split_k3, _causal_upper_mask, _make_qkv,
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


def compile_kernel(compile_func, set_kernel_name, kname, shape, cfg, dtype_name):
    """Compile a single kernel and return (binaryPath, funcName, gridDims, blockDims, shmBytes)."""
    set_kernel_name(kname)
    config = {kname: cfg}
    res = compile_func(shape, config, dtype_name)
    if not res:
        raise RuntimeError(f"Compilation failed for {kname}")
    return res


def tensor_error_summary(actual, ref):
    diff = (actual - ref).abs()
    ref_abs = ref.abs()
    rel = diff / (ref_abs + 1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
    }


def measure_gpu_time(run_once):
    start = torch_ns.Event(enable_timing=True)
    end = torch_ns.Event(enable_timing=True)
    torch_ns.synchronize()
    start.record()
    run_once()
    end.record()
    torch_ns.synchronize()
    return float(start.elapsed_time(end))


def measure_gpu_median(run_once, rounds=7):
    run_once()
    torch_ns.synchronize()
    times = [measure_gpu_time(run_once) for _ in range(rounds)]
    return float(statistics.median(times))


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
    res1 = compile_kernel(compile_k1, set_kernel_name, k1_name, shape1, cfg1, dtype_name)
    print("[combined] Compiling K2...", flush=True)
    res2 = compile_kernel(compile_k2, set_kernel_name, k2_name, shape2, cfg2, dtype_name)
    print("[combined] Compiling K3...", flush=True)
    res3 = compile_kernel(compile_k3, set_kernel_name, k3_name, shape3, cfg3, dtype_name)

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

    def _make_sig_qkv():
        return _make_qkv(1, 3, 100, 100, dt, 'cpu')

    def sig_k1():
        q_sig, k_sig, _ = _make_sig_qkv()
        return _h2o_split_k1(
            q_sig, k_sig,
            torch.randn(1, 3, 100, 1, dtype=dt), torch.randn(1, 3, 100, 1, dtype=dt))

    def sig_k2():
        q_sig, k_sig, _ = _make_sig_qkv()
        return _h2o_split_k2(
            k_sig, q_sig,
            torch.randn(1, 3, 100, 1, dtype=dt), torch.randn(1, 3, 100, 1, dtype=dt),
            torch.randn(1, 3, 100, dtype=dt))

    def sig_k3():
        q_sig, k_sig, v_sig = _make_sig_qkv()
        return _h2o_split_k3(
            q_sig, k_sig, v_sig,
            torch.randn(1, 3, 100, 1, dtype=dt), torch.randn(1, 3, 100, 1, dtype=dt),
            torch.empty(1, 3, 100, 100, dtype=dt))

    kernel1 = make_kernel(res1, k1_name, shape1, cfg1, 4, sig_k1)
    kernel2 = make_kernel(res2, k2_name, shape2, cfg2, 5, sig_k2)
    kernel3 = make_kernel(res3, k3_name, shape3, cfg3, 6, sig_k3)

    B, H, S, D = shape3
    device = dev_name(dev_id)

    q_base, k_base, v_base = _make_qkv(B, H, S, D, dt, device)
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
        scores = torch.matmul(q_base, k_base) * scale + mask
        m_val = scores.max(dim=-1, keepdim=True).values
        p_exp = torch.exp(scores - m_val)
        den = p_exp.sum(dim=-1, keepdim=True)
        ref_p = p_exp / den
        ref_row_sum = ref_p.sum(dim=2, keepdim=False)
        ref_out = torch.matmul(ref_p, v_base)

    # === Warmup ===
    kernel1.run(qq, kk, em, denom)
    kernel2.run(kk, qq, em, denom, row_sum)
    kernel3.run(qq, kk, v_base, em, denom, out)
    torch_ns.synchronize()

    # === Timed: K1 → K2 → K3 (serial) ===
    def _run_combined():
        kernel1.run(qq, kk, em, denom)
        kernel2.run(kk, qq, em, denom, row_sum)
        kernel3.run(qq, kk, v_base, em, denom, out)

    combined_time = measure_gpu_median(_run_combined, rounds=1)

    # === Baseline timing (warmup + 7 runs, take median) ===
    def _run_baseline():
        scores = torch.matmul(q_base, k_base) * scale + mask
        m_val = scores.max(dim=-1, keepdim=True).values
        p_exp = torch.exp(scores - m_val)
        den = p_exp.sum(dim=-1, keepdim=True)
        s = p_exp / den
        out_bl = torch.matmul(s, v_base)
        rs_bl = s.sum(dim=2, keepdim=False)
        return out_bl, rs_bl

    baseline_time = measure_gpu_median(_run_baseline, rounds=7)

    # === Correctness check (final outputs only) ===
    row_sum_correct = torch.allclose(row_sum, ref_row_sum, rtol=1e-3, atol=1e-3)
    out_correct = torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3)
    row_sum_error = tensor_error_summary(row_sum, ref_row_sum)
    out_error = tensor_error_summary(out, ref_out)
    overall_correct = row_sum_correct and out_correct

    speedup = baseline_time / combined_time if combined_time > 0 else 0

    result = {
        "k1_name": k1_name,
        "k2_name": k2_name,
        "k3_name": k3_name,
        "combined_time": combined_time,
        "baseline_time": baseline_time,
        "speedup": speedup
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
        f.flush()

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
