import importlib
import math
import torch
from kcg.Kernel import *
from kcg.Operators.attention import (
    AttentionBaseArgs, AttentionTuningArgs,
)


def _causal_upper_mask(S, device, dtype):
    return torch.where(
        torch.triu(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=1),
        torch.full((S, S), float("-inf"), device=device, dtype=dtype),
        torch.zeros((S, S), device=device, dtype=dtype),
    )


@kcg_kernel
def _h2o_split_k1_kernel(q_ptr, k_ptr, em_ptr, denom_ptr):
    'DUMP CODES'
    pass


@kcg_kernel
def _h2o_split_k2_kernel(k_ptr, q_ptr, em_ptr, denom_ptr, row_sum_ptr):
    'DUMP CODES'
    pass


@kcg_kernel
def _h2o_split_k3_kernel(q_ptr, k_ptr, v_ptr, em_ptr, denom_ptr, out_ptr):
    'DUMP CODES'
    pass


def _h2o_split_k1(q: torch.Tensor, k: torch.Tensor,
                   em: torch.Tensor, denom: torch.Tensor):
    return _h2o_split_k1_kernel(q, k, em, denom)


def _h2o_split_k2(k: torch.Tensor, q: torch.Tensor,
                   em: torch.Tensor, denom: torch.Tensor,
                   row_sum: torch.Tensor):
    return _h2o_split_k2_kernel(k, q, em, denom, row_sum)


def _h2o_split_k3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   em: torch.Tensor, denom: torch.Tensor, out: torch.Tensor):
    return _h2o_split_k3_kernel(q, k, v, em, denom, out)


def _make_qkv(bs, hn, sl, hd, dtype, device):
    q = 0.1 * torch.rand((bs, hn, sl, hd), dtype=dtype, device=device)
    k = 0.1 * torch.rand((bs, hn, hd, sl), dtype=dtype, device=device)
    v = 0.1 * torch.rand((bs, hn, sl, hd), dtype=dtype, device=device)
    return q, k, v


class H2OSplitOp(OpInterface):
    """Split H2O attention into three codegen'd kernels:

      H2O (Heavy-Hitter Oracle) attention computes both the standard attention
      output and per-key importance scores (row_sum).  The three kernels are:

      Kernel 1 (stats):  GEMM(Q@K^T) + causal mask + online reduce -> em, denom
        em    = exp(max(scores))            [B,H,S,1]
        denom = sum(exp(scores)) / em       [B,H,S,1]   (= expr1 in h2o notation)
        Signature: (Q_t[B,H,D,S], K[B,H,D,S], em_out[B,H,S,1], denom_out[B,H,S,1])
        C++ entry: compile_h2o_split_k1

      Kernel 2 (row_sum): GEMM(K^T@Q) + causal mask + normalize(em,denom) + row reduce -> row_sum
        Uses TRANSPOSED tiling: K as outer (blocks tile S_k), Q as inner (iterate S_q).
        This makes the col-reduce of P become a row-reduce of P^T, local to each block.
        p^T = exp(K^T@Q) / (em * denom)    [S_k, S_q]
        row_sum = reduce_sum(p^T, dim=-1)   [B,H,S]
        Signature: (K[B,H,D,S], Q_t[B,H,D,S], em[B,H,S,1], denom[B,H,S,1], row_sum[B,H,S])
        C++ entry: compile_h2o_split_k2

      Kernel 3 (output): GEMM(Q@K^T) + causal mask + normalize(em,denom) + GEMM(P@V) -> O
        Signature: (Q_t[B,H,D,S], K[B,H,D,S], V[B,H,S,D], em[B,H,S,1], denom[B,H,S,1], O[B,H,S,D])
        C++ entry: compile_h2o_split_k3

    Benchmark runs K1 -> K2 -> K3, timing their sum.
    Baseline is standard attention (with causal mask) for comparison.
    """

    def __init__(self):
        super().__init__()
        self.TuningArgs = AttentionTuningArgs()
        self.BaseArgs = AttentionBaseArgs()
        self.CompileKernelK1 = None
        self.CompileKernelK2 = None
        self.CompileKernelK3 = None
        self.SetPlatform = None
        self.SetKernelName = None
        self.fastCompile = True
        self._debug_dumped = False
        self._kernel1 = None
        self._kernel2 = None

    def _dump_tensor_stats(self, name: str, t: torch.Tensor):
        with torch.no_grad():
            nan_cnt = int(torch.isnan(t).sum().item())
            inf_cnt = int(torch.isinf(t).sum().item())
            finite_ratio = float(torch.isfinite(t).float().mean().item())
            t_safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            t_min = float(t_safe.min().item())
            t_max = float(t_safe.max().item())
        print(f"[h2o-split-debug] {name}: shape={tuple(t.shape)} min={t_min} max={t_max} "
              f"finite_ratio={finite_ratio} nan={nan_cnt} inf={inf_cnt}", flush=True)

    def GetBaselineInputTensor(self, devId: int) -> List[torch.Tensor]:
        if self.InputTensors_Baseline is None:
            [shapeList, dtypeInt] = self.BaseArgs.intValues
            assert len(shapeList) == 4
            [bs, hn, sl, hd] = shapeList
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            q, k, v = _make_qkv(bs, hn, sl, hd, ety, dev_name(devId))
            self.InputTensors_Baseline = [q, k, v]
        return self.InputTensors_Baseline

    def GetBenchmarkInputTensor(self, devId: int) -> List[torch.Tensor]:
        if self.InputTensors_Benchmark is None:
            [q, k, v] = self.GetBaselineInputTensor(devId)
            shapeList = self.BaseArgs.intValues[0]
            dtypeInt = self.BaseArgs.intValues[1]
            assert len(shapeList) == 4
            [b0, b1, m, n] = shapeList
            ety = ToTorchType(EnumKernelDType(dtypeInt))

            qq = q.transpose(-1, -2).contiguous()   # [B, H, D, S]
            kk = k                                   # [B, H, D, S]
            em = torch.empty((b0, b1, m, 1), dtype=ety, device=dev_name(devId))
            denom = torch.empty((b0, b1, m, 1), dtype=ety, device=dev_name(devId))
            row_sum = torch.empty((b0, b1, m), dtype=ety, device=dev_name(devId))
            d = torch.empty((b0, b1, m, n), dtype=ety, device=dev_name(devId))
            self.InputTensors_Benchmark = [qq, kk, v, em, denom, row_sum, d]
        return self.InputTensors_Benchmark

    def InitLibInterface(self):
        if self.CompileKernelK3 is None or self.SetPlatform is None:
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernelK1 = getattr(mod, 'compile_h2o_split_k1', None)
            self.CompileKernelK2 = getattr(mod, 'compile_h2o_split_k2', None)
            self.CompileKernelK3 = getattr(mod, 'compile_h2o_split_k3', None)
            if self.CompileKernelK1 is None:
                print("[h2o_split] WARNING: compile_h2o_split_k1 not found in libdeepgen", flush=True)
            if self.CompileKernelK2 is None:
                print("[h2o_split] WARNING: compile_h2o_split_k2 not found in libdeepgen", flush=True)
            if self.CompileKernelK3 is None:
                print("[h2o_split] WARNING: compile_h2o_split_k3 not found in libdeepgen", flush=True)
            self.SetKernelName = mod.set_kernel_name
            self.SetPlatform = mod.set_platform

    def _compile_one_kernel(self, compileFunc, kernelName, shape, config, dtype_str, backendtype, arch, opt, info):
        """Compile a single kernel via the given C++ compile function."""
        self.SetKernelName(kernelName)
        if False:
            from kcg.HIPCompiler import HIPCompiler
            hsacopath = f"{PathManager.default_dump_dir()}/hs_{kernelName}.hsaco"
            fastCompile = True if opt is None else opt.fastCompile
            res = HIPCompiler().build(Kernel.Attention, shape, config[kernelName],
                                      hsacopath, kernelName, fastCompile)
        else:
            res = compileFunc(shape, config, dtype_str)
        if not res:
            raise RuntimeError(f"Kernel compilation failed for {kernelName}")
        return res

    def Compile(self, deviceId, backendtype, arch, info, opt=None):
        assert isinstance(self.TuningArgs, AttentionTuningArgs)
        _backend = {
            EnumBackendType.CUDA.value: 1,
            EnumBackendType.HIP.value: 2,
            EnumBackendType.MLU.value: 3,
            EnumBackendType.NPU.value: 4,
        }.get(backendtype.value)
        assert _backend is not None, f'invalid backendtype {backendtype}'

        self.InitLibInterface()
        self.SetPlatform(_backend, arch)
        dataTypeInt = ToEnumIntDType(info.torchDataType)
        self.InitBaseArgs([info.baseArgs, dataTypeInt])
        shape, config = info.tsArgs
        dtype_str = EnumKernelDType(dataTypeInt).name
        dt = self.BaseArgs.getTorchDType()

        k1_name = info.kernelName + "_k1"
        k2_name = info.kernelName + "_k2"
        k3_name = info.kernelName + "_k3"

        if self.CompileKernelK1 is None or self.CompileKernelK3 is None:
            raise RuntimeError("[h2o_split] compile_h2o_split_k1/k3 not available in libdeepgen.")

        # Compile kernel 1 (stats: em, denom)
        config_k1 = {k1_name: config[info.kernelName]}
        hsaco_k1 = self._compile_one_kernel(self.CompileKernelK1, k1_name, shape, config_k1,
                                             dtype_str, backendtype, arch, opt, info)
        k1_config = KernelConfigs(hsaco_k1, k1_name, [dt] * 4, backendtype)
        k1_config.m_gridDims = list(info.gridDims)
        k1_config.m_blockDims = list(info.blockDims)
        k1_config.operatorKind = EnumOperator.Attention
        k1_config.shmBytes = info.shmBytes
        sig_k1 = self._get_k1_signature(dt)
        self._kernel1 = CompiledKernel(
            k1_config.backend, k1_config.binaryPath, k1_config.kernelFuncName,
            k1_config.sharedMem(), sig_k1, k1_config.gridDims(), k1_config.blockDims(), deviceId)

        # Compile kernel 2 (row_sum) — optional, GemmNormColSum optimizer may not exist yet
        k2_config = None
        if self.CompileKernelK2 is not None:
            try:
                config_k2 = {k2_name: config[info.kernelName]}
                hsaco_k2 = self._compile_one_kernel(self.CompileKernelK2, k2_name, shape, config_k2,
                                                     dtype_str, backendtype, arch, opt, info)
                k2_config = KernelConfigs(hsaco_k2, k2_name, [dt] * 5, backendtype)
                k2_config.m_gridDims = list(info.gridDims)
                k2_config.m_blockDims = list(info.blockDims)
                k2_config.operatorKind = EnumOperator.Attention
                k2_config.shmBytes = info.shmBytes
                sig_k2 = self._get_k2_signature(dt)
                self._kernel2 = CompiledKernel(
                    k2_config.backend, k2_config.binaryPath, k2_config.kernelFuncName,
                    k2_config.sharedMem(), sig_k2, k2_config.gridDims(), k2_config.blockDims(), deviceId)
            except Exception as e:
                print(f"[h2o_split] K2 compilation skipped ({e}), benchmark will time K1+K3 only", flush=True)
                self._kernel2 = None
                k2_config = None

        # Compile kernel 3 (output)
        config_k3 = {k3_name: config[info.kernelName]}
        hsaco_k3 = self._compile_one_kernel(self.CompileKernelK3, k3_name, shape, config_k3,
                                             dtype_str, backendtype, arch, opt, info)
        k3_config = KernelConfigs(hsaco_k3, k3_name, [dt] * 6, backendtype)
        k3_config.m_gridDims = list(info.gridDims)
        k3_config.m_blockDims = list(info.blockDims)
        k3_config.operatorKind = EnumOperator.Attention
        k3_config.shmBytes = info.shmBytes
        packedKernel = self.GetCompiledKernel(k3_config, deviceId)

        if k2_config is not None:
            return ([info.baseArgs, dataTypeInt], k3_config, packedKernel, k1_config, k2_config)
        else:
            return ([info.baseArgs, dataTypeInt], k3_config, packedKernel, k1_config)

    def _get_k1_signature(self, dtype):
        q = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtype)
        k = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtype)
        em = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtype)
        denom = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtype)
        return _h2o_split_k1(q, k, em, denom)

    def _get_k2_signature(self, dtype):
        k = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtype)
        q = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtype)
        em = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtype)
        denom = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtype)
        row_sum = torch.randn((1, 3, 100), device='cpu', dtype=dtype)
        return _h2o_split_k2(k, q, em, denom, row_sum)

    def GetCompiledKernel(self, info, deviceId):
        signature = self.GetSignature(info.dtypes)
        return CompiledKernel(
            info.backend, info.binaryPath, info.kernelFuncName,
            info.sharedMem(), signature, info.gridDims(), info.blockDims(), deviceId)

    def GetSignature(self, dtypes):
        dtypeA = dtypes[0]
        q = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        k = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        v = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        em = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtypeA)
        denom = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtypeA)
        out = torch.empty((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        return _h2o_split_k3(q, k, v, em, denom, out)

    def SetTuningArgs(self, tuningArgs):
        self.TuningArgs.assignWithList(*tuningArgs)

    def InitBaseArgs(self, args):
        shape, dtypeInt = args
        self.BaseArgs.intValues = [shape, dtypeInt]
        ety = EnumKernelDType(dtypeInt)
        self.TuningArgs = AttentionTuningArgs(ety)

    def Test_warmup(self, packedKernel, warmupCount, devId):
        tensors = self.GetBenchmarkInputTensor(devId)
        qq, kk, v, em, denom, row_sum, d = tensors
        for _ in range(warmupCount):
            if self._kernel1 is not None:
                self._kernel1.run(qq, kk, em, denom)
            if self._kernel2 is not None:
                self._kernel2.run(kk, qq, em, denom, row_sum)
            packedKernel.run(qq, kk, v, em, denom, d)

    def Test_baseline(self, devId):
        """H2O baseline: standard attention with causal mask, returns out for accuracy check."""
        [q, k, v] = self.GetBaselineInputTensor(devId)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        S = q.shape[-2]
        device, dtype = q.device, q.dtype
        mask = _causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)

        # warmup
        q_s = torch.mul(q, scale)
        p = torch.matmul(q_s, k) + mask
        p_max = torch.max(p, dim=-1, keepdim=True).values
        p_shifted = torch.sub(p, p_max)
        p_exp = torch.exp(p_shifted)
        p_sum = torch.sum(p_exp, dim=-1, keepdim=True)
        s = torch.div(p_exp, p_sum)
        _ = torch.matmul(s, v)

        # timed
        ev_start = torch_ns.Event(enable_timing=True)
        ev_end = torch_ns.Event(enable_timing=True)
        ev_start.record()
        q_scaled = torch.mul(q, scale)
        p = torch.matmul(q_scaled, k) + mask
        p_max = torch.max(p, dim=-1, keepdim=True).values
        p_shifted = torch.sub(p, p_max)
        p_exp = torch.exp(p_shifted)
        p_sum = torch.sum(p_exp, dim=-1, keepdim=True)
        s = torch.div(p_exp, p_sum)
        self.OutputTensor_Baseline = torch.matmul(s, v)
        ev_end.record()
        torch_ns.synchronize()
        return (self.OutputTensor_Baseline, ev_start.elapsed_time(ev_end))

    def Test_benchmark(self, packedKernel, benchmarkCount, devId):
        tensors = self.GetBenchmarkInputTensor(devId)
        qq, kk, v, em, denom, row_sum, d = tensors
        if not self._debug_dumped:
            self._dump_tensor_stats("Q_in", qq)
            self._dump_tensor_stats("K_in", kk)
            self._dump_tensor_stats("V_in", v)

        has_k2 = self._kernel2 is not None

        # warmup: K1 [+ K2] + K3
        assert self._kernel1 is not None, "[h2o_split] kernel1 not available — K1 config missing from pkl?"
        self._kernel1.run(qq, kk, em, denom)
        if has_k2:
            self._kernel2.run(kk, qq, em, denom, row_sum)
        packedKernel.run(qq, kk, v, em, denom, d)
        torch_ns.synchronize()

        # timed: K1 [+ K2] + K3
        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        self._kernel1.run(qq, kk, em, denom)
        if has_k2:
            self._kernel2.run(kk, qq, em, denom, row_sum)
        packedKernel.run(qq, kk, v, em, denom, d)
        et.record()
        torch_ns.synchronize()

        if not self._debug_dumped:
            self._dump_tensor_stats("em_out", em)
            self._dump_tensor_stats("denom_out", denom)
            if has_k2:
                self._dump_tensor_stats("row_sum_out", row_sum)
            self._dump_tensor_stats("O_out", d)
            self._debug_dumped = True
        return (d, st.elapsed_time(et))

    def InitInputTensorsWithDatalist(self, devId):
        self.GetBaselineInputTensor(devId)
        self.GetBenchmarkInputTensor(devId)

    def InitBaselineOutputTensor(self, devId):
        if self.OutputTensor_Baseline is None:
            b, bb, m, n = self.BaseArgs.getIntDatalist()
            dt = self.BaseArgs.getTorchDType()
            self.OutputTensor_Baseline = torch.empty(m, n, dtype=dt, device=dev_name(devId))


def _precompute_em_denom(q, k, device, dtype):
    """Compute em/denom via PyTorch for standalone K2/K3 benchmarking."""
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    S = q.shape[-2]
    mask = _causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        q_scaled = torch.mul(q, scale)
        scores = torch.matmul(q_scaled, k) + mask
        m = scores.max(dim=-1, keepdim=True).values
        em = torch.exp(m)
        sum_ex = torch.exp(scores).sum(dim=-1, keepdim=True)
        denom = sum_ex / em
    return em, denom


class _H2OSingleKernelBase(OpInterface):
    """Shared base for H2OK1Op / H2OK2Op / H2OK3Op."""

    def __init__(self):
        super().__init__()
        self.TuningArgs = AttentionTuningArgs()
        self.BaseArgs = AttentionBaseArgs()
        self.SetPlatform = None
        self.SetKernelName = None
        self.fastCompile = True
        self._debug_dumped = False
        self._compile_func = None
        self._compile_func_name = None

    def _init_lib(self):
        if self.SetPlatform is None:
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.SetKernelName = mod.set_kernel_name
            self.SetPlatform = mod.set_platform
            return mod
        return None

    def SetTuningArgs(self, tuningArgs):
        self.TuningArgs.assignWithList(*tuningArgs)

    def InitBaseArgs(self, args):
        shape, dtypeInt = args
        self.BaseArgs.intValues = [shape, dtypeInt]
        ety = EnumKernelDType(dtypeInt)
        self.TuningArgs = AttentionTuningArgs(ety)

    def _compile_single(self, deviceId, backendtype, arch, info, opt, compile_func, kname_suffix, n_dtypes):
        _backend = {
            EnumBackendType.CUDA.value: 1, EnumBackendType.HIP.value: 2,
            EnumBackendType.MLU.value: 3, EnumBackendType.NPU.value: 4,
        }.get(backendtype.value)
        assert _backend is not None
        self.SetPlatform(_backend, arch)
        dataTypeInt = ToEnumIntDType(info.torchDataType)
        self.InitBaseArgs([info.baseArgs, dataTypeInt])
        shape, config = info.tsArgs
        dtype_str = EnumKernelDType(dataTypeInt).name
        dt = self.BaseArgs.getTorchDType()

        kname = info.kernelName
        self.SetKernelName(kname)
        cfg = {kname: config[info.kernelName]}
        res = compile_func(shape, cfg, dtype_str)
        if not res:
            raise RuntimeError(f"Kernel compilation failed for {kname}")

        kconfig = KernelConfigs(res, kname, [dt] * n_dtypes, backendtype)
        kconfig.m_gridDims = list(info.gridDims)
        kconfig.m_blockDims = list(info.blockDims)
        kconfig.operatorKind = EnumOperator.Attention
        kconfig.shmBytes = info.shmBytes
        return dataTypeInt, kconfig, dt

    def InitInputTensorsWithDatalist(self, devId):
        self.GetBaselineInputTensor(devId)
        self.GetBenchmarkInputTensor(devId)

    def InitBaselineOutputTensor(self, devId):
        pass


# ====================== H2O K1 standalone ======================
class H2OK1Op(_H2OSingleKernelBase):
    """Tune H2O kernel 1 (GemmStats: Q@K^T -> em, denom) independently."""

    def InitLibInterface(self):
        mod = self._init_lib()
        if mod:
            self._compile_func = getattr(mod, 'compile_h2o_split_k1')

    def GetBaselineInputTensor(self, devId):
        if self.InputTensors_Baseline is None:
            [bs, hn, sl, hd] = self.BaseArgs.intValues[0]
            ety = ToTorchType(EnumKernelDType(self.BaseArgs.intValues[1]))
            q, k, _ = _make_qkv(bs, hn, sl, hd, ety, dev_name(devId))
            self.InputTensors_Baseline = [q, k]
        return self.InputTensors_Baseline

    def GetBenchmarkInputTensor(self, devId):
        if self.InputTensors_Benchmark is None:
            [q, k] = self.GetBaselineInputTensor(devId)
            [b0, b1, m, n] = self.BaseArgs.intValues[0]
            ety = ToTorchType(EnumKernelDType(self.BaseArgs.intValues[1]))
            qq = q.transpose(-1, -2).contiguous()
            kk = k
            em = torch.empty((b0, b1, m, 1), dtype=ety, device=dev_name(devId))
            denom = torch.empty((b0, b1, m, 1), dtype=ety, device=dev_name(devId))
            self.InputTensors_Benchmark = [qq, kk, em, denom]
        return self.InputTensors_Benchmark

    def Compile(self, deviceId, backendtype, arch, info, opt=None):
        self.InitLibInterface()
        dataTypeInt, kconfig, dt = self._compile_single(
            deviceId, backendtype, arch, info, opt, self._compile_func, "_k1", 4)
        sig = _h2o_split_k1(
            torch.randn(1,3,100,100), torch.randn(1,3,100,100),
            torch.randn(1,3,100,1), torch.randn(1,3,100,1))
        packed = CompiledKernel(
            kconfig.backend, kconfig.binaryPath, kconfig.kernelFuncName,
            kconfig.sharedMem(), sig, kconfig.gridDims(), kconfig.blockDims(), deviceId)
        return ([info.baseArgs, dataTypeInt], kconfig, packed)

    def GetSignature(self, dtypes):
        return _h2o_split_k1(
            torch.randn(1,3,100,100, dtype=dtypes[0]), torch.randn(1,3,100,100, dtype=dtypes[0]),
            torch.randn(1,3,100,1, dtype=dtypes[0]), torch.randn(1,3,100,1, dtype=dtypes[0]))

    def GetCompiledKernel(self, info, deviceId):
        sig = self.GetSignature(info.dtypes)
        return CompiledKernel(
            info.backend, info.binaryPath, info.kernelFuncName,
            info.sharedMem(), sig, info.gridDims(), info.blockDims(), deviceId)

    def Test_baseline(self, devId):
        [q, k] = self.GetBaselineInputTensor(devId)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        S = q.shape[-2]
        device, dtype = q.device, q.dtype
        mask = _causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)
        # warmup
        with torch.no_grad():
            q_s = torch.mul(q, scale)
            scores = torch.matmul(q_s, k) + mask
            _ = torch.exp(scores.max(dim=-1, keepdim=True).values)
        ev_s = torch_ns.Event(enable_timing=True)
        ev_e = torch_ns.Event(enable_timing=True)
        ev_s.record()
        q_s = torch.mul(q, scale)
        scores = torch.matmul(q_s, k) + mask
        m = scores.max(dim=-1, keepdim=True).values
        em = torch.exp(m)
        sum_ex = torch.exp(scores).sum(dim=-1, keepdim=True)
        self.OutputTensor_Baseline = sum_ex / em
        ev_e.record()
        torch_ns.synchronize()
        return (self.OutputTensor_Baseline, ev_s.elapsed_time(ev_e))

    def Test_warmup(self, packedKernel, warmupCount, devId):
        qq, kk, em, denom = self.GetBenchmarkInputTensor(devId)
        for _ in range(warmupCount):
            packedKernel.run(qq, kk, em, denom)

    def Test_benchmark(self, packedKernel, benchmarkCount, devId):
        qq, kk, em, denom = self.GetBenchmarkInputTensor(devId)
        packedKernel.run(qq, kk, em, denom)
        torch_ns.synchronize()
        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        packedKernel.run(qq, kk, em, denom)
        et.record()
        torch_ns.synchronize()
        return (denom, st.elapsed_time(et))


# ====================== H2O K2 standalone ======================
class H2OK2Op(_H2OSingleKernelBase):
    """Tune H2O kernel 2 (GemmNormColSum: K^T@Q + normalize + col reduce -> row_sum) independently."""

    def InitLibInterface(self):
        mod = self._init_lib()
        if mod:
            self._compile_func = getattr(mod, 'compile_h2o_split_k2')

    def GetBaselineInputTensor(self, devId):
        if self.InputTensors_Baseline is None:
            [bs, hn, sl, hd] = self.BaseArgs.intValues[0]
            ety = ToTorchType(EnumKernelDType(self.BaseArgs.intValues[1]))
            q, k, _ = _make_qkv(bs, hn, sl, hd, ety, dev_name(devId))
            self.InputTensors_Baseline = [q, k]
        return self.InputTensors_Baseline

    def GetBenchmarkInputTensor(self, devId):
        if self.InputTensors_Benchmark is None:
            [q, k] = self.GetBaselineInputTensor(devId)
            [b0, b1, m, _] = self.BaseArgs.intValues[0]
            ety = ToTorchType(EnumKernelDType(self.BaseArgs.intValues[1]))
            em, denom = _precompute_em_denom(q, k, q.device, ety)
            qq = q.transpose(-1, -2).contiguous()
            kk = k
            row_sum = torch.empty((b0, b1, m), dtype=ety, device=dev_name(devId))
            self.InputTensors_Benchmark = [kk, qq, em, denom, row_sum]
        return self.InputTensors_Benchmark

    def Compile(self, deviceId, backendtype, arch, info, opt=None):
        self.InitLibInterface()
        dataTypeInt, kconfig, dt = self._compile_single(
            deviceId, backendtype, arch, info, opt, self._compile_func, "_k2", 5)
        sig = _h2o_split_k2(
            torch.randn(1,3,100,100), torch.randn(1,3,100,100),
            torch.randn(1,3,100,1), torch.randn(1,3,100,1), torch.randn(1,3,100))
        packed = CompiledKernel(
            kconfig.backend, kconfig.binaryPath, kconfig.kernelFuncName,
            kconfig.sharedMem(), sig, kconfig.gridDims(), kconfig.blockDims(), deviceId)
        return ([info.baseArgs, dataTypeInt], kconfig, packed)

    def GetSignature(self, dtypes):
        return _h2o_split_k2(
            torch.randn(1,3,100,100, dtype=dtypes[0]), torch.randn(1,3,100,100, dtype=dtypes[0]),
            torch.randn(1,3,100,1, dtype=dtypes[0]), torch.randn(1,3,100,1, dtype=dtypes[0]),
            torch.randn(1,3,100, dtype=dtypes[0]))

    def GetCompiledKernel(self, info, deviceId):
        sig = self.GetSignature(info.dtypes)
        return CompiledKernel(
            info.backend, info.binaryPath, info.kernelFuncName,
            info.sharedMem(), sig, info.gridDims(), info.blockDims(), deviceId)

    def Test_baseline(self, devId):
        [q, k] = self.GetBaselineInputTensor(devId)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        S = q.shape[-2]
        device, dtype = q.device, q.dtype
        mask = _causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)
        # warmup
        with torch.no_grad():
            q_s = torch.mul(q, scale)
            scores = torch.matmul(q_s, k) + mask
            em = torch.exp(scores.max(dim=-1, keepdim=True).values)
            s_ex = torch.exp(scores).sum(dim=-1, keepdim=True)
            denom = s_ex / em
            p = torch.exp(scores) / (em * denom)
            _ = p.sum(dim=2, keepdim=False)
        ev_s = torch_ns.Event(enable_timing=True)
        ev_e = torch_ns.Event(enable_timing=True)
        ev_s.record()
        q_s = torch.mul(q, scale)
        scores = torch.matmul(q_s, k) + mask
        em = torch.exp(scores.max(dim=-1, keepdim=True).values)
        s_ex = torch.exp(scores).sum(dim=-1, keepdim=True)
        denom = s_ex / em
        p = torch.exp(scores) / (em * denom)
        self.OutputTensor_Baseline = p.sum(dim=2, keepdim=False)
        ev_e.record()
        torch_ns.synchronize()
        return (self.OutputTensor_Baseline, ev_s.elapsed_time(ev_e))

    def Test_warmup(self, packedKernel, warmupCount, devId):
        kk, qq, em, denom, row_sum = self.GetBenchmarkInputTensor(devId)
        for _ in range(warmupCount):
            packedKernel.run(kk, qq, em, denom, row_sum)

    def Test_benchmark(self, packedKernel, benchmarkCount, devId):
        kk, qq, em, denom, row_sum = self.GetBenchmarkInputTensor(devId)
        packedKernel.run(kk, qq, em, denom, row_sum)
        torch_ns.synchronize()
        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        packedKernel.run(kk, qq, em, denom, row_sum)
        et.record()
        torch_ns.synchronize()
        return (row_sum, st.elapsed_time(et))


# ====================== H2O K3 standalone ======================
class H2OK3Op(_H2OSingleKernelBase):
    """Tune H2O kernel 3 (FlashAttnSplitK2: Q@K^T + normalize + P@V -> out) independently."""

    def InitLibInterface(self):
        mod = self._init_lib()
        if mod:
            self._compile_func = getattr(mod, 'compile_h2o_split_k3')

    def GetBaselineInputTensor(self, devId):
        if self.InputTensors_Baseline is None:
            [bs, hn, sl, hd] = self.BaseArgs.intValues[0]
            ety = ToTorchType(EnumKernelDType(self.BaseArgs.intValues[1]))
            q, k, v = _make_qkv(bs, hn, sl, hd, ety, dev_name(devId))
            self.InputTensors_Baseline = [q, k, v]
        return self.InputTensors_Baseline

    def GetBenchmarkInputTensor(self, devId):
        if self.InputTensors_Benchmark is None:
            [q, k, v] = self.GetBaselineInputTensor(devId)
            [b0, b1, m, n] = self.BaseArgs.intValues[0]
            ety = ToTorchType(EnumKernelDType(self.BaseArgs.intValues[1]))
            em, denom = _precompute_em_denom(q, k, q.device, ety)
            qq = q.transpose(-1, -2).contiguous()
            kk = k
            d = torch.empty((b0, b1, m, n), dtype=ety, device=dev_name(devId))
            self.InputTensors_Benchmark = [qq, kk, v, em, denom, d]
        return self.InputTensors_Benchmark

    def Compile(self, deviceId, backendtype, arch, info, opt=None):
        self.InitLibInterface()
        dataTypeInt, kconfig, dt = self._compile_single(
            deviceId, backendtype, arch, info, opt, self._compile_func, "_k3", 6)
        sig = _h2o_split_k3(
            torch.randn(1,3,100,100), torch.randn(1,3,100,100), torch.randn(1,3,100,100),
            torch.randn(1,3,100,1), torch.randn(1,3,100,1), torch.empty(1,3,100,100))
        packed = CompiledKernel(
            kconfig.backend, kconfig.binaryPath, kconfig.kernelFuncName,
            kconfig.sharedMem(), sig, kconfig.gridDims(), kconfig.blockDims(), deviceId)
        return ([info.baseArgs, dataTypeInt], kconfig, packed)

    def GetSignature(self, dtypes):
        d = dtypes[0]
        return _h2o_split_k3(
            torch.randn(1,3,100,100, dtype=d), torch.randn(1,3,100,100, dtype=d), torch.randn(1,3,100,100, dtype=d),
            torch.randn(1,3,100,1, dtype=d), torch.randn(1,3,100,1, dtype=d), torch.empty(1,3,100,100, dtype=d))

    def GetCompiledKernel(self, info, deviceId):
        sig = self.GetSignature(info.dtypes)
        return CompiledKernel(
            info.backend, info.binaryPath, info.kernelFuncName,
            info.sharedMem(), sig, info.gridDims(), info.blockDims(), deviceId)

    def Test_baseline(self, devId):
        [q, k, v] = self.GetBaselineInputTensor(devId)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        S = q.shape[-2]
        device, dtype = q.device, q.dtype
        mask = _causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)
        # warmup
        q_s = torch.mul(q, scale)
        p = torch.matmul(q_s, k) + mask
        p_max = torch.max(p, dim=-1, keepdim=True).values
        p_exp = torch.exp(torch.sub(p, p_max))
        s = torch.div(p_exp, torch.sum(p_exp, dim=-1, keepdim=True))
        _ = torch.matmul(s, v)
        ev_s = torch_ns.Event(enable_timing=True)
        ev_e = torch_ns.Event(enable_timing=True)
        ev_s.record()
        q_s = torch.mul(q, scale)
        p = torch.matmul(q_s, k) + mask
        p_max = torch.max(p, dim=-1, keepdim=True).values
        p_exp = torch.exp(torch.sub(p, p_max))
        s = torch.div(p_exp, torch.sum(p_exp, dim=-1, keepdim=True))
        self.OutputTensor_Baseline = torch.matmul(s, v)
        ev_e.record()
        torch_ns.synchronize()
        return (self.OutputTensor_Baseline, ev_s.elapsed_time(ev_e))

    def Test_warmup(self, packedKernel, warmupCount, devId):
        qq, kk, v, em, denom, d = self.GetBenchmarkInputTensor(devId)
        for _ in range(warmupCount):
            packedKernel.run(qq, kk, v, em, denom, d)

    def Test_benchmark(self, packedKernel, benchmarkCount, devId):
        qq, kk, v, em, denom, d = self.GetBenchmarkInputTensor(devId)
        packedKernel.run(qq, kk, v, em, denom, d)
        torch_ns.synchronize()
        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        packedKernel.run(qq, kk, v, em, denom, d)
        et.record()
        torch_ns.synchronize()
        return (d, st.elapsed_time(et))
