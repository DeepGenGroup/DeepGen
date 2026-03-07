import importlib
import math
import os
import torch
from kcg.Kernel import *
from kcg.Operators.attention import (
    AttentionBaseArgs, AttentionTuningArgs,
)


@kcg_kernel
def _gemma2_split_k1_kernel(q_ptr, k_ptr, em_ptr, denom_ptr):
    'DUMP CODES'
    pass


@kcg_kernel
def _gemma2_split_k2_kernel(q_ptr, k_ptr, v_ptr, em_ptr, denom_ptr, out_ptr):
    'DUMP CODES'
    pass


def _gemma2_split_k1(q: torch.Tensor, k: torch.Tensor,
                     em: torch.Tensor, denom: torch.Tensor):
    return _gemma2_split_k1_kernel(q, k, em, denom)


def _gemma2_split_k2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     em: torch.Tensor, denom: torch.Tensor, out: torch.Tensor):
    return _gemma2_split_k2_kernel(q, k, v, em, denom, out)


def _causal_upper_mask(S, device, dtype):
    return torch.where(
        torch.triu(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=1),
        torch.full((S, S), float("-inf"), device=device, dtype=dtype),
        torch.zeros((S, S), device=device, dtype=dtype),
    )


class Gemma2SplitOp(OpInterface):
    """Split Gemma2 attention into two codegen'd kernels:

      Gemma2 differs from standard attention by applying logit softcapping:
        scores = Q@K^T / sqrt(head_dim)
        scores = tanh(scores / tanh_scale) * tanh_scale + causal_mask
        P = softmax(scores)
        O = P @ V

      Kernel 1 (stats):  GEMM(Q@K^T) + softcap + mask + online reduce -> em, denom
        Signature: (Q_t[B,H,D,S], K[B,H,D,S], em_out[B,H,S,1], denom_out[B,H,S,1])
        C++ entry: compile_gemma2_split_k1

      Kernel 2 (output): GEMM(Q@K^T) + softcap + mask + normalize(em,denom) + GEMM(P@V) -> O
        Signature: (Q_t[B,H,D,S], K[B,H,D,S], V[B,H,S,D], em[B,H,S,1], denom[B,H,S,1], O[B,H,S,D])
        C++ entry: compile_gemma2_split_k2

    Benchmark runs kernel1 then kernel2, timing their sum.
    Baseline is Gemma2-style attention for comparison.
    """

    TANH_SCALE = 50.0

    def __init__(self):
        super().__init__()
        self.TuningArgs = AttentionTuningArgs()
        self.TuningArgs.kernelNamePrefix = "Gemma2"
        self.BaseArgs = AttentionBaseArgs()
        self.CompileKernelK1 = None
        self.CompileKernelK2 = None
        self.SetPlatform = None
        self.SetKernelName = None
        self.fastCompile = True
        self._debug_dumped = False
        self._kernel1 = None
        self._kernel1_config = None

    def _dump_tensor_stats(self, name: str, t: torch.Tensor):
        with torch.no_grad():
            nan_cnt = int(torch.isnan(t).sum().item())
            inf_cnt = int(torch.isinf(t).sum().item())
            finite_ratio = float(torch.isfinite(t).float().mean().item())
            t_safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            t_min = float(t_safe.min().item())
            t_max = float(t_safe.max().item())
        print(f"[gemma2-split-debug] {name}: shape={tuple(t.shape)} min={t_min} max={t_max} "
              f"finite_ratio={finite_ratio} nan={nan_cnt} inf={inf_cnt}", flush=True)

    def GetBaselineInputTensor(self, devId: int) -> List[torch.Tensor]:
        if self.InputTensors_Baseline is None:
            [shapeList, dtypeInt] = self.BaseArgs.intValues
            assert len(shapeList) == 4
            [bs, hn, sl, hd] = shapeList
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            q = torch.ones((bs, hn, sl, hd), dtype=ety, device=dev_name(devId))
            k = torch.ones((bs, hn, hd, sl), dtype=ety, device=dev_name(devId))
            v = torch.ones((bs, hn, sl, hd), dtype=ety, device=dev_name(devId))
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
            d = torch.empty((b0, b1, m, n), dtype=ety, device=dev_name(devId))
            self.InputTensors_Benchmark = [qq, kk, v, em, denom, d]
        return self.InputTensors_Benchmark

    def InitLibInterface(self):
        if self.CompileKernelK2 is None or self.SetPlatform is None:
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernelK1 = getattr(mod, 'compile_gemma2_split_k1', None)
            self.CompileKernelK2 = getattr(mod, 'compile_gemma2_split_k2', None)
            if self.CompileKernelK1 is None:
                print("[gemma2_split] WARNING: compile_gemma2_split_k1 not found in libdeepgen", flush=True)
            if self.CompileKernelK2 is None:
                print("[gemma2_split] WARNING: compile_gemma2_split_k2 not found in libdeepgen", flush=True)
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

        if self.CompileKernelK1 is None or self.CompileKernelK2 is None:
            raise RuntimeError("[gemma2_split] compile_gemma2_split_k1/k2 not available in libdeepgen.")

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

        config_k2 = {k2_name: config[info.kernelName]}
        hsaco_k2 = self._compile_one_kernel(self.CompileKernelK2, k2_name, shape, config_k2,
                                             dtype_str, backendtype, arch, opt, info)
        k2_config = KernelConfigs(hsaco_k2, k2_name, [dt] * 6, backendtype)
        k2_config.m_gridDims = list(info.gridDims)
        k2_config.m_blockDims = list(info.blockDims)
        k2_config.operatorKind = EnumOperator.Attention
        k2_config.shmBytes = info.shmBytes
        packedKernel = self.GetCompiledKernel(k2_config, deviceId)

        return ([info.baseArgs, dataTypeInt], k2_config, packedKernel, k1_config)

    def _get_k1_signature(self, dtype):
        q = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtype)
        k = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtype)
        em = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtype)
        denom = torch.randn((1, 3, 100, 1), device='cpu', dtype=dtype)
        return _gemma2_split_k1(q, k, em, denom)

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
        return _gemma2_split_k2(q, k, v, em, denom, out)

    def SetTuningArgs(self, tuningArgs):
        self.TuningArgs.assignWithList(*tuningArgs)

    def InitBaseArgs(self, args):
        shape, dtypeInt = args
        self.BaseArgs.intValues = [shape, dtypeInt]
        ety = EnumKernelDType(dtypeInt)
        self.TuningArgs = AttentionTuningArgs(ety)
        self.TuningArgs.kernelNamePrefix = "Gemma2"

    def Test_warmup(self, packedKernel, warmupCount, devId):
        tensors = self.GetBenchmarkInputTensor(devId)
        qq, kk, v, em, denom, d = tensors
        for _ in range(warmupCount):
            if self._kernel1 is not None:
                self._kernel1.run(qq, kk, em, denom)
            packedKernel.run(qq, kk, v, em, denom, d)

    def Test_baseline(self, devId):
        """Gemma2-style baseline: scores → tanh softcap → causal mask → softmax → P@V"""
        [q, k, v] = self.GetBaselineInputTensor(devId)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        tanh_scale = self.TANH_SCALE
        S = q.shape[-2]
        device, dtype = q.device, q.dtype
        mask = _causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)

        # warmup
        scores = torch.matmul(torch.mul(q, scale), k)
        y = torch.tanh(scores / tanh_scale) * tanh_scale
        y = y + mask
        m = y.max(dim=-1, keepdim=True).values
        ex = torch.exp(y - m)
        s = ex / ex.sum(dim=-1, keepdim=True)
        _ = torch.matmul(s, v)

        # timed
        ev_start = torch_ns.Event(enable_timing=True)
        ev_end = torch_ns.Event(enable_timing=True)
        ev_start.record()
        scores = torch.matmul(torch.mul(q, scale), k)
        y = torch.tanh(scores / tanh_scale) * tanh_scale
        y = y + mask
        m = y.max(dim=-1, keepdim=True).values
        ex = torch.exp(y - m)
        s = ex / ex.sum(dim=-1, keepdim=True)
        self.OutputTensor_Baseline = torch.matmul(s, v)
        ev_end.record()
        torch_ns.synchronize()
        return (self.OutputTensor_Baseline, ev_start.elapsed_time(ev_end))

    def _compute_em_denom_pytorch(self, qq, kk, em, denom):
        """Fallback: compute em/denom via PyTorch when kernel1 is not available."""
        [q, k, _] = self.GetBaselineInputTensor(em.device.index)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        tanh_scale = self.TANH_SCALE
        S = q.shape[-2]
        device, dtype = q.device, q.dtype
        mask = _causal_upper_mask(S, device, dtype).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            scores = torch.matmul(torch.mul(q, scale), k)
            y = torch.tanh(scores / tanh_scale) * tanh_scale
            y = y + mask
            m = y.max(dim=-1, keepdim=True).values
            em.copy_(torch.exp(m))
            sum_ex = torch.exp(y).sum(dim=-1, keepdim=True)
            denom.copy_(sum_ex / em)

    def Test_benchmark(self, packedKernel, benchmarkCount, devId):
        tensors = self.GetBenchmarkInputTensor(devId)
        qq, kk, v, em, denom, d = tensors
        if not self._debug_dumped:
            self._dump_tensor_stats("Q_in", qq)
            self._dump_tensor_stats("K_in", kk)
            self._dump_tensor_stats("V_in", v)
        # warmup: K1 + K2
        assert self._kernel1 is not None, "[gemma2_split] kernel1 not available — K1 config missing from pkl?"
        self._kernel1.run(qq, kk, em, denom)
        packedKernel.run(qq, kk, v, em, denom, d)
        torch_ns.synchronize()
        # timed: K1 + K2
        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        self._kernel1.run(qq, kk, em, denom)
        packedKernel.run(qq, kk, v, em, denom, d)
        et.record()
        torch_ns.synchronize()
        if not self._debug_dumped:
            self._dump_tensor_stats("em_out", em)
            self._dump_tensor_stats("denom_out", denom)
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
