import importlib
import math
import torch
import torch.nn.functional as F
from kcg.Kernel import *
from kcg.Operators.attention import (
    _attention_kernel, _attention,
    AttentionBaseArgs, AttentionTuningArgs,
)


class AttentionV2Op(OpInterface):
    """Optimized attention: scale before mm1, div-sum after mm2.

    Compared with AttentionOp the only pipeline differences are:
      1. C++ side uses compile_attn_v2 (ElemScale+Matmul+SoftmaxExp+Matmul+RowDiv)
      2. Python side does NOT pre-scale Q — the kernel handles it via ElemScale
    """

    def __init__(self):
        super().__init__()
        self.TuningArgs = AttentionTuningArgs()
        self.BaseArgs = AttentionBaseArgs()
        self.CompileKernel = None
        self.SetPlatform = None
        self.fastCompile = True
        self._debug_dumped = False

    def _dump_tensor_stats(self, name: str, t: torch.Tensor):
        with torch.no_grad():
            nan_cnt = int(torch.isnan(t).sum().item())
            inf_cnt = int(torch.isinf(t).sum().item())
            finite_ratio = float(torch.isfinite(t).float().mean().item())
            t_safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            t_min = float(t_safe.min().item())
            t_max = float(t_safe.max().item())
        print(f"[attn-v2-debug] {name}: shape={tuple(t.shape)} min={t_min} max={t_max} "
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
            # v2: NO pre-scaling — ElemScale inside the kernel handles it
            qq = q.transpose(-1, -2).contiguous()
            kk = k
            d = torch.empty((b0, b1, m, n), dtype=ety, device=dev_name(devId))
            self.InputTensors_Benchmark = [qq, kk, v, d]
        return self.InputTensors_Benchmark

    def InitLibInterface(self):
        if self.CompileKernel is None or self.SetPlatform is None:
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernel = mod.compile_attn_v2   # <-- v2
            self.SetKernelName = mod.set_kernel_name
            self.SetPlatform = mod.set_platform

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
        self.SetKernelName(info.kernelName)

        dtype_str = EnumKernelDType(dataTypeInt).name

        if False:
            from kcg.HIPCompiler import HIPCompiler
            hsacopath = f"{PathManager.default_dump_dir()}/hs_{info.kernelName}.hsaco"
            fastCompile = True if opt is None else opt.fastCompile
            res = HIPCompiler().build(Kernel.Attention, shape, config[info.kernelName],
                                      hsacopath, info.kernelName, fastCompile)
        else:
            res = self.CompileKernel(shape, config, dtype_str)

        if not res:
            raise RuntimeError(f"Kernel compilation failed for {info.kernelName}")

        hsacoPath = res
        dt = self.BaseArgs.getTorchDType()
        inConfig = KernelConfigs(hsacoPath, info.kernelName, [dt, dt, dt, dt], backendtype)
        inConfig.m_gridDims = list(info.gridDims)
        inConfig.m_blockDims = list(info.blockDims)
        inConfig.operatorKind = EnumOperator.Attention
        inConfig.shmBytes = info.shmBytes
        packedKernel = self.GetCompiledKernel(inConfig, deviceId)
        return ([info.baseArgs, dataTypeInt], inConfig, packedKernel)

    def GetCompiledKernel(self, info, deviceId):
        signature = self.GetSignature(info.dtypes)
        return CompiledKernel(
            info.backend, info.binaryPath, info.kernelFuncName,
            info.sharedMem(), signature, info.gridDims(), info.blockDims(), deviceId)

    def GetSignature(self, dtypes):
        dtypeA = dtypes[0]
        a = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        b = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        c = torch.randn((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        d = torch.empty((1, 3, 100, 100), device='cpu', dtype=dtypeA)
        return _attention(a, b, c, d)

    def SetTuningArgs(self, tuningArgs):
        self.TuningArgs.assignWithList(*tuningArgs)

    def InitBaseArgs(self, args):
        shape, dtypeInt = args
        self.BaseArgs.intValues = [shape, dtypeInt]
        ety = EnumKernelDType(dtypeInt)
        self.TuningArgs = AttentionTuningArgs(ety)

    def Test_warmup(self, packedKernel, warmupCount, devId):
        [qq, kk, vv, out] = self.GetBenchmarkInputTensor(devId)
        for _ in range(warmupCount):
            packedKernel.run(qq, kk, vv, out)

    def Test_baseline(self, devId):
        [q, k, v] = self.GetBaselineInputTensor(devId)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        # warmup
        q_s = torch.mul(q, scale)
        p = torch.matmul(q_s, k)
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
        p = torch.matmul(q_scaled, k)
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
        [a, b, c, d] = self.GetBenchmarkInputTensor(devId)
        if not self._debug_dumped:
            self._dump_tensor_stats("Q_in", a)
            self._dump_tensor_stats("K_in", b)
            self._dump_tensor_stats("V_in", c)
        packedKernel.run(a, b, c, d)
        torch_ns.synchronize()
        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        packedKernel.run(a, b, c, d)
        et.record()
        torch_ns.synchronize()
        if not self._debug_dumped:
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
