import importlib
import math
import torch
from kcg.Kernel import *
from kcg.Operators.attention import AttentionOp, AttentionBaseArgs, AttentionTuningArgs


def _causal_upper_mask(S, device, dtype):
    return torch.where(
        torch.triu(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=1),
        torch.full((S, S), float("-inf"), device=device, dtype=dtype),
        torch.zeros((S, S), device=device, dtype=dtype),
    )


@kcg_kernel
def _h2o_origin_kernel(q_ptr, k_ptr, v_ptr, out_ptr, row_sum_ptr):
    "DUMP CODES"
    pass


def _h2o_origin(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                out: torch.Tensor, row_sum: torch.Tensor):
    return _h2o_origin_kernel(q, k, v, out, row_sum)


class H2OOriginOp(AttentionOp):
    """H2O origin fused kernel: causal attention output plus in-kernel row_sum accumulation."""

    def __init__(self):
        super().__init__()
        self.TuningArgs = AttentionTuningArgs()
        self.TuningArgs.kernelNamePrefix = "H2OOrigin"
        self.RowSum_Baseline = None
        self.RowSum_Benchmark = None

    def GetBenchmarkInputTensor(self, devId: int) -> List[torch.Tensor]:
        if self.InputTensors_Benchmark is None:
            [q, k, v] = self.GetBaselineInputTensor(devId)
            shapeList = self.BaseArgs.intValues[0]
            dtypeInt = self.BaseArgs.intValues[1]
            [b0, b1, m, n] = shapeList
            ety = ToTorchType(EnumKernelDType(dtypeInt))
            qq = q.transpose(-1, -2).contiguous()
            kk = k
            out = torch.empty((b0, b1, m, n), dtype=ety, device=dev_name(devId))
            row_sum = torch.zeros((b0, b1, m), dtype=ety, device=dev_name(devId))
            self.InputTensors_Benchmark = [qq, kk, v, out, row_sum]
        return self.InputTensors_Benchmark

    def InitLibInterface(self):
        if self.CompileKernel is None or self.SetPlatform is None:
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernel = mod.compile_h2o_origin
            self.SetKernelName = mod.set_kernel_name
            self.SetPlatform = mod.set_platform

    def Compile(self, deviceId: int, backendtype: EnumBackendType, arch: str, info: CompileNeededInfo, opt: CompileOption = None):
        _backend = {
            EnumBackendType.CUDA.value: 1,
            EnumBackendType.HIP.value: 2,
            EnumBackendType.MLU.value: 3,
            EnumBackendType.NPU.value: 4,
        }.get(backendtype.value)
        assert _backend is not None, f"invalid backendtype {backendtype}"

        self.InitLibInterface()
        self.SetPlatform(_backend, arch)
        dataTypeInt = ToEnumIntDType(info.torchDataType)
        self.InitBaseArgs([info.baseArgs, dataTypeInt])
        shape, config = info.tsArgs
        self.SetKernelName(info.kernelName)
        dtype_str = EnumKernelDType(dataTypeInt).name

        res = self.CompileKernel(shape, config, dtype_str)
        if not res:
            raise RuntimeError(f"Kernel compilation failed for {info.kernelName}")

        dt = self.BaseArgs.getTorchDType()
        inConfig = KernelConfigs(res, info.kernelName, [dt, dt, dt, dt, dt], backendtype)
        inConfig.m_gridDims = list(info.gridDims)
        inConfig.m_blockDims = list(info.blockDims)
        inConfig.operatorKind = EnumOperator.Attention
        inConfig.shmBytes = info.shmBytes
        packedKernel = self.GetCompiledKernel(inConfig, deviceId)
        return ([info.baseArgs, dataTypeInt], inConfig, packedKernel)

    def GetSignature(self, dtypes):
        dtypeA = dtypes[0]
        q = torch.randn((1, 3, 100, 100), device="cpu", dtype=dtypeA)
        k = torch.randn((1, 3, 100, 100), device="cpu", dtype=dtypeA)
        v = torch.randn((1, 3, 100, 100), device="cpu", dtype=dtypeA)
        out = torch.empty((1, 3, 100, 100), device="cpu", dtype=dtypeA)
        row_sum = torch.empty((1, 3, 100), device="cpu", dtype=dtypeA)
        return _h2o_origin(q, k, v, out, row_sum)

    def Test_baseline(self, devId: int):
        [q, k, v] = self.GetBaselineInputTensor(devId)
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        mask = _causal_upper_mask(q.shape[-2], q.device, q.dtype).unsqueeze(0).unsqueeze(0)
        _scores = torch.matmul(q, k) * scale + mask
        _p_max = torch.max(_scores, dim=-1, keepdim=True).values
        _p_shifted = torch.sub(_scores, _p_max)
        _p_exp = torch.exp(_p_shifted)
        _p_sum = torch.sum(_p_exp, dim=-1, keepdim=True)
        _probs = torch.div(_p_exp, _p_sum)
        _ = torch.matmul(_probs, v)

        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        scores = torch.matmul(q, k) * scale + mask
        p_max = torch.max(scores, dim=-1, keepdim=True).values
        p_shifted = torch.sub(scores, p_max)
        p_exp = torch.exp(p_shifted)
        p_sum = torch.sum(p_exp, dim=-1, keepdim=True)
        probs = torch.div(p_exp, p_sum)
        self.OutputTensor_Baseline = torch.matmul(probs, v)
        self.RowSum_Baseline = probs.sum(dim=2)
        et.record()
        torch_ns.synchronize()
        return (self.OutputTensor_Baseline, st.elapsed_time(et))

    def Test_benchmark(self, packedKernel, benchmarkCount, devId):
        qq, kk, v, out, row_sum = self.GetBenchmarkInputTensor(devId)
        row_sum.zero_()
        packedKernel.run(qq, kk, v, out, row_sum)
        torch_ns.synchronize()

        st = torch_ns.Event(enable_timing=True)
        et = torch_ns.Event(enable_timing=True)
        st.record()
        row_sum.zero_()
        packedKernel.run(qq, kk, v, out, row_sum)
        et.record()
        torch_ns.synchronize()
        self.RowSum_Benchmark = row_sum
        return (out, st.elapsed_time(et))

    def InitBaseArgs(self, args):
        shape, dtypeInt = args
        self.BaseArgs = AttentionBaseArgs()
        self.BaseArgs.intValues = [shape, dtypeInt]
        self.TuningArgs = AttentionTuningArgs(EnumKernelDType(dtypeInt))
        self.TuningArgs.kernelNamePrefix = "H2OOrigin"
