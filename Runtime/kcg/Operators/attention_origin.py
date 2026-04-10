import importlib
import math
import torch
from kcg.Kernel import *
from kcg.Operators.attention import AttentionOp, AttentionTuningArgs


def _causal_upper_mask(S, device, dtype):
    return torch.where(
        torch.triu(torch.ones((S, S), device=device, dtype=torch.bool), diagonal=1),
        torch.full((S, S), float("-inf"), device=device, dtype=dtype),
        torch.zeros((S, S), device=device, dtype=dtype),
    )


class AttentionOriginOp(AttentionOp):
    """Origin attention fused kernel: scale after mm1 + causal mask + softmax + PV."""

    def __init__(self):
        super().__init__()
        self.TuningArgs = AttentionTuningArgs()
        self.TuningArgs.kernelNamePrefix = "AttentionOrigin"

    def InitLibInterface(self):
        if self.CompileKernel is None or self.SetPlatform is None:
            spec = importlib.util.spec_from_file_location("deepgen", PathManager.kcg_lib_deepgen_path())
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.CompileKernel = mod.compile_attention_origin
            self.SetKernelName = mod.set_kernel_name
            self.SetPlatform = mod.set_platform

    def InitBaseArgs(self, args):
        shape, dtypeInt = args
        self.BaseArgs.intValues = [shape, dtypeInt]
        self.TuningArgs = AttentionTuningArgs(EnumKernelDType(dtypeInt))
        self.TuningArgs.kernelNamePrefix = "AttentionOrigin"

    def Compile(self, deviceId: int, backendtype: EnumBackendType, arch: str, info: CompileNeededInfo, opt: CompileOption = None):
        backend_map = {
            EnumBackendType.CUDA.value: 1,
            EnumBackendType.HIP.value: 2,
            EnumBackendType.MLU.value: 3,
            EnumBackendType.NPU.value: 4,
        }
        _backend = backend_map.get(backendtype.value)
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
        inConfig = KernelConfigs(res, info.kernelName, [dt, dt, dt, dt], backendtype)
        inConfig.m_gridDims = list(info.gridDims)
        inConfig.m_blockDims = list(info.blockDims)
        inConfig.operatorKind = EnumOperator.Attention
        inConfig.shmBytes = info.shmBytes
        packedKernel = self.GetCompiledKernel(inConfig, deviceId)
        return ([info.baseArgs, dataTypeInt], inConfig, packedKernel)

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
        et.record()
        torch_ns.synchronize()
        return (self.OutputTensor_Baseline, st.elapsed_time(et))
