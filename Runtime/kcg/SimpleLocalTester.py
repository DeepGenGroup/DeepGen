import glob
from typing import Generator
from kcg.Utils import *
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Operators import matmul, attention
import multiprocessing
import attn_FP32_test as ATT

ctx = multiprocessing.get_context('spawn')
Process = ctx.Process

class BenchmarkConfig :
    def __init__(self):
        self.benchmarkCount = 5
        self.warmupCount = 1
        self.keepTopNum = 10

def init_cuda(_devId) :
    DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
    print("init_cuda devid=",_devId)
    DeviceInfo.set_visible_devices([_devId])
    DeviceInfo.set_current_device(_devId)  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
    if not torch.cuda.is_available() :
        torch.cuda.init()
        torch.cuda.empty_cache()

def __compile_task_func(OpTy : Type[OpInterface], info : CompileNeededInfo , deviceId:int, backendtype : EnumBackendType, arch : str , index : int) :
    print("enter __compile_task_func",flush=True)
    op = OpTy()
    ba, kernlCfg, compiledKernel = op.Compile(deviceId, backendtype, arch, info)
    pklName = f"{PathManager.pikle_dir()}/{deviceId}/kfg_{index}.pkl"
    print(f"__compile_task_func : ba = {ba}")
    serialize_to_file(pklName, (ba,kernlCfg))

def compile_kernel(OpTy, tsGenerator : TsGeneratorType, deviceId:int, backendtype : EnumBackendType, arch : str) :
    # shape, dtypeInt = [[1, 32, 2048, 128], 4]
    allIndex = 0
    kernelLimit = 30
    for needInfo in tsGenerator :
        procs : List[Process] = []
        maxProcsLimit = 50
        p = Process(target=__compile_task_func,args=(OpTy,needInfo,deviceId,backendtype,arch,allIndex))
        # __compile_task_func(OpTy,needInfo,deviceId,backendtype,arch,allIndex)
        procs.append(p)
        p.start()
        allIndex += 1
        if allIndex > kernelLimit:
            break 
        if len(procs) >= maxProcsLimit :
            for pp in procs :
                pp.join()
            procs.clear()
    for pp in procs :
        pp.join()
    procs.clear()

def __runBenchmark(op : OpInterface, cfg : KernelConfigs, baseArg : List, warmupCount : int, benchCount : int,devId : int) -> float :
    op.InitBaseArgs(baseArg)
    op.GetBaselineInputTensor(devId)
    kernel = op.GetCompiledKernel(cfg,devId)
    
    op.Test_warmup(kernel,warmupCount,devId)
    r0,t0 = op.Test_baseline(devId)
    for i in range(benchCount):
        r,t = op.Test_benchmark(kernel, benchCount , devId)
    acc = 0
    if torch.allclose(r,r0,rtol=1e-3,atol=1e-3) :
        acc = t0 / t
        print(f"Test Correct! speedup = {acc}")
    else:
        print("Test Error!")
    return (acc,cfg.kernelFuncName)
    
def do_benchmark(OpTy : Type[OpInterface], devId : int, benchConfig : BenchmarkConfig):
    init_cuda(devId)
    name_format = f"{PathManager.pikle_dir()}/{devId}/*.pkl"
    pkls = glob.glob(name_format)
    op = OpTy()
    maxSppedups = []
    if len(pkls) > 0 :
        for pkl in pkls :
            (ba ,config ) = deserialize_from_file(pkl) 
            acc, funName = __runBenchmark(op,config, ba, 1, 5 , devId)
            if acc > 0 :
                obj = {"name" : funName, "speedup" : acc}
                maxSppedups.append(obj)
                maxSppedups.sort(key=lambda x: x["speedup"],reverse=True)
                if len(maxSppedups) > benchConfig.keepTopNum :
                    maxSppedups = maxSppedups[0:benchConfig.keepTopNum]
                    
    print(f" ======== benchmark end . maxSppedups = {maxSppedups} ===========")
    
def get_tuning_space(OpTy : Type[OpInterface], cfgPath : str) -> TsGeneratorType :
    if OpTy is matmul.MatmulOp :
        import NewCfgTest as ns
        return ns.getTuneSpace(cfgPath)
    if OpTy is attention.AttentionOp :
        import attn_FP32_test as ns
        return ns.getTuneSpace([1, 32, 2048, 128],[])
    assert False, f'[E] getTuningSpace : Invalid OpTy:{OpTy.__name__}'
    
if __name__ == '__main__' :
    cfgFile = "/home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json"
    opty = matmul.MatmulOp
    PathManager.init()
    print("get_tune_space",flush=True)
    ts = get_tuning_space(opty, cfgFile)
    print("compiling",flush=True)
    compile_kernel(opty,ts,7,EnumBackendType.CUDA,"80")
    print("=========== benchmark ======")
    cc = BenchmarkConfig()
    do_benchmark(opty,7,cc)