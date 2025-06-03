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
    print("[D] info.baseArgs = ", info.baseArgs)
    print("[D] info dtype =",info.torchDataType)
    ba, kernlCfg, compiledKernel = op.Compile(deviceId, backendtype, arch, info)
    pklName = f"{PathManager.pikle_dir()}/{deviceId}/kfg_{index}.pkl"
    print(f"__compile_task_func : ba = {ba}")
    serialize_to_file(pklName, (ba, kernlCfg))  # pack (baseArgs, runtime config) to a pkl


g_index : int = 0
def compile_kernel(OpTy, tsGenerator : TsGeneratorType, deviceId:int, backendtype : EnumBackendType, arch : str, kernelLimit = 10) -> bool:
    # shape, dtypeInt = [[1, 32, 2048, 128], 4]
    global g_index
    g_index = 0
    maxProcsLimit = 50
    procs : List[Process] = []
    iterationEnds = False
    print('========= compiling ============')
    while True:
        try:
            needInfo = next(tsGenerator)
        # for needInfo in tsGenerator :
            print(f"needInfo.tsArgs = {needInfo.tsArgs}") 
            # create compile process
            p = Process(target=__compile_task_func,args=(OpTy,needInfo,deviceId,backendtype,arch, g_index))
            procs.append(p)
            p.start()
            g_index += 1
            if g_index > kernelLimit:
                break 
            if len(procs) >= maxProcsLimit :
                for pp in procs :
                    pp.join()
                procs.clear()
        except StopIteration as e :
            iterationEnds = True
            break
    # wait all procs end
    for pp in procs :
        pp.join()
    procs.clear()
    return iterationEnds
    

def __runBenchmark(op : OpInterface, cfg : KernelConfigs, baseArg : List, warmupCount : int, benchCount : int,devId : int) -> float :
    op.InitBaseArgs(baseArg)
    op.GetBaselineInputTensor(devId)
    kernel = op.GetCompiledKernel(cfg,devId)
    
    # op.Test_warmup(kernel,warmupCount,devId)
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
    
def do_benchmark(OpTy : Type[OpInterface], devId : int, benchConfig : BenchmarkConfig, maxSppedups : List[Dict], tempSaveJson : str = ''):
    init_cuda(devId)
    name_format = f"{PathManager.pikle_dir()}/{devId}/*.pkl"
    pkls = glob.glob(name_format)
    if len(pkls) > 0 :
        for pkl in pkls :
            try:
                op = OpTy()
                (ba ,config ) = deserialize_from_file(pkl) 
                assert isinstance(ba, List)
                assert isinstance(config, KernelConfigs)
                print(f'[D] after desrialize : {ba}')
                acc, funName = __runBenchmark(op, config, ba, 1, 5 , devId)
                os.remove(pkl)
                if acc > 0 :
                    obj = {"name" : funName, "speedup" : acc}
                    maxSppedups.append(obj)
                    maxSppedups.sort(key=lambda x: x["speedup"],reverse=True)
                    if len(maxSppedups) > benchConfig.keepTopNum :
                        maxSppedups = maxSppedups[0:benchConfig.keepTopNum]
            except BaseException as e: 
                print('[Deepgen Exception] ',e)
            except IOError as e: 
                print('[Deepgen IOError] ',e)
                    
    print(f" ======== benchmark end . maxSppedups = {maxSppedups} ===========")
    if len(tempSaveJson) > 0 :
        with open(tempSaveJson,'w+') as f:
            result = {"res" : []}
            result['res'] = maxSppedups
            json.dump(result,f)
    
def get_tuning_space(OpTy : Type[OpInterface], cfgPath : str) -> TsGeneratorType :
    if OpTy is matmul.MatmulOp :
        import NewCfgTest as ns_mm
        return ns_mm.getTuneSpace(cfgPath)
    if OpTy is attention.AttentionOp :
        import attn_FP32_test as ns_attentiopn
        return ns_attentiopn.getTuneSpace([1, 32, 2048, 128],[])
        # return ns_attentiopn.getTuneSpace([1, 32, 128, 128],[])
    assert False, f'[E] getTuningSpace : Invalid OpTy:{OpTy.__name__}'
    

    

def test_simple() :
    hsacoPath = "/tmp/compile-ptx-src-ebe279.cubin"
    kernelName = "attention1"
    dt = torch.float32
    dev = 7
    op = attention.AttentionOp()
    
    init_cuda(dev)
    op.InitBaseArgs([[1, 32, 2048, 128],4])
    info = KernelConfigs(hsacoPath,kernelName,[dt,dt,dt,dt], EnumBackendType.CUDA)
# ==== shmBytes is 29440
# attOp: 29440,[32, 32, 1],[256, 1, 1],
    info.m_gridDims = [32, 32, 1]
    info.m_blockDims = [256,1,1]
    info.shmBytes = 29440
    kernel = op.GetCompiledKernel(info,dev)
    a,b,c =  op.GetBaselineInputTensor(dev)
    aa,bb,cc,dd = op.GetBenchmarkInputTensor(dev)
    kernel.run( aa,bb,cc,dd )
    print(dd)
    # op.Test_baseline(7)
    # op.Test_benchmark(kernel,dev)


def compile_and_benchmark(opty : Type[OpInterface], ts : TsGeneratorType , cc : BenchmarkConfig, backend : EnumBackendType , arch : str ,devId : int) :
    maxSpeedups = []
    while compile_kernel(opty,ts,devId,backend,arch, kernelLimit=10) :
        print("=========== benchmark ======")
        do_benchmark(opty,devId,cc,maxSpeedups)
    do_benchmark(opty,devId,cc,maxSpeedups)

    
if __name__ == '__main__' :
    # test_simple()
    cfgFile = "/home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json"
    opty = attention.AttentionOp
    devId = 7
    backend = EnumBackendType.CUDA
    arch = "80"
    # opty = matmul.MatmulOp
    PathManager.init(clearPkl=True)
    os.mkdir(f"{PathManager().pikle_dir()}/{devId}")
    print("get_tune_space",flush=True)
    ts = get_tuning_space(opty, cfgFile)
    cc = BenchmarkConfig()
    
    tsSize = 0
    for cfg in ts :
        tsSize += 1
    print("tsSize = ", tsSize)
    
    # # compile_and_benchmark(opty,ts,cc,backend,arch,devId)
    # compile_kernel(opty,ts,devId,EnumBackendType.CUDA,"80",kernelLimit=3)
    # do_benchmark(opty,devId,cc,[],'/home/xushilong/DeepGen/testResult.json')