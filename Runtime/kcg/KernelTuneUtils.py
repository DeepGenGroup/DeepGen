import time
import glob
import multiprocessing
from kcg.Kernel import *
from kcg.Operators import attention, matmul

ctx = multiprocessing.get_context('spawn')
Process = ctx.Process

class BenchmarkConfig :
    def __init__(self):
        self.benchmarkCount = 5
        self.warmupCount = 1
        self.keepTopNum = 10
        self.max_kernel_per_iter = 20
        self.result_json_path = ""
        self.max_try_count = 20


class TuneResult :
    def __init__(self):
        self.OpTy : Type[OpInterface] = None
        self.bestSpeedup : float = 0.0
        self.bestConfigPkl : str = None
        self.bestKernelConfig : KernelConfigs = None
        self.bestKernelBaseArg : List = None
    
    def saveToPkl(self) :
        if self.bestKernelConfig is not None :
            ba_str = ""
        for e in self.bestKernelBaseArg :
            ba_str += f"{e}:"
        pklPath = PathManager().tmp_dir() + f"/bestConfig_{self.OpTy.__name__}_{ba_str}.pkl"
        serialize_to_file(pklPath, self.bestKernelConfig)
        print(f"===== Best kernel config has been saved to {pklPath}")
        self.bestConfigPkl = pklPath
        return pklPath



def init_cuda(_devId) :
    DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
    print("init_cuda devid=",_devId)
    DeviceInfo.set_visible_devices([_devId])
    DeviceInfo.set_current_device(_devId)  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
    if not torch.cuda.is_available() :
        torch.cuda.init()
        torch.cuda.empty_cache()

def __compile_task_func(OpTy : Type[OpInterface], info : CompileNeededInfo , deviceId:int, backendtype : EnumBackendType, arch : str , index : int) :
    # print("enter __compile_task_func",flush=True)
    op = OpTy()
    # print("[D] info.baseArgs = ", info.baseArgs)
    # print("[D] info dtype =",info.torchDataType)
    ba, kernlCfg, compiledKernel = op.Compile(deviceId, backendtype, arch, info)
    pklName = f"{PathManager.pikle_dir()}/{deviceId}/kfg_{index}.pkl"
    # print(f"__compile_task_func : ba = {ba}")
    serialize_to_file(pklName, (ba, kernlCfg))  # pack (baseArgs, runtime config) to a pkl



def compile_kernel(OpTy, tsGenerator : TsGeneratorType, deviceId:int, backendtype : EnumBackendType, arch : str, kernelLimit = 10) -> bool:
    # shape, dtypeInt = [[1, 32, 2048, 128], 4]
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
            if g_index >= kernelLimit:
                break 
            if len(procs) >= maxProcsLimit :
                for pp in procs :
                    pp.join()
                procs.clear()
        except StopIteration as e :
            iterationEnds = True
            break
        except BaseException as e :
            print('[Deepgen Exception during compiling] ', e)
            traceback.print_exc()
            
            continue
    # wait all procs end
    for pp in procs :
        pp.join()
    procs.clear()
    return iterationEnds
    

def __runBenchmark(op : OpInterface, cfg : KernelConfigs, baseArg : List, warmupCount : int, benchCount : int,devId : int) -> Tuple[float, str] :
    op.InitBaseArgs(baseArg)
    op.GetBaselineInputTensor(devId)
    kernel = op.GetCompiledKernel(cfg,devId)
    
    op.Test_warmup(kernel,warmupCount,devId)
    r0,t0 = op.Test_baseline(devId)
    r,t = op.Test_benchmark(kernel, benchCount , devId)
    acc = 0
    if torch.allclose(r,r0,rtol=2e-6,atol=1e-15) :
        acc = t0 / t
        print(f"Test Correct! speedup = {acc}")
    else:
        print("Test Error!")
    return (acc,cfg.kernelFuncName)
   
def do_benchmark(OpTy : Type[OpInterface], devId : int, benchConfig : BenchmarkConfig, maxSppedups : List[Dict], tuneResult : TuneResult):
    init_cuda(devId)
    save_to = benchConfig.result_json_path
    if len(save_to) > 0 and os.path.exists(save_to) :
        with open(save_to,'r') as f:
            obj = json.load(f)
            maxSppedups = obj['testResult']
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
                if acc > EXPECTED_SPEEDUP :
                    obj = {"name" : funName, "speedup" : acc}
                    maxSppedups.append(obj)
                    maxSppedups.sort(key= lambda x: x["speedup"],reverse=True)
                    if len(maxSppedups) > benchConfig.keepTopNum :
                        maxSppedups = maxSppedups[0:benchConfig.keepTopNum]
                    # record best one
                    if acc > tuneResult.bestSpeedup :
                        tuneResult.OpTy = OpTy
                        tuneResult.bestSpeedup = acc
                        tuneResult.bestKernelBaseArg = ba
                        tuneResult.bestKernelConfig = config
            except BaseException as e: 
                print('[Deepgen Exception] ',e)
                msg = traceback.format_exc()
                print(msg, flush=True)
            except IOError as e: 
                print('[Deepgen IOError] ',e)
    print(f" ======== benchmark end . maxSppedups = {maxSppedups} ===========")
    if len(save_to) > 0 :
        with open(save_to,'w+') as f:
            rr = {"testResult" : maxSppedups}
            json.dump(rr,f,indent=2)
        
        
def get_tuning_space(OpTy : Type[OpInterface], cfgPath : str) -> TsGeneratorType :
    if OpTy is matmul.MatmulOp :
        import Runtime.kcg.tuning.NewCfgTest as ns_mm
        return ns_mm.getTuneSpace(cfgPath)
    if OpTy is attention.AttentionOp :
        import Runtime.kcg.tuning.attn_FP32_test as ns_attentiopn
        return ns_attentiopn.getTuneSpace([1, 32, 2048, 128],[])
        # return ns_attentiopn.getTuneSpace([1, 32, 128, 128],[])
    assert False, f'[Error] getTuningSpace : Invalid OpTy:{OpTy.__name__}'
    

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# def test_simple() :
#     hsacoPath = "/tmp/compile-ptx-src-ebe279.cubin"
#     kernelName = "attention1"
#     dt = torch.float32
#     dev = 7
#     op = attention.AttentionOp()
    
#     init_cuda(dev)
#     op.InitBaseArgs([[1, 32, 2048, 128],4])
#     info = KernelConfigs(hsacoPath,kernelName,[dt,dt,dt,dt], EnumBackendType.CUDA)
# # ==== shmBytes is 29440
# # attOp: 29440,[32, 32, 1],[256, 1, 1],
#     info.m_gridDims = [32, 32, 1]
#     info.m_blockDims = [256,1,1]
#     info.shmBytes = 29440
#     kernel = op.GetCompiledKernel(info,dev)
#     a,b,c =  op.GetBaselineInputTensor(dev)
#     aa,bb,cc,dd = op.GetBenchmarkInputTensor(dev)
#     kernel.run( aa,bb,cc,dd )
#     print(dd)
#     # op.Test_baseline(7)
#     # op.Test_benchmark(kernel,dev)

EXPECTED_SPEEDUP = 0

# 交替进行compile & benchmark，每次 {kernelLimit} 个 krnl
def do_compile_and_benchmark_alternatively(opty : Type[OpInterface], ts : TsGeneratorType , cc : BenchmarkConfig, backend : EnumBackendType , arch : str ,devId : int) -> TuneResult:
    maxSpeedups = []
    currIter = 0
    res = TuneResult()
    while not compile_kernel(opty,ts,devId,backend,arch, cc.max_kernel_per_iter) :
        print(f"=========== benchmark {currIter} ====== ")
        currIter+=1
        do_benchmark(opty,devId,cc,maxSpeedups,res)
    do_benchmark(opty,devId,cc,maxSpeedups,res)
    if res.bestSpeedup > EXPECTED_SPEEDUP :
        pklPath = res.saveToPkl()
        print(f"==== good tuneRes has been saved to {pklPath}")
        return res
    else:
        return None

    
def kernel_compile_tuning(opty : Type[OpInterface], cfgFile : str, devId :int, tuningSpace : TsGeneratorType) -> TuneResult :

    assert os.path.exists(cfgFile), f'Tuningparam file {cfgFile} not exist'

    if is_hip():
        backend = EnumBackendType.HIP
        arch = "906"
    else:
        backend = EnumBackendType.CUDA
        arch = "80"
    
    PathManager.init(clearPkl=True, clearCache=True)
    os.mkdir(f"{PathManager().pikle_dir()}/{devId}")
    resultPath = str(PathManager.project_dir()) + "/testResult.json"
    if os.path.exists(resultPath):
        os.remove(resultPath)

    cc = BenchmarkConfig()
    cc.keepTopNum = 1
    cc.max_kernel_per_iter = 1
    cc.result_json_path = resultPath
    
    st = time.time()
    print(f"=====  start at : {st}")
    tuneRes = do_compile_and_benchmark_alternatively(opty,tuningSpace,cc,backend,arch,devId)
    
    # compile_kernel(opty,tuningSpace,devId,backend,arch,kernelLimit=1)
    # do_benchmark(opty,devId,cc,[])
    et = time.time()
    print(f"=====  Total spends {(et - st)/ 60} minutes")
    return tuneRes
    