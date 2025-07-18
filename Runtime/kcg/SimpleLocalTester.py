import time
import glob
from typing import Generator
import multiprocessing
from kcg.Kernel import *
import kcg.Operators.matmul as kcg_mm
import kcg.Operators.attention as kcg_att
import numpy as np

ctx = multiprocessing.get_context('spawn')
Process = ctx.Process
Value  = ctx.Value

class BenchmarkConfig :
    def __init__(self):
        self.benchmarkCount = 5
        self.warmupCount = 1
        self.keepTopNum = 10
        self.max_kernel_per_iter = 20
        self.result_json_path = ""
        self.maxCount = 0

def init_cuda(_devId : List) :
    DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
    print("init_cuda devid=",_devId)
    DeviceInfo.set_visible_devices(_devId)
    DeviceInfo.set_current_device(_devId[0])  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
    if not torch.cuda.is_available() :
        torch.cuda.init()
        torch.cuda.empty_cache()

def __compile_task_func(OpTy : Type[OpInterface], info : CompileNeededInfo , deviceId:int, backendtype : EnumBackendType, arch : str , index : int, opt : CompileOption) :
    # print("enter __compile_task_func",flush=True)
    op = OpTy()
    # print("[D] info.baseArgs = ", info.baseArgs)
    # print("[D] info dtype =",info.torchDataType)
    ba, kernlCfg, compiledKernel = op.Compile(deviceId, backendtype, arch, info, opt)
    pklName = f"{PathManager.pikle_dir()}/{deviceId}/kfg_{index}.pkl"
    # print(f"__compile_task_func : ba = {ba}")
    serialize_to_file(pklName, (ba, kernlCfg))  # pack (baseArgs, runtime config) to a pkl


g_index : int = 0
def compile_kernel(OpTy, tsGenerator : TsGeneratorType, deviceId:int, backendtype : EnumBackendType, arch : str, kernelLimit = 10, globalLimit = 0, compileOpt : CompileOption = None) -> bool:
    # shape, dtypeInt = [[1, 32, 2048, 128], 4]
    global g_index
    localIndex = 0
    maxProcsLimit = 100
    procs : List[Process] = []
    iterationEnds = False
    print('========= compiling ============')
    while True:
        try:
            if globalLimit > 0 and g_index >= globalLimit:
                iterationEnds = True
                break
            needInfo = next(tsGenerator)
        # for needInfo in tsGenerator :
            print(f"needInfo.tsArgs = {needInfo.tsArgs}") 
            print(f"shmBytes = ",needInfo.shmBytes) 
            print(f"blockDims = {needInfo.blockDims}") 
            print(f"gridDims = {needInfo.gridDims}") 
            # create compile process
            p = Process(target=__compile_task_func,args=(OpTy,needInfo,deviceId,backendtype,arch, g_index, compileOpt))
            procs.append(p)
            p.start()
            localIndex += 1
            g_index += 1
            if localIndex >= kernelLimit :
                break
            if len(procs) >= maxProcsLimit :
                for pp in procs :
                    pp.join()
                procs.clear()
        except StopIteration as e :
            iterationEnds = True
            break
        except BaseException as e :
            import traceback
            print('[Deepgen Exception during compiling] ', e)
            traceback.print_exc()
            continue
    # wait all procs end
    for pp in procs :
        pp.join()
    procs.clear()
    return iterationEnds
    

# def __runBenchmark(op : OpInterface, cfg : KernelConfigs, baseArg : List, warmupCount : int, benchCount : int,devId : int) -> Tuple[float, str] :
#     op.InitBaseArgs(baseArg)
#     op.GetBaselineInputTensor(devId)
#     kernel = op.GetCompiledKernel(cfg,devId)
    
#     op.Test_warmup(kernel,warmupCount,devId)
#     r0,t0 = op.Test_baseline(devId)
#     for i in range(benchCount):
#         r,t = op.Test_benchmark(kernel, benchCount , devId)
#     acc = 0
#     if torch.allclose(r,r0,rtol=1e-3,atol=1e-3) :
#         acc = t0 / t
#         print(f"Test Correct! speedup = {acc}")
#     else:
#         print("Test Error!")
#     return (acc,cfg.kernelFuncName,t,t0)

def is_tflops_ok(b,m,n,k,t) :
    TargetTFLOPS = 12.2 * 0.6
    return (2*b*m*n*k / t / 1e-3 / 1e12) >= TargetTFLOPS
    

def _benchProcess( OpTy : Type[OpInterface] , benchConfig : BenchmarkConfig, findGoodCase, devId, time_0 ,pkls : List, checkTFLOPS = False, checkACC : float = 0) :
    init_cuda([devId])
    save_to = benchConfig.result_json_path
    maxSppedups = []
    if len(save_to) > 0 and os.path.exists(save_to) :
        with open(save_to,'r') as f:
            obj = json.load(f)
            maxSppedups = obj['testResult']
            f.close()
    for pkl in pkls :
        try:
            (ba ,config ) = deserialize_from_file(pkl) 
            os.remove(pkl)
            assert isinstance(ba, List)
            assert isinstance(config, KernelConfigs)
            print(f'[D] after desrialize : {ba}')
            if config.operatorKind is kcg_mm.MatmulOp :
                m,n,k = ba[1:4]
                b = 1
                if len(ba[0]) > 0:
                    b =  ba[0][0]
            elif config.operatorKind is kcg_att.AttentionOp :
                ...
            # init tensors
            op = OpTy()
            op.InitBaseArgs(ba)
            op.GetBaselineInputTensor(devId)
            op.GetBenchmarkInputTensor(devId)
            print("[D] tensor shape verify ======",flush=True)
            # print(tQ.shape, tK.shape, tV.shape, flush=True)
            # print(tQQ.shape, tKK.shape, tVV.shape, flush=True)
            
            kernel = op.GetCompiledKernel(config,devId)
            # warmup
            # op.Test_warmup(kernel,1,devId)
            time_base = []
            r0 = None
            t0 = 0
            for i in range(0,7):
                r0, t0 = op.Test_baseline(7)
                time_base.append(t0)
            t0 = np.median(time_base)
            if time_0.value <= 0 :
                time_0.value = t0
            else:
                t0 = time_0.value
            # benchmark
            r,t = op.Test_benchmark(kernel, 5 , devId)
            # verify result
            acc = 0
            if torch.allclose(r,r0,rtol=1e-3,atol=1e-3) :
                acc = t0 / t
                print(f"Test Correct! {op.GetKernelName()} , speedup = {acc}")
            else:
                print("Test Error!")
            funName = config.kernelFuncName
            # save result

            # kernel.release()
            if acc > 0 :
                obj = {"name" : funName, "speedup" : acc ,"time" : t, "time_base" : t0}
                maxSppedups.append(obj)
                maxSppedups.sort(key= lambda x: x["speedup"],reverse=True)
                if len(maxSppedups) > benchConfig.keepTopNum :
                    maxSppedups = maxSppedups[0:benchConfig.keepTopNum]
                if checkTFLOPS and config.operatorKind is kcg_mm.MatmulOp and is_tflops_ok(b,m,n,k,t) :
                    findGoodCase.value = 1
                if checkACC > 0 and acc >= checkACC :
                    findGoodCase.value = 1 
        except BaseException as e: 
            print('[Deepgen Exception] ',e)
            traceback.print_exc()
        except IOError as e: 
            print('[Deepgen IOError] ',e)
            traceback.print_exc()
    if len(save_to) > 0 :
        with open(save_to,'w+') as f:
            result = {"testResult" : maxSppedups}
            json.dump(result,f,indent=2)
            f.flush()
    return

g_time0 = Value('d',0.0)
tensor_input_baseline = None
tensor_input_benchmark = None
g_result = None
g_findAvaialbleCase = Value('d',0.0)
def do_benchmark(OpTy : Type[OpInterface], devId : int, benchConfig : BenchmarkConfig, maxSppedups : List[Dict], checkTflops : bool, checkAcc : float):
    global g_time0 
    global tensor_input_baseline 
    global tensor_input_benchmark 
    global g_result 
    global g_findAvaialbleCase
    name_format = f"{PathManager.pikle_dir()}/{devId}/*.pkl"
    pkls = glob.glob(name_format)
    p = Process(target = _benchProcess, args = (OpTy,benchConfig, g_findAvaialbleCase, devId, g_time0, pkls, checkTflops, checkAcc))
    p.start()
    p.join()
    # process terminated. clean undealed pkls
    # if p.is_alive():
    #     p.terminate()
    #     p.join(5)
    for pkl in pkls :
        if os.path.exists(pkl) :
            os.remove(pkl)   # delete crashed pkl
    
    
    
def get_tuning_space(OpTy : Type[OpInterface], cfgPath : str) -> TsGeneratorType :
    if OpTy is kcg_mm.MatmulOp :
        import kcg.tuning.NewCfgTest as ns_mm
        return ns_mm.getTuneSpace(cfgPath)
    if OpTy is kcg_att.AttentionOp :
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        return ns_attentiopn.getTuneSpace([1, 32, 2048, 128],cfgPath,[])
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


# 交替进行compile & benchmark，每次 {kernelLimit} 个 krnl
def do_compile_and_benchmark_alternatively(opty : Type[OpInterface], ts : TsGeneratorType , cc : BenchmarkConfig, compileOption : CompileOption, backend : EnumBackendType , arch : str ,devId : int, checktflops:bool, checkAcc:float) :
    maxSpeedups = []
    currIter = 0

    while not compile_kernel(opty,ts,devId,backend,arch, cc.max_kernel_per_iter, cc.maxCount, compileOption) :
        print(f"=========== benchmark {currIter} ====== ")
        currIter+=1
        do_benchmark(opty,devId,cc,maxSpeedups, checktflops, checkAcc)
        if g_findAvaialbleCase.value > 0 :
            print(f"=========== Find available Case ! Stopped ====== ")
            return
    do_benchmark(opty,devId,cc,maxSpeedups, checktflops, checkAcc)
    if g_findAvaialbleCase.value > 0 :
        print(f"=========== Find available Case ! Stopped ====== ")
        return


def getInputs() :
    helpmsg = "Usage : cfgFile result_json_path start maxCount checktflops(1,0) checkAcc(float)" 
    if len(sys.argv) < 4 :
        print(helpmsg)
        assert False, f"invalid input args. {helpmsg}"
        
    cfgFile = sys.argv[1]
    result_json_path = sys.argv[2]
    start = int(sys.argv[3])
    maxCount = int(sys.argv[4])
    if len(sys.argv) > 5:
        checktflops, checkAcc = int(sys.argv[5]) > 0 , float(sys.argv[6])
    else:
        checktflops, checkAcc = True, 0
    return (cfgFile,result_json_path,start,maxCount,checktflops, checkAcc)

def main():
    cfgFile,result_json_path,start,maxCount,checktflops, checkAcc = getInputs()
    # cfgFile = "/home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json"
    # opty = kcg_mm.MatmulOp
    opty = kcg_att.AttentionOp
    devId = 7

    if is_hip():
        backend = EnumBackendType.HIP
        arch = "906"
    else:
        backend = EnumBackendType.CUDA
        arch = "80"
    
    PathManager.init(clearPkl=True, clearCache=True)
    os.mkdir(f"{PathManager().pikle_dir()}/{devId}")
    print("get_tune_space",flush=True)
    tssize = 0
    for c in  get_tuning_space(opty, cfgFile):
        tssize += 1
    print(f"==== tune space size = {tssize}")
    
    # return
    print("=== checktflops, checkAcc",checktflops, checkAcc)
    ts = get_tuning_space(opty, cfgFile)
    bc = BenchmarkConfig()
    bc.keepTopNum = 10
    bc.max_kernel_per_iter = 80
    bc.result_json_path = result_json_path
    bc.maxCount = maxCount
    st = time.time()
    print(f"=====  start at : {st}")
    
    
    if start > 0:
        i=0
        for _ in ts :
            i+=1
            if i >= start:
                break 
    co = CompileOption()
    co.fastCompile = False
    do_compile_and_benchmark_alternatively(opty,ts,bc,co,backend,arch,devId,checktflops, checkAcc)
    # compile_kernel(opty,ts,devId,backend,arch,kernelLimit=1)
    # do_benchmark(opty,devId,cc,[])
    et = time.time()
    print(f"===== Complete! Total spends {(et - st)/ 60} minutes")
    
    
if __name__ == '__main__' :
    print(time.strftime('------------- %Y-%m-%d %H:%M:%S ------------- ',time.localtime(time.time())))       # 打印按指定格式排版的时间
    main()