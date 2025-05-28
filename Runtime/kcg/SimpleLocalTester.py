import glob
from kcg.Utils import *
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Operators import matmul
import multiprocessing

ctx = multiprocessing.get_context('spawn')
Process = ctx.Process

def GetTuneSpaceDict(Op : OpInterface, tuingSpaceJsonPath : str) :
    if len(tuingSpaceJsonPath) > 0 :
        with open(tuingSpaceJsonPath) as f :
            obj = json.load(f)
            tse = TuningSpaceEncoder(obj['template'])
            cfgstrs = obj['cfgs']
            for cfgString in cfgstrs :
                yield tse.decode(cfgString)
    else:
        ...



def RunKernelWithKernelConfig(op : OpInterface, cfg : KernelConfigs, devId : int) :
    op.GetBaselineInputTensor(devId)
    kernel = op.GetCompiledKernel(cfg,devId)
    r0,t0 = op.Test_baseline(devId)
    r,t = op.Test_benchmark(kernel,devId)
    if torch.allclose(r,r0,rtol=1e-3,atol=1e-3) :
        print("Test Correct!")
    else:
        print("Test Error!")
        print(r)
    # kernel.run(*args)

        
def RunKernelBinary(op : OpInterface, binpath : str, kernelFuncName : str, griddims : List[int] , blockdims : List[int], shmBytes : int ,args : List[torch.Tensor]) :
    dtypes = []
    for tensor in args :
        dtypes.append(tensor.dtype)
    backend = EnumBackendType.CUDA
    if is_hip():
        backend = EnumBackendType.HIP
    kernelConfig = KernelConfigs(binpath,kernelFuncName,dtypes,backend)
    kernelConfig.m_gridDims = griddims #[256,1,1]
    kernelConfig.m_blockDims = blockdims #[128,1,1]
    kernelConfig.shmBytes = shmBytes
    devId = args[0].get_device()
    kernel = op.GetCompiledKernel(kernelConfig,devId)
    kernel.run(*args)
    
def CompileKernelWithTuneList(op : OpInterface, deviceId:int, backendtype : EnumBackendType, arch : str, tuningArgs : List[int], baseArgs : List) -> KernelConfigs :
    op.InitBaseArgs(baseArgs)  # 本质都是给 tuningspace结构体赋值
    op.SetTuningArgs(tuningArgs) 
    _, kernlCfg, _ = op.Compile(deviceId, backendtype, arch)
    return kernlCfg

def compile_with_dict(OpTy : Type[OpInterface], baseargs : List, deviceId:int, backendtype : EnumBackendType, arch : str , configDict : Dict, index : int) :
    op = OpTy()
    op.InitBaseArgs(baseargs)
    op.TuningArgs.assignWithDict(configDict)
    _, kernlCfg, _ = op.Compile(deviceId, backendtype, arch)
    pklName = f"{PathManager.pikle_dir()}/{deviceId}/kfg_{index}.pkl"
    serialize_to_file(pklName, kernlCfg)

def CompileKernelWithSapceJson(OpTy : Type[OpInterface], deviceId:int, backendtype : EnumBackendType, arch : str, tuingSpaceJsonPath : str, baseArgs : List) -> KernelConfigs :
    index = 0
    procs : List[Process] = []
    maxProcsLimit = 50
    ts = GetTuneSpaceDict(OpTy(), tuingSpaceJsonPath)
    for configDict in ts :
        p = Process(target=compile_with_dict,args=(OpTy,baseArgs,deviceId,backendtype,arch,configDict,index))
        procs.append(p)
        p.start()
        index += 1
        if len(procs) >= maxProcsLimit :
            for pp in procs :
                pp.join()
            procs.clear()

def init_cuda(_devId) :
    DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
    print("init_cuda devid=",_devId)
    DeviceInfo.set_visible_devices([_devId])
    DeviceInfo.set_current_device(_devId)  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
    if not torch.cuda.is_available() :
        torch.cuda.init()
        torch.cuda.empty_cache()
        
def compile_matmul() :
    batch, m, n, k, dtypeInt = [1, 1024,1024,1024, 4]
    args = [batch, m, n, k, dtypeInt]
    CompileKernelWithSapceJson(matmul.MatmulOp, 7, EnumBackendType.CUDA, "80", "/home/xushilong/DeepGen/TuningCombs/ts_1.json", args )

def benchmark_mm(OpTy : Type[OpInterface], devId : int):
    init_cuda(devId)
    batch, m, n, k, dtypeInt = [1, 1024,1024,1024, 4]
    args = [batch, m, n, k, dtypeInt]
    name_format = f"{PathManager.pikle_dir()}/{devId}/*.pkl"
    pkls = glob.glob(name_format)
    op = OpTy()
    if len(pkls) > 0 :
        for pkl in pkls :
            op.InitBaseArgs(args)
            op.GetBaselineInputTensor(devId)
            config : KernelConfigs = deserialize_from_file(pkl) 
            RunKernelWithKernelConfig(op,config, devId)
            
    
if __name__ == '__main__' :

    # binPath = "/tmp/compile-ptx-src-b27d15.cubin"
    # kernelName = "GEMM_bMNK1x1024x1024x1024_DTabcfloat32xfloat32xfloat32_AT1_TTmn4x4_BTmnk32x32x8_BLmn1x1_WLmn8x8_GLWab4x4_GSW2_WSWab2x2_TSWab2x1_LSU1_BM16_UNROLL4_REGP0_SHMP0_LC0_RC0_"
    # gDims = [1024, 1, 1]
    # bDims = [64, 1, 1]
    # shmBytes = 2048
    
    

    # binPath = "/tmp/compile-ptx-src-6871fa.cubin"
    # kernelName = "GEMM_testKernel"
    # gDims = [256,1,1]
    # bDims = [128,1,1]
    # shmBytes = 8192*4
    benchmark_mm(matmul.MatmulOp, 7 )
    # test_matmul(7,binPath, kernelName, gDims, bDims, shmBytes)