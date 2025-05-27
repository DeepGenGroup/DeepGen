
from KCGTask import *
import multiprocessing 
from ConfigGenerator import BuildTuningSpace, ParseTuningSpace
import sys
from RemoteUtils import *
from RunManager import StartParam

from Operators.matmul import MatmulOp

def main_process(
    runMode,
    tuning_param_file,cacheTuningSPaceFile,tuningSpaceGenMode,
    gpu_devices,perfPathPrefix,backendType,keepTopNum,
    remoteTesterIP,
    remoteTesterSSHPort,
    remoteTesterUsername,
    remoteTesterPwd,
    needClearDir : bool
):    
    # 路径管理器初始化 & 清理缓存数据（可选）
    PathManager.init(clearPkl=needClearDir, clearTmp=needClearDir, clearCache=needClearDir,clearDump=needClearDir)
    ######################################################################################
    st = time.time()

    # 调优空间生成
    totalLen = 0
    if runMode != EnumRunMode.AsRemotePerftester:
        print(f'===== Waiting for tuning space build with {tuning_param_file} ... ',flush=True)
        totalLen = BuildTuningSpace(tuning_param_file, cacheTuningSPaceFile, tuningSpaceGenMode)
        print(f'===== Tuning space build OK! size = {totalLen} ==== ',flush=True)
        if totalLen <= 0 :
            return
    
    # 编译及benchmark启动
    isAsRemoteTester = False
    remoteBenchmarker = RemoteSSHConnect(remoteTesterIP, remoteTesterSSHPort, remoteTesterUsername,remoteTesterPwd)

    if runMode.value != EnumRunMode.GetTuneSpace_Local_Only.value :
        need_compile = True
        need_bencmark = True
        if runMode.value == EnumRunMode.AsRemotePerftester.value :
            need_compile = False
            isAsRemoteTester = True
            remoteBenchmarker = None
        if runMode.value == EnumRunMode.CallRemotePerftester.value :
            need_compile = True
            isAsRemoteTester = False
            assert remoteBenchmarker is not None
        
        tm =  ParallelTaskManager(
            runMode,
            gpu_devices,
            totalLen, cacheTuningSPaceFile, perfPathPrefix, 
            benchmarkcnt=10,  # 单个case执行次数
            warmupcnt=1,  # 每轮执行warmup次数
            keepTopNum = keepTopNum,  # 最佳结果保留前xx（取中位数）
            torchDynamicLogPath='',  # 是否周期性记录torch对应kernel的性能变化
            nTorchEpsInitTest=300,  # 测量torch的baseline时所运行次数（取中位数）
            atol=1e-3,  # 绝对误差
            rtol=1e-3,   # 相对误差
            remoteTestser = remoteBenchmarker
        )
        op = MatmulOp()
        tm.setTargetOp(op)
        print("===== set target op ===========")
        tm.run(
            backendType ,  # 后端类型
            archInfo=arch,
            maxProcess= maxCompilingProcess , # 编译kernel的最大进程数 
            needCompile=need_compile, # 是否执行编译过程
            needPerfTest=need_bencmark, # 是否执行benchmark过程
            startFrom=0,     # 从空间里编号为x的config开始执行
            isAsRemoteTester=isAsRemoteTester
        )
        et = time.time()
        print(f">>>> ====== Total Time Costs : {(et-st)/3600} Hours")
    return
    
if __name__ == '__main__' :
    tuning_param_file_list = []
    perfPathPrefix_list = []
    cacheTuningSPaceFile_list = []
    

    
    # Tuning 参数空间配置文件
    tuning_param_file = f'{PathManager.project_dir()}/TuningConfigs/GEMM_configs_2.json'
    # perf文件路径前缀(用于记录当前最佳性能的case)
    perfPathPrefix = f'{PathManager.project_dir()}/_gemmPerf'
    # 调优空间存储文件
    cacheTuningSPaceFile = f'{PathManager.project_dir()}/TuningCombs/tuingspace_gemm_debug.json'
    # 最大编译进程数
    maxCompilingProcess = 100
    # 可见设备列表
    gpu_devices = [7]  
    # 调优空间生成策略（0：先生成space再剪枝 1：直接生成剪枝后的space）
    tuningSpaceGenMode = 1  
    # 当前后端类型 & 架构信息
    backendType = EnumBackendType.CUDA  
    arch = "80"
    remoteTesterIP = "10.18.96.58"
    remoteTesterSSHPort = 2133
    remoteTesterUsername = "xushilong"
    remoteTesterPwd = "xushilong"
    runMode = EnumRunMode.CallRemotePerftester
    keepTopNum = 100

    tuning_param_file_list.append(tuning_param_file)
    perfPathPrefix_list.append(perfPathPrefix)
    cacheTuningSPaceFile_list.append(cacheTuningSPaceFile)
    
    param = StartParam()
    if len(sys.argv) > 1 :
        startParamJsonPath = sys.argv[1]
        needClearDir = int(sys.argv[2]) > 0
        print(f"[D] needClearDir = {needClearDir}")
        param.parseFromJson(startParamJsonPath)
        # Tuning 参数空间配置文件
        tuning_param_file_list =  param.tuning_param_file
        # perf文件路径前缀(用于记录当前最佳性能的case)
        perfPathPrefix_list =  param.perfPathPrefix
        # 调优空间存储文件
        cacheTuningSPaceFile_list =  param.cacheTuningSPaceFile
        
        def addProjectDirAhead(arr : List) :
            for i in range(len(arr)):
                arr[i] = str(PathManager.project_dir()) + "/" + arr[i]
        
        addProjectDirAhead(tuning_param_file_list) 
        addProjectDirAhead(perfPathPrefix_list) 
        addProjectDirAhead(cacheTuningSPaceFile_list) 
        
        # 最大编译进程数
        maxCompilingProcess = param.maxCompilingProcess
        # 可见设备列表
        gpu_devices = param.gpu_devices
        # 调优空间生成策略（0：先生成space再剪枝 1：直接生成剪枝后的space）
        tuningSpaceGenMode = param.tuningSpaceGenMode
        # 当前后端类型 & 架构信息
        backendType = param.backendType
        arch = param.arch
        runMode = param.runMode
        keepTopNum = param.keepTopNum

    
    for i in range(len(tuning_param_file_list)) :
        tuning_param_file = tuning_param_file_list[i]
        perfPathPrefix = perfPathPrefix_list[i]
        cacheTuningSPaceFile = cacheTuningSPaceFile_list[i]
        main_process(
            runMode,
            tuning_param_file,cacheTuningSPaceFile,tuningSpaceGenMode,
            gpu_devices,perfPathPrefix,backendType,keepTopNum,
            remoteTesterIP,remoteTesterSSHPort,remoteTesterUsername,remoteTesterPwd,
            needClearDir
        )
    
