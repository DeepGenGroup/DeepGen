
from KCGTask import *
import multiprocessing 
from ConfigGenerator import BuildTuningSpace, ParseTuningSpace
import sys
from RemoteUtils import *

def main():    
    # 路径管理器初始化 & 清理缓存数据（可选）
    PathManager.init(clearPkl=True, clearTmp=True, clearCache=True,clearDump=True)

    # Tuning 参数空间配置文件
    tuning_param_file = f'{PathManager.project_dir()}/TuningConfigs/GEMM_configs_2.json'
    # perf文件路径前缀(用于记录当前最佳性能的case)
    perfPathPrefix = f'{PathManager.project_dir()}/__test_bmm'
    # 调优空间存储文件
    cacheTuningSPaceFile = f'{PathManager.project_dir()}/TuningCombs/test_bmm_1024.json'
    # 最大编译进程数
    maxCompilingProcess = 100
    # 可见设备列表
    gpu_devices = [7]  
    # 调优空间生成策略（0：先生成space再剪枝 1：直接生成剪枝后的space）
    tuningSpaceGenMode = 1  
    # 当前后端类型 & 架构信息
    backendType = EnumBackendType.CUDA  
    arch = "80"
    M = N = K = 1024
    batch = 1
    elementType = torch.float32
    sshsender = RemoteFileSender("10.18.96.58","2133","xushilong","xushilong")
    runMode = EnumRunMode.AsRemotePerftester
    ######################################################################################
    # 调优空间生成
    totalLen = 0
    if runMode != EnumRunMode.AsRemotePerftester:
        print('===== Waiting for tuning space build ... ',flush=True)
        totalLen = BuildTuningSpace(tuning_param_file, cacheTuningSPaceFile, tuningSpaceGenMode)
        print(f'===== Tuning space build OK! size = {totalLen} ==== ',flush=True)
        if totalLen <= 0 :
            return
    
    # 编译及benchmark启动
    if runMode != EnumRunMode.GetTuneSpace_Local_Only :
        need_compile = True
        need_bencmark = True
        if runMode == EnumRunMode.AsRemotePerftester :
            need_compile = False
        tm =  ParallelTaskManager(
            gpu_devices,
            totalLen, cacheTuningSPaceFile, perfPathPrefix, 
            benchmarkcnt=10,  # 单个case执行次数
            warmupcnt=1,  # 每轮执行warmup次数
            keepTopNum = 15,  # 最佳结果保留前xx（取中位数）
            torchDynamicLogPath='',  # 是否周期性记录torch对应kernel的性能变化
            nTorchEpsInitTest=300,  # 测量torch的baseline时所运行次数（取中位数）
            atol=1e-3,  # 绝对误差
            rtol=1e-3,   # 相对误差
            sender = sshsender
        )
        tm.run(
            backendtype=backendType ,  # 后端类型
            archInfo=arch,
            maxProcess= maxCompilingProcess , # 编译kernel的最大进程数 
            needCompile=need_compile, # 是否执行编译过程
            needPerfTest=need_bencmark, # 是否执行benchmark过程
            startFrom=0,     # 从空间里编号为x的config开始执行
            baselineInitInfo= [batch, M, N ,K , elementType]    # 用于pytorch基准的测试。因为PerfTester设计上的通用性（不关注kernel的具体参数），理论上该值只能运行时查找，且不一定保证唯一。考虑到实现的复杂性，这里先简单处理，后期改进（结合其他算子、各种参数再重新设计）
        )

if __name__ == '__main__' :
    main()