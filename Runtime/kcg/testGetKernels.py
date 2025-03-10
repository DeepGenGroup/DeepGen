
from KCGTask import *
import multiprocessing 
from ConfigGenerator import BuildTuningSpace, ParseTuningSpace
import sys

def main():    
    # 路径管理器初始化 & 清理缓存数据（可选）
    PathManager.init(clearPkl=True, clearTmp=True, clearCache=True,clearDump=True)

    # Tuning 参数空间配置文件
    tuning_param_file = f'{PathManager.project_dir()}/TuningConfigs/GEMM_configs_2.json'
    # perf文件路径前缀(用于记录当前最佳性能的case)
    perfPAth = f'{PathManager.project_dir()}/__test_bmm'
    # 调优空间存储文件
    # cacheTuningSPaceFile = f'{PathManager.project_dir()}/TuningCombs/tuingspace_gemm_1024LSU.json'
    cacheTuningSPaceFile = f'{PathManager.project_dir()}/TuningCombs/test_bmm_1024.json'
    # 是否只进行调优空间生成并存入 cacheTuningSPaceFile，不执行kernel编译以及benchmark
    onlyGenerateCfg = False 
    # 最大进程数
    nProcess = 100
    # 可见设备列表
    gpu_devices = [7]  
    # 调优空间生成策略（0：先生成space再剪枝 1：直接生成剪枝后的space）
    tuningSpaceGenMode = 1  
    # 当前后端类型 & 架构信息
    backendType = EnumBackendType.HIP  
    arch = "906"
    M = N = K = 256
    batch = 2
    elementType = torch.float32
    ######################################################################################
    # 命令行参数解析
    if len(sys.argv) > 1 :
        tuning_param_file = sys.argv[1]
        cacheTuningSPaceFile = sys.argv[2]
        onlyGenerateCfg = int(sys.argv[3]) >= 1
        msg = f'''
        ====== User Commandline Inputs =========
        [tuning_param_file] = {tuning_param_file}
        [cacheTuningSPaceFile] = {cacheTuningSPaceFile}
        [onlyGenerateCfg] = {onlyGenerateCfg}
        [tuning space gen mode] = {tuningSpaceGenMode}
        [gpu_devices] = {gpu_devices}
        =======================================
        '''
        print(msg, flush=True)
    # 调优空间生成
    print('===== Waiting for tuning space build ... ',flush=True)
    totalLen = BuildTuningSpace(tuning_param_file, cacheTuningSPaceFile, tuningSpaceGenMode)
    print(f'===== Tuning space build OK! size = {totalLen} ==== ',flush=True)
    if totalLen <= 0 :
        return
    # 编译及benchmark启动
    if not onlyGenerateCfg :
        tm =  ParallelTaskManager(
            gpu_devices,
            totalLen, cacheTuningSPaceFile, perfPAth, 
            benchmarkcnt=10,  # 单个case执行次数
            warmupcnt=1,  # 每轮执行warmup次数
            keepTopNum = 15,  # 最佳结果保留前xx（取中位数）
            torchDynamicLogPath='',  # 是否周期性记录torch对应kernel的性能变化
            nTorchEpsInitTest=300,  # 测量torch的baseline时所运行次数（取中位数）
            atol=1e-3,  # 绝对误差
            rtol=1e-3   # 相对误差
        )
        tm.run(
            backendtype=backendType ,  # 后端类型
            archInfo=arch,
            maxProcess= nProcess , # 编译kernel的最大进程数 
            needCompile=True, # 是否执行编译过程
            needPerfTest=True, # 是否执行benchmark过程
            startFrom=0,     # 从空间里编号为x的config开始执行
            baselineInitInfo= [batch, M, N ,K , elementType]    # 用于pytorch基准的测试。因为PerfTester设计上的通用性（不关注kernel的具体参数），理论上该值只能运行时查找，且不一定保证唯一。考虑到实现的复杂性，这里先简单处理，后期改进（结合其他算子、各种参数再重新设计）
        )

if __name__ == '__main__' :
    main()