if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import BuildTuningSpace, ParseTuningSpace
    import sys
    # 路径管理器初始化 & 清理缓存数据（可选）
    PathManager.init(clearPkl=True, clearTmp=True, clearCache=True)
    # Tuning 参数空间配置文件
    tuning_param_file = '/home/xushilong/DeepGen/TuningConfigs/GEMM_configs_2.json'
    # perf文件路径前缀(用于记录当前最佳性能的case)
    perfPAth = '/home/xushilong/DeepGen/perfRecordlog_7'
    # 调优空间存储文件
    cacheTuningSPaceFile = '/home/xushilong/DeepGen/TuningCombs/tuingspace_gemm_debug.json'
    # 是否只进行调优空间生成并存入 cacheTuningSPaceFile，不执行kernel编译以及benchmark
    onlyGenerateCfg = False 
    # 最大进程数
    nProcess = 1
    # 可见设备列表
    gpu_devices = [6]  
    # 调优空间生成策略（0：先生成space再剪枝 1：直接生成剪枝后的space）
    tuningSpaceGenMode = 1  
    # 当前后端类型
    backendType = EnumBackendType.CUDA
    
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
    
    # 编译及benchmark启动
    if not onlyGenerateCfg :
        tm =  ParallelTaskManager(
            gpu_devices,
            totalLen, cacheTuningSPaceFile, perfPAth, 
            benchmarkcnt=10,  # 单个case执行次数
            warmupcnt=1,  # 每轮执行warmup次数
            keepTopNum = 15,  # 最佳结果保留前xx（取中位数）
            torchDynamicLogPath='',  # 是否周期性记录torch对应kernel的性能变化
            nTorchEpsInitTest=30  # 测量torch的baseline时所运行次数（取中位数）
        )
        tm.run(backendtype=backendType ,  # 后端类型
               maxProcess= nProcess , # 编译kernel的最大进程数 
               needCompile=True, # 是否执行编译过程
               needPerfTest=True # 是否执行benchmark过程
        )

