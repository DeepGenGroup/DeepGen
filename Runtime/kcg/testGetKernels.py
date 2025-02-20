if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import BuildTuningSpace, ParseTuningSpace
    import sys
    # 路径管理器初始化 & 清理缓存数据（可选）
    PathManager.init(clearPkl=True, clearTmp=True, clearCache=True)
    # Tuning 参数空间配置文件
    tuning_param_file = '/home/xushilong/DeepGen/TuningConfigs/GEMM_configs_2.json'
    # perf文件路径(用于记录当前最佳性能的case)
    perfPAth = '/home/xushilong/DeepGen/perfRecordlog_7'
    # 调优空间存储文件
    cacheTuningSPaceFile = '/home/xushilong/DeepGen/TuningCombs/tuingspace_gemm_debug.json'
    # 是否只生产 tuning space 并存入 cacheTuningSPaceFile
    onlyGenerateCfg = False 
    # 最大进程数
    nProcess = 100 
    # 可见设备
    gpu_devices = [6,7]  
    # 调优空间生成策略（0：先生成space再剪枝 1：直接生成剪枝后的space）
    tuningSpaceGenMode = 1  
    
    ######################################################################################
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
        
    print('===== Waiting for tuning space build ... ',flush=True)
    totalLen = BuildTuningSpace(tuning_param_file, cacheTuningSPaceFile, tuningSpaceGenMode)
    print(f'===== Tuning space build OK! ==== ',flush=True)
    
    if not onlyGenerateCfg :
        tm =  ParallelTaskManager(
            gpu_devices,
            totalLen, cacheTuningSPaceFile, perfPAth, 
            benchmarkcnt=10, 
            warmupcnt=1, 
            keepTopNum = 15,
            torchDynamicLogPath='', 
            nTorchEpsInitTest=30
        )
        tm.run(maxProcess= nProcess , startFromSubjson = '', needCompile=True, needPerfTest=True)

