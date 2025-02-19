if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import BuildTuningSpace, ParseTuningSpace
    import sys
    PathManager.init(clearPkl=True, clearTmp=True, clearCache=True)
    # Tuning 参数空间配置文件
    tuning_param_file = '/home/xushilong/DeepGen/TuningConfigs/GEMM_configs_1024.json'
    # perf文件路径(用于记录当前最佳性能的case)
    perfPAth = '/home/xushilong/DeepGen/perfRecordlog_7'
    cacheTuningSPaceFile = '/home/xushilong/DeepGen/TuningCombs/tuingspace_gemm_debug.json'
    onlyGenerateCfg = True # 是否只生产 tuning space 并存入 cacheTuningSPaceFile
    nProcess = 100 # 最大进程数
    gpu_devices = '0,7'  # 可见设备
    tuningSpaceGenMode = 1  # 调优空间生成策略（0：先生成space再剪枝 1：直接生成剪枝后的space）
    
    '''
        tuning_param_file 列举调优空间参数及其可选值
        cacheTuningSPaceFile 是调优空间文件，内含所有可选参数值的组合
    '''
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
        =======================================
        '''
        print(msg)
        
    DeviceInfo.set_current_device(gpu_devices)
    print(f'===== Set current device to {DeviceInfo.get_current_device()} =======',flush=True)
    print('==== waiting for config_gen ==== ',flush=True)
    totalLen = BuildTuningSpace(tuning_param_file, cacheTuningSPaceFile, tuningSpaceGenMode)
    print(f'==== config_gen Done! ==== ',flush=True)
    
    if not onlyGenerateCfg :
        tm =  ParallelTaskManager(
            totalLen, cacheTuningSPaceFile, perfPAth, 
            benchmarkcnt=10, 
            warmupcnt=1, 
            devId=DeviceInfo.get_current_device(), 
            keepTopNum = 15,
            torchDynamicLogPath='', 
            nTorchEpsInitTest=10
        )
        tm.run(maxProcess= nProcess , startFromSubjson = '', needCompile=True, needPerfTest=True)

