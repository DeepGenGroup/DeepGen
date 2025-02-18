if __name__ == '__main__' :
    from KCGTask import *
    import multiprocessing 
    from ConfigGenerator import BuildTuningSpace, ParseTuningSpace
    import sys
    PathManager.init(clearPkl=True, clearTmp=True, clearCache=True)
    # Tuning 参数空间配置文件
    tuning_param_file = '/home/xushilong/CodeGenDemo/TuningConfigs/GEMM_configs_2048.json'
    # perf文件路径(用于记录当前最佳性能的case)
    perfPAth = '/home/xushilong/CodeGenDemo/perfRecordlog_7'
    cacheTuningSPaceFile = '/home/xushilong/CodeGenDemo/TuningCombs/tuingspace_gemm_2048x2048.json'
    onlyGenerateCfg = False # 是否只生产 tuning space 并存入 cacheTuningSPaceFile
    nProcess = 100 # 最大进程数
    smallJsonLen = nProcess * 5  # 单个小json文件含有的cofigs上限
    
    '''
        文件生成关系：
        tuning_param_file -> cfg_combinations.json -> 拆分为 subjson, 存到tmp/中
        tuning_param_file 定义调优空间
        cfg_combinations.json 是所有参数的组合
        subjson 是 cfg_combinations.json 拆成的一堆小文件
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
        =======================================
        '''
        print(msg)
        
    # DeviceInfo.set_current_device(7)
    print(f'===== Set current device to {DeviceInfo.get_current_device()} =======')
    print('==== waiting for config_gen ==== ')
    t0 = time.time()
    totalLen = BuildTuningSpace(tuning_param_file, cacheTuningSPaceFile)
    t1 = time.time()
    print(f'==== config_gen Done! Elapsed {(t1-t0)} seconds ==== ')
    
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

