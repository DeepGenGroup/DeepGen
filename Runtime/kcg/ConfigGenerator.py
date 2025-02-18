import itertools
import json
from Utils import *
import multiprocessing
from Operators.matmul import *
# 功能：根据 /home/xushilong/CodeGenDemo/TuningConfigs/GEMM_configs_2.json， 输出所有参数的组合并校验。校验通过的加入到 tuning space
# tuning space 为一个文件夹，文件夹内含很多小json，共同组成一个很大的调优空间
# tuning space 的命名由用户输入
# 运行测试时，tuning space可以由缓存载入，不用重复生成
# 可以通过名字区分算子以及space的配置

def _process_cfg(encoder : TuningSpaceEncoder_Matmul, cfgs : List[Dict], check_funcs : List[callable], tempfilename:str ) :
    ret = {'results':[]}
    for config in cfgs :
        isOK = True
        for check_func in check_funcs : 
            if not check_func(config) :
                isOK = False;break
        if isOK :
            encodestr = encoder.encode(config)
            ret['results'].append(encodestr)
    if len(ret['results']) > 0:
        with open(tempfilename,'w') as f:
            json.dump(ret,f)

class TuningSpaceManager :
    def __init__(self,spacename : str ,tuningConfigFilePath : str, cacheFileName : str):
        self.m_spaceName = spacename
        self.m_cacheFileName = cacheFileName
        self.m_tuningConfigFileName = tuningConfigFilePath
        self.m_encoder = None
        

    def _read_params(self, userInputJsonPath : str) :
        with open(userInputJsonPath, 'r') as file:
            json_data = json.load(file)
            return json_data
        return None
    
    def _getTmpfilename(self,st) -> str :
        return f'{PathManager().tmp_dir()}/tmp_{st}.json'
    
    def generateSpaceParallel(self, maxProcess) -> int:
        param_options = self._read_params(self.m_tuningConfigFileName)
        self.m_encoder = TuningSpaceEncoder_Matmul(param_options)
        # 获取所有参数名和对应的可选值
        keys = list(param_options.keys())
        values = list(param_options.values())
        all_estimate_count = 1
        for val in values :
            all_estimate_count *= len(val)
        print('All_estimate_count = ',all_estimate_count)
        # 生成所有可能的参数组合
        checkers = [TuningSpaceChecker_Matmul.check_shm_size,
                    TuningSpaceChecker_Matmul.check_size,
                    TuningSpaceChecker_Matmul.check_warp,
                    TuningSpaceChecker_Matmul.check_glw
                    ]
        progress_granularity = 0.01
        totalProgress = progress_granularity
        
        maxProc = maxProcess
        groupSize = 500
        subProcs = []
        tempCfgs = []
        i = 0
        dealed = 0
        import time
        t0 = time.time()
        estimateTimeTotal = 0
        spentTime = 0
        for combination in itertools.product(*values): 
            config = dict(zip(keys, combination))
            tempCfgs.append(config)
            i+=1
            if i >= groupSize :
                p = multiprocessing.Process(target=_process_cfg,args=(self.m_encoder,tempCfgs,checkers,self._getTmpfilename(dealed)))
                dealed+=groupSize
                subProcs.append(p)
                p.start()
                if len(subProcs) >= maxProc:
                    for e in subProcs :
                        e.join()
                tempCfgs.clear(); i=0
                subProcs.clear()
                if dealed >= totalProgress * all_estimate_count :
                    t1 = time.time()
                    eps = t1 - t0
                    if estimateTimeTotal <= 0:
                        estimateTimeTotal = (1 / progress_granularity)*eps
                    spentTime = eps
                    print(f'>>>>>  Progress : {totalProgress*100}%. Estimate total time(s) : {estimateTimeTotal}, currentElapsed : {spentTime}')
                    totalProgress += progress_granularity
        if len(tempCfgs) > 0:
            p = multiprocessing.Process(target=_process_cfg,args=(self.m_encoder,tempCfgs,checkers,self._getTmpfilename(dealed)))
            p.start()
            subProcs.append(p)
        if len(subProcs) > 0 :
            for e in subProcs :
                e.join()
        files = os.listdir(PathManager().tmp_dir())
        obj = {
            'template' : "-" , 
            'cfgs' : []
        }
        for fname in files :
            fpath = PathManager().tmp_dir()+ "/" + fname
            with open(fpath) as f :
                result = json.load(f)
                obj['cfgs'] += result['results']
            os.remove(fpath)
        obj['template'] = self._read_params(self.m_tuningConfigFileName)
        with open(self.m_cacheFileName,'w') as f :
            json.dump(obj,f)
        return len(obj['cfgs'])
    

def BuildTuningSpace(fname : str , outfname : str) -> int:
    totalLen = 0
    if not os.path.exists(outfname) :
        tsm = TuningSpaceManager('spacename',fname,outfname)
        totalLen = tsm.generateSpaceParallel(maxProcess = 80)
        print(f'Tuning space generate OK! Stored in {outfname}')
    else:
        with open(outfname) as f:
            o = json.load(f)
            totalLen = len(o['cfgs'])
            print(f'Tuning space already exists, skip generate. Read from {outfname}')
    return totalLen

def ParseTuningSpace(fname : str) :
    space = None
    with open(fname) as f :
        space = json.load(f)
    te = TuningSpaceEncoder_Matmul(space['template'])
    cfgs = []
    for cfgstr in space['cfgs'] :
        cfgs.append(te.decode(cfgstr))
    return cfgs


