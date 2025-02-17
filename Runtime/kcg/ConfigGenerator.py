import itertools
import json
from Utils import *
import multiprocessing

# 功能：根据 /home/xushilong/CodeGenDemo/TuningConfigs/GEMM_configs_2.json， 输出所有参数的组合并校验。校验通过的加入到 tuning space
# tuning space 为一个文件夹，文件夹内含很多小json，共同组成一个很大的调优空间
# tuning space 的命名由用户输入
# 运行测试时，tuning space可以由缓存载入，不用重复生成
# 可以通过名字区分算子以及space的配置


class TuningSpaceManager :
    def __init__(self):
        self.m_spaceName = ""
        self.m_cacheDir = ""
        self.m_tuningConfigFilePath = ""
        self.m_smallJsonItemsCount = 200
    
    def _getSmallJsonFileName(self,startIndex : int, endIndex : int, outDir:str) :
        return f'{outDir}/tmp_{self.m_spaceName}_{str(startIndex)}_{str(endIndex)}.json'

    def _process_cfg(self,cfgs,st,smallJsonLen, check_funcs : List[callable] ) :
        # config = dict(zip(keys, combination))
        res = {'cfgs' : []}
        for config in cfgs :
            isOK = True
            for check_func in check_funcs : 
                if not check_func(config) :
                    isOK = False;break
            if isOK :
                res['cfgs'].append(config)
        save_to_json(res, self._getSmallJsonFileName(st, st + smallJsonLen, self.m_cacheDir))

    
    def _read_params(self, userInputJsonPath : str) :
        with open(userInputJsonPath, 'r') as file:
            json_data = json.load(file)
            return json_data
        return None
    
    def generateSpace(self) :
        param_options = self.read_params(self.m_tuningConfigFilePath)

        # 获取所有参数名和对应的可选值
        keys = list(param_options.keys())
        values = list(param_options.values())

        # 生成所有可能的参数组合
        subconfigs = []
        st = 0
        maxProc = 50
        nProc = 0
        procList = []
        for combination in itertools.product(*values):
            if len(subconfigs) < g_singleLen :
                config = dict(zip(keys, combination))
                subconfigs.append(config)
            else:
                procList.append(multiprocessing.Process(target=self._process_cfg, args=(subconfigs,st,[])))
                subconfigs.clear()
                nProc += 1
                if nProc >= maxProc :
                    nProc = 0
                    for p in procList :
                        p.join()
                    procList.clear()


# 输出为 JSON 文件
def save_to_json(combinations, output_file="config_combinations.json"):
    with open(output_file, "w") as f:
        json.dump(combinations, f, indent=4)


def split_bigjson_to_temp(bigcfgs : Dict, startIndex : int, endIndex : int, outDir:str) -> str:
    if outDir[-1] == '/':
        outDir = outDir[:-1]
    tmpjson = {'cfgs' : []}
    maxLen = len(bigcfgs['cfgs'])
    for i in range(startIndex,endIndex) :
        if i >= maxLen : 
            break
        else:
            tmpjson['cfgs'].append(bigcfgs['cfgs'][i])
    filename = getTempJsonFileName(startIndex,endIndex,outDir)
    save_to_json(tmpjson, filename)
    return filename

def config_gen(tuning_cfg_file :str, preGeneratedJson : str, singleLength : int, cacheTuningSPaceFile : str, splitBigJson = True) :
    combs = {'cfgs' : []}
    tempFileNames = []
    g_singleLen = singleLength
    # if len(startSubjson) > 0:
    #     if os.path.exists(PathManager.tmp_dir()) and os.path.isdir(PathManager.tmp_dir()):
    #     # 遍历目录中的所有文件和子目录
    #         for filename in os.listdir(PathManager.tmp_dir()):
    #             file_path = os.path.join(PathManager.tmp_dir(), filename)
    #             # 如果是文件，删除它
    #             if os.path.isfile(file_path):
    #                 tempFileNames.append(file_path)
    #     return (tempFileNames,len(items))
    if len(preGeneratedJson) > 0 :
        print(f'======== Use pre-generated json combinations {preGeneratedJson} ==========')
        combs = read_params(preGeneratedJson)
    else:
        print(f'======== Generate combinations by {tuning_cfg_file} ==========')
        combs['cfgs'] = get_cfgs(tuning_cfg_file)
        if len(cacheTuningSPaceFile) > 0 :
            print(f'====== store tuning combinations to file : {cacheTuningSPaceFile} =======')
            with open(cacheTuningSPaceFile,'w') as f:
                json.dump(combs,f)
    items = combs['cfgs']
    
    if splitBigJson :
        print(f'====== split big tuning combinations to {PathManager.tmp_dir()} ==========')
        for i in range(0,len(items), singleLength) :
            fname = split_bigjson_to_temp(combs,i,i+singleLength,PathManager.tmp_dir())
            tempFileNames.append(fname)
        
    print(f"Generated {len(items)} configurations and saved subfiles Done")
    return (tempFileNames,len(items))


# # 主函数
# if __name__ == "__main__":
#     save_to_json(combinations)
#     print(f"Generated {len(combinations)} configurations and saved to 'config_combinations.json'.")
    