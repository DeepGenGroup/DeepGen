from kcg.Operators.matmul import TuningSpaceEncoder_Matmul
import json
import sys

def convert_and_save(perflogpath:str, outputspacepath : str) :
    perfdata = None
    with open(perflogpath) as f :
        perfdata = json.load(f)
    template = perfdata['results'][0]['config']
    tuningDict = dict()
    for key in template.keys() :
        tuningDict[key] = set()
        tuningDict[key].add(template[key])
    
    for data in perfdata['results'] :
        cfg = data['config']
        for key in cfg.keys() :
            tuningDict[key].add(cfg[key])
    
    for key in tuningDict.keys():
        tuningDict[key] = list(tuningDict[key])
    
    te = TuningSpaceEncoder_Matmul(tuningDict)
    space={"template" : None, 'cfgs' : []}
    space['template'] = tuningDict
    for data in perfdata['results'] :
        cfg = data['config']
        space['cfgs'].append(int(te.encode(cfg)))
        
    with open(outputspacepath,'w') as f :
        json.dump(space,f)

if __name__ == "__main__" :
    if len(sys.argv) > 1 :
        input = sys.argv[1]  # 输入 perflog 文件
        output = sys.argv[2]  # 输出 tuningspace 文件
        convert_and_save(input, output)
        