from kcg.Operators.matmul import TuningSpaceEncoder_Matmul
import json
import sys

SAVE_AS_SPACE = 1
SAVE_AS_TEMPLATE = 2

def convert_and_save(perflogpath:str, outputspacepath : str,mode : int) :
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
        
    if mode == SAVE_AS_SPACE : # save as space
        te = TuningSpaceEncoder_Matmul(tuningDict)
        space={"template" : None, 'cfgs' : []}
        space['template'] = tuningDict
        for data in perfdata['results'] :
            cfg = data['config']
            space['cfgs'].append(int(te.encode(cfg)))
        if len(outputspacepath) > 0:    
            with open(outputspacepath,'w') as f :
                json.dump(space,f)
        else:
            return space
    elif mode == SAVE_AS_TEMPLATE : # extract config from bests and save
        if len(outputspacepath) > 0:
            with open(outputspacepath,'w') as f :
                json.dump(tuningDict ,f)
        else:
            return tuningDict
    else:
        assert False, f"Invalid mode {mode}"
    return None

if __name__ == "__main__" :
    if len(sys.argv) > 3 :
        input = sys.argv[1]  # 输入 perflog 文件
        output = sys.argv[2]  # 输出 tuningspace 文件
        mode = sys.argv[3]
        convert_and_save(input, output, mode)
    else :
        helpmsg = \
f'''
command error. args format: $inputPerflog $outputspace $mode[1|2]
    --inputPerflog : input perflog path
    --outputspace : output file name.
    --mode : 
        {SAVE_AS_SPACE} - 将bestperfs保存为 tuning space (即template+encoding)
        {SAVE_AS_TEMPLATE} - 将bestperfs中的config抽取出来并保存 (即只有template)
    '''
        print(helpmsg)
