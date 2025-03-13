from SavePerflogAsTuningSpace import *

# 功能：将 bestperfs的数据抽取为 config params。对param中参数数量=1的key值进行扩空间。扩充规则为：以当前值为中心，在该项目内增加 *2 和 /2的值
# 相当于遗传算法中的杂交（抽取为包含最优值的config）+变异（扩展参数）
keys_noDiversify = [
    "WARP_SIZE",
    "DATATYPE_A",
    "DATATYPE_B",
    "DATATYPE_C",
    "M_SIZE",
    "N_SIZE",
    "K_SIZE",
    "BATCH_SIZE",
    "IS_ATRANS",
    "UNROLL_NUM",
]

keys_mustDiversify = {
    "LOCAL_SPLIT_U" : [1,2],
    "REG_PREFETCH" : [1,0],
    "SHARED_PREFETCH" : [1,0],
    "LOAD_CONTINUOUS" : [1,0],
    "REDUCE_C_CONTINUOUS" : [1,0],
}

def analyzeValueCounts(perflogpath:str ) :
    perfdata = None
    with open(perflogpath) as f :
        perfdata = json.load(f)
    template = convert_and_save(perflogpath,"",SAVE_AS_TEMPLATE)
    counter = dict()
    for key in template.keys() :
        for v in template[key]:
            counter[key] = {str(v) : 1}
    for data in perfdata['results'] :
        cfg = data['config']
        for key in cfg.keys() :
            val = str(cfg[key])
            if val not in counter[key].keys() :
                counter[key][val] = 1
            else:
                counter[key][val] += 1
    return (counter,template)

def diversifyTuningParams(perflogPath : str) :
    valCounter, params = analyzeValueCounts(perflogPath)
    for key in valCounter.keys():
        if len(valCounter[key]) <= 1 :
            if key in keys_noDiversify :
                continue
            elif key in keys_mustDiversify.keys() :
                params[key] = keys_mustDiversify[key]
            else:
                v = int(params[key][0])
                if v // 2 > 0:
                    params[key].append(v//2)
                params[key].append(v * 2)
    return params            

# dd = diversifyTuningParams('/home/xushilong/DeepGenRun/__perf-20250304-dim2048_card0.json')
# with open('/home/xushilong/DeepGenRun/myooo.json','w+') as f:
#     json.dump(dd,f)
if __name__ == '__main__' :
    if len(sys.argv) > 2:
        perflog = sys.argv[1]
        outpath = sys.argv[2]
        dd = diversifyTuningParams(perflog)
        with open(outpath,'w+') as f:
            json.dump(dd, f)
    else:
        print("Invalid args!")