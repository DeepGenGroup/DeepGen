from SavePerflogAsTuningSpace import *

keys_noDiversify = [
    "WARP_SIZE",
    "DATATYPE_A",
    "DATATYPE_B",
    "DATATYPE_C",
    "M_SIZE",
    "N_SIZE",
    "K_SIZE",
    "IS_ATRANS",
]

keys_mustDiversify = {
    "LOCAL_SPLIT_U" : [1,2],
    "UNROLL_NUM" : [8,16,32],
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
