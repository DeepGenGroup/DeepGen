from Utils import *
from Operators.matmul import *
import time
import itertools
import json
from ConfigGenerator import *
import numpy as np

jsonPath = "/home/xushilong/DeepGen/combs_ALL.json"

keywords = [
"BLOCK_SIZE_M",
"BLOCK_SIZE_N",
"BLOCK_SIZE_K",
"THREAD_SIZE_M",
"THREAD_SIZE_N",
"WARP_SIZE",
"BLOCK_LAYOUT_M",
"BLOCK_LAYOUT_N",
"WARP_LAYOUT_M",
"WARP_LAYOUT_N",
"DATATYPE_A",
"DATATYPE_B",
"DATATYPE_C",
"M_SIZE",
"N_SIZE",
"K_SIZE",
"IS_ATRANS",
"GLOB_LOAD_WIDTH_A",
"GLOB_LOAD_WIDTH_B",
"WARP_SCATTER_WIDTH_A",
"WARP_SCATTER_WIDTH_B",
"THREAD_SCATTER_WIDTH_A",
"THREAD_SCATTER_WIDTH_B",
"LOCAL_SPLIT_U",
"BLOCK_MAPPING",
"GLOB_STORE_WIDTH",
"UNROLL_NUM",
"REG_PREFETCH",
"SHARED_PREFETCH",
"LOAD_CONTINUOUS",
"REDUCE_C_CONTINUOUS",]

def getVal() :
    # str 编码：文件大小 4300023
    # int : 4100023
    ret = '0'
    for i in range(len(keywords)) :
        ret += '1'
    return int(ret)

obj = {'confs' : []}

def buildTuningSpace(fname : str , outfname : str):
    tsm = TuningSpaceManager('spacename',fname,outfname)
    tsm.generateSpaceParallel()

def parseTuningSpace(fname : str) :
    space = None
    with open(fname) as f :
        space = json.load(f)
    te = TuningSpaceEncoder_Matmul(space['template'])
    cfgs = []
    times = []
    for cfgstr in space['cfgs'] :
        t0 = time.time()
        cfgs.append(te.decode(cfgstr))
        t1 = time.time()
        times.append((t1-t0) * 1000) 
    print(f'deal med time (ms): {np.median(times)}') 
    return cfgs





# ret = parseSpace('/home/xushilong/DeepGen/TuningConfigs/GEMM_configs_2.jsonrrr')
# print(len(ret))
getSpace('/home/xushilong/DeepGen/TuningConfigs/GEMM_configs_2.json', '/home/xushilong/DeepGen/TuningConfigs/GEMM_configs_2.jsonrrr')
