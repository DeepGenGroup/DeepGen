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
t0 = time.time()

for i in range(1000000) :
    # obj["confs"].append('0000000000000000000020000121000')  # file 33.38MB ,time: 0.14580845832824707 / 20.436734894380212%,  0.5676541328430176 / 79.56326510561979%, total 0.7134625911712646.
    obj["confs"].append(int('0000000000000000000020000121000'))  # file 12.4MB  time: 0.4025759696960449 / 38.05594690234771%,  0.6552770137786865 / 61.94405309765229%, total 1.0578529834747314.

t1 = time.time()
with open('/home/xushilong/DeepGen/te.json','w') as f :
    json.dump(obj,f)
t2 = time.time()
print(f'time: {t1-t0} / {(t1-t0)/(t2-t0)*100}%,  {t2-t1} / {(t2-t1)/(t2-t0)*100}%, total {t2-t0}.' )