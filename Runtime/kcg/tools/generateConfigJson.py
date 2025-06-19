# /home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json

import json
from typing import List

def generate_json(b : List,m,n,k, dstPath : str) :
    src = '/home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json'
    with open(src) as f:
        config = json.load(f)
    # "M_SIZE" : [4096],
    # "N_SIZE" : [8192],
    # "K_SIZE" : [512],
    # "BATCH_SIZE" : [1],
    config['M_SIZE'] = [m]
    config["N_SIZE"] = [n]
    config["K_SIZE"] = [k]
    config["BATCH_SIZE"] = b
    
    with open(dstPath,'w') as f:
        json.dump(config,f,indent=4)
    
if __name__ == "__main__" :
    lists = [
        # model test
        # [[1, 12], 1024, 1024, 64 ],
        # [[1, 12], 1024, 64, 1024 ],
        # [[1], 1024, 1024, 1024 ],
        # [[1, 16], 1024, 1024, 64 ],
        # [[1, 16], 1024, 64, 1024 ],
        # [[1], 1024, 4096, 1024 ],
        # [[1], 1024, 1024, 4096 ],
        # [[1], 2048, 4096, 4096 ],
        # [[1, 32], 2048, 2048, 128 ],
        # [[1, 32], 2048, 128, 2048 ],
        
        #  conv test
        [[8],16, 65536,1024],
        [[8], 4, 1048576, 256],
        [[8], 4, 262144, 1024],
        [[1], 256, 65536, 4096],
        [[1], 256, 16384, 4096],
        [[2], 512, 4096, 8192],
        [[1], 512, 2048, 8192],
        [[1], 1024, 256, 65536],
        [[4], 256, 512, 16384 ],
        [[8], 128, 1024, 8192],
        [[2], 512, 1024, 8192],
        [[4], 512, 512,8192],
    ]
    targetdir = '/home/xushilong/DeepGen/TuningConfigs/conv'
    index = 0
    for item in lists :
        b,m,n,k = item
        bb = ""
        for _b in b :
            bb += str(_b) + "."
        prefix = f"mm_{bb}_{m}_{n}_{k}"
        filename = f"{targetdir}/{prefix}.json"
        generate_json(b,m,n,k,filename)
        
        msg = f'''
            cfg={filename}
            saveTo=$targetdir/result-{prefix}_$st-$max.json
            python SimpleLocalTester.py $cfg $saveTo $st $max > log{index}.log 2>&1
        ''' 
        index+=1
        print(msg)
