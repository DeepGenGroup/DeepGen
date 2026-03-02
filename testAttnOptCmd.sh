#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/attn_llama2_0227.json ./testattn_opt.json 0 0 0 0 attn_v2 4 float32