#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/attn_llama2_0227_debug.json ./testattn_naive.json 0 0 0 0 attn_v1 3