#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/attn_split.json ./testattn_split.json 0 0 0 0 attn_split 3 float32
