#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/gemma2_split.json ./testattn_gemma2_split.json 0 0 0 0 gemma2_split 4 float32
