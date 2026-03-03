#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/h2o_split.json ./testattn_h2o_split.json 0 1 0 0 h2o_split 5 float32
