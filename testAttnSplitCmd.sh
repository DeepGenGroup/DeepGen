#!/bin/bash
SEQLEN=${SEQLEN:-4096}
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/attn_split.json ./testattn_split_${SEQLEN}.json 0 1 0 0 attn_split 3 float32 --seqlen "$SEQLEN"
