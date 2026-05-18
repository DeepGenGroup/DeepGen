#!/bin/bash
SEQLEN=${SEQLEN:-2048}
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py  /home/xushilong/DeepGen/TuningConfigs/xbk/attn.json  ./xbk_testattn_naive_oooo.json 0 0 0 0 attn_v1 3 --seqlen "$SEQLEN" float32 notune
# python SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/attn_llama2_0227_debug.json ./xbk_testattn_naive.json 0 0 0 0 attn_v1 3 --seqlen "$SEQLEN"