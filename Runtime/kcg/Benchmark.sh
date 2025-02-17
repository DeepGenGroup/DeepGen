#! /bin/bash
export PYTHONPATH=/home/xushilong/DeepGen/Runtime
cd /home/xushilong/DeepGen/Runtime/kcg
export HIP_VISIBLE_DEVICES=7
nohup python testGetKernels.py > log.log 2>&1 &
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &
