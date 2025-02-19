#! /bin/bash
export PYTHONPATH=/home/xiebaokang/projects/mymlir/DeepGen
cd /home/xushilong/DeepGen/Runtime/python/kcg
export HIP_VISIBLE_DEVICES=7
nohup python testGetKernels.py > log.log 2>&1 &
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &
