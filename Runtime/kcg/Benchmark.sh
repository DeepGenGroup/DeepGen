#! /bin/bash
mydir="/home/xushilong/DeepGen"
export PYTHONPATH=$mydir/Runtime
cd ${mydir}/Runtime/kcg
export HIP_VISIBLE_DEVICES=7
tuning_param_file=$mydir/TuningConfigs/GEMM_configs_2048.json
# perf文件路径(用于记录当前最佳性能的case)
cacheTuningSPaceFile=$mydir/TuningCombs/tuingspace_gemm_2048x2048.json
onlyGenerateCfg=1 # 是否只生产 tuning space 并存入 cacheTuningSPaceFile

nohup python testGetKernels.py $tuning_param_file $cacheTuningSPaceFile $onlyGenerateCfg  > ${mydir}/log.log 2>&1 &
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &
