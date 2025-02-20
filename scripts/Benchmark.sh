#! /bin/bash
mydir="/home/xushilong/DeepGen"
export PYTHONPATH=$mydir/Runtime
cd ${mydir}/Runtime/kcg

tuning_param_file=$mydir/TuningConfigs/GEMM_configs_1024.json
# perf文件路径(用于记录当前最佳性能的case)
cacheTuningSPaceFile=$mydir/TuningCombs/tuingspace_gemm_1024x1024.json
onlyGenerateCfg=0 # 是否只生产 tuning space 并存入 cacheTuningSPaceFile

nohup python testGetKernels.py $tuning_param_file $cacheTuningSPaceFile $onlyGenerateCfg  > ${mydir}/log.log 2>&1 &
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &
