#! /bin/bash
temp=$(dirname "$0")
cd ${temp}/..
mydir=`pwd`
echo $mydir

cd ${mydir} 
# sh Compile.sh
export PYTHONPATH=${mydir}/Runtime
cd ${mydir}/Runtime/kcg

tuning_param_file=$mydir/TuningConfigs/GEMM_configs_1024.json # 指定调优参数配置
cacheTuningSPaceFile=$mydir/TuningCombs/tuingspace_gemm_1024x1024.json # 指定调优空间文件名字（不存在会创建，存在则直接使用）
onlyGenerateCfg=0 # 是否只进行调优空间生成并存入 cacheTuningSPaceFile，不执行编译和benchmark

# 启动指令1 ：使用Benchmark脚本参数启动，会话进程分离，用于长期执行
# nohup python testGetKernels.py $tuning_param_file $cacheTuningSPaceFile $onlyGenerateCfg  > ${mydir}/log.log 2>&1 &
nohup python testGetKernels.py > ${mydir}/log.log 2>&1 &

# 启动指令2 ： 使用python内的参数启动， 会话进程不分离
# python testGetKernels.py > ${mydir}/log.log 2>&1 &

# hipprof测试指令
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &