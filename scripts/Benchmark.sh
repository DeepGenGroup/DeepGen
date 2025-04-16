#! /bin/bash
startParamfile=$1
logFile=$2
temp=$(dirname "$0")
cd ${temp}/..
deepgendir=`pwd`
echo $deepgendir ; cd ${deepgendir} ;
source ~/anaconda3/etc/profile.d/conda.sh ; conda activate triton_rocm
# source ~/anaconda3/etc/profile.d/conda.sh ; conda activate torch-mlir
export PYTHONPATH=${deepgendir}/Runtime
cd ${deepgendir}/Runtime/kcg
echo nvcc_path=`which nvcc`
# 启动指令1 ：使用Benchmark脚本参数启动，会话进程分离，用于长期执行
nohup python deepGenMain.py $startParamfile > ${logFile} 2>&1 & 

# hipprof测试指令
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &