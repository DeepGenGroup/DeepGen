#! /bin/bash
startParamfile=$1  # filename
needClearDir=$2  # 1 or 0
temp=$(dirname "$0")
cd ${temp}/..
mydir=`pwd`
echo $mydir ; cd ${mydir} 
# sh Compile.sh
source ~/anaconda3/etc/profile.d/conda.sh ; conda activate py310
export PYTHONPATH=${mydir}/Runtime
cd ${mydir}
# echo nvcc_path=`which nvcc`
# 启动指令1 ：使用Benchmark脚本参数启动，会话进程分离，用于长期执行
logname=${startParamfile##*/}Log.log
nohup python ./Runtime/kcg/deepGenMain.py $startParamfile $needClearDir > ${logname} 2>&1 & 

# hipprof测试指令
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &