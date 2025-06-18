#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg

# cfg=/home/xushilong/DeepGen/TuningConfigs/mm-b2-2048-2048-1024.json
# st=0
# max=0
# saveTo=/home/xushilong/DeepGen/result-mm-b2-2048-2048-1024_${st}_${max}.json
# python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

max=11000
st=1000
targetdir=/home/xushilong/DeepGen/result

for st in $(seq 1000 $max 30600); do
    # echo $st

    cfg=/home/xushilong/DeepGen/TuningConfigs/mm-2048-2048-2048.json
    saveTo=${targetdir}/result-mm-2048-2048-2048_${st}_${max}.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

    cfg=/home/xushilong/DeepGen/TuningConfigs/mm-2048-4096-1024.json
    saveTo=${targetdir}/result-mm-2048-4096-1024_${st}_${max}.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

    cfg=/home/xushilong/DeepGen/TuningConfigs/mm-4096-8192-512.json
    saveTo=${targetdir}/result-mm-4096-8192-512_${st}_${max}.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

    cfg=/home/xushilong/DeepGen/TuningConfigs/mm-8192-2048-1024.json
    saveTo=${targetdir}/result-mm-8192-2048-1024_${st}_${max}.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

    cfg=/home/xushilong/DeepGen/TuningConfigs/mm-b4-1024-1024-2048.json
    saveTo=${targetdir}/result-mm-b4-1024-1024-2048_${st}_${max}.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

    cfg=/home/xushilong/DeepGen/TuningConfigs/mm-b8-2048-1024-1024.json
    saveTo=${targetdir}/result-mm-b8-2048-1024-1024_${st}_${max}.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

    cfg=/home/xushilong/DeepGen/TuningConfigs/mm-b16-1024-4096-256.json
    saveTo=${targetdir}/result-mm-b16-1024-4096-256_${st}_${max}.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > log.log 2>&1 

done
