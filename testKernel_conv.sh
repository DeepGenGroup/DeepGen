#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg
max=6000
st=1000
targetdir=/home/xushilong/DeepGen/result_conv

for st in $(seq 0 $max 30600); do

    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_1._256_65536_4096.json
    saveTo=$targetdir/result-mm_1._256_65536_4096_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 9512


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_1._256_16384_4096.json
    saveTo=$targetdir/result-mm_1._256_16384_4096_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 9512


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_2._512_4096_8192.json
    saveTo=$targetdir/result-mm_2._512_4096_8192_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 24016


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_1._512_2048_8192.json
    saveTo=$targetdir/result-mm_1._512_2048_8192_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 24016


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_1._1024_256_65536.json
    saveTo=$targetdir/result-mm_1._1024_256_65536_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 30608


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_4._256_512_16384.json
    saveTo=$targetdir/result-mm_4._256_512_16384_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 9512


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_8._128_1024_8192.json
    saveTo=$targetdir/result-mm_8._128_1024_8192_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 800


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_2._512_1024_8192.json
    saveTo=$targetdir/result-mm_2._512_1024_8192_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 24016


    cfg=/home/xushilong/DeepGen/TuningConfigs/conv/mm_4._512_512_8192.json
    saveTo=$targetdir/result-mm_4._512_512_8192_$st-$max.json
    python SimpleLocalTester.py $cfg $saveTo $st $max > convlog.log 2>&1  
    # ==== tune space size = 24016

done







































