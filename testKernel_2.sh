#!/bin/bash
cd /home/xushilong/DeepGen/Runtime
export PYTHONPATH=`pwd`
cd kcg

# cfg=/home/xushilong/DeepGen/TuningConfigs/mm-b2-2048-2048-1024.json
# st=0
# max=0
# saveTo=/home/xushilong/DeepGen/result-mm-b2-2048-2048-1024_${st}_${max}.json
# python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog.log 2>&1 

max=3900 # 7800 / 2
targetdir=/home/xushilong/DeepGen/result_model


for st in $(seq 0 $max 30600); do
        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.12._1024_1024_64.json
        saveTo=$targetdir/result-mm_1.12._1024_1024_64_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog0.log 2>&1
    
        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.12._1024_64_1024.json
        saveTo=$targetdir/result-mm_1.12._1024_64_1024_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog1.log 2>&1

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._1024_1024_1024.json
        saveTo=$targetdir/result-mm_1._1024_1024_1024_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog2.log 2>&1
    

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.16._1024_1024_64.json
        saveTo=$targetdir/result-mm_1.16._1024_1024_64_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog3.log 2>&1
    

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.16._1024_64_1024.json
        saveTo=$targetdir/result-mm_1.16._1024_64_1024_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog4.log 2>&1
    

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._1024_4096_1024.json
        saveTo=$targetdir/result-mm_1._1024_4096_1024_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog5.log 2>&1
    

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._1024_1024_4096.json
        saveTo=$targetdir/result-mm_1._1024_1024_4096_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog6.log 2>&1
    

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._2048_4096_4096.json
        saveTo=$targetdir/result-mm_1._2048_4096_4096_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog7.log 2>&1
    

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.32._2048_2048_128.json
        saveTo=$targetdir/result-mm_1.32._2048_2048_128_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog8.log 2>&1
    

        cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.32._2048_128_2048.json
        saveTo=$targetdir/result-mm_1.32._2048_128_2048_$st-$max.json
        python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog9.log 2>&1

done
