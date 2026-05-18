#!/bin/bash
deepgenDir=/home/${USER}/DeepGen
cd ${deepgenDir}/Runtime
export PYTHONPATH=`pwd`
cd kcg

# cfg=${deepgenDir}/TuningConfigs/mmTest/mm-2048-2048-2048.json
cfg=${deepgenDir}/TuningConfigs/xbk/gemm_best.json
st=0
max=0
# saveTo=/home/xushilong/DeepGen/result-xbk-mm-b1-2048-sq_${st}_${max}.json
# python SimpleLocalTester.py $cfg $saveTo $st $max 0 0 > xbk_mlog.log 2>&1 
python SimpleLocalTester.py $cfg ./xbk_best_gemm2048_${SEQLEN}.json 0 $max 0 0 matmul 4 float32 --seqlen 2048 notune  


# max=3900 # 7800 / 2
# targetdir=/home/xushilong/DeepGen/result_model


# for st in $(seq 0 $max 30600); do
#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.12._1024_1024_64.json
#         saveTo=$targetdir/result-mm_1.12._1024_1024_64_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog0.log 2>&1
    
#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.12._1024_64_1024.json
#         saveTo=$targetdir/result-mm_1.12._1024_64_1024_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog1.log 2>&1

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._1024_1024_1024.json
#         saveTo=$targetdir/result-mm_1._1024_1024_1024_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog2.log 2>&1
    

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.16._1024_1024_64.json
#         saveTo=$targetdir/result-mm_1.16._1024_1024_64_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog3.log 2>&1
    

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.16._1024_64_1024.json
#         saveTo=$targetdir/result-mm_1.16._1024_64_1024_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog4.log 2>&1
    

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._1024_4096_1024.json
#         saveTo=$targetdir/result-mm_1._1024_4096_1024_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog5.log 2>&1
    

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._1024_1024_4096.json
#         saveTo=$targetdir/result-mm_1._1024_1024_4096_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog6.log 2>&1
    

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1._2048_4096_4096.json
#         saveTo=$targetdir/result-mm_1._2048_4096_4096_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog7.log 2>&1
    

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.32._2048_2048_128.json
#         saveTo=$targetdir/result-mm_1.32._2048_2048_128_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog8.log 2>&1
    

#         cfg=/home/xushilong/DeepGen/TuningConfigs/modelTest/mm_1.32._2048_128_2048.json
#         saveTo=$targetdir/result-mm_1.32._2048_128_2048_$st-$max.json
#         python SimpleLocalTester.py $cfg $saveTo $st $max 0 1.25 > mlog9.log 2>&1

# done
