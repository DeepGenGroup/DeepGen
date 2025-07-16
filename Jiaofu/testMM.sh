
export PYTHONPATH=/home/xushilong/DeepGen/Runtime
project_path=/home/xushilong/DeepGen
cd $project_path/Runtime/kcg
# python SimpleLocalTester.py $project_path/TuningConfigs/mmTest/mm-512-2048-8192.json  $project_path/项目交付/gemm_result/res-mm-512-2048-8192_larger.json 0 0 1 2.0 > $project_path/项目交付/log-512-2048-8192_larger.log 2>&1 
# python SimpleLocalTester.py $project_path/TuningConfigs/mmTest/mm-1024-2048-4096.json  $project_path/项目交付/gemm_result/res-mm-1024-2048-4096.json 0 0 0 2.0 > $project_path/项目交付/log-1024-2048-4096.log 2>&1 
# python SimpleLocalTester.py $project_path/TuningConfigs/mmTest/mm-1024-4096-2048.json  $project_path/项目交付/gemm_result/res-mm-1024-4096-2048.json 0 0 0 2.0 > $project_path/项目交付/log-1024-4096-2048.log 2>&1 
# python SimpleLocalTester.py $project_path/TuningConfigs/mmTest/mm-2048-2048-2048.json  $project_path/项目交付/gemm_result/res-mm-2048-2048-2048.json 0 0 0 2.0 > $project_path/项目交付/log-2048-2048-2048.log 2>&1 
# python SimpleLocalTester.py $project_path/TuningConfigs/mmTest/mm-2048-8192-2048.json  $project_path/项目交付/gemm_result/res_mm-2048-8192-2048.json 0 0 0 2.0 > $project_path/项目交付/log-2048_8192_2048.json.log 2>&1 
# python SimpleLocalTester.py $project_path/TuningConfigs/mmTest/mm-2048-8192-2048.json  $project_path/项目交付/gemm_result/res__opt_mm-2048-8192-2048.json 0 0 0 2.0 > $project_path/项目交付/log-opt-2048_8192_2048.json.log 2>&1 
python /home/xushilong/DeepGen/Runtime/kcg/SimpleLocalTester.py /home/xushilong/DeepGen/TuningConfigs/mmTest/mm-4096-4096-4096.json  /home/xushilong/DeepGen/Jiaofu/gemm_result/res_mm-4096-4096-4096.json 0 0 0 0 > /home/xushilong/DeepGen/Jiaofu/log-4096-4096-4096.log 2>&1 

