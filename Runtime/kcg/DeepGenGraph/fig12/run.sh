#!/bin/bash
set -e

FIG_DIR=$(dirname $(readlink -f $0))
PROJECT_DIR=$(dirname $FIG_DIR)
HOST=$(hostname)
echo "hostname: $HOST"
echo "project_dir: $PROJECT_DIR"


LOG_DIR=${FIG_DIR}/ae_logs

if [ -d ${LOG_DIR} ]; then
  echo "Removing existing log directory: ${LOG_DIR}"
  rm -rf ${LOG_DIR}
fi
mkdir -p ${LOG_DIR}

if [[ "$HOST" == "yes" ]]; then
device=a100
max_tasks=7
elif [[ "$HOST" == "fuse0" ]]; then
device=h100
max_tasks=2
else
  echo "unknown host: $HOST"
  exit -1
fi


# kernel
echo Running for kernel
source ${PROJECT_DIR}/baseline_env.sh
for sys in {torch,dynamo,tensorrt,tvm}; do
for model in {attn,h2o,roco,kf,snapkv,corm,gemma2}; do

while [ "$(squeue -u $(whoami) --noheader | wc -l)" -ge "${max_tasks}" ]; do
  echo "waiting for a slot to free up... current tasks:" $(squeue -u $(whoami) --noheader | wc -l)
  sleep 20
done

echo "Running ${model} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_kernel.py -m $model -s $sys 2>&1 | tee ${LOG_DIR}/kernel.${device}.${model}.${sys}.log &

done
done

wait


sys=our
source ${PROJECT_DIR}/our_env.sh
for model in {attn,h2o,roco,kf,snapkv,corm,gemma2}; do

while [ "$(squeue -u $(whoami) --noheader | wc -l)" -ge "${max_tasks}" ]; do
  echo "waiting for a slot to free up... current tasks:" $(squeue -u $(whoami) --noheader | wc -l)
  sleep 20
done


echo "Running ${model} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_kernel.py -m $model -s $sys 2>&1 | tee ${LOG_DIR}/kernel.${device}.${model}.${sys}.log &

done

wait


# e2e
echo Running for e2e
source ${PROJECT_DIR}/baseline_env.sh
for sys in {torch,dynamo,tensorrt,tvm}; do
for model in {attn,h2o,roco,kf,snapkv,corm,gemma2}; do

while [ "$(squeue -u $(whoami) --noheader | wc -l)" -ge "${max_tasks}" ]; do
  echo "waiting for a slot to free up... current tasks:" $(squeue -u $(whoami) --noheader | wc -l)
  sleep 20
done


echo "Running ${model} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_e2e.py -p $HOST -m $model -s $sys 2>&1 | tee ${LOG_DIR}/e2e.${device}.${model}.${sys}.log &

done
done

wait


sys=our
source ${PROJECT_DIR}/our_env.sh
for model in {attn,h2o,roco,kf,snapkv,corm,gemma2}; do

while [ "$(squeue -u $(whoami) --noheader | wc -l)" -ge "${max_tasks}" ]; do
  echo "waiting for a slot to free up... current tasks:" $(squeue -u $(whoami) --noheader | wc -l)
  sleep 20
done


echo "Running ${model} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_e2e.py -p $HOST -m $model -s $sys 2>&1 | tee ${LOG_DIR}/e2e.${device}.${model}.${sys}.log &

done

wait
