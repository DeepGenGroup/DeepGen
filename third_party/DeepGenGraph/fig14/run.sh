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

if [[ "$HOST" == "node-9658" ]]; then
device=a100
elif [[ "$HOST" == "fuse0" ]]; then
device=h100
else
  echo "unknown host: $HOST"
  exit -1
fi


for sys in {flashattn,flashinfer,tensorrt}; do
for model in {attn,gemma2}; do
echo "Running ${model} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_kernel.py -m $model -s $sys 2>&1 | tee ${LOG_DIR}/${device}.${model}.${sys}.log
done
done


sys=our
source ${PROJECT_DIR}/our_env.sh
for model in {attn,gemma2}; do
echo "Running ${model} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_kernel.py -m $model -s $sys 2>&1 | tee ${LOG_DIR}/${device}.${model}.${sys}.log
done

