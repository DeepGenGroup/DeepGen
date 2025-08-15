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
elif [[ "$HOST" == "fuse0" ]]; then
device=h100
else
  echo "unknown host: $HOST"
  exit -1
fi

source ${PROJECT_DIR}/baseline_env.sh
for sys in {torch,dynamo,tensorrt}; do
for seqlen in {1024,2048,4096,8192}; do
echo "Running ${seqlen} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_kernel.py -m h2o -s $sys --seqlen $seqlen 2>&1 | tee ${LOG_DIR}/${device}.${sys}.${seqlen}.log
done
done

sys=our
source ${PROJECT_DIR}/our_env.sh
for seqlen in {1024,2048,4096,8192}; do
echo "Running ${seqlen} on ${sys}"
$SRUN python ${PROJECT_DIR}/run_kernel.py -m h2o -s $sys --seqlen $seqlen 2>&1 | tee ${LOG_DIR}/${device}.${sys}.${seqlen}.log
done
