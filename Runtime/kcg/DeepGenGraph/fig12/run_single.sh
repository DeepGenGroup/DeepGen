#!/bin/bash
set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <exp: kernel|e2e> <model> <sys: torch|dynamo|tensorrt|tvm|our>"
  exit 1
fi

EXP=$1
MODEL=$2
SYS=$3

FIG_DIR=$(dirname $(readlink -f $0))
PROJECT_DIR=$(dirname $FIG_DIR)
HOST=$(hostname)
echo "hostname: $HOST"
echo "project_dir: $PROJECT_DIR"


LOG_DIR=${FIG_DIR}/ae_logs
mkdir -p ${LOG_DIR}

if [[ "$HOST" == "yes" ]]; then
device=a100
elif [[ "$HOST" == "fuse0" ]]; then
device=h100
else
  echo "unknown host: $HOST"
  exit -1
fi


if [[ "$SYS" == "our" ]]; then
  source ${PROJECT_DIR}/our_env.sh
else
  source ${PROJECT_DIR}/baseline_env.sh
fi

if [[ "$EXP" == "kernel" ]]; then
  echo "Running kernel experiment: model=${MODEL}, sys=${SYS}"
  $SRUN python ${PROJECT_DIR}/run_kernel.py -m $MODEL -s $SYS 2>&1 | tee ${LOG_DIR}/kernel.${device}.${MODEL}.${SYS}.log
elif [[ "$EXP" == "e2e" ]]; then
  echo "Running e2e experiment: model=${MODEL}, sys=${SYS}"
  $SRUN python ${PROJECT_DIR}/run_e2e.py -p $HOST -m $MODEL -s $SYS 2>&1 | tee ${LOG_DIR}/e2e.${device}.${MODEL}.${SYS}.log
else
  echo "Invalid experiment type: $EXP. Must be 'kernel' or 'e2e'."
  exit -1
fi
