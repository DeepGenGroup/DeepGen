#!/bin/bash
set -e

FIG_DIR=$(dirname $(readlink -f $0))
PROJECT_DIR=$(dirname $FIG_DIR)
source ${PROJECT_DIR}/our_env.sh

DEFAULT_DIR=backup_logs
LOG_DIR=${1:-${DEFAULT_DIR}}

python ${FIG_DIR}/plot.py --log_dir ${LOG_DIR}