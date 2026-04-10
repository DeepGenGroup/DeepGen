#!/bin/bash
set -e

DEV=${DEV:-3}
DTYPE=${DTYPE:-float32}
SEQLEN=${SEQLEN:-4096}
PROJECT_DIR=${PROJECT_DIR:-/home/xushilong/DeepGen}
CFG=${CFG:-${PROJECT_DIR}/TuningConfigs/attn_split.json}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd ${PROJECT_DIR}/Runtime
export PYTHONPATH=`pwd`
cd kcg

echo "=============================================="
echo " Attention Origin Full Tuning - Device ${DEV}, dtype ${DTYPE}, seqlen ${SEQLEN}"
echo " Config: ${CFG}"
echo "=============================================="

echo ""
echo "=== [1/1] Tuning Origin Fused Kernel ==="
date
${PYTHON_BIN} SimpleLocalTester.py \
  ${CFG} ./testattn_origin_${SEQLEN}.json \
  0 0 0 0 attention_origin ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=============================================="
echo " Attention Origin Full Tuning Complete"
echo " Results: testattn_origin_${SEQLEN}.json"
echo "=============================================="
date
