#!/bin/bash
set -e

DEV=3
DTYPE=float32
SEQLEN=${SEQLEN:-4096}
PROJECT_DIR=/home/xushilong/DeepGen
CFG=${PROJECT_DIR}/TuningConfigs/attn_split_tuning.json
PYTHON_BIN=${PYTHON_BIN:-python3}

cd ${PROJECT_DIR}/Runtime
export PYTHONPATH=`pwd`
cd kcg

echo "=============================================="
echo " Attention Split Full Tuning - Device ${DEV}, dtype ${DTYPE}, seqlen ${SEQLEN}"
echo " Config: ${CFG}"
echo "=============================================="

echo ""
echo "=== [1/3] Tuning K1 (GemmStats) ==="
date
${PYTHON_BIN} SimpleLocalTester.py ${CFG} ./testattn_split_k1.json 0 1 0 0 attn_k1 ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=== [2/3] Tuning K2 (FlashAttnSplitK2) ==="
date
${PYTHON_BIN} SimpleLocalTester.py ${CFG} ./testattn_split_k2.json 0 1 0 0 attn_k2 ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=== [3/3] Combined K1+K2 Benchmark ==="
date
${PYTHON_BIN} AttentionSplitCombinedBenchmark.py \
  ./testattn_split_k1.json ./testattn_split_k2.json \
  ./testattn_split_combined.json ${DEV} ${DTYPE}

echo ""
echo "=============================================="
echo " Attention Split Full Tuning Complete"
echo " Results: testattn_split_k1/k2/combined.json"
echo "=============================================="
date
