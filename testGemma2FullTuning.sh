#!/bin/bash
set -e

DEV=7
DTYPE=float32
SEQLEN=${SEQLEN:-4096}
PROJECT_DIR=/home/xushilong/DeepGen
CFG=${PROJECT_DIR}/TuningConfigs/gemma2_tuning.json
PYTHON_BIN=${PYTHON_BIN:-python3}

cd ${PROJECT_DIR}/Runtime
export PYTHONPATH=`pwd`
cd kcg

echo "=============================================="
echo " Gemma2 Full Tuning - Device ${DEV}, dtype ${DTYPE}, seqlen ${SEQLEN}"
echo " Config: ${CFG}"
echo "=============================================="

echo ""
echo "=== [1/3] Tuning K1 (GemmStats + softcap) ==="
date
${PYTHON_BIN} SimpleLocalTester.py ${CFG} ./testattn_gemma2_k1.json 0 1 0 0 gemma2_k1 ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=== [2/3] Tuning K2 (FlashAttnSplitK2 + softcap) ==="
date
${PYTHON_BIN} SimpleLocalTester.py ${CFG} ./testattn_gemma2_k2.json 0 0 0 0 gemma2_k2 ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=== [3/3] Combined K1+K2 Benchmark ==="
date
${PYTHON_BIN} Gemma2CombinedBenchmark.py \
  ./testattn_gemma2_k1.json ./testattn_gemma2_k2.json \
  ./testattn_gemma2_combined.json ${DEV} ${DTYPE}

echo ""
echo "=============================================="
echo " Gemma2 Full Tuning Complete"
echo " Results: testattn_gemma2_k1/k2/combined.json"
echo "=============================================="
date
