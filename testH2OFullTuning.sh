#!/bin/bash
set -e

DEV=7
DTYPE=float32
SEQLEN=${SEQLEN:-2048}
PROJECT_DIR=/home/xushilong/DeepGen
CFG=${PROJECT_DIR}/TuningConfigs/h2o_tuning.json

cd ${PROJECT_DIR}/Runtime
export PYTHONPATH=`pwd`
cd kcg

echo "=============================================="
echo " H2O Full Tuning - Device ${DEV}, dtype ${DTYPE}, seqlen ${SEQLEN}"
echo " Config: ${CFG}"
echo "=============================================="

echo ""
echo "=== [1/4] Tuning K1 (GemmStats) ==="
date
python SimpleLocalTester.py ${CFG} ./testattn_h2o_k1_${SEQLEN}.json 0 0 0 0 h2o_k1 ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=== [2/4] Tuning K2 (GemmNormColSum) ==="
date
python SimpleLocalTester.py ${CFG} ./testattn_h2o_k2_${SEQLEN}.json 0 0 0 0 h2o_k2 ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=== [3/4] Tuning K3 (FlashAttnSplitK2) ==="
date
python SimpleLocalTester.py ${CFG} ./testattn_h2o_k3_${SEQLEN}.json 0 0 0 0 h2o_k3 ${DEV} ${DTYPE} --seqlen "$SEQLEN"

echo ""
echo "=== [4/4] Combined K1+K2+K3 Benchmark ==="
date
python H2OCombinedBenchmark.py \
  ./testattn_h2o_k1_${SEQLEN}.json ./testattn_h2o_k2_${SEQLEN}.json ./testattn_h2o_k3_${SEQLEN}.json \
  ./testattn_h2o_combined_${SEQLEN}.json ${DEV} ${DTYPE}

echo ""
echo "=============================================="
echo " H2O Full Tuning Complete"
echo " Results: testattn_h2o_k1_${SEQLEN}/k2_${SEQLEN}/k3_${SEQLEN}/combined_${SEQLEN}.json"
echo "=============================================="
date
