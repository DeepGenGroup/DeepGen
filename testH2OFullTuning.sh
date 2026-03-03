#!/bin/bash
set -e

DEV=4
DTYPE=float32
PROJECT_DIR=/home/xushilong/DeepGen
CFG=${PROJECT_DIR}/TuningConfigs/h2o_tuning.json

cd ${PROJECT_DIR}/Runtime
export PYTHONPATH=`pwd`
cd kcg

echo "=============================================="
echo " H2O Full Tuning - Device ${DEV}, dtype ${DTYPE}"
echo " Config: ${CFG}"
echo "=============================================="

echo ""
echo "=== [1/4] Tuning K1 (GemmStats) ==="
date
python SimpleLocalTester.py ${CFG} ./testattn_h2o_k1.json 0 0 0 0 h2o_k1 ${DEV} ${DTYPE}

echo ""
echo "=== [2/4] Tuning K2 (GemmNormColSum) ==="
date
python SimpleLocalTester.py ${CFG} ./testattn_h2o_k2.json 0 0 0 0 h2o_k2 ${DEV} ${DTYPE}

echo ""
echo "=== [3/4] Tuning K3 (FlashAttnSplitK2) ==="
date
python SimpleLocalTester.py ${CFG} ./testattn_h2o_k3.json 0 0 0 0 h2o_k3 ${DEV} ${DTYPE}

echo ""
echo "=== [4/4] Combined K1+K2+K3 Benchmark ==="
date
python H2OCombinedBenchmark.py \
  ./testattn_h2o_k1.json ./testattn_h2o_k2.json ./testattn_h2o_k3.json \
  ./testattn_h2o_combined.json ${DEV} ${DTYPE}

echo ""
echo "=============================================="
echo " H2O Full Tuning Complete"
echo " Results: testattn_h2o_k1/k2/k3/combined.json"
echo "=============================================="
date
