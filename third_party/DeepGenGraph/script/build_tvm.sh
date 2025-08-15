#!/bin/bash
set -x

PROJECT_DIR=$(dirname $(dirname $(readlink -f $0)))

# tvm v0.16.0
# git submodule update --init --recursive

cd ${PROJECT_DIR}/3rd/tvm

mkdir build
cp ${PROJECT_DIR}/script/tvm_config.cmake build/config.cmake

cd build
cmake .. -G Ninja
ninja


# python

# xgboost=2.1.0 remove rabit
pip install xgboost==2.0.0
cd ${PROJECT_DIR}/3rd/tvm/python
python setup.py install