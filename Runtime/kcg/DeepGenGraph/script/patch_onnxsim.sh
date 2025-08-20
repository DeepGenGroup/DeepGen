#!/bin/bash
set -x

ONNXSIM_PATH=$(python -c "import onnxsim; print(onnxsim.__path__[0])")

# ref: https://github.com/daquexian/onnx-simplifier/issues/171
cp ${ONNXSIM_PATH}/onnx_simplifier.py ${ONNXSIM_PATH}/onnx_simplifier.py.bak
sed -i 's/sess_options = rt\.SessionOptions()/sess_options = rt\.SessionOptions(); sess_options\.intra_op_num_threads = 1; sess_options\.inter_op_num_threads = 1/g' ${ONNXSIM_PATH}/onnx_simplifier.py
