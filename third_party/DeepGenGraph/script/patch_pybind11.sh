#!/bin/bash
set -x

TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
PYBIND11_PATH="${TORCH_PATH}/include/pybind11"

cp ${PYBIND11_PATH}/cast.h ${PYBIND11_PATH}/cast.h.bak
sed -i 's/return caster\.operator typename make_caster<T>::template cast_op_type<T>();/return caster;/' ${PYBIND11_PATH}/cast.h

