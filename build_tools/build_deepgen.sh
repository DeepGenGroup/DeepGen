#! /bin/bash
project_dir="$HOME/DeepGen"
cd $project_dir
is_as_pymodule='ON'
buildType=MinSizeRel  # Debug Release MinSizeRel 
mkdir build
mkdir _dump
cd build  
cmake .. \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCOMPILE_AS_PYMODULE=$is_as_pymodule \
    -DCMAKE_BUILD_TYPE=$buildType \
    -Dpybind11_DIR=$HOME/anaconda3/envs/py310/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 \
    -Dnanobind_DIR=/home/xushilong/anaconda3/envs/triton_rocm/lib/python3.8/site-packages/nanobind/cmake
make -j16
