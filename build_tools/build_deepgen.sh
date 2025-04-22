#! /bin/bash
project_dir="$HOME/DeepGen"
cd $project_dir
is_as_pymodule='ON'
enable_debug_mode='OFF'
mkdir build
mkdir _dump
cd build  
cmake .. \
    -DCOMPILE_AS_PYMODULE=$is_as_pymodule \
    -DENABLE_DEBUG_LOG=$enable_debug_mode \
    -Dpybind11_DIR=$HOME/anaconda3/envs/py310/lib/python3.10/site-packages/pybind11/share/cmake/pybind11
make -j16
