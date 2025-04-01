#! /bin/bash
cd $(dirname "$0")
is_as_pymodule='ON'
is_debug_mode='ON'
mkdir build
mkdir _dump
cd build  
cmake .. -DCOMPILE_AS_PYMODULE=$is_as_pymodule -DIS_DEBUG_MODE=$is_debug_mode
make -j16
