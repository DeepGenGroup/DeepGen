#! /bin/bash
cd $(dirname "$0")
is_as_pymodule='ON'
is_debug_mode='OFF'
mkdir build
mkdir _dump
cd build  
cmake .. -DCOMPILE_AS_PYMODULE=$is_as_pymodule -DDEBUG_MODE=$is_debug_mode
make -j16
