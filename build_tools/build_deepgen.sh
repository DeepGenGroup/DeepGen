#! /bin/bash
cd $(dirname "$0")
is_as_pymodule='ON'
enable_debug_mode='OFF'
mkdir build
mkdir _dump
cd build  
cmake .. -DCOMPILE_AS_PYMODULE=$is_as_pymodule -DENABLE_DEBUG_LOG=$enable_debug_mode
make -j16
