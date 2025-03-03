#! /bin/bash
cd $(dirname "$0")
is_as_pymodule='ON'

mkdir build
mkdir _dump
cd build  
cmake .. -DCOMPILE_AS_PYMODULE=$is_as_pymodule
make -j16
