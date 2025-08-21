#! /bin/bash
project_dir="$HOME/DeepGen"
cd $project_dir
is_as_pymodule='ON'
buildType=Release  # Debug Release MinSizeRel 
mkdir build
mkdir _dump
cd build  
cmake .. \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCOMPILE_AS_PYMODULE=$is_as_pymodule \
    -DCMAKE_BUILD_TYPE=$buildType \
    -DENABLE_GRAPH_OPT=OFF
make -j16
