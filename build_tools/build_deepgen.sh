#! /bin/bash
project_dir="$HOME/DeepGen"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "CONDA_PREFIX is not set. Please activate the target conda environment first."
    exit 1
fi

cd "$project_dir"
is_as_pymodule='ON'
buildType=Release  # Debug Release MinSizeRel 
mkdir -p build
mkdir -p _dump
cd build
cmake .. \
    -DMLIR_DIR=/home/xushilong/rocm-llvm-install/lib/cmake/mlir \
    -DLLVM_DIR=/home/xushilong/rocm-llvm-install/lib/cmake/llvm \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCOMPILE_AS_PYMODULE="$is_as_pymodule" \
    -DCMAKE_BUILD_TYPE="$buildType" \
    -DENABLE_GRAPH_OPT=OFF \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DZLIB_ROOT="$CONDA_PREFIX" \
    -Dzstd_ROOT="$CONDA_PREFIX"
make -j16
