#!/bin/bash
set -euo pipefail

project_dir="$HOME/DeepGen"
plugin_src_dir="${project_dir}/tensorrt_plugins"
build_dir="${BUILD_DIR:-${project_dir}/build-trt-plugin-standalone}"
build_type="${BUILD_TYPE:-Release}"
build_jobs="${BUILD_JOBS:-8}"

# Default paths for the current machine. Environment variables may still override them.
python_bin="${Python_EXECUTABLE:-$HOME/anaconda3/envs/deepgen/bin/python}"
trt_include_dir="${TENSORRT_INCLUDE_DIR:-$HOME/.local/TensorRT-10.15-src/include}"
trt_library_dir="${TENSORRT_LIBRARY_DIR:-$HOME/anaconda3/envs/deepgen/lib/python3.12/site-packages/tensorrt_libs}"
cuda_include_dir="${CUDA_INCLUDE_DIR:-$HOME/.local/cuda-12.2/include}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Cannot find Python at ${python_bin}. Set Python_EXECUTABLE=/path/to/python." >&2
  exit 1
fi

if [[ ! -d "${trt_include_dir}" ]]; then
  echo "Cannot find TensorRT headers at ${trt_include_dir}. Set TENSORRT_INCLUDE_DIR or TENSORRT_ROOT_DIR." >&2
  exit 1
fi

if [[ ! -d "${trt_library_dir}" ]]; then
  echo "Cannot find TensorRT libraries at ${trt_library_dir}. Set TENSORRT_LIBRARY_DIR." >&2
  exit 1
fi

if [[ ! -d "${cuda_include_dir}" ]]; then
  echo "Cannot find CUDA headers at ${cuda_include_dir}. Set CUDA_INCLUDE_DIR or CUDA_HOME." >&2
  exit 1
fi

cd "${project_dir}"

mkdir -p "${build_dir}"

cmake \
  -S "${plugin_src_dir}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DPython_EXECUTABLE="${python_bin}" \
  -DTENSORRT_INCLUDE_DIR="${trt_include_dir}" \
  -DTENSORRT_LIBRARY_DIR="${trt_library_dir}" \
  -DCUDA_INCLUDE_DIR="${cuda_include_dir}"

cmake --build "${build_dir}" --target deepgen_trt_attention_plugin -j"${build_jobs}"

echo "Built TensorRT attention plugin:"
echo "  ${project_dir}/bin/libdeepgen_trt_attention_plugin.so"
