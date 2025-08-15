#!/bin/bash
set -e
PROJECT_DIR=$(dirname $(dirname $(readlink -f $0)))
HOST=$(hostname)
echo "hostname: $HOST"
echo "project_dir: $PROJECT_DIR"

cd ${PROJECT_DIR} && pwd

echo "init spack"
if spack find --loaded | grep -q .; then
  spack unload --all 2>/dev/null
else
  echo "No loaded packages to unload."
fi

if [[ "$HOST" == "yes" ]]; then

spack load gcc@12.2.0
spack load cuda@12.1.1/odp466h
spack load llvm@18.1.2
spack load cmake@3.27.9/w5bq4at
spack load ninja@1.11.1/antp5sf
spack load openmpi@4.1.5%gcc@12.2.0/cext65e
spack load python@3.10.13/blgzgdt
spack load cudnn@8.9.7.29-12/xzdefne

elif [[ "$HOST" == "fuse0" ]]; then

spack load cuda@12.1.1/bym4wir
spack load llvm@18.1.2/myuyx62
spack load cmake@3.28.6/zy5zg7l
spack load ninja@1.12.1/cxtc6lz
spack load python@3.10.13/brtehs2
spack load cudnn@8.9.7.29-12/gjbn4bq

else
  echo "unknown host: $HOST"
  exit -1
fi

which nvcc

# force using g++ instead of clang in llvm
export CC=$(which gcc)
export CXX=$(which g++)

# git submodule update --init --recursive

echo "create baseline venv and install dependencies"
if [ -d "baseline_venv" ]; then
  echo "baseline_venv directory exists, deleting it..."
  rm -rf baseline_venv
  echo "baseline_venv has been deleted."
else
  echo "baseline_venv directory does not exist."
fi
python -m venv baseline_venv
source baseline_venv/bin/activate
pip install packaging==24.2 wheel==0.45.0 torch==2.2.2
TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install -r ${PROJECT_DIR}/requirements-pedantic.txt
${PROJECT_DIR}/script/patch_pybind11.sh
${PROJECT_DIR}/script/patch_onnxsim.sh
TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install ${PROJECT_DIR} -v
${PROJECT_DIR}/script/build_tvm.sh
deactivate


echo "create our venv and install dependencies"
if [ -d "our_venv" ]; then
  echo "our_venv directory exists, deleting it..."
  rm -rf our_venv
  echo "our_venv has been deleted."
else
  echo "our_venv directory does not exist."
fi
python -m venv our_venv
source our_venv/bin/activate
pip install packaging==24.2 wheel==0.45.0 torch==2.2.2
TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install -r ${PROJECT_DIR}/requirements-pedantic.txt 
${PROJECT_DIR}/script/patch_pybind11.sh
${PROJECT_DIR}/script/patch_onnxsim.sh
TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install ${PROJECT_DIR} -v
${PROJECT_DIR}/script/build_our.sh
deactivate
