#!/bin/bash
HOST=$(hostname)
echo "hostname: $HOST"
PROJECT_DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))
echo "project_dir: $PROJECT_DIR"

if spack find --loaded | grep -q .; then
  spack unload --all 2>/dev/null
else
  echo "No loaded packages to unload."
fi

if command -v deactivate &>/dev/null; then
  deactivate
else
  echo "venv not activated"
fi


if [[ "$HOST" == "yes" ]]; then

spack load gcc@12.2.0
spack load cuda@12.1.1/odp466h
spack load llvm@18.1.2
spack load cmake@3.27.9/w5bq4at
spack load ninja@1.11.1/antp5sf
spack load python@3.10.13/blgzgdt
spack load cudnn@8.9.7.29-12/xzdefne

# force using g++ instead of clang in llvm
export CC=$(which gcc)
export CXX=$(which g++)

source ${PROJECT_DIR}/our_venv/bin/activate

export SRUN="srun -p long -A priority --gres=gpu:a100:1"

elif [[ "$HOST" == "fuse0" ]]; then

spack load cuda@12.1.1/bym4wir
spack load llvm@18.1.2/myuyx62
spack load cmake@3.28.6/zy5zg7l
spack load ninja@1.12.1/cxtc6lz
spack load python@3.10.13/brtehs2
spack load cudnn@8.9.7.29-12/gjbn4bq

# force using g++ instead of clang in llvm
export CC=$(which gcc)
export CXX=$(which g++)

source ${PROJECT_DIR}/our_venv/bin/activate

export SRUN="srun -p Long --gres=gpu:H100:1"

else
  echo "unknown host: $HOST"
fi
