#!/bin/bash
set -x

PROJECT_DIR=$(dirname $(dirname $(readlink -f $0)))

git submodule update --init --recursive

cd ${PROJECT_DIR}/3rd/deepgengraph
pip install -e . -v 2>&1 | tee build.log

pip uninstall triton -y 
pip uninstall triton-nightly -y
# pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly==3.0.0.post20240708181524
# pip install /home/dataset/ae_pkgs/triton_nightly-3.0.0.post20240708181524-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl