import glob
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import torch.utils.cpp_extension as torch_cpp_ext


if not os.environ.get("MAX_JOBS"):
  max_num_jobs_cores = max(1, os.cpu_count() // 2)
  os.environ["MAX_JOBS"] = str(max_num_jobs_cores)
print(f"{os.environ.get('MAX_JOBS')=}")


enable_bf16 = True
if enable_bf16:
  torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")

def remove_unwanted_pytorch_nvcc_flags():
  REMOVE_NVCC_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
  ]
  for flag in REMOVE_NVCC_FLAGS:
    try:
      torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
      print(f"no nvcc flag: {flag}")

src = [os.path.join('csrc', 'ffi.cpp')]
for fpath in glob.glob(os.path.join('csrc', '**'), recursive=True):
  if fpath.endswith('.cu'):
    src.append(fpath)

extra_compile_args = {'cxx': ['-O3'], 'nvcc': ['-O3', '-std=c++17', '-Xcompiler', '-fPIC,-Wall,-O3', '--use_fast_math']}
print(f'{src=}')

PROJECT_NAME="asuka_exp"

remove_unwanted_pytorch_nvcc_flags()
setup(
  name=PROJECT_NAME,
  version='0.0.1',
  packages=find_packages(),
  ext_modules=[
    CUDAExtension(
      name=f'{PROJECT_NAME}._csrc',
      sources=src,
      extra_compile_args=extra_compile_args
    )
  ],
  cmdclass={'build_ext': BuildExtension},
  zip_safe=False,
)

