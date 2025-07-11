import contextlib
import sys
import io
import sysconfig
import os
import shutil
import setuptools
import subprocess
import tempfile
from pathlib import Path
import torch
import statistics

dirname = os.path.dirname(os.path.realpath(__file__))
# torch_include_dir = cpp_extension.include_paths()
# print(torch_include_dir)
# compile dependent
dep_lib = {
  "rocm": (["/opt/rocm/lib"], ["/opt/rocm/include"], ["hsa-runtime64", "amdhip64"]), 
  "cuda": (["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"], ["/usr/local/cuda/include"], ["cuda", "cudart"])
}

@contextlib.contextmanager
def quiet():
  old_stdout, old_stderr = sys.stdout, sys.stderr
  sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
  try:
    yield
  finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
  suffix = sysconfig.get_config_var('EXT_SUFFIX')
  so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
  # try to avoid setuptools if possible
  cc = os.environ.get("CC")
  if cc is None:
    # TODO: support more things here.
    clang = shutil.which("clang")
    gcc = shutil.which("g++")
    cc = gcc if gcc is not None else clang
    if cc is None:
      raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
  # This function was renamed and made public in Python 3.10
  if hasattr(sysconfig, 'get_default_scheme'):
    scheme = sysconfig.get_default_scheme()
  else:
    scheme = sysconfig._get_default_scheme()
  # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
  # path changes to include 'local'. This change is required to use triton with system-wide python.
  if scheme == 'posix_local':
    scheme = 'posix_prefix'
  py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
  include_dirs = include_dirs + [srcdir, py_include_dir]
  # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
  cc_cmd = ["g++", src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-Wno-unused-result", "-o", so]
  cc_cmd += [f'-l{lib}' for lib in libraries]
  cc_cmd += [f"-L{dir}" for dir in library_dirs]
  cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
  ret = subprocess.check_call(cc_cmd)
  if ret == 0:
    return so
  # fallback on setuptools
  extra_compile_args = []
  # extra arguments
  extra_link_args = []
  # create extension module
  ext = setuptools.Extension(
    name=name,
    language='c',
    sources=[src],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args + ['-O3'],
    extra_link_args=extra_link_args,
    library_dirs=library_dirs,
    libraries=libraries,
  )
  # build extension module
  args = ['build_ext']
  args.append('--build-temp=' + srcdir)
  args.append('--build-lib=' + srcdir)
  args.append('-q')
  args = dict(
    name=name,
    ext_modules=[ext],
    script_args=args,
  )
  with quiet():
    setuptools.setup(**args)
  return so


def loadLibModule(name: str, path: str):
  # load func in lib file
  import importlib.util
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod

def compileModuleFromFile(name: str, src_path: str, lib_path:str, target="rocm"):
  # compile from src file
  name_ = f"{name}_{target}"
  dep = dep_lib[target]
  if not Path(lib_path).exists():
    os.mkdir(lib_path)
  so = _build(name_, src_path, lib_path, *dep)
  return loadLibModule(name, so)


def compileModuleFromSrc(name: str, src: str, target="rocm"):
  # src is code string, using g++ compile host code, return mod
  with tempfile.TemporaryDirectory() as tmpdir:
    src_path = os.path.join(tmpdir, "main.c")
    with open(src_path, "w") as f:
      f.write(src)
    name_ = f"{name}_{target}"
    dep = dep_lib[target]
    so = _build(name_, src_path, tmpdir, *dep)
    mod = loadLibModule(name, so)
  return mod


def getGPUInfo(target="rocm"):
  # get gpu info and device count
  utilPath = os.path.join(dirname, "../utils")
  suffix = sysconfig.get_config_var('EXT_SUFFIX')
  so = os.path.join(utilPath, "../lib", '{name}{suffix}'.format(name=f"gpu_info_{target}", suffix=suffix))
  if os.path.exists(so):
    mod = loadLibModule("gpu_info", so)
    return mod.get_device_count(), mod.get_gpu_info()
  lib_dir = os.path.join(utilPath, "lib")
  src_dir = os.path.join(utilPath, f"{target}.cc")
  mod = compileModuleFromFile("gpu_info", src_dir, lib_dir, target)
  return mod.get_device_count(), mod.get_gpu_info()


def perf(kernelFunc, inputs, warmup=1, iter=3):
  for _ in range(warmup):
    kernelFunc(*inputs)
  times = []
  for _ in range(iter):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    kernelFunc(*inputs)
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end)
    times.append(elapsed_time_ms)
  median = statistics.median(times)
  return median

# if __name__ == "__main__":
#   dev_count, info = getGPUInfo(target="cuda")
#   print(dev_count, info.name)