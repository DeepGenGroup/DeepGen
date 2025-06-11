import time
import glob
from typing import Generator
from kcg.Utils import *
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Operators import matmul, attention
import multiprocessing
import attn_FP32_test as ATT

def compile_model_kernels(mdl) :
    ...

