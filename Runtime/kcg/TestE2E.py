import time
import glob
from typing import Generator
from kcg.Utils import *
from kcg.HIPLauncher import *
from kcg.CUDALauncher import *
from kcg.Operators import matmul, attention
import multiprocessing
import Runtime.kcg.tuning.attn_FP32_test as ATT
from torch import nn

def compile_model_kernels(model : nn.Module) :
    ...

