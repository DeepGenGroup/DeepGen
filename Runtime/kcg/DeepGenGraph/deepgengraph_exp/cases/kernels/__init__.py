from .attn import Attn
from .corm import Corm
from .gemma2 import Gemma2
from .h2o import H2O
from .kf import  KeyFormer
from .roco import RoCo
from .snapkv import SnapKV


KERNEL_ZOO = {
  'attn': Attn,
  'corm': Corm,
  'gemma2': Gemma2,
  'h2o': H2O,
  'kf': KeyFormer,
  'roco': RoCo,
  'snapkv': SnapKV,
}