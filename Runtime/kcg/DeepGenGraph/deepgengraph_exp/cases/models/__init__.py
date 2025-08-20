from .attn import Llama as Attn
from .corm import Llama as Corm
from .gemma2 import Llama as Gemma2
from .h2o import Llama as H2O
from .kf import  Llama as KeyFormer
from .roco import Llama as RoCo
from .snapkv import Llama as SnapKV


MODEL_ZOO = {
  'attn': Attn,
  'corm': Corm,
  'gemma2': Gemma2,
  'h2o': H2O,
  'kf': KeyFormer,
  'roco': RoCo,
  'snapkv': SnapKV,
}