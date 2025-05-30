import json
import importlib.util
from typing import Any, Generator, List
from kcg.Operators.matmul import MatmulTuningArgs
from kcg.Utils import *

kw = ConfigKeywords

def readConfigJson(path):
  # read json file
  with open(path, 'r') as f:
    json_str = f.read().strip()
  cfg_dict = json.loads(json_str)
  return cfg_dict

# create configs
class CreateMatmulConfig:
  def __init__(self, cfg_dict, word_width):
    self.cfg_dict = cfg_dict
    self.word_width = word_width  # 一个字 4 byte
    self.max_reg_size = 256 * self.word_width   # byte
    self.max_sm_size = 64 * 1024  # byte
    # self.encoder = TuningSpaceEncoder(self.cfg_dict)

  def getThreadTile(self, halfTag=True, squareTag=True):
    # 获取线程tile的大小，halfTag为是tile只是用一半，另一半对称的不使用，squareTag且尽量方形
    thread_tiles = []
    for tm in self.cfg_dict[kw.KEY_THREAD_SIZE_M]:
      for tn in self.cfg_dict[kw.KEY_THREAD_SIZE_N]:
        rate = int(tm / tn)
        if squareTag and halfTag:
          if rate <= 2 and rate > 0:
            thread_tiles.append((tm, tn))
        elif not squareTag and halfTag:
          if rate > 0:
            thread_tiles.append((tm, tn))
        elif squareTag and not halfTag:
          if rate <= 2:
            thread_tiles.append((tm, tn))
        else:
          thread_tiles.append((tm, tn))
    # (thread_size_m, thread_size_n)
    return thread_tiles
    

  def getBlockTile(self, halfTag=True, squareTag=True):
    # block tile size 也靠近方正，且不对称
    block_tiles = []
    for bm in self.cfg_dict[kw.KEY_BLOCK_SIZE_M]:
      for bn in self.cfg_dict[kw.KEY_BLOCK_SIZE_N]:
        rate = int(bm / bn)
        if squareTag and halfTag:
          if rate <= 2 and rate > 0:
            block_tiles.append((bm, bn))
        elif not squareTag and halfTag:
          if rate > 0:
            block_tiles.append((bm, bn))
        elif squareTag and not halfTag:
          if rate <= 2:
            block_tiles.append((bm, bn))
        else:
          block_tiles.append((bm, bn))
    # (block_size_m, block_size_n)
    return block_tiles
    
  
  def getSplitUAndLayout(self, ttiles, bthiles, max_thread_num=256):
    # 生成splitu和layout参数（收到splitU参数的制约）
    tileAndlayouts = []
    for spu in self.cfg_dict[kw.KEY_LOCAL_SPLIT_U]:
      for bt in bthiles:
        for tt in ttiles:
          ty_num = int(bt[0] / tt[0])   # block y 方向的 thread 数量  block_size_m / thread_size_m
          tx_num = int(bt[1] / tt[1])
          if max_thread_num >= ty_num * tx_num * spu:  # 线程个数和splitU相关，layout不随着splitU的增大而翻倍
            for wlm in self.cfg_dict[kw.KEY_WARP_LAYOUT_M]:
              for wln in self.cfg_dict[kw.KEY_WARP_LAYOUT_N]:
                if (wlm * wln == self.cfg_dict[kw.KEY_WARP_SIZE][0]):  # warp_x * warp_y == 64
                  blm = int(ty_num / wlm)
                  bln = int(tx_num / wln)
                  # block_y % warp_y == 0 && block 至少有一个warp
                  if ty_num % wlm == 0 and tx_num % wln == 0 and blm >= 1 and bln >= 1:
                    if blm in self.cfg_dict[kw.KEY_BLOCK_LAYOUT_M] and bln in self.cfg_dict[kw.KEY_BLOCK_LAYOUT_N]:  # 符合json中的设置
                      tileAndlayouts.append((bt, tt, (blm, bln), (wlm, wln), spu))
    # (block_size_m/n, thread_size_m/n, block_layout_m/n, warp_layout_m/n, splitU)
    return tileAndlayouts


  def getBlockK(self, tileAndlayouts):
    # 添加 block k 变量，受到 load_width 的限制
    new_tals = []
    for tal in tileAndlayouts:
      for bk in self.cfg_dict[kw.KEY_BLOCK_SIZE_K]:
        # thread_num =  block_layout_m * warp_layout_m * block_layout_m * warp_layout_m * splitU
        th_num = tal[2][0] * tal[3][0] * tal[2][1] * tal[3][1] * tal[4]
        smA_size = tal[0][0] * bk  # block_size_m * block_size_k
        smB_size = tal[0][1] * bk  # block_szie_n * block_size_k
        total_load_width_a = int(smA_size / th_num)
        total_load_width_b = int(smB_size / th_num)
        for lwa in self.cfg_dict[kw.KEY_GLOB_LOAD_WIDTH_A]:
          for lwb in self.cfg_dict[kw.KEY_GLOB_LOAD_WIDTH_B]:
            # 每个线程加载的总数量 >= 设置的load_width && 能整除
            if (lwa <= total_load_width_a and total_load_width_a % lwa == 0) and (lwb <= total_load_width_b and total_load_width_b % lwb == 0):
              block_size = (tal[0][0], tal[0][1], bk)  # (block_size_m, block_size_n, block_size_k)
              load_width = (lwa, lwb)  # (glob_load_width_a, glob_load_width_b)
              new_tals.append((block_size, tal[1], load_width, tal[2], tal[3], tal[4]))
    # (block_size_m/n/k, thread_size_m/n, glob_load_width_a/b, block_layout_m/n, warp_layout_m/n, splitU)
    return new_tals


  def getScatterWidth(self, tileAndlayouts):
    # 离散化的宽度，受到 thread tile 的限制
    new_tals = []
    for tal in tileAndlayouts:
      for wswa in self.cfg_dict[kw.KEY_WARP_SCATTER_WIDTH_A]:
        for tswa in self.cfg_dict[kw.KEY_THREAD_SCATTER_WIDTH_A]:
          # WARP_SCATTER_WIDTH_A >= THREAD_SCATTER_WIDTH_A  &  WARP_SCATTER_WIDTH_A <= thread_size_m
          if wswa >= tswa and wswa <= tal[1][0]:
            for wswb in self.cfg_dict[kw.KEY_WARP_SCATTER_WIDTH_B]:
              for tswb in self.cfg_dict[kw.KEY_THREAD_SCATTER_WIDTH_B]:
                if wswb >= tswb and wswb <= tal[1][1]:
                  wsw = (wswa, wswb)  # (warp_scatter_a/b)
                  tsw = (tswa, tswb)  # (thread_scatter_a/b)
                  new_tals.append((tal[0], tal[1], tal[2], tal[3], tal[4], wsw, tsw, tal[5]))
    # (block_size_m/n/k, thread_size_m/n, glob_load_width_a/b, block_layout_m/n, 
    #  warp_layout_m/n, warp_scatter_a/b, thread_scatter_a/b, splitU)
    return new_tals


  def getPrefetchAndContinuous(self, tileAndlayouts):
    # 确定两个预取和load是否连续，同时实现了对memory的限制
    new_tals = []
    for tal in tileAndlayouts:
      for lc in self.cfg_dict[kw.KEY_LOAD_CONTINUOUS]:
        for pshread in self.cfg_dict[kw.KEY_SHARED_PREFETCH]:
          for preg in self.cfg_dict[kw.KEY_REG_PREFETCH]:
            sm_size = (tal[0][0] + tal[0][1]) * tal[0][2]  # 基础sm的大小  (block_size_m + block_size_n) * block_size_k
            reg_size = (tal[1][0] * tal[0][1])             # 基础reg的大小  (thread_size_m * thread_size_n)
            if pshread == 1:    # double shared buf
              sm_size = sm_size * 2
            if tal[-1] > 1:  # splitU > 1  =>  compare sm_size with smC_size
              # smC_size = splitU * block_size_m * block_szie_n
              sm_size = max(sm_size, tal[-1] * tal[0][0] * tal[0][1])  # splitU * block_size_m * block_size_n
            if preg == 1:   # double registers buf
              reg_size = reg_size * 2
            if reg_size <= self.max_reg_size and sm_size <= self.max_sm_size:  # reg & shared memory limit
              line = (tal[0], tal[1], tal[2], tal[3], tal[4], tal[5], tal[6], (pshread, preg, lc), tal[7])
              new_tals.append(line)
    # (block_size_m/n/k, thread_size_m/n, glob_load_width_a/b, block_layout_m/n, 
    #  warp_layout_m/n, warp_scatter_a/b, thread_scatter_a/b, (preftch_sm/reg, glob_load_continuous), splitU)
    return new_tals


  def getOther(self, tileAndlayouts):
    # 将剩下的参数补上
    new_tals = []
    temp_tals = []
    for tal in tileAndlayouts:
      if tal[-1] > 1:             # have splitU
        single_smC_size = tal[0][0] * tal[0][1]  # 单个smC分块的大小 block_size_m * block_size_n
        th_num = tal[3][0] * tal[3][1] * tal[4][0] * tal[4][1] * tal[-1]
        # 每个线程存储元素的个数，因为reduceC是由循环控制累加，所以所有线程参与到一个小的分块上
        total_store_width = int(single_smC_size / th_num)  
        for sw in self.cfg_dict[kw.KEY_GLOB_STORE_WIDTH ]:
          # 存储 smC 到 glob 设置的宽度 store_width，每个线程存储的 width % store_width == 0 && store_width <= width
          if sw <= total_store_width and total_store_width % sw == 0:
            for rcc in self.cfg_dict[kw.KEY_REDUCE_C_CONTINUOUS]:
              line = (tal[0], tal[1], tal[2], tal[3], tal[4], tal[5], tal[6], tal[7], (tal[-1], sw, rcc))
              temp_tals.append(line)
      else:
        line = (tal[0], tal[1], tal[2], tal[3], tal[4], tal[5], tal[6], tal[7], (tal[-1], 0, 0))
        temp_tals.append(line)
        # (block_size_m/n/k, thread_size_m/n, glob_load_width_a/b, block_layout_m/n, 
        #  warp_layout_m/n, warp_scatter_a/b, thread_scatter_a/b, (preftch_sm/reg, glob_load_continuous), 
        # (splitU, glob_store_width, reduce_c_continuous))

    # block maping & unroll
    for ttal in temp_tals:
      for unroll in self.cfg_dict[kw.KEY_UNROLL_NUM]:
        for bmap in self.cfg_dict[kw.KEY_BLOCK_MAPPING]:
          by_num = self.cfg_dict[kw.KEY_M][0] / ttal[0][0]  # M / block_size_m
          if by_num >= bmap:   # mapping block 的个数不能超过 grid y
            line = (ttal[0], ttal[1], ttal[2], ttal[3], ttal[4], ttal[5], ttal[6], ttal[7], ttal[8], (bmap, unroll))
            new_tals.append(line)
    # (block_size_m/n/k, thread_size_m/n, glob_load_width_a/b, block_layout_m/n, 
    #  warp_layout_m/n, warp_scatter_a/b, thread_scatter_a/b, (preftch_sm/reg, glob_load_continuous), 
    # (splitU, glob_store_width, reduce_c_continuous), (unroll, block_mapping))
    return new_tals

  def createMatMulConfig(self, thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256) -> TsGeneratorType :
    # main
    ttiles = self.getThreadTile(halfTag=thalfTag, squareTag=tsquareTag)
    btiles = self.getBlockTile(halfTag=bhalfTag, squareTag=bsquareTag)
    tals = self.getSplitUAndLayout(ttiles, btiles, max_thread_num=max_thread_num)
    temp_tals = self.getBlockK(tals)
    temp_tals = self.getScatterWidth(temp_tals)
    temp_tals = self.getPrefetchAndContinuous(temp_tals)
    temp_tals = self.getOther(temp_tals)
    for tal in temp_tals:
      ta = MatmulTuningArgs(self.cfg_dict[kw.KEY_M][0], self.cfg_dict[kw.KEY_N][0], self.cfg_dict[kw.KEY_K][0], self.cfg_dict[kw.KEY_BATCH][0] ,            # M, N, K, batch
                            self.cfg_dict[kw.KEY_DTYPE_A][0]) 
                            # self.cfg_dict[kw.KEY_DTYPE_B][0], 
                            # self.cfg_dict[kw.KEY_DTYPE_C][0]) # typeA, typeB, typeC
      config = (
        tal[0][0], tal[0][1], tal[0][2],  # block_size
        tal[1][0], tal[1][1],             # thread_size
        tal[2][0], tal[2][1],             # load_width
        tal[3][0], tal[3][1],             # block_layout
        tal[4][0], tal[4][1],             # warp_layout
        tal[5][0], tal[5][1],             # warp_scatter
        tal[6][0], tal[6][1],             # thread_scatter
        tal[7][0], tal[7][1], tal[7][2],  # sm_prefatch, reg_prefatch, load_continuous
        tal[8][0], tal[8][1], tal[8][2],  # splitU, store_width, store_continuos
        tal[9][0], tal[9][1],             # block_mapping, unroll
        self.cfg_dict[kw.KEY_WARP_SIZE][0], self.cfg_dict[kw.KEY_IS_A_TRANSPOSE][0],   # warp_size, is_Atran
      )
      ta.assignWithList(*config)
      ret = CompileNeededInfo()
      ret.baseArgs = [ta.M, ta.N, ta.K]
      ret.tsArgs = [            
          ta.BLOCK_SIZE_M,
          ta.BLOCK_SIZE_N,
          ta.BLOCK_SIZE_K,
          ta.THREAD_SIZE_M,
          ta.THREAD_SIZE_N,
          ta.WARP_SIZE,
          ta.BLOCK_LAYOUT_M,
          ta.BLOCK_LAYOUT_N,
          ta.WARP_LAYOUT_M,
          ta.WARP_LAYOUT_N,
          ta.GLOB_LOAD_WIDTH_A,
          ta.GLOB_LOAD_WIDTH_B,
          ta.WARP_SCATTER_WIDTH_A,
          ta.WARP_SCATTER_WIDTH_B,
          ta.THREAD_SCATTER_WIDTH_A,
          ta.THREAD_SCATTER_WIDTH_B,
          ta.LOCAL_SPLIT_U,
          ta.BLOCK_MAPPING,
          ta.GLOB_STORE_WIDTH,
          ta.UNROLL_NUM,
          ta.REG_PREFETCH,
          ta.SHARED_PREFETCH,
          ta.LOAD_CONTINUOUS,
          ta.REDUCE_C_CONTINUOUS,
          ta.dtA, # A
          ta.dtB, # B
          ta.dtC, # C
          ta.M, ta.N, ta.K, ta.batch,
          ta.isATranspose
        ]
      ret.dataType = ta.dtC
      yield ret
    

def getTuneSpace(geemConfigPath : str) -> TsGeneratorType :
  cfg_dict = readConfigJson(geemConfigPath)
  cmc = CreateMatmulConfig(cfg_dict, 4)
  kams = cmc.createMatMulConfig(thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256)  # KernelArgMatmul
  return kams

# # example code 
# if "__main__" == __name__:
#   path = "/home/xiebaokang/projects/mlir/DeepGen/TuningConfigs/GEMM_configs_1024.json"
#   cfg_dict = readConfigJson(path)
#   cmc = CreateMatmulConfig(cfg_dict, 4)
#   kams = cmc.createMatMulConfig(thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256)  # KernelArgMatmul
  # print(len(kams))
#   compileFunc = getCompileFunc()
#   for kam in kams:
#     packedKernel = compile(compileFunc, kam, 7)
#     break

