import json
import importlib.util

from kcg.Operators.matmul import MatmulTuningArgs, ConfigKeywords
from kcg.Utils import PathManager
from kcg.Operators.matmul import TuningSpaceEncoder
from typing import List



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
    self.encoder = TuningSpaceEncoder(self.cfg_dict)

  def getThreadTile(self, halfTag=True, squareTag=True):
    # 获取线程tile的大小，halfTag为是tile只是用一般，另一半对称的不使用，squareTag且尽量方形
    thread_tiles = []
    for tm in self.cfg_dict[ConfigKeywords.KEY_THREAD_SIZE_M]:
      for tn in self.cfg_dict[ConfigKeywords.KEY_THREAD_SIZE_N]:
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
    return thread_tiles

  def getBlockTile(self, halfTag=True, squareTag=True):
    # block tile size 也靠近方正，且不对称
    block_tiles = []
    for bm in self.cfg_dict[ConfigKeywords.KEY_BLOCK_SIZE_M]:
      for bn in self.cfg_dict[ConfigKeywords.KEY_BLOCK_SIZE_N]:
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
    return block_tiles

  def getLayout(self, ttiles, bthiles, max_thread_num=256):
    # 计算出根据 thread tile 与 block tile 计算出 两个 layout，有最大线程数的限制
    tileAndlayouts = []
    for bt in bthiles:
      for tt in ttiles:
        ty_num = int(bt[0] / tt[0])   # block y 方向的 thread 数量
        tx_num = int(bt[1] / tt[1])
        if max_thread_num >= ty_num * tx_num:   # max thread count
          for wlm in self.cfg_dict[ConfigKeywords.KEY_WARP_LAYOUT_M]:
            for wln in self.cfg_dict[ConfigKeywords.KEY_WARP_LAYOUT_N]:
              if (wlm * wln == self.cfg_dict[ConfigKeywords.KEY_WARP_SIZE][0]):  # warp_x * warp_y == 64
                blm = int(ty_num / wlm)
                bln = int(tx_num / wln)
                if ty_num % wlm == 0 and tx_num % wln == 0 and blm >= 1 and bln >= 1:   # block_y % warp_y == 0 && block 至少有一个warp
                  if blm in self.cfg_dict[ConfigKeywords.KEY_BLOCK_LAYOUT_M] and bln in self.cfg_dict[ConfigKeywords.KEY_BLOCK_LAYOUT_N]:  # 符合json中的设置
                    tileAndlayouts.append((bt, tt, (blm, bln), (wlm, wln)))
    return tileAndlayouts

  def getBlockK(self, tileAndlayouts):
    # 添加 block k 变量，受到 load_width 的限制
    new_tals = []
    for tal in tileAndlayouts:
      for bk in self.cfg_dict[ConfigKeywords.KEY_BLOCK_SIZE_K]:
        th_num = tal[2][0] * tal[3][0] * tal[2][1] * tal[3][1]
        smA_size = tal[0][0] * bk
        smB_size = tal[0][1] * bk
        load_width_a = int(smA_size / th_num)
        load_width_b = int(smB_size / th_num)
        for lwa in self.cfg_dict[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_A]:
          for lwb in self.cfg_dict[ConfigKeywords.KEY_GLOB_LOAD_WIDTH_B]:
            # 每个线程加载的总数量 >= 设置的load_width && 能整除
            if (lwa <= load_width_a and load_width_a % lwa == 0) and (lwb <= load_width_b and load_width_b % lwb == 0):
              block_size = (tal[0][0], tal[0][1], bk)
              load_width = (lwa, lwb)
              new_tals.append((block_size, tal[1], load_width, tal[2], tal[3]))
    return new_tals

  def getScatterWidth(self, tileAndlayouts):
    # 离散化的宽度，受到 thread tile 的限制
    new_tals = []
    for tal in tileAndlayouts:
      for wswa in self.cfg_dict[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_A]:
        for tswa in self.cfg_dict[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_A]:
          if wswa >= tswa and wswa <= tal[1][0]:   # thread size >= warp 离散化的w >= thread 离散化的w
            for wswb in self.cfg_dict[ConfigKeywords.KEY_WARP_SCATTER_WIDTH_B]:
              for tswb in self.cfg_dict[ConfigKeywords.KEY_THREAD_SCATTER_WIDTH_B]:
                if wswb >= tswb and wswb <= tal[1][1]:
                  wsw = (wswa, wswb)
                  tsw = (tswa, tswb)
                  new_tals.append((tal[0], tal[1], tal[2], tal[3], tal[4], wsw, tsw))
    return new_tals

  def getPrefetchAndContinuous(self, tileAndlayouts):
    # 确定两个预取和load是否连续
    new_tals = []
    for tal in tileAndlayouts:
      for lc in self.cfg_dict[ConfigKeywords.KEY_LOAD_CONTINUOUS]:
        for pshread in self.cfg_dict[ConfigKeywords.KEY_SHARED_PREFETCH]:
          for preg in self.cfg_dict[ConfigKeywords.KEY_REG_PREFETCH]:
            factor = 1
            if preg == 1 :  # reg prefetch
              factor = 2
            if factor * tal[1][0] * tal[1][1] * self.word_width <= self.max_reg_size:  # reg 限制
              line = (tal[0], tal[1], tal[2], tal[3], tal[4], tal[5], tal[6], (pshread, preg, lc))
              new_tals.append(line)
    return new_tals

  def getSplitU(self, tileAndlayouts,max_thread_num):
    # splitU + sm limit
    new_tals = []
    for tal in tileAndlayouts:
      for spU in self.cfg_dict[ConfigKeywords.KEY_LOCAL_SPLIT_U]:
        single_sm_size = tal[0][2] * (tal[0][0] + tal[0][1])  # 单个 sm 的大小
        if spU != 1:                                          # 有 splitU
          smC_size = tal[0][0] * tal[0][1]   # smC 的大小
          if tal[-1][0] == 1:                               # 有 double smAB buf
            sm_size = max(2 * single_sm_size, smC_size)
          else:                                             # 无 double smAB buf
            sm_size = max(single_sm_size, smC_size)
        else:                                                 # 无 splitU
          if tal[-1][0] == 1:                               # 有 double smAB buf
            sm_size = 2 * single_sm_size
          else:                                             # 无 double smAB buf
            sm_size = single_sm_size
        if sm_size * self.word_width <= self.max_sm_size :   # sm size 是否超出 sm 总大小 
          th_num = tal[3][0] * tal[3][1] * tal[4][0] * tal[4][1]
          if spU * th_num <= max_thread_num : 
            line = (tal[0], tal[1], tal[2], tal[3], tal[4], tal[5], tal[6], tal[7], spU)
            new_tals.append(line)
    return new_tals

  def getOther(self, tileAndlayouts):
    # 将剩下的参数补上
    new_tals = []
    temp_tals = []
    for tal in tileAndlayouts:
      if tal[-1] != 1:  # have splitU
        smC_size = tal[0][0] * tal[0][1]
        th_num = tal[3][0] * tal[3][1] * tal[4][0] * tal[4][1]
        asw = int(smC_size / th_num)
        for sw in self.cfg_dict[ConfigKeywords.KEY_GLOB_STORE_WIDTH]:
          # 存储 smC 到 glob 设置的宽度 store_width，每个线程存储的 width % store_width == 0 && store_width <= width
          if sw <= asw and asw % sw == 0:
            for rcc in self.cfg_dict[ConfigKeywords.KEY_REDUCE_C_CONTINUOUS]:
              line = (tal[0], tal[1], tal[2], tal[3], tal[4], tal[5], tal[6], tal[7], (tal[-1], sw, rcc))
              temp_tals.append(line)
      else:
        line = (tal[0], tal[1], tal[2], tal[3], tal[4], tal[5], tal[6], tal[7], (tal[-1], 0, 0))
        temp_tals.append(line)

    # block maping & unroll
    for ttal in temp_tals:
      for unroll in self.cfg_dict[ConfigKeywords.KEY_UNROLL_NUM]:
        for bmap in self.cfg_dict[ConfigKeywords.KEY_BLOCK_MAPPING]:
          by_num = self.cfg_dict[ConfigKeywords.KEY_M][0] / ttal[0][0]
          if by_num >= bmap:   # mapping block 的个数不能超过 grid y
            line = (ttal[0], ttal[1], ttal[2], ttal[3], ttal[4], ttal[5], ttal[6], ttal[7], ttal[8], (bmap, unroll))
            new_tals.append(line)
            
    return new_tals

  def createMatMulConfig(self, thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256) -> List[int]:
    # main
    ttiles = self.getThreadTile(halfTag=thalfTag, squareTag=tsquareTag)
    btiles = self.getBlockTile(halfTag=bhalfTag, squareTag=bsquareTag)
    tals = self.getLayout(ttiles, btiles, max_thread_num=max_thread_num)
    temp_tals = self.getBlockK(tals)
    temp_tals = self.getScatterWidth(temp_tals)
    temp_tals = self.getPrefetchAndContinuous(temp_tals)
    temp_tals = self.getSplitU(temp_tals, max_thread_num = max_thread_num)
    temp_tals = self.getOther(temp_tals)

    kams = []
    for tal in temp_tals:
      kam = MatmulTuningArgs(
        self.cfg_dict[ConfigKeywords.KEY_M][0], self.cfg_dict[ConfigKeywords.KEY_N][0], self.cfg_dict[ConfigKeywords.KEY_K][0],self.cfg_dict[ConfigKeywords.KEY_BATCH][0],  # M, N, K,batch
        self.cfg_dict[ConfigKeywords.KEY_DTYPE_A][0]) 
        # self.cfg_dict[ConfigKeywords.KEY_DTYPE_B][0], 
        # self.cfg_dict[ConfigKeywords.KEY_DTYPE_C][0]) # typeA, typeB, typeC
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
        self.cfg_dict[ConfigKeywords.KEY_WARP_SIZE][0], self.cfg_dict[ConfigKeywords.KEY_IS_A_TRANSPOSE][0],   # warp_size, is_Atran
      )
      kam.assignWithList(*config)
      kamEncodedStr = self.encoder.encode(kam.jsonfy())
      kams.append(int(kamEncodedStr))
    return kams
    

# def getCompileFunc():
#   # 获取编译程序
#   spec = importlib.util.spec_from_file_location("KCGCompiler", PathManager.kcg_compiler_path())
#   mod = importlib.util.module_from_spec(spec)
#   spec.loader.exec_module(mod)
#   return mod.compile_kernel_matmul

# def compile(compileFunc, config, device=7):
#   # 编译 这里获取一次 func 所以速度应该会提升
#   hsacoPath, kernelName, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ = compileFunc(
#     config.BLOCK_SIZE_M, config.BLOCK_SIZE_N, config.BLOCK_SIZE_K,
#     config.THREAD_SIZE_M, config.THREAD_SIZE_N,
#     config.WARP_SIZE,
#     config.BLOCK_LAYOUT_M, config.BLOCK_LAYOUT_N,
#     config.WARP_LAYOUT_M, config.WARP_LAYOUT_N,
#     config.GLOB_LOAD_WIDTH_A, config.GLOB_LOAD_WIDTH_B,
#     config.WARP_SCATTER_WIDTH_A, config.WARP_SCATTER_WIDTH_B,
#     config.THREAD_SCATTER_WIDTH_A, config.THREAD_SCATTER_WIDTH_B,
#     config.LOCAL_SPLIT_U,
#     config.BLOCK_MAPPING,
#     config.GLOB_STORE_WIDTH,
#     config.UNROLL_NUM,
#     config.REG_PREFETCH, config.SHARED_PREFETCH, config.LOAD_CONTINUOUS,
#     config.REDUCE_C_CONTINUOUS,
#     config.dtype('A'), config.dtype('B'), config.dtype('C'), config.M,config.N,config.K,config.batch,
#     config.isATranspose
#   )[0]
#   inConfig = UserInputs(hsacoPath, kernelName, config)
#   inConfig.m_gridDims = [gridDimX, gridDimY, gridDimZ]
#   inConfig.m_blockDims = [blockDimX, blockDimY, blockDimZ]
#   inConfig.operatorKind = EnumOperator.Matmul
#   packedKernel = CompiledKernelFactory.getKernel(inConfig, device)
#   return packedKernel

# # example code 
# if "__main__" == __name__:
#   path = "/home/xiebaokang/projects/mymlir/DeepGen/TuningConfigs/test.json"
#   cfg_dict = readConfigJson(path)
#   cmc = CreateMatmulConfig(cfg_dict, 4)
#   kams = cmc.createMatMulConfig(thalfTag=False, tsquareTag=False, bhalfTag=False, bsquareTag=False, max_thread_num=256)  # KernelArgMatmul
#   print(len(kams))
#   compileFunc = getCompileFunc()
#   for kam in kams:
#     packedKernel = compile(compileFunc, kam, 7)
#     break

