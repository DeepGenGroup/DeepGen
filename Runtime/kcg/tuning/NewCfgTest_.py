import json
import math
from tqdm import tqdm
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
  def __init__(self, cfg_dict, type_width=4, smem_size=65536, sm_num=60):
    self.cfg_dict = cfg_dict
    self.type_width = type_width  # 一个字 4 byte
    self.max_reg_size = 256 * self.type_width   # byte
    self.max_sm_size = smem_size  # byte
    self.sm_num = sm_num
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
            for wlm in self.cfg_dict[kw.KEY_WARP_LAYOUT_Y]:
              for wln in self.cfg_dict[kw.KEY_WARP_LAYOUT_X]:
                if (wlm * wln == self.cfg_dict[kw.KEY_WARP_SIZE][0]):  # warp_x * warp_y == 64
                  blm = int(ty_num / wlm)
                  bln = int(tx_num / wln)
                  # block_y % warp_y == 0 && block 至少有一个warp
                  if ty_num % wlm == 0 and tx_num % wln == 0 and blm >= 1 and bln >= 1:
                    if blm in self.cfg_dict[kw.KEY_BLOCK_LAYOUT_Y] and bln in self.cfg_dict[kw.KEY_BLOCK_LAYOUT_X]:  # 符合json中的设置
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
      for wswa in self.cfg_dict[kw.KEY_BLOCK_SCATTER_WIDTH_M]:
        for tswa in self.cfg_dict[kw.KEY_WARP_SCATTER_WIDTH_M]:
          # WARP_SCATTER_WIDTH_A >= THREAD_SCATTER_WIDTH_A  &  WARP_SCATTER_WIDTH_A <= thread_size_m
          if wswa >= tswa and wswa <= tal[1][0]:
            for wswb in self.cfg_dict[kw.KEY_BLOCK_SCATTER_WIDTH_N]:
              for tswb in self.cfg_dict[kw.KEY_WARP_SCATTER_WIDTH_N]:
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
            for rcc in self.cfg_dict[kw.KEY_STORE_CONTINUOUS]:
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


  def getBankConflict(self, sm_size, load_width, block_size_k, splitu, 
                      block_layout_x, warp_layout_x, block_repeat, warp_repeat, 
                      warp_size, thread_num, getAddrFunc, mat):
    # sm_size(byte), word：一个bank(4byte)
    def calculate_bank_conflict_count(bank_list):
      # 获取bank_list中重复的bank_id的最大值
      bank_set = set(bank_list)
      max_bc_num = 0
      for id_ in bank_set:
        count_ = bank_list.count(id_)
        if max_bc_num < count_:
          max_bc_num = count_
      return max_bc_num
    
    bank_width = 4
    bank_num = 32  # bank的数量
    word_num = int(sm_size / bank_width)  # 这块smem有多少个word
    # print(word_num)
    addr_bank_dict = {}
    # 计算每一个word的地址对应的bank号
    for word_id in range(word_num):
      word_addr = word_id * bank_width  # byte
      bank_id = word_id % 32
      addr_bank_dict[word_addr] = bank_id
    # print(addr_bank_dict)
    
    all_bank_list = []
    for tz in range(splitu):  # tz
      for bk in range(block_size_k):  # bk
        for i in range(block_repeat):
          for j in range(warp_repeat):
            bank_list = []
            for tid in range(thread_num):
              wrap_id = int(tid / warp_size)
              lane_id = tid % warp_size
              warp_, lane_ = 0, 0
              if mat == "A":
                warp_ = int(wrap_id / block_layout_x)
                lane_ = int(lane_id / warp_layout_x)
              else:
                warp_ = wrap_id % block_layout_x
                lane_ = lane_id % warp_layout_x

              thAddr = getAddrFunc(tz, bk, i, j, warp_, lane_)
              # print(thAddr)
              bank_list.append(addr_bank_dict[thAddr])
            all_bank_list.append(bank_list)
    
    bank_conflict_count = 0
    transcation_num = int(warp_size / bank_num)
    group_size = load_width * transcation_num
    for bank_list in all_bank_list:
      for tid in range(len(bank_list)):
        new_bank_lists = [bank_list[i:i + warp_size] for i in range(0, len(bank_list), warp_size)]
        for new_bank_list in new_bank_lists:
          new_bank_lists_ = [new_bank_list[i:i + group_size] for i in range(0, len(new_bank_list), group_size)]
          for new_bank_list_ in new_bank_lists_:
            bank_conflict_count += calculate_bank_conflict_count(new_bank_list_)
    return bank_conflict_count
  
  
  def ultimate(self, space):
    print(len(space))
    new_space = []
    batch = self.cfg_dict[kw.KEY_BATCH][0]
    m, n, k = self.cfg_dict[kw.KEY_M][0], self.cfg_dict[kw.KEY_N][0], self.cfg_dict[kw.KEY_K][0]
    fix_wave_saturate = lambda x: self.sm_num if x == 0 else x  # 最后一波的sm数量
    best_wave, best_wave_num, best_bc_count = None, None, None
    for case in tqdm(space):
      tm, tn = case[1][0], case[1][1]
      bm, bn, bk = case[0][0], case[0][1], case[0][2] 
      thread_num = int(bm * bn / tm / tn)
      if bm <= 128 or bn <= 128:
        # wave
        waves_num = math.ceil(math.ceil(m/bm) * math.ceil(n/bn) * batch / self.sm_num)
        last_wave_util = fix_wave_saturate(math.ceil(m/bm) * math.ceil(n/bn) % self.sm_num)
        
        # # bank_conflict
        # def getAIdx(tz, bk, i, j, wid, lid):
        #   if case[7][1] == 1:
        #     return (((bk + 1) * case[8][0] + tz) * case[0][0] + 
        #           (i * case[3][0] + wid) * case[4][0] * case[5][0] +  
        #           (j * case[4][0] + lid) * case[6][0]) * self.type_width
        #   else:
        #     return ((bk * case[8][0] + tz) * case[0][0] + 
        #           (i * case[3][0] + wid) * case[4][0] * case[5][0] +  
        #           (j * case[4][0] + lid) * case[6][0]) * self.type_width

        # def getBIdx(tz, bk, i, j, wid, lid):
        #   if case[7][1] == 1:
        #     return (((bk + 1) * case[8][0] + tz) * case[0][1] + 
        #           (i * case[3][1] + wid) * case[4][1] * case[5][1] +  
        #           (j * case[4][1] + lid) * case[6][1]) * self.type_width
        #   else:
        #     return ((bk * case[8][0] + tz) * case[0][1] + 
        #           (i * case[3][1] + wid) * case[4][1] * case[5][1] +  
        #           (j * case[4][1] + lid) * case[6][1]) * self.type_width
        # bk = case[0][2]
        # smA_size = case[0][0] * case[0][2] * self.type_width
        # smB_size = case[0][1] * case[0][2] * self.type_width
        # if case[7][1] == 1:
        #   bk -= 1
        # bc_count_a = self.getBankConflict(smA_size, case[6][0], bk, case[8][0], 
        #                                   case[3][1], case[4][1], int(case[1][0] / case[5][0]), int(case[5][0] / case[6][0]), 
        #                                   self.cfg_dict[kw.KEY_WARP_SIZE][0], thread_num, getAIdx, "A")
        # bc_count_b = self.getBankConflict(smB_size, case[6][1], bk, case[8][0], 
        #                                   case[3][1], case[4][1], int(case[1][0] / case[5][1]), int(case[5][1] / case[6][1]), 
        #                                   self.cfg_dict[kw.KEY_WARP_SIZE][0], thread_num, getBIdx, "B")
 
        # total_bc_count = bc_count_a + bc_count_b
        # iter
        # if best_wave is None or best_wave_num is None or best_bc_count is None:
        if best_wave is None or best_wave_num is None:
          best_wave = last_wave_util
          best_wave_num = waves_num
          # best_bc_count = total_bc_count
        # elif last_wave_util < best_wave and waves_num < best_wave_num and best_bc_count > total_bc_count:
        elif last_wave_util < best_wave and waves_num < best_wave_num:
          best_wave = last_wave_util
          best_wave_num = waves_num
          # best_bc_count = total_bc_count
    
    for case in space:
      tm, tn = case[1][0], case[1][1]
      bm, bn, bk = case[0][0], case[0][1], case[0][2] 
      thread_num = bm * bn / tm / tn
      if bm <= 128 or bn <= 128:
        # float4
        load_width_per_thread_a = bm * bk / thread_num
        load_width_per_thread_b = bn * bk / thread_num
        if load_width_per_thread_a >= 4 and load_width_per_thread_b >= 4:
          waves_num = math.ceil(math.ceil(m/bm) * math.ceil(n/bn) * batch / self.sm_num)
          last_wave_util = fix_wave_saturate(math.ceil(m/bm) * math.ceil(n/bn) % self.sm_num)
          if waves_num == best_wave_num and last_wave_util == best_wave:
            new_space.append(case)
    print(len(new_space))
    return new_space


  def createMatMulConfig(self, thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256) -> TsGeneratorType :
    # main
    ttiles = self.getThreadTile(halfTag=thalfTag, squareTag=tsquareTag)
    btiles = self.getBlockTile(halfTag=bhalfTag, squareTag=bsquareTag)
    tals = self.getSplitUAndLayout(ttiles, btiles, max_thread_num=max_thread_num)
    temp_tals = self.getBlockK(tals)
    temp_tals = self.getScatterWidth(temp_tals)
    temp_tals = self.getPrefetchAndContinuous(temp_tals)
    temp_tals = self.getOther(temp_tals)
    temp_tals = self.ultimate(temp_tals)
    for tal in temp_tals:
      ta = MatmulTuningArgs(
        m = self.cfg_dict[kw.KEY_M][0], 
        n = self.cfg_dict[kw.KEY_N][0],
        k= self.cfg_dict[kw.KEY_K][0],
        batch= self.cfg_dict[kw.KEY_BATCH] ,
        enumDType= self.cfg_dict[kw.KEY_DTYPE_A][0]
      ) 
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
      for e in ta.batch:
        if e == 1:
          ta.batch.remove(e)
      kernelName = ta.generateKernelName()
      configDict = {
        kernelName : ta.jsonfy()
      }
      ret = CompileNeededInfo()
      ret.kernelName = kernelName
      ret.baseArgs = [ta.batch, ta.M, ta.N, ta.K, int(ta.dtA)]
      ret.tsArgs = [[ta.batch, ta.M, ta.N, ta.K] , configDict  ]
      ret.torchDataType = ToTorchType(ta.dtA)
      gridDim = ta.M / ta.BLOCK_SIZE_M * ta.N / ta.BLOCK_SIZE_N
      blockDim = (ta.BLOCK_SIZE_M / ta.THREAD_SIZE_M) * ( ta.BLOCK_SIZE_N / ta.THREAD_SIZE_N )
      shmBytes = (ta.BLOCK_SIZE_M + ta.BLOCK_SIZE_N) * ta.BLOCK_SIZE_K
      if ta.SHARED_PREFETCH > 0 :
        shmBytes *= 2
      if ta.LOCAL_SPLIT_U > 1 :
        blockDim *= ta.LOCAL_SPLIT_U
        shm_reduce = ta.BLOCK_SIZE_M * ta.BLOCK_SIZE_N * ta.LOCAL_SPLIT_U
        if shm_reduce > shmBytes :
          shmBytes = shm_reduce
      ret.blockDims = [int(blockDim),1,1]
      
      ret.gridDims = [int(gridDim)]
      if int(gridDim) >= 65535 :
        continue
      if int(blockDim) >= 65535 :
        continue
      if len(ta.batch) > 0 :
        ret.gridDims += ta.batch  # 处理方式： 将batch维度加到griddim的y,z上. 即batch数组的维度不超过2
      assert len(ret.gridDims) <= 3
      while len(ret.gridDims) < 3:
        ret.gridDims.append(1)   # 不够三维的部分用1 补全
      ret.shmBytes = int(shmBytes * sizeof(ta.dtA))
      
      yield ret
    

def getTuneSpace(geemConfigPath : str) -> TsGeneratorType :
  cfg_dict = readConfigJson(geemConfigPath)
  cmc = CreateMatmulConfig(cfg_dict, 4)
  kams = cmc.createMatMulConfig(thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256)  # KernelArgMatmul
  return kams

def getTuneSpaceWithBaseargs(geemConfigPath : str, baseargs : List) -> TsGeneratorType :
  cfg_dict = readConfigJson(geemConfigPath)
  datatype = baseargs[-1]
  batch,m,n,k = baseargs[0:-1]
  cfg_dict[kw.KEY_BATCH] = batch
  cfg_dict[kw.KEY_M] = [m]
  cfg_dict[kw.KEY_N] = [n]
  cfg_dict[kw.KEY_K] = [k]
  cfg_dict[kw.KEY_DTYPE_A] = [ToEnumIntDType(datatype)]
  cfg_dict[kw.KEY_DTYPE_B] = [ToEnumIntDType(datatype)]
  cfg_dict[kw.KEY_DTYPE_C] = [ToEnumIntDType(datatype)]
  if is_hip() :
    cfg_dict[kw.KEY_WARP_SIZE] = [64]
  else:
    cfg_dict[kw.KEY_WARP_SIZE] = [32]
  cmc = CreateMatmulConfig(cfg_dict, 4)
  kams = cmc.createMatMulConfig(thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256)  # KernelArgMatmul
  return kams

# # example code 
if "__main__" == __name__:
  path = "/home/xiebaokang/projects/evaluate/DeepGen/TuningConfigs/GEMM_cfg_32.json"
  cfg_dict = readConfigJson(path)
  cmc = CreateMatmulConfig(cfg_dict, 4, 65536, 60)
  ttiles = cmc.getThreadTile(halfTag=True, squareTag=True)
  btiles = cmc.getBlockTile(halfTag=True, squareTag=True)
  tals = cmc.getSplitUAndLayout(ttiles, btiles, max_thread_num=256)
  temp_tals = cmc.getBlockK(tals)
  temp_tals = cmc.getScatterWidth(temp_tals)
  temp_tals = cmc.getPrefetchAndContinuous(temp_tals)
  temp_tals = cmc.getOther(temp_tals)
  for cfg in cmc.ultimate(temp_tals):
    print(cfg)

