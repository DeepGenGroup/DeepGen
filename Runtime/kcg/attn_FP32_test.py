import json
import importlib.util
from typing import List

from kcg.Kernel import *

class CreateConfig:
  def __init__(self, json_path, shape):
    f = open(json_path, "r")
    cfg_str = f.read().strip()
    self.cfg = json.loads(cfg_str)
    self.batch = shape[0]
    self.head_num = shape[1]
    self.seq_len = shape[2]
    self.head_dim = shape[3]
    
  def a(self):
    ptiles = []
    for br in self.cfg["Br"]:
      for bc in self.cfg["Bc"]:
        if br <= self.seq_len and bc <= self.seq_len:
          for ptr in self.cfg["PTr"]:
            for ptc in self.cfg["PTc"]:
              if (br*bc/ptr/ptc >= 32):
                ptiles.append([br, bc, ptr, ptc])
    return ptiles

  def b(self):
    otiles = []
    for br in self.cfg["Br"]:
      if br <= self.seq_len:
        for otr in self.cfg["OTr"]:
          for otc in self.cfg["OTc"]:
            if (br*self.cfg["Hd"][0]/otr/otc >= 32):
              otiles.append([br, self.cfg["Hd"][0], otr, otc])
    return otiles

  def c(self, ptiles, otiles):
    potiles = []
    for ptile in ptiles:
      for otile in otiles:
        thnum1 = ptile[0]*ptile[1]/ptile[2]/ptile[3]
        thnum2 = otile[0]*otile[1]/otile[2]/otile[3]
        if (thnum1 == thnum2 and ptile[0] == otile[0]):
          potiles.append([ptile[0], ptile[1], otile[1], ptile[2], ptile[3], otile[2], otile[3]])
          # br, bc, hd, ptr, ptc, otr, otc
    return potiles
  
  def d(self, potiles):
    results = []
    for potile in potiles:
      for s1 in self.cfg["Slice1"]:
        for s2 in self.cfg["Slice2"]:
          smQSize = potile[0] * s1
          smKSize = potile[1] * s1
          smVSize = potile[2] * s2
          pby = potile[0] / potile[3]  # br / ptr
          pbx = potile[1] / potile[4]  # bc / ptc
          for qldw in self.cfg["GLOB_LOAD_WIDTH_Q"]:
            for kldw in self.cfg["GLOB_LOAD_WIDTH_K"]:
              for vldw in self.cfg["GLOB_LOAD_WIDTH_V"]:
                thread_num = pby*pbx
                if smQSize / thread_num >= qldw and smKSize / thread_num >= kldw and smVSize / thread_num >= vldw:
                  oby = potile[0] / potile[5]  # br / otr
                  obx = potile[2] / potile[6]  # hd / otc
                  block_layout = (pby, pbx, oby, obx)
                  item = potile.copy()
                  item.insert(0, block_layout)
                  item.insert(4, s1)
                  item.insert(5, s2)
                  item.extend([qldw, kldw, vldw])
                  results.append(item)
    # (pthy, pthx, othy, othx), br, bc, hd, slice1, slice2, ptr, ptc, otr, otc, glob_load_q, glob_load_k, glob_load_v
    return results

  def e(self, oldResults):
    results = []
    for oldResult in oldResults:
      for pbly in self.cfg["BLOCK_LAYOUT_P_Y"]:
        for pblx in self.cfg["BLOCK_LAYOUT_P_X"]:
          for pwly in self.cfg["WARP_LAYOUT_P_Y"]:
            for pwlx in self.cfg["WARP_LAYOUT_P_X"]:
              ty = pbly * pwly
              tx = pblx * pwlx
              if ty == oldResult[0][0] and tx == oldResult[0][1]:
                for qbsw in self.cfg["BLOCK_SCATTER_WIDTH_Q"]:
                  for kbsw in self.cfg["BLOCK_SCATTER_WIDTH_K"]:
                    for qwsw in self.cfg["WARP_SCATTER_WIDTH_Q"]:
                      for kwsw in self.cfg["WARP_SCATTER_WIDTH_K"]:
                        if qbsw <= oldResult[6] and qwsw <= qbsw and kbsw <= oldResult[7] and kwsw <= kbsw:
                          item = oldResult.copy()
                          item.extend([pbly, pblx, pwly, pwlx, qbsw, kbsw, qwsw, kwsw])
                          results.append(item)
    # (pthy, pthx, othy, othx), br, bc, hd, slice1, slice2, ptr, ptc, otr, otc, glob_load_q, glob_load_k, glob_load_v, blokc_layout_py, px, wpy, wpx, block_scatter_wq, wk, wwq, wwk
    return results

  def f(self, oldResults):
    results = []
    for oldResult in oldResults:
      for obly in self.cfg["BLOCK_LAYOUT_O_Y"]:
        for oblx in self.cfg["BLOCK_LAYOUT_O_X"]:
          for owly in self.cfg["WARP_LAYOUT_O_Y"]:
            for owlx in self.cfg["WARP_LAYOUT_O_X"]:
              ty = obly * owly
              tx = oblx * owlx
              if ty == oldResult[0][2] and tx == oldResult[0][3]:
                for pbsw in self.cfg["BLOCK_SCATTER_WIDTH_P"]:
                  for vbsw in self.cfg["BLOCK_SCATTER_WIDTH_V"]:
                    for pwsw in self.cfg["WARP_SCATTER_WIDTH_P"]:
                      for vwsw in self.cfg["WARP_SCATTER_WIDTH_V"]:
                        if pbsw <= oldResult[8] and pwsw <= pbsw and vbsw <= oldResult[9] and vwsw <= vbsw:
                          item = oldResult.copy()
                          item.extend([obly, oblx, owly, owlx, pbsw, vbsw, pwsw, vwsw])
                          results.append(item)
    # (pthy, pthx, othy, othx), br, bc, hd, slice1, slice2, ptr, ptc, otr, otc, glob_load_q, glob_load_k, glob_load_v, {blokc_layout_py, px, wpy, wpx, block_scatter_wq, wk, wwq, wwk}, {blokc_layout_oy, ox, woy, wox, block_scatter_wp, wv, wwp, wwv}
    return results
  
  def g(self, oldResults):
    results = []
    for oldResult in oldResults:
      for unroll in self.cfg["UNROLL_NUM"]:
        for ldcp in self.cfg["LOAD_CONTINUOUS_P"]:
          for ldco in self.cfg["LOAD_CONTINUOUS_O"]:
            for sm_prefatch in self.cfg["SHARED_PREFETCH_P"]:
              for reg_prefatch_p in self.cfg["REG_PREFETCH_P"]:
                for reg_prefatch_o in self.cfg["REG_PREFETCH_O"]:
                  item = oldResult.copy()
                  item.extend([unroll, self.cfg["WARP_SIZE"][0], ldcp, ldco, sm_prefatch, reg_prefatch_p, reg_prefatch_o])
                  results.append(item)
    # [..., unroll, warp_size, load_continuous_p, load_continuous_o, shared_prefetch_p, reg_prefetch_p, reg_prefetch_o]
    return results

  def h(self, oldResults):
    results = []
    for oldResult in oldResults:
    #   if oldResult[31] == 1 and oldResult[0][0] * oldResult[0][1] >= oldResult[1] and oldResult[0][0] * oldResult[0][1] >= oldResult[2]:
    #     if oldResult[32] == 1 and oldResult[0][2] * oldResult[0][3] >= oldResult[3]:
    #       results.append(oldResult)
      thread_num = int(oldResult[0][0] * oldResult[0][1])
      if thread_num <= 256:
        LDS_SZIE = oldResult[1] * oldResult[4] + oldResult[2] * oldResult[4] + oldResult[1] * oldResult[2] + oldResult[3] * oldResult[5] + 3 * oldResult[1]
        if oldResult[33] == 1:
          LDS_SZIE += oldResult[1] * oldResult[4] + oldResult[2] * oldResult[4]
        if LDS_SZIE <= 96 * 1024 / 4:
          item = oldResult[1:].copy()
          item.append((thread_num, LDS_SZIE * 4))
          results.append(item)
    return results

  def main(self):
    ptiles = self.a()
    otiles = self.b()
    potiles = self.c(ptiles, otiles)
    results = self.d(potiles)
    results = self.e(results)
    results = self.f(results)
    results = self.g(results)
    results = self.h(results)
    return results


spec = importlib.util.spec_from_file_location("attention", "/home/xushilong/DeepGen/bin/libdeepgen.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
compile_kernel_FA = mod.compile_attn

def get_cfgs(shape = [1, 32, 2048, 128]) -> List:
  path = "/home/xushilong/DeepGen/_TempCodes/config.json"
  cc = CreateConfig(path, shape)
  cfgs = cc.main()
  return cfgs

def compile(shape : List[int] , cfgs : List):
  # shape = [1, 32, 2048, 128]
  if len(cfgs) <= 0:
    cfgs = get_cfgs(shape)
  tse = TuningSpaceEncoder(cfgs[0]['attention1'])
  
  for cfg in cfgs:
    kam = 
    config = {
      "attention1": {
        "Br": cfg[0], "Bc": cfg[1], "Hd": cfg[2], "Slice1": cfg[3], "Slice2": cfg[4], 
        "PTr": cfg[5], "PTc": cfg[6], "OTr": cfg[7], "OTc": cfg[8],
        # global to shared
        "GLOB_LOAD_WIDTH_Q": cfg[9], "GLOB_LOAD_WIDTH_K": cfg[10], "GLOB_LOAD_WIDTH_V": cfg[11],
        # P = Q * K
        "BLOCK_LAYOUT_P_Y": cfg[12], "BLOCK_LAYOUT_P_X": cfg[13], "WARP_LAYOUT_P_Y": cfg[14], "WARP_LAYOUT_P_X": cfg[15],
        "BLOCK_SCATTER_WIDTH_Q": cfg[16], "BLOCK_SCATTER_WIDTH_K": cfg[17], "WARP_SCATTER_WIDTH_Q": cfg[18], "WARP_SCATTER_WIDTH_K": cfg[19],
        # O = P * V
        "BLOCK_LAYOUT_O_Y": cfg[20], "BLOCK_LAYOUT_O_X": cfg[21], "WARP_LAYOUT_O_Y": cfg[22], "WARP_LAYOUT_O_X": cfg[23], 
        "BLOCK_SCATTER_WIDTH_P": cfg[24], "BLOCK_SCATTER_WIDTH_V": cfg[25], "WARP_SCATTER_WIDTH_P": cfg[26], "WARP_SCATTER_WIDTH_V": cfg[27],

        "UNROLL_NUM": cfg[28], "WARP_SIZE": cfg[29], 
        "LOAD_CONTINUOUS_P": cfg[30], "LOAD_CONTINUOUS_O": cfg[31], 
        # prefecth
        "SHARED_PREFETCH_P": cfg[32], "REG_PREFETCH_P": cfg[33], "REG_PREFETCH_O": cfg[34],
      }
    }
    
  # def createMatMulConfig(self, thalfTag=True, tsquareTag=True, bhalfTag=True, bsquareTag=True, max_thread_num=256) -> List[int]:
  #   # main
  #   ttiles = self.getThreadTile(halfTag=thalfTag, squareTag=tsquareTag)
  #   btiles = self.getBlockTile(halfTag=bhalfTag, squareTag=bsquareTag)
  #   tals = self.getSplitUAndLayout(ttiles, btiles, max_thread_num=max_thread_num)
  #   temp_tals = self.getBlockK(tals)
  #   temp_tals = self.getScatterWidth(temp_tals)
  #   temp_tals = self.getPrefetchAndContinuous(temp_tals)
  #   temp_tals = self.getOther(temp_tals)
  #   kams = []
  #   for tal in temp_tals:
  #     kam = MatmulTuningArgs(self.cfg_dict[kw.KEY_M][0], self.cfg_dict[kw.KEY_N][0], self.cfg_dict[kw.KEY_K][0], self.cfg_dict[kw.KEY_BATCH][0] ,            # M, N, K, batch
  #                           self.cfg_dict[kw.KEY_DTYPE_A][0]) 
  #                           # self.cfg_dict[kw.KEY_DTYPE_B][0], 
  #                           # self.cfg_dict[kw.KEY_DTYPE_C][0]) # typeA, typeB, typeC
  #     config = (
  #       tal[0][0], tal[0][1], tal[0][2],  # block_size
  #       tal[1][0], tal[1][1],             # thread_size
  #       tal[2][0], tal[2][1],             # load_width
  #       tal[3][0], tal[3][1],             # block_layout
  #       tal[4][0], tal[4][1],             # warp_layout
  #       tal[5][0], tal[5][1],             # warp_scatter
  #       tal[6][0], tal[6][1],             # thread_scatter
  #       tal[7][0], tal[7][1], tal[7][2],  # sm_prefatch, reg_prefatch, load_continuous
  #       tal[8][0], tal[8][1], tal[8][2],  # splitU, store_width, store_continuos
  #       tal[9][0], tal[9][1],             # block_mapping, unroll
  #       self.cfg_dict[kw.KEY_WARP_SIZE][0], self.cfg_dict[kw.KEY_IS_A_TRANSPOSE][0],   # warp_size, is_Atran
  #     )
  #     kam.setArgs(*config)
  #     kamEncodedStr = self.encoder.encode(kam.jsonfy())
  #     kams.append(int(kamEncodedStr))
  #   return kams
    
    tse.encode(config['attention1'])
    
    gridSize = [int(shape[2]/cfg[1]), shape[1], shape[0]]  # bx, by, bz
    blockSize = [cfg[-1][0]]  # tx
    sharedSize = cfg[-1][1]  # shared memroy size

    print(config)
    print("gridSize: ", gridSize)
    print("blockSize: ", blockSize)
    print(f"sharedSize: {sharedSize} byte")
    
    compile_kernel_FA(shape,config)

