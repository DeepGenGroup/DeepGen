import json
import importlib.util
from typing import List

from kcg.Operators.attention import *

def readConfigJson(path):
  # read json file
  with open(path, 'r') as f:
    json_str = f.read().strip()
  cfg_dict = json.loads(json_str)
  return cfg_dict

class CreateAttnConfig:
  def __init__(self, cfg_dict ,shape, max_thread=256, type_width=4, smem_size=65536, sm_num=108):
    self.cfg = cfg_dict
    self.batch, self.head_num, self.seq_len, self.head_dim = shape
    # print(self.seq_len, shape)
    self.max_thread = max_thread
    self.type_width = type_width  # 一个字 4 byte
    self.max_reg_size = 256 * self.type_width   # byte
    self.max_sm_size = smem_size  # byte
    self.sm_num = sm_num
  
  def threadTile(self):
    result = []
    for ptr in self.cfg["PTr"]:
      for ptc in self.cfg["PTc"]:
        for otr in self.cfg["OTr"]:
          for otc in self.cfg["OTc"]:
            result.append((ptr, ptc, otr, otc))
    return result

  def blockTile(self, old_cfgs):
    result = []
    for old_cfg in old_cfgs:
      for br in self.cfg["Br"]:
        for bc in self.cfg["Bc"]:
          if br <= self.seq_len and bc <= self.seq_len:
            thread_p = int(br / old_cfg[0]) * int(bc / old_cfg[1])
            thread_o = int(br / old_cfg[2]) * int(self.cfg["Hd"][0] / old_cfg[3])
            if thread_p == thread_o and self.max_thread >= thread_p:  # 线程相等
              thread_num = thread_p
              for s1 in self.cfg["Slice1"]:
                for s2 in self.cfg["Slice2"]:
                  for glwq in self.cfg["GLOB_LOAD_WIDTH_Q"]:
                    for glwk in self.cfg["GLOB_LOAD_WIDTH_K"]:
                      for glwv in self.cfg["GLOB_LOAD_WIDTH_V"]:
                        ldtw_q = br * s1 / thread_num
                        ldtw_k = bc * s1 / thread_num
                        ldtw_v = self.cfg["Hd"][0] * s2 / thread_num
                        if ldtw_q >= glwq and ldtw_k >= glwk and ldtw_v >= glwv:  # 保证能够正常glob load                    
                          result.append((thread_num, (br, bc, self.cfg["Hd"][0], s1, s2), old_cfg, (glwq, glwk, glwv)))
    return result # [thread_num, (br, bc, hd, slice1, slice2), (ptr, ptc, otr, otc), (glob_load_w_q, glwk, glwv)]
  
  def layoutAndScatterP(self, old_cfgs):
    result = []
    for old_cfg in old_cfgs:
      for bly in self.cfg["BLOCK_LAYOUT_P_Y"]:
        for blx in self.cfg["BLOCK_LAYOUT_P_X"]:
          for wly in self.cfg["WARP_LAYOUT_P_Y"]:
            for wlx in self.cfg["WARP_LAYOUT_P_X"]:
              by = old_cfg[1][0] / old_cfg[2][0]  # by = br / ptr
              bx = old_cfg[1][1] / old_cfg[2][1]  # bx = bc / ptc
              if wly * wlx == self.cfg["WARP_SIZE"][0] and blx == 1 and bly * wly == by and wlx == bx:
                for bswq in self.cfg["BLOCK_SCATTER_WIDTH_Q"]:
                  for bswk in self.cfg["BLOCK_SCATTER_WIDTH_K"]:
                    for wswq in self.cfg["WARP_SCATTER_WIDTH_Q"]:
                      for wswk in self.cfg["WARP_SCATTER_WIDTH_K"]:
                        if bswq <= old_cfg[2][0] and bswk <= old_cfg[2][1] and wswq <= bswq and wswk <= bswk:
                          result.append(old_cfg + ((bly, blx, wly, wlx, bswq, bswk, wswq, wswk), ))
    # (th_num, (br, bc, hd, s1, s2), (ptr, ptc, otr, otc), (glwq, glwk, glwv), (bly, blx, wly, wlx, bswq, bswk, wswq, wswk))
    return result

  def layoutAndScatterO(self, old_cfgs):
    result = []
    for old_cfg in old_cfgs:
      for bly in self.cfg["BLOCK_LAYOUT_O_Y"]:
        for blx in self.cfg["BLOCK_LAYOUT_O_X"]:
          for wly in self.cfg["WARP_LAYOUT_O_Y"]:
            for wlx in self.cfg["WARP_LAYOUT_O_X"]:
              by = old_cfg[1][0] / old_cfg[2][2]  # by = br / otr
              bx = old_cfg[1][2] / old_cfg[2][3]  # bx = hd / otc
              if wly * wlx == self.cfg["WARP_SIZE"][0] and bly * wly == by and blx * wlx == bx:
                for bswp in self.cfg["BLOCK_SCATTER_WIDTH_P"]:
                  for bswv in self.cfg["BLOCK_SCATTER_WIDTH_V"]:
                    for wswp in self.cfg["WARP_SCATTER_WIDTH_P"]:
                      for wswv in self.cfg["WARP_SCATTER_WIDTH_V"]:
                        if bswp <= old_cfg[2][2] and bswv <= old_cfg[2][3] and wswp <= bswp and wswv <= bswv:
                          result.append(old_cfg + ((bly, blx, wly, wlx, bswp, bswv, wswp, wswv), ))
    # (th_num, (br, bc, hd, s1, s2), (ptr, ptc, otr, otc), (glwq, glwk, glwv), (bly, blx, wly, wlx, bswq, bswk, wswq, wswk), 
    # (bly, blx, wly, wlx, bswp, bswv, wswp, wswv))
    return result
  
  def storeSizeAndOther(self, old_cfgs):
    result = []
    for old_cfg in old_cfgs:
      for spp in self.cfg["SHARED_PREFETCH_P"]:
        smem_size = old_cfg[1][0] * old_cfg[1][3] + old_cfg[1][1] * old_cfg[1][3] + \
                    old_cfg[1][0] * old_cfg[1][1] + old_cfg[1][2] * old_cfg[1][4] + 3 * old_cfg[1][0]
        if spp == 1:
          smem_size += old_cfg[1][0] * old_cfg[1][3] + old_cfg[1][1] * old_cfg[1][3]
        if smem_size * self.type_width <= self.max_sm_size:  # shared memory size
          for rpp in self.cfg["REG_PREFETCH_P"]:
            for rpo in self.cfg["REG_PREFETCH_O"]:
              for unroll in self.cfg["UNROLL_NUM"]:
                result.append(old_cfg + ((unroll, self.cfg["WARP_SIZE"][0], 1, 1, spp, rpp, rpo), smem_size * self.type_width))  # 只能连续访存
    # (th_num, (br, bc, hd, s1, s2), (ptr, ptc, otr, otc), (glwq, glwk, glwv), (bly, blx, wly, wlx, bswq, bswk, wswq, wswk), 
    # (bly, blx, wly, wlx, bswp, bswv, wswp, wswv), (unroll ,warp_size, load_continuous_p, lc_o, sm_prefetch_p, reg_pf_p, reg, pf_o))
    return result

  # def storeSizeAndOther_(self, old_cfgs):
  #   result = []
  #   for old_cfg in old_cfgs:
  #     smem_size = old_cfg[1][0] * old_cfg[1][3] + old_cfg[1][1] * old_cfg[1][3] + \
  #                 old_cfg[1][0] * old_cfg[1][1] + old_cfg[1][2] * old_cfg[1][4] + 3 * old_cfg[1][0]
  #     if smem_size * self.type_width <= self.max_sm_size:  # shared memory size
  #       result.append(old_cfg + ((16, self.cfg["WARP_SIZE"][0], 1, 1, 0, 0, 0), ))  # 只能连续访存
  #   # (th_num, (br, bc, hd, s1, s2), (ptr, ptc, otr, otc), (glwq, glwk, glwv), (bly, blx, wly, wlx, bswq, bswk, wswq, wswk), 
  #   # (bly, blx, wly, wlx, bswp, bswv, wswp, wswv), (unroll ,warp_size, load_continuous_p, lc_o, sm_prefetch_p, reg_pf_p, reg_pf_o))
  #   return result
  
  def cut(self, old_cfgs):
    result = []
    # max_util = 0
    # for old in old_cfgs:
    #   # print(old)
    #   sm_util = (self.seq_len / old[1][0])
    #   if (sm_util >= max_util):
    #     max_util = sm_util
    
    for old in old_cfgs:
      # print(old)
      # sm_util = (self.seq_len / old[1][0])  # block_num
      # sm占用率 / 离散化约束
      # if (sm_util == max_util):
      rep_p_y = (old[2][0] // old[4][4]) * (old[4][4] // old[4][6])  # (ptr / bswq) * (bswq / wswq)
      rep_p_x = (old[2][1] // old[4][5]) * (old[4][5] // old[4][7])  # (ptc / bswk) * (bswk / wswk)
      rep_o_y = (old[2][2] // old[5][4]) * (old[5][4] // old[5][6])  # (otr / bswp) * (bswp / wswp)
      rep_o_x = (old[2][3] // old[5][5]) * (old[5][5] // old[5][7])  # (otc / bswv) * (bswv / wswv)
      best_rep_p_y = (old[4][2] * old[2][0] * (4 // self.type_width)) // 32  # wly * ptr * (fp32w / typew) / 32
      best_rep_p_x = (old[4][3] * old[2][1] * (4 // self.type_width)) // 32  # wly * ptr * (fp32w / typew) / 32
      best_rep_o_y = (old[5][2] * old[2][2] * (4 // self.type_width)) // 32  # wly * otr * (fp32w / typew) / 32
      best_rep_o_x = (old[5][3] * old[2][3] * (4 // self.type_width)) // 32  # wly * otc * (fp32w / typew) / 32
      best_rep_p_y = 1 if best_rep_p_y == 0 else best_rep_p_y
      best_rep_p_x = 1 if best_rep_p_x == 0 else best_rep_p_x
      best_rep_o_y = 1 if best_rep_o_y == 0 else best_rep_o_y
      best_rep_o_x = 1 if best_rep_o_x == 0 else best_rep_o_x
      # print(rep_p_y, best_rep_p_y, rep_p_x, best_rep_p_x, rep_o_y, best_rep_o_y, rep_o_x, best_rep_o_x)
      if rep_p_y == best_rep_p_y and rep_p_x == best_rep_p_x and rep_o_y == best_rep_o_y and rep_o_x == best_rep_o_x:
        result.append(old)
    return result

    
  def main(self):
    result = self.threadTile()
    result = self.blockTile(result)
    result = self.layoutAndScatterP(result)
    result = self.layoutAndScatterO(result)
    result = self.storeSizeAndOther(result)
    result = self.cut(result)
    if len(result):
      return result
    return None

 
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
              if ty == oldResult[0][0] and tx == oldResult[0][1] and pwly * pwlx == 32:
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
              if ty == oldResult[0][2] and tx == oldResult[0][3] and owly * owlx == 32:
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


# spec = importlib.util.spec_from_file_location("attention", "/home/xushilong/DeepGen/bin/libdeepgen.so")
# mod = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(mod)
# compile_kernel_FA = mod.compile_attn

def get_cfgs(cfgfilepath : str, shape = [1, 32, 4096, 128]) -> List:
  # cfgfilepath = str(PathManager.project_dir()) + "/TuningConfigs/attn_llama2.json"
  cfg_dict = readConfigJson(cfgfilepath)
  print(f'get_cfgs shape = {shape}',flush=True)
  cfg_dict['Hd'] = [shape[3]]
  print(f"attn Hd = {cfg_dict['Hd']}",flush=True)
  cc = CreateAttnConfig(cfg_dict, shape)
  cfgs = cc.main()
  return cfgs

def getTuneSpace(shape : List[int] , cfgfile : str, cfgs : List) -> TsGeneratorType : 
  # shape = [1, 32, 2048, 128]
  # batch(几个句子), seqLen（句子长度）, (hiddenDim(一个单词编码以后的向量长度) -> headnum * headDim),   
  kw = ConfigKeywords
  if len(cfgs) <= 0:
    cfgs = get_cfgs(cfgfile,shape)
  for cfg in cfgs:
    (th_num, (br, bc, hd, s1, s2), (ptr, ptc, otr, otc), (glwq, glwk, glwv), (bly_p, blx_p, wly_p, wlx_p, bswq_p, bswk_p, wswq_p, wswk_p), 
    (bly_o, blx_o, wly_o, wlx_o, bswp_o, bswv_o, wswp_o, wswv_o), (unroll ,warp_size, load_continuous_p, load_continuous_o, sm_prefetch_p, reg_prefetch_p, reg_prefetch_o), smem_size) = cfg
    valDict = {
        "Br": br, "Bc": bc, "Hd": hd, "Slice1": s1, "Slice2": s2, 
        "PTr": ptr, "PTc": ptc, "OTr": otr, "OTc": otc,
        # global to shared
        "GLOB_LOAD_WIDTH_Q": glwq, "GLOB_LOAD_WIDTH_K": glwk, "GLOB_LOAD_WIDTH_V": glwv,
        # P = Q * K
        "BLOCK_LAYOUT_P_Y": bly_p, "BLOCK_LAYOUT_P_X": blx_p, "WARP_LAYOUT_P_Y": wly_p, "WARP_LAYOUT_P_X": wlx_p,
        "BLOCK_SCATTER_WIDTH_Q": bswq_p, "BLOCK_SCATTER_WIDTH_K": bswk_p, "WARP_SCATTER_WIDTH_Q": wswq_p, "WARP_SCATTER_WIDTH_K": wswk_p,
        # O = P * V
        "BLOCK_LAYOUT_O_Y": bly_o, "BLOCK_LAYOUT_O_X": blx_o, "WARP_LAYOUT_O_Y": wly_o, "WARP_LAYOUT_O_X": wlx_o, 
        "BLOCK_SCATTER_WIDTH_P": bswp_o, "BLOCK_SCATTER_WIDTH_V": bswv_o, "WARP_SCATTER_WIDTH_P": wswp_o, "WARP_SCATTER_WIDTH_V": wswv_o,

        "UNROLL_NUM": unroll, "WARP_SIZE": warp_size, 
        "LOAD_CONTINUOUS_P": load_continuous_p, "LOAD_CONTINUOUS_O": load_continuous_o, 
        # prefecth
        "SHARED_PREFETCH_P": sm_prefetch_p, "REG_PREFETCH_P": reg_prefetch_p, "REG_PREFETCH_O": reg_prefetch_o,
        # baseArgs
        kw.KEY_BLOCK_DIM_X : th_num, kw.KEY_BLOCK_DIM_Y : 1, kw.KEY_BLOCK_DIM_Z : 1,
        kw.KEY_SHM_BYTES :  smem_size ,
        kw.KEY_GRID_DIM_X : int(shape[2]/br),
        kw.KEY_GRID_DIM_Y : shape[1],
        kw.KEY_GRID_DIM_Z : shape[0]
      }
    temp = AttentionTuningArgs(ToEnumIntDType(torch.float32))
    temp.assignWithDict(valDict)
    temp.basearg.argDict['shape'] = shape
    temp.basearg.argDict['dtype'] = EnumKernelDType.float32
    kernelName = temp.generateKernelName()
    config = {kernelName : valDict}
    
    ret = CompileNeededInfo()
    ret.kernelName = kernelName
    ret.baseArgs = shape
    ret.torchDataType = torch.float32
    ret.tsArgs = [shape,config]
    ret.blockDims = [ valDict[kw.KEY_BLOCK_DIM_X], valDict[kw.KEY_BLOCK_DIM_Y], valDict[kw.KEY_BLOCK_DIM_Z] ]  # tx
    ret.gridDims = [ valDict[ kw.KEY_GRID_DIM_X], valDict[ kw.KEY_GRID_DIM_Y], valDict[ kw.KEY_GRID_DIM_Z] ]
    ret.shmBytes = smem_size
    yield ret
    
    # compile_kernel_FA(shape,config)

if __name__ == '__main__' :
  ...
#   compile([1, 32, 2048, 128],[])
  # thread_num, cfgs = get_cfgs()
  # print(len(cfgs))
  # for cfg in cfgs:
  #   if cfg[0] == 256:
  #     print(cfg)
  
  