import json

def readConfigJson(path):
  # read json file
  with open(path, 'r') as f:
    json_str = f.read().strip()
  cfg_dict = json.loads(json_str)
  return cfg_dict

class CreateAttnConfig:
  def __init__(self, cfg_dict ,shape, max_thread=256, type_width=4, smem_size=65536, sm_num=60):
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
      
def buildSapce(kernel: str, inputs, gpu_info):
  configs = []
  if kernel == "matmul":
    tmp_dict = {
      "shape": [128, 1024, 1024, 128], 
      "type": ["fp32", "fp32", "fp32"], 
      "grid": [64, 128, 1], 
      "block": [256, 1, 1],
      "smem": 16384,
      "config": {
        "matmul": {
          "BLOCK_SIZE_M": 128, "THREAD_SIZE_M": 8, "BLOCK_SIZE_N": 128, "THREAD_SIZE_N": 8,
          "LOCAL_SPLIT_U": 1, "BLOCK_SIZE_K": 16, 
          "GLOB_LOAD_WIDTH_A": 4, "GLOB_LOAD_WIDTH_B": 4, "GLOB_STORE_WIDTH": 4,
          "BLOCK_LAYOUT_Y": 4, "BLOCK_LAYOUT_X": 2, "WARP_LAYOUT_Y": 4, "WARP_LAYOUT_X": 8,
          "BLOCK_SCATTER_WIDTH_M": 8, "WARP_SCATTER_WIDTH_M": 8, "BLOCK_SCATTER_WIDTH_N": 4, "WARP_SCATTER_WIDTH_N": 4,
          "WARP_SIZE": 32, "LOAD_CONTINUOUS": 1, "STORE_CONTINUOUS": 1, "SHARED_PREFETCH": 0, "REG_PREFETCH": 0, 
          "BLOCK_MAPPING": 4, "UNROLL_NUM": 16
        }
      }
    }
    configs.append(tmp_dict)
  elif kernel == "attention":
    # input_shape: [bs, hn, sl, hd]
    batch_size, head_num, head_dim, seq_len = inputs[0].shape  # Q shape: [batch_size, head_num, head_dim, seq_len]
    shape = [batch_size, head_num, seq_len, head_dim]
    path = f"/home/xiebaokang/projects/DeepGen/python/deepgen/cfg_jsons/attn.json"
    cfg_dict = readConfigJson(path)
    cfg_dict["Hd"] = [head_dim]
    cfg_dict["WARP_SIZE"] = [gpu_info.warp_size]
    # print(shape, gpu_info.shared_mem_per_block, gpu_info.compute_units)
    cc = CreateAttnConfig(cfg_dict, shape, smem_size=gpu_info.shared_mem_per_block, sm_num=gpu_info.compute_units)
    cfgs = cc.main()
    for cfg in cfgs:
      smem_size = cfg[7]
      configs.append({
        "shape": shape, 
        "type": ["fp32", "fp32", "fp32", "fp32"], 
        "grid": [int(seq_len / cfg[1][0]), head_num, batch_size], 
        "block": [cfg[0], 1, 1],
        "smem": smem_size,
        "config": {
          "attention": {
            "Br": cfg[1][0], "Bc": cfg[1][1], "Hd": cfg[1][2], "Slice1": cfg[1][3], "Slice2": cfg[1][4], 
            "PTr": cfg[2][0], "PTc": cfg[2][1], "OTr": cfg[2][2], "OTc": cfg[2][3], 
            "GLOB_LOAD_WIDTH_Q": cfg[3][0], "GLOB_LOAD_WIDTH_K": cfg[3][1], "GLOB_LOAD_WIDTH_V": cfg[3][2], 

            "BLOCK_LAYOUT_P_Y": cfg[4][0], "BLOCK_LAYOUT_P_X": cfg[4][1], "WARP_LAYOUT_P_Y": cfg[4][2], "WARP_LAYOUT_P_X": cfg[4][3],
            "BLOCK_SCATTER_WIDTH_Q": cfg[4][4], "BLOCK_SCATTER_WIDTH_K": cfg[4][5], "WARP_SCATTER_WIDTH_Q": cfg[4][6], "WARP_SCATTER_WIDTH_K": cfg[4][7],

            "BLOCK_LAYOUT_O_Y": cfg[5][0], "BLOCK_LAYOUT_O_X": cfg[5][1], "WARP_LAYOUT_O_Y": cfg[5][2], "WARP_LAYOUT_O_X": cfg[5][3],
            "BLOCK_SCATTER_WIDTH_P": cfg[5][4], "BLOCK_SCATTER_WIDTH_V": cfg[5][5], "WARP_SCATTER_WIDTH_P": cfg[5][6], "WARP_SCATTER_WIDTH_V": cfg[5][7],
            "UNROLL_NUM": cfg[6][0], "WARP_SIZE": cfg[6][1], "LOAD_CONTINUOUS_P": cfg[6][2], "LOAD_CONTINUOUS_O": cfg[6][3], 
            "SHARED_PREFETCH_P": cfg[6][4], "REG_PREFETCH_P": cfg[6][5], "REG_PREFETCH_O": cfg[6][6],
          }
        }
      })
  return configs


# cfg_dict = {
#   "Br": [32, 64, 128], 
#   "Bc": [32, 64, 128], 
#   "Hd": [128], 
#   "Slice1": [8, 16, 32], 
#   "Slice2": [8, 16, 32], 

#   "PTr": [4, 8], 
#   "PTc": [4, 8], 
#   "OTr": [4, 8], 
#   "OTc": [4, 8],

#   "GLOB_LOAD_WIDTH_Q": [4], 
#   "GLOB_LOAD_WIDTH_K": [4], 
#   "GLOB_LOAD_WIDTH_V": [4],

#   "BLOCK_LAYOUT_P_Y": [1, 2, 4], 
#   "BLOCK_LAYOUT_P_X": [1, 2, 4], 
#   "WARP_LAYOUT_P_Y": [2, 4, 8, 16], 
#   "WARP_LAYOUT_P_X": [2, 4, 8, 16],
#   "BLOCK_SCATTER_WIDTH_Q": [2, 4, 8], 
#   "BLOCK_SCATTER_WIDTH_K": [2, 4, 8], 
#   "WARP_SCATTER_WIDTH_Q": [1, 2, 4], 
#   "WARP_SCATTER_WIDTH_K": [1, 2, 4],

#   "BLOCK_LAYOUT_O_Y": [1, 2, 4], 
#   "BLOCK_LAYOUT_O_X": [1, 2, 4], 
#   "WARP_LAYOUT_O_Y": [2, 4, 8, 16], 
#   "WARP_LAYOUT_O_X": [2, 4, 8, 16], 
#   "BLOCK_SCATTER_WIDTH_P": [2, 4, 8], 
#   "BLOCK_SCATTER_WIDTH_V": [2, 4, 8], 
#   "WARP_SCATTER_WIDTH_P": [1, 2, 4], 
#   "WARP_SCATTER_WIDTH_V": [1, 2, 4],

#   "UNROLL_NUM": [8, 16], 
#   "WARP_SIZE": [32], 
#   "LOAD_CONTINUOUS_P": [1], 
#   "LOAD_CONTINUOUS_O": [1], 
#   "SHARED_PREFETCH_P": [0, 1], 
#   "REG_PREFETCH_P": [0, 1], 
#   "REG_PREFETCH_O": [0, 1]
# }

# cc = CreateAttnConfig(cfg_dict, [1, 32, 2048, 128], smem_size=49152, sm_num=108)
# result = cc.main()
# print(len(result))
# for i in result:
#   print(i)