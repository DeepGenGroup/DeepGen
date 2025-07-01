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
  
  # def storeSizeAndOther(self, old_cfgs):
  #   result = []
  #   for old_cfg in old_cfgs:
  #     for spp in self.cfg["SHARED_PREFETCH_P"]:
  #       smem_size = old_cfg[1][0] * old_cfg[1][3] + old_cfg[1][1] * old_cfg[1][3] + \
  #                   old_cfg[1][0] * old_cfg[1][1] + old_cfg[1][2] * old_cfg[1][4] + 3 * old_cfg[1][0]
  #       if spp == 1:
  #         smem_size += old_cfg[1][0] * old_cfg[1][3] + old_cfg[1][1] * old_cfg[1][3]
  #       if smem_size <= self.max_sm_size:  # shared memory size
  #         for rpp in self.cfg["REG_PREFETCH_P"]:
  #           for rpo in self.cfg["REG_PREFETCH_O"]:
  #             for unroll in self.cfg["UNROLL_NUM"]:
  #               result.append(old_cfg + ((unroll, self.cfg["WARP_SIZE"][0], 1, 1, spp, rpp, rpo), ))  # 只能连续访存
  #   # (th_num, (br, bc, hd, s1, s2), (ptr, ptc, otr, otc), (glwq, glwk, glwv), (bly, blx, wly, wlx, bswq, bswk, wswq, wswk), 
  #   # (bly, blx, wly, wlx, bswp, bswv, wswp, wswv), (unroll ,warp_size, load_continuous_p, lc_o, sm_prefetch_p, reg_pf_p, reg, pf_o))
  #   return result

  def storeSizeAndOther_(self, old_cfgs):
    result = []
    for old_cfg in old_cfgs:
      smem_size = old_cfg[1][0] * old_cfg[1][3] + old_cfg[1][1] * old_cfg[1][3] + \
                  old_cfg[1][0] * old_cfg[1][1] + old_cfg[1][2] * old_cfg[1][4] + 3 * old_cfg[1][0]
      if smem_size * self.type_width <= self.max_sm_size:  # shared memory size
        result.append(old_cfg + ((16, self.cfg["WARP_SIZE"][0], 1, 1, 0, 0, 0), ))  # 只能连续访存
    # (th_num, (br, bc, hd, s1, s2), (ptr, ptc, otr, otc), (glwq, glwk, glwv), (bly, blx, wly, wlx, bswq, bswk, wswq, wswk), 
    # (bly, blx, wly, wlx, bswp, bswv, wswp, wswv), (unroll ,warp_size, load_continuous_p, lc_o, sm_prefetch_p, reg_pf_p, reg_pf_o))
    return result
  
  def cut(self, old_cfgs):
    result = []
    for old in old_cfgs:
      # print(old)
      sm_util = (self.seq_len / old[1][0])
      # sm占用率 / 离散化约束
      if (sm_util >= self.sm_num):
        br_div_bk = old[1][0] / 32  #  br / bank num(32)
        bc_div_bk = old[1][1] / 32  #  br / bank num(32)
        hd_div_bk = old[1][2] / 32  #  hd / bank num(32)
        brep_q = old[2][0] / old[4][4]
        brep_k = old[2][1] / old[4][5]
        brep_p = old[2][2] / old[5][4]
        brep_v = old[2][3] / old[5][5]
        if (brep_q == br_div_bk and brep_k == bc_div_bk and brep_p == br_div_bk and brep_v == hd_div_bk):
          # if old[4][4] == old[4][6] and old[4][5] == old[4][7] and old[5][4] == old[5][6] and old[5][5] == old[5][7]:
          result.append(old)
    return result

    
  def main(self):
    result = self.threadTile()
    result = self.blockTile(result)
    result = self.layoutAndScatterP(result)
    result = self.layoutAndScatterO(result)
    # result = self.storeSizeAndOther(result)
    result = self.storeSizeAndOther_(result)
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
    path = "/home/xiebaokang/projects/evaluate/DeepGen/python/deepgen/cfg_jsons/attn.json"
    cfg_dict = readConfigJson(path)
    cfg_dict["Hd"] = [head_dim]
    cc = CreateAttnConfig(cfg_dict, shape, smem_size=gpu_info.shared_mem_per_block, sm_num=gpu_info.compute_units)
    cfgs = cc.main()
    for cfg in cfgs:
      smem_size = cfg[1][0]*cfg[1][3] + cfg[1][1]*cfg[1][3] + cfg[1][0]*cfg[1][1] + cfg[1][2]*cfg[1][4] + 3*cfg[1][0]
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
            "UNROLL_NUM": 16, "WARP_SIZE": 64, "LOAD_CONTINUOUS_P": 1, "LOAD_CONTINUOUS_O": 1, 
            "SHARED_PREFETCH_P": 0, "REG_PREFETCH_P": 0, "REG_PREFETCH_O": 0,
          }
        }
      })
  return configs
  

