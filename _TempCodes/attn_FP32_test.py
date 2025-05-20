import json
import importlib.util

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
      if oldResult[0][0] * oldResult[0][1] <= 256:
        LDS_SZIE = oldResult[1] * oldResult[4] + oldResult[2] * oldResult[4] + oldResult[1] * oldResult[2] + oldResult[3] * oldResult[5] + 3 * oldResult[1]
        if oldResult[33] == 1:
          LDS_SZIE += oldResult[1] * oldResult[4] + oldResult[2] * oldResult[4]
        if LDS_SZIE <= 96 * 1024 / 4:
          results.append(oldResult[1:])
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

# def compile():
#   spec = importlib.util.spec_from_file_location("KCGCompiler", )
#   mod = importlib.util.module_from_spec(spec)
#   spec.loader.exec_module(mod)
#   self.__compile_kernel_matmul = mod.compile_kernel_matmul

if __name__ == "__main__":
  path = "./config.json"
  shape = [1, 32, 2048, 128]
  cc = CreateConfig(path, shape)
  cfgs = cc.main()
  print(len(cfgs))
  # for cfg in cfgs:
  #   print(cfg)
