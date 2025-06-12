#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Python.h"

using namespace KernelCodeGen;
using TuneConfig = std::map<std::string, std::map<std::string, int64_t>>;
using TileConfig = std::map<std::string, std::map<std::string, int64_t>>;

KernelCodeGen::KernelCodeGenerator generator;

std::string compile_kernel(TuneConfig tuneCfg, TileConfig tileCfg, std::vector<KernelData> kds, std::vector<FuseKernelData> fkds={}) {
  // compile func
  mlir::ModuleOp module = generator.createModule();
  auto noSupKernels = generator.createKernels(module, kds);  // create kernels
  auto result = generator.fusing(module, fkds);  // fusing
  result = generator.mapping(module, tileCfg);  // mpping
  generator.optimize(module, tuneCfg);  // optimize
  result = generator.transform(module);
  llvm::outs() << "transform result: " << result << "\n";
  llvm::outs() << module << "\n";
  // result = generator.lowering_(module);  // lowering
  // llvm::outs() << "lowering_ result: " << result << "\n";
  return "";
  // generator.lowering(module);  // lowering
  // auto path = generator.translate(module);  // translate
  // // std::cout << "[lib] ===========4" << std::endl;
  // return path;
}

std::string matmul(std::vector<int64_t> shape, const TuneConfig& config) {
  // matmul compile func
  // shape: {batch, ..., m, n, k}
  auto mm = config.at("matmul");
  TileConfig tileConfig  = {
    {"matmul", {{"BLOCK_SIZE_Y", mm.at(KEY_BLOCK_SIZE_M)}, {"THREAD_SIZE_Y", mm.at(KEY_THREAD_SIZE_M)}, 
                {"BLOCK_SIZE_X", mm.at(KEY_BLOCK_SIZE_N)}, {"THREAD_SIZE_X", mm.at(KEY_THREAD_SIZE_N)}}}
  };
  // create new shapes
  int len = shape.size(), bl = shape.size()-3;
  int64_t m = shape[len-3], n = shape[len-2], k = shape[len-1];  // m, n, k
  std::vector<int64_t> b(shape.begin(), shape.begin()+bl);  // batch
  std::vector<int64_t> sha{m, k}, shb{n, k}, shc{m, n};
  for (int i=b.size()-1; i>=0; i--) {  // add batch
    sha.insert(sha.begin(), b[i]); shb.insert(shc.begin(), b[i]); shc.insert(shc.begin(), b[i]);
  }
  // create kernel info
  KernelData kd = {
    "matmul", "Matmul", {sha, shb, shc}, {"float32", "float32", "float32"}, {true, false}, 1
  };
  std::vector<KernelData> kds{kd};
  // compile kernel
  return compile_kernel(config, tileConfig, kds);
}

std::string attention(std::vector<int64_t> shape, const TuneConfig& config) {
  // attn compile func
  // shape: {batch, head_num, seq_len, head_dim}
  auto attn = config.at("attention");
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}}, 
    {"softmax1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")}, 
                {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
  };
  // create new shapes
  int len = shape.size(), bl = shape.size()-2;
  int64_t sl = shape[len-2], hd = shape[len-1];
  std::vector<int64_t> b(shape.begin(), shape.begin()+bl);
  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh1{sl, sl}, sh2{sl, sl};
  std::vector<int64_t> sha1{sl, sl}, shb1{sl, hd}, shc1{sl, hd};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh1.insert(sh1.begin(), b[i]); sh2.insert(sh2.begin(), b[i]); 
    sha1.insert(sha1.begin(), b[i]); shb1.insert(shb1.begin(), b[i]); shc1.insert(shc1.begin(), b[i]);
  }
  // kernel info
  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {"float32", "float32", "float32"}, {true, false}, 1
  };
  KernelData kd2 = {
    "softmax1", "Softmax", {sh1, sh2}, {"float32", "float32"}, {false}, 1
  };
  KernelData kd3 = {
    "matmul2", "Matmul", {sha1, shb1, shc1}, {"float32", "float32", "float32"}, {false, false}, 1
  };
  // kernel fusing
  FuseKernelData fkd = {
    "attention", "FlashAttn", {"matmul1", "softmax1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd3.shapes[1], kd3.shapes[2]}, {kd1.shapes[2]},
    {"float32", "float32", "float32", "float32"}, {"float32"},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {{"matmul2", {2}}}},
    {{{"matmul1", {2}}, {"softmax1", {0, 1}} , {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd3.isTrans[1]}, {"y"}, 1
  };
  std::vector<KernelData> kds{kd1, kd2, kd3};
  std::vector<FuseKernelData> fkds{fkd};
  // compile kernel
  return compile_kernel(config, tileConfig, kds, fkds);
}

int main() {
  std::vector<int64_t> shape{1, 32, 4096, 128};
  generator.setPaltform(Target::ROCm, "906");
  TuneConfig attn_cfg = {
    {"attention", 
      {{"Br", 128}, {"Bc", 128}, {"Hd", 128}, {"Slice1", 16}, {"Slice2", 16}, 
       {"PTr", 2}, {"PTc", 8}, {"OTr", 4}, {"OTc", 8}, 
       // global to shared
       {"GLOB_LOAD_WIDTH_Q", 4}, {"GLOB_LOAD_WIDTH_K", 4}, {"GLOB_LOAD_WIDTH_V", 4}, 
       {"LOAD_CONTINUOUS_P", 1}, {"LOAD_CONTINUOUS_O", 1}, 
       // prefecth
       {"SHARED_PREFETCH_P", 1}, {"REG_PREFETCH_P", 1}, {"SHARED_PREFETCH_O", 1}, {"REG_PREFETCH_O", 1},
       // P = Q * K
       {"BLOCK_LAYOUT_P_Y", 8}, {"BLOCK_LAYOUT_P_X", 1}, {"WARP_LAYOUT_P_Y", 4}, {"WARP_LAYOUT_P_X", 8},
       {"BLOCK_SCATTER_WIDTH_Q", 2}, {"BLOCK_SCATTER_WIDTH_K", 2}, {"WARP_SCATTER_WIDTH_Q", 1}, {"WARP_SCATTER_WIDTH_K", 1},
       // O = P * V
       {"BLOCK_LAYOUT_O_Y", 2}, {"BLOCK_LAYOUT_O_X", 4}, {"WARP_LAYOUT_O_Y", 8}, {"WARP_LAYOUT_O_X", 4},
       {"BLOCK_SCATTER_WIDTH_P", 2}, {"BLOCK_SCATTER_WIDTH_V", 2}, {"WARP_SCATTER_WIDTH_P", 1}, {"WARP_SCATTER_WIDTH_V", 1},
       {"WARP_SIZE", 64}, {"UNROLL_NUM", 16}}}
  };
  std::string path = attention(shape, attn_cfg);
  llvm::outs() << "DCU hsaco file path: " << path << "\n";
}