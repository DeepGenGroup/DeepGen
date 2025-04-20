#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Common/ThreadPool.h"

using namespace KernelCodeGen;

int main(){
  int64_t bs = 1;
  int64_t hn = 16;
  int64_t sl = 4096;
  int64_t hd = 128;

  KernelCodeGenerator generator(Target::CUDA, "");
  mlir::ModuleOp module = generator.createModule();
  std::vector<KernelData> kds;
  std::vector<FuseKernelData> fkds;

  // ======  kernel  ======
  KernelData kd1, kd2, kd3;
  // matmul1
  kd1.name = "matmul1";
  kd1.type = "Matmul";
  kd1.argNames = {"A1", "B1", "C1"};
  kd1.shapes = {{bs, hn, sl, hd}, {bs, hn, hd, sl}, {bs, hn, sl, sl}};
  kd1.dtypes = {"float32", "float32", "float32"};
  kd1.isTrans = {false, false};
  kd1.outputArgNum = 1;
  kds.push_back(kd1);
  //Softmax1
  kd2.name = "softmax1";
  kd2.type = "Softmax";
  kd2.argNames = {"C1", "C2"};
  kd2.shapes = {{bs, hn, sl, sl}, {bs, hn, sl, sl}};
  kd2.dtypes = {"float32", "float32"};
  kd2.isTrans = {false};
  kd2.outputArgNum = 1;
  kds.push_back(kd2);
  // matmul2
  kd3.name = "matmul2";
  kd3.type = "Matmul";
  kd3.argNames = {"C2", "B2", "C3"};
  kd3.shapes = {{bs, hn, sl, sl}, {bs, hn, sl, hd}, {bs, hn, sl, hd}};
  kd3.dtypes = {"float32", "float32", "float32"};
  kd3.isTrans = {false, false};
  kd3.outputArgNum = 1;
  kds.push_back(kd3);

  // ======  fuse kernel  ======
  FuseKernelData fkd = {
    "attention1",
    "FlashAttn",
    {"matmul1", "softmax1", "matmul2"},
    {{bs, hn, sl, hd}, {bs, hn, hd, sl}, {bs, hn, sl, hd}, {bs, hn, sl, hd}},
    {{bs, hn, sl, sl}},
    {"float32", "float32", "float32", "float32"},
    {"float32"},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {{"matmul2", {2}}}}, 
    {{{"matmul1", {2}}, {"softmax1", {0, 1}} , {"matmul2", {0}}}},
    {false, false, false},
    {"y"},
    1
  };
  fkds.push_back(fkd);

  // create kernels
  auto noSupKernels = generator.createKernels(module, kds);
  // fusing
  auto result = generator.fusing(module, fkds);
  // llvm::outs() << module << "\n";

  std::vector<mlir::ModuleOp> mods;

  std::map<std::string, std::map<std::string, int64_t>> tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", 64}, {"THREAD_SIZE_Y", 4}, {"BLOCK_SIZE_X", 64}, {"THREAD_SIZE_X", 4}}}, 
    {"softmax1", {{"BLOCK_SIZE_Y", 64}, {"THREAD_SIZE_Y", 4}, {"BLOCK_SIZE_X", 64}, {"THREAD_SIZE_X", 4}}},
    {"matmul2", {{"BLOCK_SIZE_Y", 64}, {"THREAD_SIZE_Y", 4}, {"BLOCK_SIZE_X", 128}, {"THREAD_SIZE_X", 8}}},
  };
  
  //
  std::map<std::string, std::map<std::string, int64_t>> tuneConfig = {
    {"attention1", 
      {{"Br", 64}, {"Bc", 64}, {"Hd", 128}, {"Slice1", 32}, {"Slice2", 32}, 
       {"PTr", 4}, {"PTc", 4}, {"OTr", 4}, {"OTc", 8}, 
       // global to shared
       {"GLOB_LOAD_WIDTH_Q", 4}, {"GLOB_LOAD_WIDTH_K", 4}, {"GLOB_LOAD_WIDTH_V", 4},
       {"LOAD_CONTINUOUS_P", 1}, {"LOAD_CONTINUOUS_O", 1}, 
       // prefecth
       {"SHARED_PREFETCH_P", 1}, {"REG_PREFETCH_P", 1}, {"SHARED_PREFETCH_O", 1}, {"REG_PREFETCH_O", 1},
       // P = Q * K
       {"BLOCK_LAYOUT_P_Y", 8}, {"BLOCK_LAYOUT_P_X", 1}, {"WARP_LAYOUT_P_Y", 2}, {"WARP_LAYOUT_P_X", 16},
       {"BLOCK_SCATTER_WIDTH_Q", 4}, {"BLOCK_SCATTER_WIDTH_K", 4}, {"WARP_SCATTER_WIDTH_Q", 2}, {"WARP_SCATTER_WIDTH_K", 2},
       // O = P * V
       {"BLOCK_LAYOUT_O_Y", 2}, {"BLOCK_LAYOUT_O_X", 4}, {"WARP_LAYOUT_O_Y", 8}, {"WARP_LAYOUT_O_X", 4},
       {"BLOCK_SCATTER_WIDTH_P", 4}, {"BLOCK_SCATTER_WIDTH_V", 4}, {"WARP_SCATTER_WIDTH_P", 2}, {"WARP_SCATTER_WIDTH_V", 2},
       {"WARP_SIZE", 32}, {"UNROLL_NUM", 16}}}
  };
  // std::map<std::string, std::map<std::string, int64_t>> tuneConfig = {
  //   {"matmul1", {{"BLOCK_SIZE_M", 64}, {"THREAD_SIZE_M", 4}, {"BLOCK_SIZE_N", 64}, {"THREAD_SIZE_N", 4}, /* thread_num: 256 */
  //                {"LOCAL_SPLIT_U", 1}, {"BLOCK_SIZE_K", 32},     /* u>1 -> thread_num *= u */
  //                {"GLOB_LOAD_WIDTH_A", 4}, {"GLOB_LOAD_WIDTH_B", 4}, {"GLOB_STORE_WIDTH", 4},
  //                {"BLOCK_LAYOUT_Y", 2}, {"BLOCK_LAYOUT_X", 4}, {"WARP_LAYOUT_Y", 8}, {"WARP_LAYOUT_X", 4},  /* 不受splitu的影响th256 */
  //                {"BLOCK_SCATTER_WIDTH_M", 4}, {"WARP_SCATTER_WIDTH_M", 2}, {"BLOCK_SCATTER_WIDTH_N", 4}, {"WARP_SCATTER_WIDTH_N", 2},
  //                {"WARP_SIZE", 32}, {"LOAD_CONTINUOUS", 1}, {"STORE_CONTINUOUS", 1}, {"SHARED_PREFETCH", 1}, {"REG_PREFETCH", 1}, 
  //                {"BLOCK_MAPPING", 8}, {"UNROLL_NUM", 16}}}
  // };
#if 0
  // 如果是调优的话就需要将模型切成小模型 splitModule
  mods = generator.splitModule(module);
#endif

#if 1
  // 如果已经有所有kernel的最优config了，就不需要运行 splitModule
  mods.push_back(module);
#endif

for (auto mod : mods) {
  // mpping
  result = generator.mapping(mod, tileConfig);
  // llvm::outs() << mod << "\n";
  
  // optimize
  generator.optimize(mod, tuneConfig);
  // llvm::outs() << mod << "\n";

  // lowering
  generator.lowering(mod);
  // llvm::outs() << mod << "\n";

  // translate
  auto path = generator.translate(mod);
  llvm::outs() << "bin path: " << path << "\n";
}

  return 0;
}


