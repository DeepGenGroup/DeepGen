#define PY_SSIZE_T_CLEAN
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include "config.h"
#include "KernelCodeGen.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Python.h"
#include "llvm/Support/raw_ostream.h"

using namespace KernelCodeGen;
using TuneConfig = std::map<std::string, std::map<std::string, int64_t>>;
using TileConfig = std::map<std::string, std::map<std::string, int64_t>>;

KernelCodeGen::KernelCodeGenerator generator;

std::string __GlobalKernelName = "attention1";

namespace {
std::atomic<uint64_t> gIrDumpSeq{0};

bool isIrDumpEnabled() {
  const char* enabledEnv = std::getenv("KCG_DUMP_IR");
  if (!enabledEnv || !*enabledEnv) {
    return false;
  }
  std::string value(enabledEnv);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::string defaultIrDumpDir() {
  std::filesystem::path projectRoot = std::filesystem::path(BC_DUMP_PATH).parent_path();
  return (projectRoot / "_TempIRCodes").string();
}

std::string sanitizePathToken(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if ((c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') ||
        c == '_' || c == '-' || c == '.') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? std::string("unknown") : out;
}

void dumpModuleIRIfEnabled(mlir::ModuleOp module, const std::string& stage) {
  if (!isIrDumpEnabled()) {
    return;
  }

  const char* dumpDirEnv = std::getenv("KCG_DUMP_IR_DIR");
  const std::string dumpDir = (dumpDirEnv && *dumpDirEnv) ? std::string(dumpDirEnv) : defaultIrDumpDir();
  std::error_code ec;
  std::filesystem::create_directories(dumpDir, ec);
  if (ec) {
    llvm::errs() << "[ir-dump] create dir failed: " << dumpDir << " err=" << ec.message() << "\n";
    return;
  }

  const uint64_t seq = ++gIrDumpSeq;
  const std::string kernel = sanitizePathToken(__GlobalKernelName);
  const std::string tag = sanitizePathToken(stage);
  const auto outPath = (std::filesystem::path(dumpDir) /
      (kernel + ".s" + std::to_string(seq) + "." + tag + ".mlir")).string();
  const auto latestPath = (std::filesystem::path(dumpDir) /
      (kernel + ".latest." + tag + ".mlir")).string();

  auto writeOne = [&](const std::string& path) -> bool {
    std::error_code fec;
    llvm::raw_fd_ostream os(path, fec);
    if (fec) {
      llvm::errs() << "[ir-dump] open failed: " << path << " err=" << fec.message() << "\n";
      return false;
    }
    module.print(os);
    os.flush();
    return true;
  };

  bool wroteSeq = writeOne(outPath);
  bool wroteLatest = writeOne(latestPath);
  if (wroteSeq || wroteLatest) {
    llvm::errs() << "[ir-dump] wrote " << outPath;
    if (wroteLatest) {
      llvm::errs() << " and " << latestPath;
    }
    llvm::errs() << "\n";
  }
}
}  // namespace

std::string compile_kernel(TuneConfig tuneCfg, TileConfig tileCfg, std::vector<KernelData> kds, std::vector<FuseKernelData> fkds={}) {
  // compile func
  mlir::ModuleOp module = generator.createModule();
  auto noSupKernels = generator.createKernels(module, kds);  // create kernels
  llvm::errs() << "[pipeline] createKernels done\n";
  dumpModuleIRIfEnabled(module, "00_create");
  auto result = generator.fusing(module, fkds);  // fusing
  llvm::errs() << "[pipeline] fusing done\n";
  dumpModuleIRIfEnabled(module, "01_fusing");
  result = generator.mapping(module, tileCfg);  // mpping
  llvm::errs() << "[pipeline] mapping done\n";
  dumpModuleIRIfEnabled(module, "02_mapping");
  llvm::errs() << "[pipeline] about to optimize\n";
  if (!generator.optimize(module, tuneCfg)) {
    llvm::errs() << "[pipeline] optimize FAILED\n";
    dumpModuleIRIfEnabled(module, "03_optimize_failed");
    return "";
  }
  llvm::errs() << "[pipeline] optimize done\n";
  dumpModuleIRIfEnabled(module, "03_optimize");
  if (mlir::failed(module.verify())) {
    llvm::errs() << "[pipeline] ERROR: IR already broken after optimize!\n";
    dumpModuleIRIfEnabled(module, "03_verify_failed");
    return "";
  }
  llvm::errs() << "[pipeline] verify OK after optimize\n";
  llvm::errs() << "[pipeline] about to transform\n";
  if (!generator.transform(module)) {
    llvm::errs() << "[pipeline] transform FAILED\n";
    dumpModuleIRIfEnabled(module, "04_transform_failed");
    return "";
  }
  llvm::errs() << "[pipeline] transform done\n";
  dumpModuleIRIfEnabled(module, "04_transform");
  llvm::errs() << "[pipeline] about to lowering\n";
  if (!generator.lowering_(module)) {
    llvm::errs() << "[pipeline] lowering FAILED\n";
    dumpModuleIRIfEnabled(module, "05_lowering_failed");
    return "";
  }
  llvm::errs() << "[pipeline] lowering done\n";
  dumpModuleIRIfEnabled(module, "05_lowering");
  auto path = generator.translate(module);
  llvm::errs() << "[pipeline] translate done\n";
  return path;
}

std::string matmul(std::vector<int64_t> shape, const TuneConfig& config) {
  // matmul compile func
  auto mm = config.at(__GlobalKernelName);
  // auto mm = config.at("matmul");
  TileConfig tileConfig  = {
    {__GlobalKernelName, {{"BLOCK_SIZE_Y", mm.at(KEY_BLOCK_SIZE_M)}, {"THREAD_SIZE_Y", mm.at(KEY_THREAD_SIZE_M)}, 
                {"BLOCK_SIZE_X", mm.at(KEY_BLOCK_SIZE_N)}, {"THREAD_SIZE_X", mm.at(KEY_THREAD_SIZE_N)}}}
  };
  // create new shapes
  int len = shape.size(), bl = shape.size()-3;
  int64_t m = shape[len-3], n = shape[len-2], k = shape[len-1];  // m, n, k
  std::vector<int64_t> b(shape.begin(), shape.begin()+bl);  // batch
  std::vector<int64_t> sha{k, m}, shb{k, n}, shc{m, n};
  for (int i=b.size()-1; i>=0; i--) {  // add batch
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
  }
  // create kernel info
  KernelData kd = {
    __GlobalKernelName, "Matmul", {sha, shb, shc}, {"float32", "float32", "float32"}, {true, false}, 1
  };
  std::vector<KernelData> kds{kd};
  // compile kernel
  return compile_kernel(config, tileConfig, kds);
}

std::string attention(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // attn compile func
  // shape supports both {batch, head_num, seq_len, head_dim}
  // and {batch, seq_len, head_num, head_dim}.
  auto attn = config.at(__GlobalKernelName);
  // auto attn = config.at("attention");
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}}, 
    {"softmax1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")}, 
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")}, 
                {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
  };
  // create new shapes
  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  // Infer (head_num, seq_len) from the two pre-hd dimensions.
  // Typical attention always has seq_len >= head_num.
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);
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
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "softmax1", "Softmax", {sh1, sh2}, {dtype, dtype}, {false}, 1
  };
  KernelData kd3 = {
    "matmul2", "Matmul", {sha1, shb1, shc1}, {dtype, dtype, dtype}, {false, false}, 1
  };
  // kernel fusing
  FuseKernelData fkd = {
    __GlobalKernelName, "FlashAttn", {"matmul1", "softmax1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd3.shapes[1], kd3.shapes[2]}, {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {{"matmul2", {2}}}},
    {{{"matmul1", {2}}, {"softmax1", {0, 1}} , {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd3.isTrans[1]}, {"y"}, 1
  };
  std::vector<KernelData> kds{kd1, kd2, kd3};
  std::vector<FuseKernelData> fkds{fkd};
  // v1: scale scores AFTER mm1 (standard attention)
  TuneConfig configV1 = config;
  configV1[__GlobalKernelName]["SCALE_SCORES"] = 1;
  return compile_kernel(configV1, tileConfig, kds, fkds);
}

std::string attention_v2(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // Same 3-operator fuse structure as attention(), but:
  //   - Python passes UNSCALED Q (just transposed)
  //   - Optimizer fuses scale into Q-load pipeline (SCALE_Q flag)
  //   - No extra global memory round-trip for ElemScale

  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"softmax1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")},
                {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);
  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh1{sl, sl}, sh2{sl, sl};
  std::vector<int64_t> sha1{sl, sl}, shb1{sl, hd}, shc1{sl, hd};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh1.insert(sh1.begin(), b[i]); sh2.insert(sh2.begin(), b[i]);
    sha1.insert(sha1.begin(), b[i]); shb1.insert(shb1.begin(), b[i]); shc1.insert(shc1.begin(), b[i]);
  }
  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "softmax1", "Softmax", {sh1, sh2}, {dtype, dtype}, {false}, 1
  };
  KernelData kd3 = {
    "matmul2", "Matmul", {sha1, shb1, shc1}, {dtype, dtype, dtype}, {false, false}, 1
  };
  FuseKernelData fkd = {
    __GlobalKernelName, "FlashAttn", {"matmul1", "softmax1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd3.shapes[1], kd3.shapes[2]}, {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {{"matmul2", {2}}}},
    {{{"matmul1", {2}}, {"softmax1", {0, 1}} , {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd3.isTrans[1]}, {"y"}, 1
  };
  std::vector<KernelData> kds{kd1, kd2, kd3};
  std::vector<FuseKernelData> fkds{fkd};

  // Add SCALE_Q flag so optimizer fuses scale into Q-load pipeline
  TuneConfig configV2 = config;
  configV2[__GlobalKernelName]["SCALE_Q"] = 1;
  return compile_kernel(configV2, tileConfig, kds, fkds);
}

std::string attention_split_k1(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // Split attention kernel 1: GEMM(Q@K^T) + online reduce → em, denom
  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"softmaxstats1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);

  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh_em{sl, 1}, sh_denom{sl, 1};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh_em.insert(sh_em.begin(), b[i]); sh_denom.insert(sh_denom.begin(), b[i]);
  }

  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "softmaxstats1", "SoftmaxStats", {shc, sh_em, sh_denom}, {dtype, dtype, dtype}, {false}, 2
  };

  FuseKernelData fkd = {
    __GlobalKernelName, "GemmStats", {"matmul1", "softmaxstats1"},
    {kd1.shapes[0], kd1.shapes[1], sh_em, sh_denom},
    {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"softmaxstats1", {1}}}, {{"softmaxstats1", {2}}}},
    {{{"matmul1", {2}}, {"softmaxstats1", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1]}, {"y"}, 2
  };

  std::vector<KernelData> kds{kd1, kd2};
  std::vector<FuseKernelData> fkds{fkd};

  TuneConfig configK1 = config;
  configK1[__GlobalKernelName]["SCALE_Q"] = 1;
  configK1[__GlobalKernelName]["CAUSAL_MASK"] = 1;
  return compile_kernel(configK1, tileConfig, kds, fkds);
}

std::string attention_split_k2(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // Split attention kernel 2: scale-Q GEMM(Q@K^T) + causal mask + tmp=exp(scores)/em
  //                           + GEMM(tmp@V) + row divide by denom → O
  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")},
                {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);

  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh_em{sl, 1}, sh_denom{sl, 1};
  std::vector<int64_t> sha1{sl, sl}, shb1{sl, hd}, shc1{sl, hd};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh_em.insert(sh_em.begin(), b[i]); sh_denom.insert(sh_denom.begin(), b[i]);
    sha1.insert(sha1.begin(), b[i]); shb1.insert(shb1.begin(), b[i]); shc1.insert(shc1.begin(), b[i]);
  }

  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "matmul2", "Matmul", {sha1, shb1, shc1}, {dtype, dtype, dtype}, {false, false}, 1
  };

  // No middle operator — just 2 matmuls fused via midBuf (scores/tmp).
  // em/denom are extra func args with empty index maps.
  // The optimizer applies exp/em on tileP, then row-divides O by denom.
  FuseKernelData fkd = {
    __GlobalKernelName, "FlashAttnSplitK2", {"matmul1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd2.shapes[1], sh_em, sh_denom, kd2.shapes[2]},
    {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {}, {}, {{"matmul2", {2}}}},
    {{{"matmul1", {2}}, {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd2.isTrans[1]}, {"y"}, 1
  };

  std::vector<KernelData> kds{kd1, kd2};
  std::vector<FuseKernelData> fkds{fkd};

  TuneConfig configK2 = config;
  configK2[__GlobalKernelName]["SCALE_Q"] = 1;
  configK2[__GlobalKernelName]["CAUSAL_MASK"] = 1;
  return compile_kernel(configK2, tileConfig, kds, fkds);
}

std::string gemma2_split_k1(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // Gemma2 split kernel 1: GEMM(Q@K^T) + softcap(tanh) + causal mask + online reduce → em, denom
  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"softmaxstats1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);

  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh_em{sl, 1}, sh_denom{sl, 1};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh_em.insert(sh_em.begin(), b[i]); sh_denom.insert(sh_denom.begin(), b[i]);
  }

  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "softmaxstats1", "SoftmaxStats", {shc, sh_em, sh_denom}, {dtype, dtype, dtype}, {false}, 2
  };

  FuseKernelData fkd = {
    __GlobalKernelName, "GemmStats", {"matmul1", "softmaxstats1"},
    {kd1.shapes[0], kd1.shapes[1], sh_em, sh_denom},
    {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"softmaxstats1", {1}}}, {{"softmaxstats1", {2}}}},
    {{{"matmul1", {2}}, {"softmaxstats1", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1]}, {"y"}, 2
  };

  std::vector<KernelData> kds{kd1, kd2};
  std::vector<FuseKernelData> fkds{fkd};

  TuneConfig configK1 = config;
  configK1[__GlobalKernelName]["SCALE_Q"] = 1;
  configK1[__GlobalKernelName]["SOFTCAP_TANH"] = 1;
  configK1[__GlobalKernelName]["CAUSAL_MASK"] = 1;
  return compile_kernel(configK1, tileConfig, kds, fkds);
}

std::string gemma2_split_k2(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // Gemma2 split kernel 2: GEMM(Q@K^T) + softcap(tanh) + causal mask + broadcast normalize(em,denom) + GEMM(P@V) → O
  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")},
                {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);

  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh_em{sl, 1}, sh_denom{sl, 1};
  std::vector<int64_t> sha1{sl, sl}, shb1{sl, hd}, shc1{sl, hd};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh_em.insert(sh_em.begin(), b[i]); sh_denom.insert(sh_denom.begin(), b[i]);
    sha1.insert(sha1.begin(), b[i]); shb1.insert(shb1.begin(), b[i]); shc1.insert(shc1.begin(), b[i]);
  }

  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "matmul2", "Matmul", {sha1, shb1, shc1}, {dtype, dtype, dtype}, {false, false}, 1
  };

  FuseKernelData fkd = {
    __GlobalKernelName, "FlashAttnSplitK2", {"matmul1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd2.shapes[1], sh_em, sh_denom, kd2.shapes[2]},
    {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {}, {}, {{"matmul2", {2}}}},
    {{{"matmul1", {2}}, {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd2.isTrans[1]}, {"y"}, 1
  };

  std::vector<KernelData> kds{kd1, kd2};
  std::vector<FuseKernelData> fkds{fkd};

  TuneConfig configK2 = config;
  configK2[__GlobalKernelName]["SCALE_Q"] = 1;
  configK2[__GlobalKernelName]["SOFTCAP_TANH"] = 1;
  configK2[__GlobalKernelName]["CAUSAL_MASK"] = 1;
  return compile_kernel(configK2, tileConfig, kds, fkds);
}

std::string h2o_split_k1(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // H2O split kernel 1: GEMM(Q@K^T) + causal mask + online reduce → em, denom
  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"softmaxstats1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);

  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh_em{sl, 1}, sh_denom{sl, 1};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh_em.insert(sh_em.begin(), b[i]); sh_denom.insert(sh_denom.begin(), b[i]);
  }

  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "softmaxstats1", "SoftmaxStats", {shc, sh_em, sh_denom}, {dtype, dtype, dtype}, {false}, 2
  };

  FuseKernelData fkd = {
    __GlobalKernelName, "GemmStats", {"matmul1", "softmaxstats1"},
    {kd1.shapes[0], kd1.shapes[1], sh_em, sh_denom},
    {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"softmaxstats1", {1}}}, {{"softmaxstats1", {2}}}},
    {{{"matmul1", {2}}, {"softmaxstats1", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1]}, {"y"}, 2
  };

  std::vector<KernelData> kds{kd1, kd2};
  std::vector<FuseKernelData> fkds{fkd};

  TuneConfig configK1 = config;
  configK1[__GlobalKernelName]["SCALE_Q"] = 1;
  configK1[__GlobalKernelName]["CAUSAL_MASK"] = 1;
  return compile_kernel(configK1, tileConfig, kds, fkds);
}

std::string h2o_split_k2(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // H2O split kernel 2: GEMM(K^T@Q) + causal mask + normalize(em,denom) + row reduce → row_sum
  //
  // TRANSPOSED TILING: K is funcArg[0] (A, outer), Q is funcArg[1] (B, inner).
  //   GEMM produces P^T = K^T @ Q  [S_k, S_q]
  //   - Outer blocks tile along S_k (each block "owns" Br rows of row_sum)
  //   - Inner loop iterates along S_q
  //   - em/denom index S_q (inner dim), loaded per iteration
  //   - reduce_sum along S_q (inner dim) → row_sum[S_k], local to each block
  //   - No cross-block atomics needed.
  //
  // Python passes: (K[B,H,D,S], Q_t[B,H,D,S], em[B,H,S,1], denom[B,H,S,1], row_sum[B,H,S])
  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"reducesum1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);

  // sha/shb are symmetric ({hd, sl}), but semantically:
  //   funcArg[0] → A = K, funcArg[1] → B = Q_t
  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh_em{sl, 1}, sh_denom{sl, 1};
  std::vector<int64_t> sh_rowsum{sl};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh_em.insert(sh_em.begin(), b[i]); sh_denom.insert(sh_denom.begin(), b[i]);
    sh_rowsum.insert(sh_rowsum.begin(), b[i]);
  }

  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "reducesum1", "ReduceSum", {shc, sh_rowsum}, {dtype, dtype}, {false}, 2
  };

  // funcArgs: [K, Q_t, em, denom, row_sum]
  //   K → matmul1.arg[0] (A, outer blocks tile S_k)
  //   Q_t → matmul1.arg[1] (B, inner loop iterates S_q)
  //   em, denom → handled by optimizer (index inner dim S_q)
  //   row_sum → reducesum1.arg[1] (output, indexed by outer dim S_k)
  FuseKernelData fkd = {
    __GlobalKernelName, "GemmNormColSum", {"matmul1", "reducesum1"},
    {kd1.shapes[0], kd1.shapes[1], sh_em, sh_denom, sh_rowsum},
    {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {}, {}, {{"reducesum1", {1}}}},
    {{{"matmul1", {2}}, {"reducesum1", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1]}, {"y"}, 2
  };

  std::vector<KernelData> kds{kd1, kd2};
  std::vector<FuseKernelData> fkds{fkd};

  TuneConfig configK2 = config;
  configK2[__GlobalKernelName]["SCALE_Q"] = 1;
  configK2[__GlobalKernelName]["CAUSAL_MASK"] = 1;
  return compile_kernel(configK2, tileConfig, kds, fkds);
}

std::string h2o_split_k3(std::vector<int64_t> shape, const TuneConfig& config, const std::string& dtype = "float32") {
  // H2O split kernel 3: GEMM(Q@K^T) + causal mask + broadcast normalize(em,denom) + GEMM(P@V) → O
  auto attn = config.at(__GlobalKernelName);
  TileConfig tileConfig = {
    {"matmul1", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("PTr")},
                {"BLOCK_SIZE_X", attn.at("Bc")}, {"THREAD_SIZE_X", attn.at("PTc")}}},
    {"matmul2", {{"BLOCK_SIZE_Y", attn.at("Br")}, {"THREAD_SIZE_Y", attn.at("OTr")},
                {"BLOCK_SIZE_X", attn.at("Hd")}, {"THREAD_SIZE_X", attn.at("OTc")}}},
  };

  int len = shape.size();
  int64_t hd = shape[len-1];
  int64_t d0 = shape[len-3], d1 = shape[len-2];
  int64_t head_num = std::min(d0, d1);
  int64_t sl = std::max(d0, d1);
  std::vector<int64_t> b(shape.begin(), shape.begin() + (len - 3));
  b.push_back(head_num);

  std::vector<int64_t> sha{hd, sl}, shb{hd, sl}, shc{sl, sl};
  std::vector<int64_t> sh_em{sl, 1}, sh_denom{sl, 1};
  std::vector<int64_t> sha1{sl, sl}, shb1{sl, hd}, shc1{sl, hd};
  for (int i=b.size()-1; i>=0; i--) {
    sha.insert(sha.begin(), b[i]); shb.insert(shb.begin(), b[i]); shc.insert(shc.begin(), b[i]);
    sh_em.insert(sh_em.begin(), b[i]); sh_denom.insert(sh_denom.begin(), b[i]);
    sha1.insert(sha1.begin(), b[i]); shb1.insert(shb1.begin(), b[i]); shc1.insert(shc1.begin(), b[i]);
  }

  KernelData kd1 = {
    "matmul1", "Matmul", {sha, shb, shc}, {dtype, dtype, dtype}, {true, false}, 1
  };
  KernelData kd2 = {
    "matmul2", "Matmul", {sha1, shb1, shc1}, {dtype, dtype, dtype}, {false, false}, 1
  };

  FuseKernelData fkd = {
    __GlobalKernelName, "FlashAttnSplitK2", {"matmul1", "matmul2"},
    {kd1.shapes[0], kd1.shapes[1], kd2.shapes[1], sh_em, sh_denom, kd2.shapes[2]},
    {kd1.shapes[2]},
    {dtype, dtype, dtype, dtype, dtype, dtype}, {dtype},
    {{{"matmul1", {0}}}, {{"matmul1", {1}}}, {{"matmul2", {1}}}, {}, {}, {{"matmul2", {2}}}},
    {{{"matmul1", {2}}, {"matmul2", {0}}}},
    {kd1.isTrans[0], kd1.isTrans[1], kd2.isTrans[1]}, {"y"}, 1
  };

  std::vector<KernelData> kds{kd1, kd2};
  std::vector<FuseKernelData> fkds{fkd};

  TuneConfig configK3 = config;
  configK3[__GlobalKernelName]["SCALE_Q"] = 1;
  configK3[__GlobalKernelName]["CAUSAL_MASK"] = 1;
  return compile_kernel(configK3, tileConfig, kds, fkds);
}

// bind python module
static bool py_list_to_vector(PyObject* py_list, std::vector<int64_t>& vec) {
  // list to vector
  if (!PyList_Check(py_list)) {
    PyErr_SetString(PyExc_TypeError, "Expected a list");
    return false;
  }
  Py_ssize_t size = PyList_Size(py_list);
  vec.resize(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "List items must be integers");
      return false;
    }
    vec[i] = PyLong_AsLongLong(item);
  }
  return true;
}

static bool py_dict_to_config(PyObject* py_dict, TuneConfig& config) {
  // dict to config
  if (!PyDict_Check(py_dict)) {
    PyErr_SetString(PyExc_TypeError, "Expected a dictionary");
    return false;
  }
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(py_dict, &pos, &key, &value)) {
    if (!PyUnicode_Check(key)) {
      PyErr_SetString(PyExc_TypeError, "Dictionary keys must be strings");
      return false;
    }
    if (!PyDict_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Expected nested dictionary");
      return false;
    }
    const char* outer_key = PyUnicode_AsUTF8(key);
    std::map<std::string, int64_t> inner_map;
    PyObject *inner_key, *inner_value;
    Py_ssize_t inner_pos = 0;
    while (PyDict_Next(value, &inner_pos, &inner_key, &inner_value)) {
      if (!PyUnicode_Check(inner_key)) {
        PyErr_SetString(PyExc_TypeError, "Nested dictionary keys must be strings");
        return false;
      }
      if (!PyLong_Check(inner_value)) {
        PyErr_SetString(PyExc_TypeError, "Nested dictionary values must be integers");
        return false;
      }
      const char* key_str = PyUnicode_AsUTF8(inner_key);
      int64_t val = PyLong_AsLongLong(inner_value);
      inner_map[key_str] = val;
    }
      config[outer_key] = inner_map;
    }
  return true;
}

static PyObject* py_compile_attn(PyObject* self, PyObject* args) {
  // bind compile_attn func
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) {
    return NULL;
  }
  if (!py_dict_to_config(py_config, config)) {
    return NULL;
  }
  std::string result = attention(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_mm(PyObject* self, PyObject* args) {
  // bind compile_attn func
  PyObject* py_shape;
  PyObject* py_config;
  if (!PyArg_ParseTuple(args, "OO", &py_shape, &py_config)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) {
    return NULL;
  }
  if (!py_dict_to_config(py_config, config)) {
    return NULL;
  }
  std::string result = matmul(shape, config);
  return PyUnicode_FromString(result.c_str());
}

static PyObject* set_platform(PyObject* self, PyObject* args) {
  int target = 0;
  char* platInfo = NULL;
  KernelCodeGen::Target enumTarget;
  if(PyArg_ParseTuple(args, "is", &target, &platInfo)){
    if(target == 2){
      enumTarget = KernelCodeGen::Target::ROCm;
    }
    else if(target == 1){
      enumTarget = KernelCodeGen::Target::CUDA;
    }
    else if(target == 3){
      enumTarget = KernelCodeGen::Target::Hanwuji;
    }
    else if(target == 4){
      enumTarget = KernelCodeGen::Target::Huawei;
    }
    else{
      std::cout << "DeepGen Error : Invalid Platform id " << target << std::endl;
      std::abort();
    }
    generator.setPaltform(enumTarget, std::string(platInfo));
  }
  return Py_None;
}

static PyObject* set_kernel_name(PyObject* self, PyObject* args) {
  char* kernelName = NULL;
  if(PyArg_ParseTuple(args, "s", &kernelName)){
    __GlobalKernelName = std::string(kernelName);
    if(__GlobalKernelName.size() <= 0){
      std::cout << "DeepGen Error : Invalid KernelName " << __GlobalKernelName << std::endl;
      std::abort();
    }
    else{
      std::cout << "[lib] setKernelNAme : "<< __GlobalKernelName << std::endl;
    }
  }
  return Py_None;
}
static PyObject* py_compile_attn_v2(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) {
    return NULL;
  }
  if (!py_dict_to_config(py_config, config)) {
    return NULL;
  }
  std::string result = attention_v2(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_attn_split_k1(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) return NULL;
  if (!py_dict_to_config(py_config, config)) return NULL;
  std::string result = attention_split_k1(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_attn_split_k2(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) return NULL;
  if (!py_dict_to_config(py_config, config)) return NULL;
  std::string result = attention_split_k2(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_gemma2_split_k1(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) return NULL;
  if (!py_dict_to_config(py_config, config)) return NULL;
  std::string result = gemma2_split_k1(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_gemma2_split_k2(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) return NULL;
  if (!py_dict_to_config(py_config, config)) return NULL;
  std::string result = gemma2_split_k2(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_h2o_split_k1(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) return NULL;
  if (!py_dict_to_config(py_config, config)) return NULL;
  std::string result = h2o_split_k1(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_h2o_split_k2(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) return NULL;
  if (!py_dict_to_config(py_config, config)) return NULL;
  std::string result = h2o_split_k2(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyObject* py_compile_h2o_split_k3(PyObject* self, PyObject* args) {
  PyObject* py_shape;
  PyObject* py_config;
  const char* dtype_str = "float32";
  if (!PyArg_ParseTuple(args, "OO|s", &py_shape, &py_config, &dtype_str)) {
    return NULL;
  }
  std::vector<int64_t> shape;
  TuneConfig config;
  if (!py_list_to_vector(py_shape, shape)) return NULL;
  if (!py_dict_to_config(py_config, config)) return NULL;
  std::string result = h2o_split_k3(shape, config, std::string(dtype_str));
  return PyUnicode_FromString(result.c_str());
}

static PyMethodDef DeepgenMethods[] = {
    {"compile_attn", py_compile_attn, METH_VARARGS, "Compile attention with given shape and config"},
    {"compile_attn_v2", py_compile_attn_v2, METH_VARARGS, "Compile optimized attention (scale before mm1, div after mm2)"},
    {"compile_mm", py_compile_mm, METH_VARARGS, "Compile matmul with given shape and config"},
    {"set_platform", set_platform, METH_VARARGS, "Set target platform and architecture"},
    {"set_kernel_name", set_kernel_name, METH_VARARGS, "Set global kernel name"},
    {"compile_attn_split_k1", py_compile_attn_split_k1, METH_VARARGS, "Compile split attention kernel 1 (GEMM + reduce -> em, denom)"},
    {"compile_attn_split_k2", py_compile_attn_split_k2, METH_VARARGS, "Compile split attention kernel 2 (GEMM + broadcast norm + GEMM -> O)"},
    {"compile_gemma2_split_k1", py_compile_gemma2_split_k1, METH_VARARGS, "Compile Gemma2 split kernel 1 (GEMM + softcap + mask + reduce -> em, denom)"},
    {"compile_gemma2_split_k2", py_compile_gemma2_split_k2, METH_VARARGS, "Compile Gemma2 split kernel 2 (GEMM + softcap + mask + broadcast norm + GEMM -> O)"},
    {"compile_h2o_split_k1", py_compile_h2o_split_k1, METH_VARARGS, "Compile H2O split kernel 1 (GEMM + mask + reduce -> em, denom)"},
    {"compile_h2o_split_k2", py_compile_h2o_split_k2, METH_VARARGS, "Compile H2O split kernel 2 (GEMM + mask + normalize + col reduce -> row_sum)"},
    {"compile_h2o_split_k3", py_compile_h2o_split_k3, METH_VARARGS, "Compile H2O split kernel 3 (GEMM + mask + broadcast norm + GEMM -> O)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef deepgenmodule = {
  PyModuleDef_HEAD_INIT,
  "deepgen",
  NULL,
  -1,
  DeepgenMethods
};

PyMODINIT_FUNC PyInit_deepgen(void) {
  return PyModule_Create(&deepgenmodule);
}
