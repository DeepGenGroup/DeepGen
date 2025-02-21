#pragma once

#include "Common/Utils.h"

using namespace mlir;

namespace KernelCodeGen {
  std::pair<std::string, std::string> generatePTXAndCubinFromLLIRFile(const std::string llvmIR, int capability, int version);

}