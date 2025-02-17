#pragma once

#include "Common/Utils.h"

using namespace mlir;

namespace KernelCodeGen {
std::string generateAmdgcnAndHsacoFromLLIRFile(
        const std::string module,
        const std::string& gfx_arch,
        const std::string& gfx_triple,
        const std::string& gfx_features);

std::tuple<std::string, std::string> translateLLVMIRToHSACO(
        const std::string llvmIR, 
        std::string gfx_arch, 
        std::string gfx_triple, 
        std::string gfx_features);

}