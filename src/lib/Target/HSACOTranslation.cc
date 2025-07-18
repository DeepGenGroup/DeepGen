#include "config.h"
#include "Target/HSACOTranslation.h"
#include <exception>
#include "KernelCodeGen.h"
#include <mutex>
#include <stack>
#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Bytecode/BytecodeWriter.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/SourceMgr.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/CodeGen.h"
#include <vector>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <iterator>
#include <fstream>
#include <iostream>
#include "Target/LLVMIRTranslation.h"
#include <cstdio>
// #ifdef USE_CUDA
// #include <cuda_runtime.h>
// #endif
using namespace mlir;

namespace KernelCodeGen
{

std::unique_ptr<llvm::TargetMachine>
initialize_module(llvm::Module *module, const std::string &triple,
                    const std::string &proc, const std::string &features)
{
    // verify and store llvm
    llvm::legacy::PassManager pm;
    pm.add(llvm::createVerifierPass());
    pm.run(*module);

    llvm::Triple tr;
    tr.setArch(llvm::Triple::ArchType::amdgcn);
    tr.setVendor(llvm::Triple::VendorType::AMD);
    tr.setOS(llvm::Triple::OSType::AMDHSA);
    // tr.setEnvironment(llvm::Triple::EnvironmentType::UnknownEnvironment);
    module->setTargetTriple(tr);

    std::string error;
    auto target =
        llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
    if (target == nullptr)
    {
        llvm::errs() << "LookupTarget fail: " << error << '\n';
        return nullptr;
    }
    llvm::TargetOptions opt;
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    opt.UnsafeFPMath = false;
    opt.NoInfsFPMath = false;
    opt.NoNaNsFPMath = true;
    llvm::TargetMachine *machine = target->createTargetMachine(
        module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
        std::nullopt, llvm::CodeGenOptLevel::Aggressive);

    module->setDataLayout(machine->createDataLayout());

    for (llvm::Function &f : module->functions())
        f.addFnAttr(llvm::Attribute::AlwaysInline);

    return std::unique_ptr<llvm::TargetMachine>(machine);
}

void init_llvm()
{
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
}

std::string generate_amdgcn_assembly(llvm::Module *module,
                                        const std::string &triple,
                                        const std::string &proc,
                                        const std::string &features)
{
    auto machine = initialize_module(module, triple, proc, features);

    if (machine == nullptr){
        assert(false && "generate_amdgcn_assembly error!");
        return "";
    }

    llvm::SmallVector<char, 0> buffer;
    llvm::legacy::PassManager pass;
    llvm::raw_svector_ostream stream(buffer);

    // emit
    machine->addPassesToEmitFile(pass, stream, nullptr,
                                    llvm::CodeGenFileType::AssemblyFile);
    pass.run(*module);
    std::string amdgcn(buffer.begin(), buffer.end());
    return amdgcn;
}

std::string generate_hsaco(llvm::Module *module, const std::string &triple,
                            const std::string &proc,
                            const std::string &features)
{
    auto machine = initialize_module(module, triple, proc, features);
    std::string dump_path = BC_DUMP_PATH;

    // create unique dir for kernel's binary and hsaco
    std::error_code ec;
    llvm::SmallString<128> fsrc;
    llvm::sys::fs::createTemporaryFile("kcg_kernel", "", fsrc);
    llvm::FileRemover remover(fsrc);
    std::filesystem::path kernel_dir{fsrc.data()};

    std::string kernel_name{fsrc};

    // Save GCN ISA binary.
    std::filesystem::path isa_binary(kernel_name + ".o");
    std::string isabin_path;
    // if (!dump_path.empty())
    //     isabin_path = (dump_path / isa_binary).string();
    // else
    isabin_path = isa_binary.string();
    std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
        new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
    if (ec)
    {
        llvm::errs() << isabin_path
                        << " was not created. error code: " << ec.category().name()
                        << ':' << ec.value() << '\n';
    }

    // Write out bitcode
    std::filesystem::path bitcode_filename(kernel_name + ".bc");
    std::string bitcode_path;
    // if (!dump_path.empty())
    //     bitcode_path = (dump_path / bitcode_filename).string();
    // else
    bitcode_path = bitcode_filename.string();
    std::unique_ptr<llvm::raw_fd_ostream> bitecode_fs(
        new llvm::raw_fd_ostream(bitcode_path, ec, llvm::sys::fs::OF_Text));
    if (ec)
    {
        llvm::errs() << bitcode_path
                        << " was not created. error code: " << ec.category().name()
                        << ':' << ec.value() << '\n';
    }

    llvm::WriteBitcodeToFile(*module, *bitecode_fs);

    // emit
    llvm::legacy::PassManager pass;
    machine->addPassesToEmitFile(pass, *isabin_fs, nullptr,
                                    llvm::CodeGenFileType::ObjectFile);
    pass.run(*module);

    // module->print(llvm::outs(), nullptr);

    // generate HASCO file
    std::filesystem::path hsaco(kernel_name + ".hsaco");
    std::string hsaco_path = hsaco.string();
    std::string error_message;

    // Check in triton/third_party/rocm/llvm/bin first.  For whls this will be the
    // correct location. If not found, go back to using ROCM_PATH or /opt/rocm
    static const auto this_library_path = []
    {
        Dl_info fileinfo;
        if (dladdr(reinterpret_cast<void *>(generate_hsaco), &fileinfo) == 0)
        {
            return std::filesystem::path();
        }
        return std::filesystem::path(fileinfo.dli_fname);
    }();

    std::string lld_path = USER_LLD_PATH;
    if (!std::filesystem::exists(lld_path))
    {
        std::string rocm_path = getenv("HIP_PATH");
        auto ROCM_DEFAULT_DIR = "/opt/dtk";
        lld_path = (rocm_path.empty()) ? ROCM_DEFAULT_DIR : rocm_path;
        lld_path += "/llvm/bin/ld.lld";
        if (!std::filesystem::exists(lld_path)){
            std::cout << "[FatalError] ld.lld not found" << std::endl;
            std::abort();
        }
    }

    int lld_result =
        llvm::sys::ExecuteAndWait(lld_path,
                                    {lld_path, "-flavor", "gnu",
                                    "-shared", "-o", hsaco_path, isabin_path},
                                    std::nullopt, {}, 0, 0, &error_message);
    if (lld_result)
    {
        llvm::errs() << "ld.lld execute fail: " << '\n'
                        << error_message << "Code: " << lld_result << '\n';
    }
    isabin_fs->close();
    bitecode_fs->close();
    if (remove(bitcode_path.c_str()) == 0) {
#ifdef KCG_DEBUG
        std::cout << "file deleted: " << bitcode_path << std::endl;
#endif
    } else {
        perror("file deleted error"); // 打印错误信息
    }
    if (remove(isabin_path.c_str()) == 0) {
#ifdef KCG_DEBUG
        std::cout << "file deleted: " << isabin_path << std::endl;
#endif
    } else {
        perror("file deleted error"); // 打印错误信息
    }

    return hsaco_path;
}

std::tuple<std::string, std::string>
llir_to_amdgcn_and_hsaco(llvm::Module *module, std::string gfx_arch,
                            std::string gfx_triple, std::string gfx_features)
{
    init_llvm();

    // verify and store llvm
    auto module_obj = llvm::CloneModule(*module);
    if (!module_obj) {
        llvm::errs() << "Error: clonging LLIR failed\n";
    }
    auto amdgcn = generate_amdgcn_assembly(module, gfx_triple, gfx_arch, gfx_features);
    auto hsaco_path = generate_hsaco(module_obj.get(), gfx_triple, gfx_arch, gfx_features);

    return std::make_tuple(amdgcn, hsaco_path);
}

std::tuple<std::string, std::string> translateLLVMIRToHSACO(
    const std::string llvmIR, 
    std::string gfx_arch, 
    std::string gfx_triple, 
    std::string gfx_features) 
{
    llvm::LLVMContext context;
    std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
    llvm::SMDiagnostic error;
    std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer->getMemBufferRef(), error, context);

    auto hsacoCode = llir_to_amdgcn_and_hsaco(module.get(), gfx_arch, gfx_triple, gfx_features);
    return hsacoCode;
}

std::string generateAmdgcnAndHsacoFromLLIRFile(
    const std::string &llvmIR,
    const std::string &gfx_arch,
    const std::string &gfx_triple,
    const std::string &gfx_features)
{
    auto ret = translateLLVMIRToHSACO(llvmIR, gfx_arch, gfx_triple, gfx_features);
    std::string amdgcn = std::get<0>(ret);
    std::string hsacoPath = std::get<1>(ret);

    std::string amdgcnPath{"/home/xushilong/DeepGen/_TempCodes/rocmshuffle/testgcn.s"};
    std::ofstream outasm(amdgcnPath);
    if (outasm.is_open()) {
        outasm << amdgcn;
        outasm.close();
        // std::cout << "write amdgcn success!" << std::endl;
    } else {
        // std::cout << "write amdgcn error!" << std::endl;
    }
    std::cout << "amdgcnpath=" << amdgcnPath << std::endl;
    // std::cout << "hsacopath=" << hsacoPath << std::endl;
    return hsacoPath;
}



}