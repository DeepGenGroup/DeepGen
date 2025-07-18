# project config
###################################################################
cmake_minimum_required(VERSION 3.15.0)
project(KernelCodeGen LANGUAGES CXX C)    # delete CUDA
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_COMPILER /usr/bin/gcc)
############################ User config #####################
set(LLVM_INSTALL_DIR "/home/xushilong/llvm-install")
set(DEBUG_AMDGCN_OUTPUT_PATH "/home/xushilong/DeepGen/test.amdgcn") 
# set(USER_LLD_PATH "${CMAKE_SOURCE_DIR}/third_party/hip/bin/ld.lld") 
# After change llvm repo, the ld.lld must be compiled from llvm-project. Default ld in dtk is not available
set(USER_LLD_PATH "${LLVM_INSTALL_DIR}/bin/ld.lld") 
set(USER_PTXAS_PATH "/home/xushilong/anaconda3/bin//ptxas") 
set(CUDA_CAP        80)
set(PTXAS_VERSION   82)
set(CUDA_INCLUDE_DIR "/home/xushilong/anaconda3/include")
option(USE_STABLEHLO_EMBEDDED "Use embedded stableHLO(ON) or external stableHLO(OFF)" ON)

#############################################################
find_package(Python COMPONENTS Interpreter Development)
# set(pybind11_DIR ${PYBIND11_INSTALL_DIR}/share/cmake/pybind11)
# find_package(pybind11 REQUIRED)
if(pybind11_FOUND)
  message("==== Found pybind11 : ${pybind11_INCLUDE_DIRS}")
endif()
if(Python_FOUND)
    message(STATUS "=== Found Python: ${Python_EXECUTABLE}")
    message(STATUS "=== Python Include: ${Python_INCLUDE_DIRS}")
    message(STATUS "=== Python Libraries: ${Python_LIBRARIES}")
endif()

# # 查找 Python
# find_package(PythonInterp REQUIRED)
# find_package(PythonLibs REQUIRED)

# # 获取 Python 版本信息
# execute_process(
#     COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
#     OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# message(STATUS "===== Found PythonIncludeDIr ${PYTHON_INCLUDE_DIR}")
# message(STATUS "===== Found PyLib ${Python_LIBRARIES}")

option(COMPILE_AS_PYMODULE "Compile kcg_compiler to DynamicLib or Exe" ON)
# close some warnings
add_compile_options(
  -Wno-maybe-uninitialized
  -Wno-unused-function
  -Wno-unused-variable
  -Wno-unused-result
  -Wno-sign-compare
  -Wno-unused-but-set-variable
  -Wno-return-local-addr
  -Wno-parentheses
  -Wno-cast-qual
  -Wno-unused-but-set-parameter
  -Wno-deprecated-declarations
  -Wno-unused-value
  )

##########################################################################
  
set(BC_DUMP_PATH ${CMAKE_SOURCE_DIR}/_dump)
set(HIP_BITCODE_PATH ${CMAKE_SOURCE_DIR}/third_party/hip/bitcode)
set(CUDA_BITCODE_PATH ${CMAKE_SOURCE_DIR}/third_party/cuda/lib)

# generate config.h with user-config variables
configure_file(config.h.in ${CMAKE_SOURCE_DIR}/include/config.h)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG HINTS ${LLVM_INSTALL_DIR}/lib/cmake)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "" )
set(CMAKE_BUILD_TYPE "Debug")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "==== enable debug_log =======")
  add_definitions(-DKCG_DEBUG)
  add_compile_options(-g)
else()
  message(STATUS "==== disable debug_log =======")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(KernelCodeGen_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(KernelCodeGen_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${KernelCodeGen_SOURCE_DIR}/include)
include_directories(${KernelCodeGen_BINARY_DIR}/include)
include_directories(${KernelCodeGen_BINARY_DIR})
include_directories(${KernelCodeGen_SOURCE_DIR}/build/include)

# link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${LLVM_INSTALL_DIR}/lib)
add_definitions(${LLVM_DEFINITIONS})
add_definitions()
message("add-def:${LLVM_DEFINITIONS}")



# Configure CMake.
list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build_tools/cmake)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)

# Use embedded stablehlo
# if (USE_STABLEHLO_EMBEDDED)
#   set(STABLEHLO_BUILD_EMBEDDED ON)
#   set(STABLEHLO_ENABLE_BINDINGS_PYTHON ON)
#   add_subdirectory(${KernelCodeGen_SOURCE_DIR}/externals/rocm-stablehlo
#     ${KernelCodeGen_BINARY_DIR}/rocm-stablehlo
#     EXCLUDE_FROM_ALL)
#   include_directories(${CMAKE_CURRENT_SOURCE_DIR}/externals/rocm-stablehlo)
# endif()

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

if(ENABLE_DEBUG_LOG)
  message("==== Enable debug log ====")
  add_definitions(-DKCG_DEBUG)
endif()

# add_subdirectory(include)
add_subdirectory(src/lib)
add_subdirectory(src)
