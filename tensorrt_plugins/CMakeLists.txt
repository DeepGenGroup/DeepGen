if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.15)
  project(DeepGenTensorRTAttentionPlugin LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  find_package(Python COMPONENTS Interpreter)
  get_filename_component(DEEPGEN_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
else()
  set(DEEPGEN_PROJECT_ROOT "${PROJECT_SOURCE_DIR}")
endif()

set(DEEPGEN_TRT_PLUGIN_TARGET deepgen_trt_attention_plugin)

set(TENSORRT_ROOT_DIR "" CACHE PATH "TensorRT root directory containing include/ and lib/")
set(TENSORRT_INCLUDE_DIR "" CACHE PATH "TensorRT include directory containing NvInfer.h")
set(TENSORRT_LIBRARY_DIR "" CACHE PATH "TensorRT library directory containing libnvinfer.so")

if(Python_EXECUTABLE)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import tensorrt, pathlib; print((pathlib.Path(tensorrt.__file__).resolve().parent.parent / 'tensorrt_libs').as_posix())"
    OUTPUT_VARIABLE TENSORRT_PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

if(CUDA_INCLUDE_DIR)
  get_filename_component(DEEPGEN_CUDA_ROOT "${CUDA_INCLUDE_DIR}" DIRECTORY)
else()
  if(DEFINED ENV{CUDA_HOME})
    set(DEEPGEN_CUDA_ROOT "$ENV{CUDA_HOME}")
    set(CUDA_INCLUDE_DIR "$ENV{CUDA_HOME}/include")
  elseif(EXISTS "$ENV{HOME}/.local/cuda-12.2/include")
    set(DEEPGEN_CUDA_ROOT "$ENV{HOME}/.local/cuda-12.2")
    set(CUDA_INCLUDE_DIR "$ENV{HOME}/.local/cuda-12.2/include")
  else()
    set(DEEPGEN_CUDA_ROOT "")
  endif()
endif()

find_path(
  DEEPGEN_TRT_INCLUDE_DIR
  NvInfer.h
  HINTS ${TENSORRT_INCLUDE_DIR} ${TENSORRT_ROOT_DIR}
  PATH_SUFFIXES include
)

if(NOT DEEPGEN_TRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT headers not found. Pass -DTENSORRT_INCLUDE_DIR=/path/to/include or -DTENSORRT_ROOT_DIR=/path/to/TensorRT.")
endif()

find_library(
  DEEPGEN_NVINFER_LIB
  nvinfer
  HINTS ${TENSORRT_LIBRARY_DIR} ${TENSORRT_ROOT_DIR} ${TENSORRT_PYTHON_LIB_DIR}
  PATH_SUFFIXES "" lib lib64 tensorrt_libs
)

find_library(
  DEEPGEN_NVINFER_PLUGIN_LIB
  nvinfer_plugin
  HINTS ${TENSORRT_LIBRARY_DIR} ${TENSORRT_ROOT_DIR} ${TENSORRT_PYTHON_LIB_DIR}
  PATH_SUFFIXES "" lib lib64 tensorrt_libs
)

if(NOT DEEPGEN_NVINFER_LIB)
  find_file(
    DEEPGEN_NVINFER_LIB
    NAMES libnvinfer.so.10 libnvinfer.so
    HINTS ${TENSORRT_LIBRARY_DIR} ${TENSORRT_ROOT_DIR} ${TENSORRT_PYTHON_LIB_DIR}
    PATH_SUFFIXES "" lib lib64 tensorrt_libs
  )
endif()

if(NOT DEEPGEN_NVINFER_PLUGIN_LIB)
  find_file(
    DEEPGEN_NVINFER_PLUGIN_LIB
    NAMES libnvinfer_plugin.so.10 libnvinfer_plugin.so
    HINTS ${TENSORRT_LIBRARY_DIR} ${TENSORRT_ROOT_DIR} ${TENSORRT_PYTHON_LIB_DIR}
    PATH_SUFFIXES "" lib lib64 tensorrt_libs
  )
endif()

find_library(
  DEEPGEN_CUDA_DRIVER_LIB
  cuda
  HINTS ${DEEPGEN_CUDA_ROOT}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs stubs
)

find_library(
  DEEPGEN_CUDART_LIB
  cudart
  HINTS ${DEEPGEN_CUDA_ROOT}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib
)

if(NOT DEEPGEN_NVINFER_LIB OR NOT DEEPGEN_NVINFER_PLUGIN_LIB)
  message(FATAL_ERROR "TensorRT runtime libraries not found. Pass -DTENSORRT_LIBRARY_DIR=/path/to/lib or ensure the Python TensorRT package is installed.")
endif()

if(NOT DEEPGEN_CUDA_DRIVER_LIB OR NOT DEEPGEN_CUDART_LIB)
  message(FATAL_ERROR "CUDA driver/runtime libraries not found. Check CUDA installation paths.")
endif()

add_library(${DEEPGEN_TRT_PLUGIN_TARGET} SHARED AttentionCorePlugin.cc)

target_include_directories(
  ${DEEPGEN_TRT_PLUGIN_TARGET}
  PRIVATE
    ${DEEPGEN_TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  ${DEEPGEN_TRT_PLUGIN_TARGET}
  PRIVATE
    ${DEEPGEN_NVINFER_LIB}
    ${DEEPGEN_NVINFER_PLUGIN_LIB}
    ${DEEPGEN_CUDA_DRIVER_LIB}
    ${DEEPGEN_CUDART_LIB}
    dl
)

set_target_properties(
  ${DEEPGEN_TRT_PLUGIN_TARGET}
  PROPERTIES
    OUTPUT_NAME "deepgen_trt_attention_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${DEEPGEN_PROJECT_ROOT}/bin"
)
