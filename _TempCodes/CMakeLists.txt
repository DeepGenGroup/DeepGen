project(caller)  # output.so 的调用者。包含main函数

link_directories("/home/xushilong/rocm-llvm-install/lib")
link_libraries(dl mlir_cuda_runtime)
add_executable(${PROJECT_NAME} main.cpp)
