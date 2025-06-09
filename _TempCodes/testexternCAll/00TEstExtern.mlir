// failed : 以下函数可以产出可执行文件，但运行时粗错：（貌似为 mlir_cuda_runtime 的问题）
// 'cuStreamSynchronize(stream)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
// 'cuStreamDestroy(stream)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
// 'cuModuleUnload(module)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
// Segmentation fault (core dumped

module  {
  func.func private @say_hello() -> i32
  func.func @main() -> i32 {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc_A = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32, 1>
    %alloc_B = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32, 1>
    %alloc_C = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32, 1>
    %1 = func.call @say_hello() : ()->i32
    func.return %1 : i32
  }
}
