// failed : 以下函数可以产出可执行文件，但运行时粗错：（貌似为 mlir_cuda_runtime 的问题）
// 'cuStreamSynchronize(stream)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
// 'cuStreamDestroy(stream)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
// 'cuModuleUnload(module)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
// Segmentation fault (core dumped

module attributes {gpu.container_module} {
  func.func @main() -> memref<1024x1024xf32,1> {
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc_A = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32, 1>
    %alloc_B = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32, 1>
    %alloc_C = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32, 1>

    gpu.launch_func  @main_graph_kernel::@GEMM_testKernel blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1)  args(%alloc_A : memref<1024x1024xf32,1>, %alloc_B : memref<1024x1024xf32,1>, %alloc_C : memref<1024x1024xf32,1>)
    return %alloc_C : memref<1024x1024xf32,1>
  }

  gpu.module @main_graph_kernel{
    memref.global "public" @kcg_shm0 : memref<8192xf32, 3> {alignment = 16 : i64}
    gpu.func @GEMM_testKernel(%arg0: memref<1024x1024xf32, 1>, %arg1: memref<1024x1024xf32, 1>, %arg2: memref<1024x1024xf32, 1>) kernel attributes {func.block.dim = array<i32: 128>, func.grid.dim = array<i32: 256>, func.op.name = "Matmul", func.state = "gpu"} 
    {
      %c7 = arith.constant 7 : index
      %c6 = arith.constant 6 : index
      %c5 = arith.constant 5 : index
      %c4 = arith.constant 4 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %block_id_x = gpu.block_id  x
      %0 = affine.apply affine_map<(d0) -> ((d0 floordiv 128) * 8 + d0 mod 8)>(%block_id_x)
      %1 = affine.apply affine_map<(d0) -> ((d0 mod 128) floordiv 8)>(%block_id_x)
      %2 = memref.get_global @kcg_shm0 : memref<8192xf32, 3>
      %thread_id_x = gpu.thread_id  x
      %alloca = memref.alloca() {alignment = 16 : i64} : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c0 * 8 + %c7 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c1 * 8 + %c7 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c2 * 8 + %c7 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c3 * 8 + %c7 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c4 * 8 + %c7 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c5 * 8 + %c7 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c6 * 8 + %c7 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c0 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c1 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c2 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c3 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c4 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c5 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c6 + 48] : memref<112xf32>
      affine.store %cst, %alloca[%c7 * 8 + %c7 + 48] : memref<112xf32>
      %3 = affine.vector_load %arg0[%thread_id_x floordiv 16 + %c0 * 8, %0 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      affine.vector_store %3, %2[(%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      %4 = affine.vector_load %arg0[%thread_id_x floordiv 16 + %c1 * 8, %0 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      affine.vector_store %4, %2[(%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      %5 = affine.vector_load %arg1[%thread_id_x floordiv 16 + %c0 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      affine.vector_store %5, %2[(%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
      %6 = affine.vector_load %arg1[%thread_id_x floordiv 16 + %c1 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      affine.vector_store %6, %2[(%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
      gpu.barrier
      %7 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c0 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c0 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %7, %alloca[%c0 * 4 + %c0 * 2 + 16] : memref<112xf32>, vector<2xf32>
      %8 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c0 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c1 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %8, %alloca[%c0 * 4 + %c1 * 2 + 16] : memref<112xf32>, vector<2xf32>
      %9 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c1 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c0 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %9, %alloca[%c1 * 4 + %c0 * 2 + 16] : memref<112xf32>, vector<2xf32>
      %10 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c1 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c1 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %10, %alloca[%c1 * 4 + %c1 * 2 + 16] : memref<112xf32>, vector<2xf32>
      %11 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c0 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c0 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %11, %alloca[%c0 * 4 + %c0 * 2 + 32] : memref<112xf32>, vector<2xf32>
      %12 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c0 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c1 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %12, %alloca[%c0 * 4 + %c1 * 2 + 32] : memref<112xf32>, vector<2xf32>
      %13 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c1 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c0 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %13, %alloca[%c1 * 4 + %c0 * 2 + 32] : memref<112xf32>, vector<2xf32>
      %14 = affine.vector_load %2[(%thread_id_x floordiv 64) * 64 + (%c1 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c1 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
      affine.vector_store %14, %alloca[%c1 * 4 + %c1 * 2 + 32] : memref<112xf32>, vector<2xf32>
      affine.for %arg3 = 16 to 1040 step 16 {
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg3) {
          %487 = affine.vector_load %arg0[%thread_id_x floordiv 16 + %c0 * 8 + %arg3, %0 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %487, %alloca[%c0 * 4] : memref<112xf32>, vector<4xf32>
          %488 = affine.vector_load %arg0[%thread_id_x floordiv 16 + %c1 * 8 + %arg3, %0 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %488, %alloca[%c1 * 4] : memref<112xf32>, vector<4xf32>
          %489 = affine.vector_load %arg1[%thread_id_x floordiv 16 + %c0 * 8 + %arg3, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %489, %alloca[%c0 * 4 + 8] : memref<112xf32>, vector<4xf32>
          %490 = affine.vector_load %arg1[%thread_id_x floordiv 16 + %c1 * 8 + %arg3, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %490, %alloca[%c1 * 4 + 8] : memref<112xf32>, vector<4xf32>
        }
        affine.for %arg4 = 0 to 14 step 2 {
          %487 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c0 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c0 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %487, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c0 * 4 + %c0 * 2 + 16] : memref<112xf32>, vector<2xf32>
          %488 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c0 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c1 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %488, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c0 * 4 + %c1 * 2 + 16] : memref<112xf32>, vector<2xf32>
          %489 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c1 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c0 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %489, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c1 * 4 + %c0 * 2 + 16] : memref<112xf32>, vector<2xf32>
          %490 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c1 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c1 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %490, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c1 * 4 + %c1 * 2 + 16] : memref<112xf32>, vector<2xf32>
          %491 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c0 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c0 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %491, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c0 * 4 + %c0 * 2 + 32] : memref<112xf32>, vector<2xf32>
          %492 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c0 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c1 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %492, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c0 * 4 + %c1 * 2 + 32] : memref<112xf32>, vector<2xf32>
          %493 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c1 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c0 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %493, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c1 * 4 + %c0 * 2 + 32] : memref<112xf32>, vector<2xf32>
          %494 = affine.vector_load %2[((%arg3 floordiv 16 - 1) mod 2) * 1024 + (%arg4 + %thread_id_x floordiv 64 + 2) * 64 + (%c1 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c1 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
          affine.vector_store %494, %alloca[((%arg4 floordiv 2 + 1) mod 2) * 8 + %c1 * 4 + %c1 * 2 + 32] : memref<112xf32>, vector<2xf32>
          %495 = affine.load %alloca[%c0 * 8 + %c0 + 48] : memref<112xf32>
          %496 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %497 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %498 = arith.mulf %496, %497 : f32
          %499 = arith.addf %498, %495 : f32
          affine.store %499, %alloca[%c0 * 8 + %c0 + 48] : memref<112xf32>
          %500 = affine.load %alloca[%c0 * 8 + %c1 + 48] : memref<112xf32>
          %501 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %502 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %503 = arith.mulf %501, %502 : f32
          %504 = arith.addf %503, %500 : f32
          affine.store %504, %alloca[%c0 * 8 + %c1 + 48] : memref<112xf32>
          %505 = affine.load %alloca[%c0 * 8 + %c2 + 48] : memref<112xf32>
          %506 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %507 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %508 = arith.mulf %506, %507 : f32
          %509 = arith.addf %508, %505 : f32
          affine.store %509, %alloca[%c0 * 8 + %c2 + 48] : memref<112xf32>
          %510 = affine.load %alloca[%c0 * 8 + %c3 + 48] : memref<112xf32>
          %511 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %512 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %513 = arith.mulf %511, %512 : f32
          %514 = arith.addf %513, %510 : f32
          affine.store %514, %alloca[%c0 * 8 + %c3 + 48] : memref<112xf32>
          %515 = affine.load %alloca[%c0 * 8 + %c4 + 48] : memref<112xf32>
          %516 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %517 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %518 = arith.mulf %516, %517 : f32
          %519 = arith.addf %518, %515 : f32
          affine.store %519, %alloca[%c0 * 8 + %c4 + 48] : memref<112xf32>
          %520 = affine.load %alloca[%c0 * 8 + %c5 + 48] : memref<112xf32>
          %521 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %522 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %523 = arith.mulf %521, %522 : f32
          %524 = arith.addf %523, %520 : f32
          affine.store %524, %alloca[%c0 * 8 + %c5 + 48] : memref<112xf32>
          %525 = affine.load %alloca[%c0 * 8 + %c6 + 48] : memref<112xf32>
          %526 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %527 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %528 = arith.mulf %526, %527 : f32
          %529 = arith.addf %528, %525 : f32
          affine.store %529, %alloca[%c0 * 8 + %c6 + 48] : memref<112xf32>
          %530 = affine.load %alloca[%c0 * 8 + %c7 + 48] : memref<112xf32>
          %531 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 16] : memref<112xf32>
          %532 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %533 = arith.mulf %531, %532 : f32
          %534 = arith.addf %533, %530 : f32
          affine.store %534, %alloca[%c0 * 8 + %c7 + 48] : memref<112xf32>
          %535 = affine.load %alloca[%c1 * 8 + %c0 + 48] : memref<112xf32>
          %536 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %537 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %538 = arith.mulf %536, %537 : f32
          %539 = arith.addf %538, %535 : f32
          affine.store %539, %alloca[%c1 * 8 + %c0 + 48] : memref<112xf32>
          %540 = affine.load %alloca[%c1 * 8 + %c1 + 48] : memref<112xf32>
          %541 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %542 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %543 = arith.mulf %541, %542 : f32
          %544 = arith.addf %543, %540 : f32
          affine.store %544, %alloca[%c1 * 8 + %c1 + 48] : memref<112xf32>
          %545 = affine.load %alloca[%c1 * 8 + %c2 + 48] : memref<112xf32>
          %546 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %547 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %548 = arith.mulf %546, %547 : f32
          %549 = arith.addf %548, %545 : f32
          affine.store %549, %alloca[%c1 * 8 + %c2 + 48] : memref<112xf32>
          %550 = affine.load %alloca[%c1 * 8 + %c3 + 48] : memref<112xf32>
          %551 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %552 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %553 = arith.mulf %551, %552 : f32
          %554 = arith.addf %553, %550 : f32
          affine.store %554, %alloca[%c1 * 8 + %c3 + 48] : memref<112xf32>
          %555 = affine.load %alloca[%c1 * 8 + %c4 + 48] : memref<112xf32>
          %556 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %557 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %558 = arith.mulf %556, %557 : f32
          %559 = arith.addf %558, %555 : f32
          affine.store %559, %alloca[%c1 * 8 + %c4 + 48] : memref<112xf32>
          %560 = affine.load %alloca[%c1 * 8 + %c5 + 48] : memref<112xf32>
          %561 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %562 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %563 = arith.mulf %561, %562 : f32
          %564 = arith.addf %563, %560 : f32
          affine.store %564, %alloca[%c1 * 8 + %c5 + 48] : memref<112xf32>
          %565 = affine.load %alloca[%c1 * 8 + %c6 + 48] : memref<112xf32>
          %566 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %567 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %568 = arith.mulf %566, %567 : f32
          %569 = arith.addf %568, %565 : f32
          affine.store %569, %alloca[%c1 * 8 + %c6 + 48] : memref<112xf32>
          %570 = affine.load %alloca[%c1 * 8 + %c7 + 48] : memref<112xf32>
          %571 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 16] : memref<112xf32>
          %572 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %573 = arith.mulf %571, %572 : f32
          %574 = arith.addf %573, %570 : f32
          affine.store %574, %alloca[%c1 * 8 + %c7 + 48] : memref<112xf32>
          %575 = affine.load %alloca[%c2 * 8 + %c0 + 48] : memref<112xf32>
          %576 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %577 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %578 = arith.mulf %576, %577 : f32
          %579 = arith.addf %578, %575 : f32
          affine.store %579, %alloca[%c2 * 8 + %c0 + 48] : memref<112xf32>
          %580 = affine.load %alloca[%c2 * 8 + %c1 + 48] : memref<112xf32>
          %581 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %582 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %583 = arith.mulf %581, %582 : f32
          %584 = arith.addf %583, %580 : f32
          affine.store %584, %alloca[%c2 * 8 + %c1 + 48] : memref<112xf32>
          %585 = affine.load %alloca[%c2 * 8 + %c2 + 48] : memref<112xf32>
          %586 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %587 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %588 = arith.mulf %586, %587 : f32
          %589 = arith.addf %588, %585 : f32
          affine.store %589, %alloca[%c2 * 8 + %c2 + 48] : memref<112xf32>
          %590 = affine.load %alloca[%c2 * 8 + %c3 + 48] : memref<112xf32>
          %591 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %592 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %593 = arith.mulf %591, %592 : f32
          %594 = arith.addf %593, %590 : f32
          affine.store %594, %alloca[%c2 * 8 + %c3 + 48] : memref<112xf32>
          %595 = affine.load %alloca[%c2 * 8 + %c4 + 48] : memref<112xf32>
          %596 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %597 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %598 = arith.mulf %596, %597 : f32
          %599 = arith.addf %598, %595 : f32
          affine.store %599, %alloca[%c2 * 8 + %c4 + 48] : memref<112xf32>
          %600 = affine.load %alloca[%c2 * 8 + %c5 + 48] : memref<112xf32>
          %601 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %602 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %603 = arith.mulf %601, %602 : f32
          %604 = arith.addf %603, %600 : f32
          affine.store %604, %alloca[%c2 * 8 + %c5 + 48] : memref<112xf32>
          %605 = affine.load %alloca[%c2 * 8 + %c6 + 48] : memref<112xf32>
          %606 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %607 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %608 = arith.mulf %606, %607 : f32
          %609 = arith.addf %608, %605 : f32
          affine.store %609, %alloca[%c2 * 8 + %c6 + 48] : memref<112xf32>
          %610 = affine.load %alloca[%c2 * 8 + %c7 + 48] : memref<112xf32>
          %611 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 16] : memref<112xf32>
          %612 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %613 = arith.mulf %611, %612 : f32
          %614 = arith.addf %613, %610 : f32
          affine.store %614, %alloca[%c2 * 8 + %c7 + 48] : memref<112xf32>
          %615 = affine.load %alloca[%c3 * 8 + %c0 + 48] : memref<112xf32>
          %616 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %617 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %618 = arith.mulf %616, %617 : f32
          %619 = arith.addf %618, %615 : f32
          affine.store %619, %alloca[%c3 * 8 + %c0 + 48] : memref<112xf32>
          %620 = affine.load %alloca[%c3 * 8 + %c1 + 48] : memref<112xf32>
          %621 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %622 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %623 = arith.mulf %621, %622 : f32
          %624 = arith.addf %623, %620 : f32
          affine.store %624, %alloca[%c3 * 8 + %c1 + 48] : memref<112xf32>
          %625 = affine.load %alloca[%c3 * 8 + %c2 + 48] : memref<112xf32>
          %626 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %627 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %628 = arith.mulf %626, %627 : f32
          %629 = arith.addf %628, %625 : f32
          affine.store %629, %alloca[%c3 * 8 + %c2 + 48] : memref<112xf32>
          %630 = affine.load %alloca[%c3 * 8 + %c3 + 48] : memref<112xf32>
          %631 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %632 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %633 = arith.mulf %631, %632 : f32
          %634 = arith.addf %633, %630 : f32
          affine.store %634, %alloca[%c3 * 8 + %c3 + 48] : memref<112xf32>
          %635 = affine.load %alloca[%c3 * 8 + %c4 + 48] : memref<112xf32>
          %636 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %637 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %638 = arith.mulf %636, %637 : f32
          %639 = arith.addf %638, %635 : f32
          affine.store %639, %alloca[%c3 * 8 + %c4 + 48] : memref<112xf32>
          %640 = affine.load %alloca[%c3 * 8 + %c5 + 48] : memref<112xf32>
          %641 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %642 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %643 = arith.mulf %641, %642 : f32
          %644 = arith.addf %643, %640 : f32
          affine.store %644, %alloca[%c3 * 8 + %c5 + 48] : memref<112xf32>
          %645 = affine.load %alloca[%c3 * 8 + %c6 + 48] : memref<112xf32>
          %646 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %647 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %648 = arith.mulf %646, %647 : f32
          %649 = arith.addf %648, %645 : f32
          affine.store %649, %alloca[%c3 * 8 + %c6 + 48] : memref<112xf32>
          %650 = affine.load %alloca[%c3 * 8 + %c7 + 48] : memref<112xf32>
          %651 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 16] : memref<112xf32>
          %652 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %653 = arith.mulf %651, %652 : f32
          %654 = arith.addf %653, %650 : f32
          affine.store %654, %alloca[%c3 * 8 + %c7 + 48] : memref<112xf32>
          %655 = affine.load %alloca[%c4 * 8 + %c0 + 48] : memref<112xf32>
          %656 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %657 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %658 = arith.mulf %656, %657 : f32
          %659 = arith.addf %658, %655 : f32
          affine.store %659, %alloca[%c4 * 8 + %c0 + 48] : memref<112xf32>
          %660 = affine.load %alloca[%c4 * 8 + %c1 + 48] : memref<112xf32>
          %661 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %662 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %663 = arith.mulf %661, %662 : f32
          %664 = arith.addf %663, %660 : f32
          affine.store %664, %alloca[%c4 * 8 + %c1 + 48] : memref<112xf32>
          %665 = affine.load %alloca[%c4 * 8 + %c2 + 48] : memref<112xf32>
          %666 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %667 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %668 = arith.mulf %666, %667 : f32
          %669 = arith.addf %668, %665 : f32
          affine.store %669, %alloca[%c4 * 8 + %c2 + 48] : memref<112xf32>
          %670 = affine.load %alloca[%c4 * 8 + %c3 + 48] : memref<112xf32>
          %671 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %672 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %673 = arith.mulf %671, %672 : f32
          %674 = arith.addf %673, %670 : f32
          affine.store %674, %alloca[%c4 * 8 + %c3 + 48] : memref<112xf32>
          %675 = affine.load %alloca[%c4 * 8 + %c4 + 48] : memref<112xf32>
          %676 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %677 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %678 = arith.mulf %676, %677 : f32
          %679 = arith.addf %678, %675 : f32
          affine.store %679, %alloca[%c4 * 8 + %c4 + 48] : memref<112xf32>
          %680 = affine.load %alloca[%c4 * 8 + %c5 + 48] : memref<112xf32>
          %681 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %682 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %683 = arith.mulf %681, %682 : f32
          %684 = arith.addf %683, %680 : f32
          affine.store %684, %alloca[%c4 * 8 + %c5 + 48] : memref<112xf32>
          %685 = affine.load %alloca[%c4 * 8 + %c6 + 48] : memref<112xf32>
          %686 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %687 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %688 = arith.mulf %686, %687 : f32
          %689 = arith.addf %688, %685 : f32
          affine.store %689, %alloca[%c4 * 8 + %c6 + 48] : memref<112xf32>
          %690 = affine.load %alloca[%c4 * 8 + %c7 + 48] : memref<112xf32>
          %691 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 16] : memref<112xf32>
          %692 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %693 = arith.mulf %691, %692 : f32
          %694 = arith.addf %693, %690 : f32
          affine.store %694, %alloca[%c4 * 8 + %c7 + 48] : memref<112xf32>
          %695 = affine.load %alloca[%c5 * 8 + %c0 + 48] : memref<112xf32>
          %696 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %697 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %698 = arith.mulf %696, %697 : f32
          %699 = arith.addf %698, %695 : f32
          affine.store %699, %alloca[%c5 * 8 + %c0 + 48] : memref<112xf32>
          %700 = affine.load %alloca[%c5 * 8 + %c1 + 48] : memref<112xf32>
          %701 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %702 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %703 = arith.mulf %701, %702 : f32
          %704 = arith.addf %703, %700 : f32
          affine.store %704, %alloca[%c5 * 8 + %c1 + 48] : memref<112xf32>
          %705 = affine.load %alloca[%c5 * 8 + %c2 + 48] : memref<112xf32>
          %706 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %707 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %708 = arith.mulf %706, %707 : f32
          %709 = arith.addf %708, %705 : f32
          affine.store %709, %alloca[%c5 * 8 + %c2 + 48] : memref<112xf32>
          %710 = affine.load %alloca[%c5 * 8 + %c3 + 48] : memref<112xf32>
          %711 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %712 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %713 = arith.mulf %711, %712 : f32
          %714 = arith.addf %713, %710 : f32
          affine.store %714, %alloca[%c5 * 8 + %c3 + 48] : memref<112xf32>
          %715 = affine.load %alloca[%c5 * 8 + %c4 + 48] : memref<112xf32>
          %716 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %717 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %718 = arith.mulf %716, %717 : f32
          %719 = arith.addf %718, %715 : f32
          affine.store %719, %alloca[%c5 * 8 + %c4 + 48] : memref<112xf32>
          %720 = affine.load %alloca[%c5 * 8 + %c5 + 48] : memref<112xf32>
          %721 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %722 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %723 = arith.mulf %721, %722 : f32
          %724 = arith.addf %723, %720 : f32
          affine.store %724, %alloca[%c5 * 8 + %c5 + 48] : memref<112xf32>
          %725 = affine.load %alloca[%c5 * 8 + %c6 + 48] : memref<112xf32>
          %726 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %727 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %728 = arith.mulf %726, %727 : f32
          %729 = arith.addf %728, %725 : f32
          affine.store %729, %alloca[%c5 * 8 + %c6 + 48] : memref<112xf32>
          %730 = affine.load %alloca[%c5 * 8 + %c7 + 48] : memref<112xf32>
          %731 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 16] : memref<112xf32>
          %732 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %733 = arith.mulf %731, %732 : f32
          %734 = arith.addf %733, %730 : f32
          affine.store %734, %alloca[%c5 * 8 + %c7 + 48] : memref<112xf32>
          %735 = affine.load %alloca[%c6 * 8 + %c0 + 48] : memref<112xf32>
          %736 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %737 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %738 = arith.mulf %736, %737 : f32
          %739 = arith.addf %738, %735 : f32
          affine.store %739, %alloca[%c6 * 8 + %c0 + 48] : memref<112xf32>
          %740 = affine.load %alloca[%c6 * 8 + %c1 + 48] : memref<112xf32>
          %741 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %742 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %743 = arith.mulf %741, %742 : f32
          %744 = arith.addf %743, %740 : f32
          affine.store %744, %alloca[%c6 * 8 + %c1 + 48] : memref<112xf32>
          %745 = affine.load %alloca[%c6 * 8 + %c2 + 48] : memref<112xf32>
          %746 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %747 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %748 = arith.mulf %746, %747 : f32
          %749 = arith.addf %748, %745 : f32
          affine.store %749, %alloca[%c6 * 8 + %c2 + 48] : memref<112xf32>
          %750 = affine.load %alloca[%c6 * 8 + %c3 + 48] : memref<112xf32>
          %751 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %752 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %753 = arith.mulf %751, %752 : f32
          %754 = arith.addf %753, %750 : f32
          affine.store %754, %alloca[%c6 * 8 + %c3 + 48] : memref<112xf32>
          %755 = affine.load %alloca[%c6 * 8 + %c4 + 48] : memref<112xf32>
          %756 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %757 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %758 = arith.mulf %756, %757 : f32
          %759 = arith.addf %758, %755 : f32
          affine.store %759, %alloca[%c6 * 8 + %c4 + 48] : memref<112xf32>
          %760 = affine.load %alloca[%c6 * 8 + %c5 + 48] : memref<112xf32>
          %761 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %762 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %763 = arith.mulf %761, %762 : f32
          %764 = arith.addf %763, %760 : f32
          affine.store %764, %alloca[%c6 * 8 + %c5 + 48] : memref<112xf32>
          %765 = affine.load %alloca[%c6 * 8 + %c6 + 48] : memref<112xf32>
          %766 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %767 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %768 = arith.mulf %766, %767 : f32
          %769 = arith.addf %768, %765 : f32
          affine.store %769, %alloca[%c6 * 8 + %c6 + 48] : memref<112xf32>
          %770 = affine.load %alloca[%c6 * 8 + %c7 + 48] : memref<112xf32>
          %771 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 16] : memref<112xf32>
          %772 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %773 = arith.mulf %771, %772 : f32
          %774 = arith.addf %773, %770 : f32
          affine.store %774, %alloca[%c6 * 8 + %c7 + 48] : memref<112xf32>
          %775 = affine.load %alloca[%c7 * 8 + %c0 + 48] : memref<112xf32>
          %776 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %777 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c0 + 32] : memref<112xf32>
          %778 = arith.mulf %776, %777 : f32
          %779 = arith.addf %778, %775 : f32
          affine.store %779, %alloca[%c7 * 8 + %c0 + 48] : memref<112xf32>
          %780 = affine.load %alloca[%c7 * 8 + %c1 + 48] : memref<112xf32>
          %781 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %782 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c1 + 32] : memref<112xf32>
          %783 = arith.mulf %781, %782 : f32
          %784 = arith.addf %783, %780 : f32
          affine.store %784, %alloca[%c7 * 8 + %c1 + 48] : memref<112xf32>
          %785 = affine.load %alloca[%c7 * 8 + %c2 + 48] : memref<112xf32>
          %786 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %787 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c2 + 32] : memref<112xf32>
          %788 = arith.mulf %786, %787 : f32
          %789 = arith.addf %788, %785 : f32
          affine.store %789, %alloca[%c7 * 8 + %c2 + 48] : memref<112xf32>
          %790 = affine.load %alloca[%c7 * 8 + %c3 + 48] : memref<112xf32>
          %791 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %792 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c3 + 32] : memref<112xf32>
          %793 = arith.mulf %791, %792 : f32
          %794 = arith.addf %793, %790 : f32
          affine.store %794, %alloca[%c7 * 8 + %c3 + 48] : memref<112xf32>
          %795 = affine.load %alloca[%c7 * 8 + %c4 + 48] : memref<112xf32>
          %796 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %797 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c4 + 32] : memref<112xf32>
          %798 = arith.mulf %796, %797 : f32
          %799 = arith.addf %798, %795 : f32
          affine.store %799, %alloca[%c7 * 8 + %c4 + 48] : memref<112xf32>
          %800 = affine.load %alloca[%c7 * 8 + %c5 + 48] : memref<112xf32>
          %801 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %802 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c5 + 32] : memref<112xf32>
          %803 = arith.mulf %801, %802 : f32
          %804 = arith.addf %803, %800 : f32
          affine.store %804, %alloca[%c7 * 8 + %c5 + 48] : memref<112xf32>
          %805 = affine.load %alloca[%c7 * 8 + %c6 + 48] : memref<112xf32>
          %806 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %807 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c6 + 32] : memref<112xf32>
          %808 = arith.mulf %806, %807 : f32
          %809 = arith.addf %808, %805 : f32
          affine.store %809, %alloca[%c7 * 8 + %c6 + 48] : memref<112xf32>
          %810 = affine.load %alloca[%c7 * 8 + %c7 + 48] : memref<112xf32>
          %811 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 16] : memref<112xf32>
          %812 = affine.load %alloca[((%arg4 floordiv 2) mod 2) * 8 + %c7 + 32] : memref<112xf32>
          %813 = arith.mulf %811, %812 : f32
          %814 = arith.addf %813, %810 : f32
          affine.store %814, %alloca[%c7 * 8 + %c7 + 48] : memref<112xf32>
        }
        affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg3) {
          %487 = affine.vector_load %alloca[%c0 * 4] : memref<112xf32>, vector<4xf32>
          affine.vector_store %487, %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
          %488 = affine.vector_load %alloca[%c1 * 4] : memref<112xf32>, vector<4xf32>
          affine.vector_store %488, %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
          %489 = affine.vector_load %alloca[%c0 * 4 + 8] : memref<112xf32>, vector<4xf32>
          affine.vector_store %489, %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
          %490 = affine.vector_load %alloca[%c1 * 4 + 8] : memref<112xf32>, vector<4xf32>
          affine.vector_store %490, %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
          gpu.barrier
        }
        %159 = affine.load %alloca[%c0 * 8 + %c0 + 48] : memref<112xf32>
        %160 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %161 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %162 = arith.mulf %160, %161 : f32
        %163 = arith.addf %162, %159 : f32
        affine.store %163, %alloca[%c0 * 8 + %c0 + 48] : memref<112xf32>
        %164 = affine.load %alloca[%c0 * 8 + %c1 + 48] : memref<112xf32>
        %165 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %166 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %167 = arith.mulf %165, %166 : f32
        %168 = arith.addf %167, %164 : f32
        affine.store %168, %alloca[%c0 * 8 + %c1 + 48] : memref<112xf32>
        %169 = affine.load %alloca[%c0 * 8 + %c2 + 48] : memref<112xf32>
        %170 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %171 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %172 = arith.mulf %170, %171 : f32
        %173 = arith.addf %172, %169 : f32
        affine.store %173, %alloca[%c0 * 8 + %c2 + 48] : memref<112xf32>
        %174 = affine.load %alloca[%c0 * 8 + %c3 + 48] : memref<112xf32>
        %175 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %176 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %177 = arith.mulf %175, %176 : f32
        %178 = arith.addf %177, %174 : f32
        affine.store %178, %alloca[%c0 * 8 + %c3 + 48] : memref<112xf32>
        %179 = affine.load %alloca[%c0 * 8 + %c4 + 48] : memref<112xf32>
        %180 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %181 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %182 = arith.mulf %180, %181 : f32
        %183 = arith.addf %182, %179 : f32
        affine.store %183, %alloca[%c0 * 8 + %c4 + 48] : memref<112xf32>
        %184 = affine.load %alloca[%c0 * 8 + %c5 + 48] : memref<112xf32>
        %185 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %186 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %187 = arith.mulf %185, %186 : f32
        %188 = arith.addf %187, %184 : f32
        affine.store %188, %alloca[%c0 * 8 + %c5 + 48] : memref<112xf32>
        %189 = affine.load %alloca[%c0 * 8 + %c6 + 48] : memref<112xf32>
        %190 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %191 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %192 = arith.mulf %190, %191 : f32
        %193 = arith.addf %192, %189 : f32
        affine.store %193, %alloca[%c0 * 8 + %c6 + 48] : memref<112xf32>
        %194 = affine.load %alloca[%c0 * 8 + %c7 + 48] : memref<112xf32>
        %195 = affine.load %alloca[%c0 + 24] : memref<112xf32>
        %196 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %197 = arith.mulf %195, %196 : f32
        %198 = arith.addf %197, %194 : f32
        affine.store %198, %alloca[%c0 * 8 + %c7 + 48] : memref<112xf32>
        %199 = affine.load %alloca[%c1 * 8 + %c0 + 48] : memref<112xf32>
        %200 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %201 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %202 = arith.mulf %200, %201 : f32
        %203 = arith.addf %202, %199 : f32
        affine.store %203, %alloca[%c1 * 8 + %c0 + 48] : memref<112xf32>
        %204 = affine.load %alloca[%c1 * 8 + %c1 + 48] : memref<112xf32>
        %205 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %206 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %207 = arith.mulf %205, %206 : f32
        %208 = arith.addf %207, %204 : f32
        affine.store %208, %alloca[%c1 * 8 + %c1 + 48] : memref<112xf32>
        %209 = affine.load %alloca[%c1 * 8 + %c2 + 48] : memref<112xf32>
        %210 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %211 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %212 = arith.mulf %210, %211 : f32
        %213 = arith.addf %212, %209 : f32
        affine.store %213, %alloca[%c1 * 8 + %c2 + 48] : memref<112xf32>
        %214 = affine.load %alloca[%c1 * 8 + %c3 + 48] : memref<112xf32>
        %215 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %216 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %217 = arith.mulf %215, %216 : f32
        %218 = arith.addf %217, %214 : f32
        affine.store %218, %alloca[%c1 * 8 + %c3 + 48] : memref<112xf32>
        %219 = affine.load %alloca[%c1 * 8 + %c4 + 48] : memref<112xf32>
        %220 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %221 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %222 = arith.mulf %220, %221 : f32
        %223 = arith.addf %222, %219 : f32
        affine.store %223, %alloca[%c1 * 8 + %c4 + 48] : memref<112xf32>
        %224 = affine.load %alloca[%c1 * 8 + %c5 + 48] : memref<112xf32>
        %225 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %226 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %227 = arith.mulf %225, %226 : f32
        %228 = arith.addf %227, %224 : f32
        affine.store %228, %alloca[%c1 * 8 + %c5 + 48] : memref<112xf32>
        %229 = affine.load %alloca[%c1 * 8 + %c6 + 48] : memref<112xf32>
        %230 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %231 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %232 = arith.mulf %230, %231 : f32
        %233 = arith.addf %232, %229 : f32
        affine.store %233, %alloca[%c1 * 8 + %c6 + 48] : memref<112xf32>
        %234 = affine.load %alloca[%c1 * 8 + %c7 + 48] : memref<112xf32>
        %235 = affine.load %alloca[%c1 + 24] : memref<112xf32>
        %236 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %237 = arith.mulf %235, %236 : f32
        %238 = arith.addf %237, %234 : f32
        affine.store %238, %alloca[%c1 * 8 + %c7 + 48] : memref<112xf32>
        %239 = affine.load %alloca[%c2 * 8 + %c0 + 48] : memref<112xf32>
        %240 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %241 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %242 = arith.mulf %240, %241 : f32
        %243 = arith.addf %242, %239 : f32
        affine.store %243, %alloca[%c2 * 8 + %c0 + 48] : memref<112xf32>
        %244 = affine.load %alloca[%c2 * 8 + %c1 + 48] : memref<112xf32>
        %245 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %246 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %247 = arith.mulf %245, %246 : f32
        %248 = arith.addf %247, %244 : f32
        affine.store %248, %alloca[%c2 * 8 + %c1 + 48] : memref<112xf32>
        %249 = affine.load %alloca[%c2 * 8 + %c2 + 48] : memref<112xf32>
        %250 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %251 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %252 = arith.mulf %250, %251 : f32
        %253 = arith.addf %252, %249 : f32
        affine.store %253, %alloca[%c2 * 8 + %c2 + 48] : memref<112xf32>
        %254 = affine.load %alloca[%c2 * 8 + %c3 + 48] : memref<112xf32>
        %255 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %256 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %257 = arith.mulf %255, %256 : f32
        %258 = arith.addf %257, %254 : f32
        affine.store %258, %alloca[%c2 * 8 + %c3 + 48] : memref<112xf32>
        %259 = affine.load %alloca[%c2 * 8 + %c4 + 48] : memref<112xf32>
        %260 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %261 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %262 = arith.mulf %260, %261 : f32
        %263 = arith.addf %262, %259 : f32
        affine.store %263, %alloca[%c2 * 8 + %c4 + 48] : memref<112xf32>
        %264 = affine.load %alloca[%c2 * 8 + %c5 + 48] : memref<112xf32>
        %265 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %266 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %267 = arith.mulf %265, %266 : f32
        %268 = arith.addf %267, %264 : f32
        affine.store %268, %alloca[%c2 * 8 + %c5 + 48] : memref<112xf32>
        %269 = affine.load %alloca[%c2 * 8 + %c6 + 48] : memref<112xf32>
        %270 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %271 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %272 = arith.mulf %270, %271 : f32
        %273 = arith.addf %272, %269 : f32
        affine.store %273, %alloca[%c2 * 8 + %c6 + 48] : memref<112xf32>
        %274 = affine.load %alloca[%c2 * 8 + %c7 + 48] : memref<112xf32>
        %275 = affine.load %alloca[%c2 + 24] : memref<112xf32>
        %276 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %277 = arith.mulf %275, %276 : f32
        %278 = arith.addf %277, %274 : f32
        affine.store %278, %alloca[%c2 * 8 + %c7 + 48] : memref<112xf32>
        %279 = affine.load %alloca[%c3 * 8 + %c0 + 48] : memref<112xf32>
        %280 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %281 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %282 = arith.mulf %280, %281 : f32
        %283 = arith.addf %282, %279 : f32
        affine.store %283, %alloca[%c3 * 8 + %c0 + 48] : memref<112xf32>
        %284 = affine.load %alloca[%c3 * 8 + %c1 + 48] : memref<112xf32>
        %285 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %286 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %287 = arith.mulf %285, %286 : f32
        %288 = arith.addf %287, %284 : f32
        affine.store %288, %alloca[%c3 * 8 + %c1 + 48] : memref<112xf32>
        %289 = affine.load %alloca[%c3 * 8 + %c2 + 48] : memref<112xf32>
        %290 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %291 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %292 = arith.mulf %290, %291 : f32
        %293 = arith.addf %292, %289 : f32
        affine.store %293, %alloca[%c3 * 8 + %c2 + 48] : memref<112xf32>
        %294 = affine.load %alloca[%c3 * 8 + %c3 + 48] : memref<112xf32>
        %295 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %296 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %297 = arith.mulf %295, %296 : f32
        %298 = arith.addf %297, %294 : f32
        affine.store %298, %alloca[%c3 * 8 + %c3 + 48] : memref<112xf32>
        %299 = affine.load %alloca[%c3 * 8 + %c4 + 48] : memref<112xf32>
        %300 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %301 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %302 = arith.mulf %300, %301 : f32
        %303 = arith.addf %302, %299 : f32
        affine.store %303, %alloca[%c3 * 8 + %c4 + 48] : memref<112xf32>
        %304 = affine.load %alloca[%c3 * 8 + %c5 + 48] : memref<112xf32>
        %305 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %306 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %307 = arith.mulf %305, %306 : f32
        %308 = arith.addf %307, %304 : f32
        affine.store %308, %alloca[%c3 * 8 + %c5 + 48] : memref<112xf32>
        %309 = affine.load %alloca[%c3 * 8 + %c6 + 48] : memref<112xf32>
        %310 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %311 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %312 = arith.mulf %310, %311 : f32
        %313 = arith.addf %312, %309 : f32
        affine.store %313, %alloca[%c3 * 8 + %c6 + 48] : memref<112xf32>
        %314 = affine.load %alloca[%c3 * 8 + %c7 + 48] : memref<112xf32>
        %315 = affine.load %alloca[%c3 + 24] : memref<112xf32>
        %316 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %317 = arith.mulf %315, %316 : f32
        %318 = arith.addf %317, %314 : f32
        affine.store %318, %alloca[%c3 * 8 + %c7 + 48] : memref<112xf32>
        %319 = affine.load %alloca[%c4 * 8 + %c0 + 48] : memref<112xf32>
        %320 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %321 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %322 = arith.mulf %320, %321 : f32
        %323 = arith.addf %322, %319 : f32
        affine.store %323, %alloca[%c4 * 8 + %c0 + 48] : memref<112xf32>
        %324 = affine.load %alloca[%c4 * 8 + %c1 + 48] : memref<112xf32>
        %325 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %326 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %327 = arith.mulf %325, %326 : f32
        %328 = arith.addf %327, %324 : f32
        affine.store %328, %alloca[%c4 * 8 + %c1 + 48] : memref<112xf32>
        %329 = affine.load %alloca[%c4 * 8 + %c2 + 48] : memref<112xf32>
        %330 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %331 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %332 = arith.mulf %330, %331 : f32
        %333 = arith.addf %332, %329 : f32
        affine.store %333, %alloca[%c4 * 8 + %c2 + 48] : memref<112xf32>
        %334 = affine.load %alloca[%c4 * 8 + %c3 + 48] : memref<112xf32>
        %335 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %336 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %337 = arith.mulf %335, %336 : f32
        %338 = arith.addf %337, %334 : f32
        affine.store %338, %alloca[%c4 * 8 + %c3 + 48] : memref<112xf32>
        %339 = affine.load %alloca[%c4 * 8 + %c4 + 48] : memref<112xf32>
        %340 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %341 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %342 = arith.mulf %340, %341 : f32
        %343 = arith.addf %342, %339 : f32
        affine.store %343, %alloca[%c4 * 8 + %c4 + 48] : memref<112xf32>
        %344 = affine.load %alloca[%c4 * 8 + %c5 + 48] : memref<112xf32>
        %345 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %346 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %347 = arith.mulf %345, %346 : f32
        %348 = arith.addf %347, %344 : f32
        affine.store %348, %alloca[%c4 * 8 + %c5 + 48] : memref<112xf32>
        %349 = affine.load %alloca[%c4 * 8 + %c6 + 48] : memref<112xf32>
        %350 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %351 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %352 = arith.mulf %350, %351 : f32
        %353 = arith.addf %352, %349 : f32
        affine.store %353, %alloca[%c4 * 8 + %c6 + 48] : memref<112xf32>
        %354 = affine.load %alloca[%c4 * 8 + %c7 + 48] : memref<112xf32>
        %355 = affine.load %alloca[%c4 + 24] : memref<112xf32>
        %356 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %357 = arith.mulf %355, %356 : f32
        %358 = arith.addf %357, %354 : f32
        affine.store %358, %alloca[%c4 * 8 + %c7 + 48] : memref<112xf32>
        %359 = affine.load %alloca[%c5 * 8 + %c0 + 48] : memref<112xf32>
        %360 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %361 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %362 = arith.mulf %360, %361 : f32
        %363 = arith.addf %362, %359 : f32
        affine.store %363, %alloca[%c5 * 8 + %c0 + 48] : memref<112xf32>
        %364 = affine.load %alloca[%c5 * 8 + %c1 + 48] : memref<112xf32>
        %365 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %366 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %367 = arith.mulf %365, %366 : f32
        %368 = arith.addf %367, %364 : f32
        affine.store %368, %alloca[%c5 * 8 + %c1 + 48] : memref<112xf32>
        %369 = affine.load %alloca[%c5 * 8 + %c2 + 48] : memref<112xf32>
        %370 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %371 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %372 = arith.mulf %370, %371 : f32
        %373 = arith.addf %372, %369 : f32
        affine.store %373, %alloca[%c5 * 8 + %c2 + 48] : memref<112xf32>
        %374 = affine.load %alloca[%c5 * 8 + %c3 + 48] : memref<112xf32>
        %375 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %376 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %377 = arith.mulf %375, %376 : f32
        %378 = arith.addf %377, %374 : f32
        affine.store %378, %alloca[%c5 * 8 + %c3 + 48] : memref<112xf32>
        %379 = affine.load %alloca[%c5 * 8 + %c4 + 48] : memref<112xf32>
        %380 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %381 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %382 = arith.mulf %380, %381 : f32
        %383 = arith.addf %382, %379 : f32
        affine.store %383, %alloca[%c5 * 8 + %c4 + 48] : memref<112xf32>
        %384 = affine.load %alloca[%c5 * 8 + %c5 + 48] : memref<112xf32>
        %385 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %386 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %387 = arith.mulf %385, %386 : f32
        %388 = arith.addf %387, %384 : f32
        affine.store %388, %alloca[%c5 * 8 + %c5 + 48] : memref<112xf32>
        %389 = affine.load %alloca[%c5 * 8 + %c6 + 48] : memref<112xf32>
        %390 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %391 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %392 = arith.mulf %390, %391 : f32
        %393 = arith.addf %392, %389 : f32
        affine.store %393, %alloca[%c5 * 8 + %c6 + 48] : memref<112xf32>
        %394 = affine.load %alloca[%c5 * 8 + %c7 + 48] : memref<112xf32>
        %395 = affine.load %alloca[%c5 + 24] : memref<112xf32>
        %396 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %397 = arith.mulf %395, %396 : f32
        %398 = arith.addf %397, %394 : f32
        affine.store %398, %alloca[%c5 * 8 + %c7 + 48] : memref<112xf32>
        %399 = affine.load %alloca[%c6 * 8 + %c0 + 48] : memref<112xf32>
        %400 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %401 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %402 = arith.mulf %400, %401 : f32
        %403 = arith.addf %402, %399 : f32
        affine.store %403, %alloca[%c6 * 8 + %c0 + 48] : memref<112xf32>
        %404 = affine.load %alloca[%c6 * 8 + %c1 + 48] : memref<112xf32>
        %405 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %406 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %407 = arith.mulf %405, %406 : f32
        %408 = arith.addf %407, %404 : f32
        affine.store %408, %alloca[%c6 * 8 + %c1 + 48] : memref<112xf32>
        %409 = affine.load %alloca[%c6 * 8 + %c2 + 48] : memref<112xf32>
        %410 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %411 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %412 = arith.mulf %410, %411 : f32
        %413 = arith.addf %412, %409 : f32
        affine.store %413, %alloca[%c6 * 8 + %c2 + 48] : memref<112xf32>
        %414 = affine.load %alloca[%c6 * 8 + %c3 + 48] : memref<112xf32>
        %415 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %416 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %417 = arith.mulf %415, %416 : f32
        %418 = arith.addf %417, %414 : f32
        affine.store %418, %alloca[%c6 * 8 + %c3 + 48] : memref<112xf32>
        %419 = affine.load %alloca[%c6 * 8 + %c4 + 48] : memref<112xf32>
        %420 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %421 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %422 = arith.mulf %420, %421 : f32
        %423 = arith.addf %422, %419 : f32
        affine.store %423, %alloca[%c6 * 8 + %c4 + 48] : memref<112xf32>
        %424 = affine.load %alloca[%c6 * 8 + %c5 + 48] : memref<112xf32>
        %425 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %426 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %427 = arith.mulf %425, %426 : f32
        %428 = arith.addf %427, %424 : f32
        affine.store %428, %alloca[%c6 * 8 + %c5 + 48] : memref<112xf32>
        %429 = affine.load %alloca[%c6 * 8 + %c6 + 48] : memref<112xf32>
        %430 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %431 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %432 = arith.mulf %430, %431 : f32
        %433 = arith.addf %432, %429 : f32
        affine.store %433, %alloca[%c6 * 8 + %c6 + 48] : memref<112xf32>
        %434 = affine.load %alloca[%c6 * 8 + %c7 + 48] : memref<112xf32>
        %435 = affine.load %alloca[%c6 + 24] : memref<112xf32>
        %436 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %437 = arith.mulf %435, %436 : f32
        %438 = arith.addf %437, %434 : f32
        affine.store %438, %alloca[%c6 * 8 + %c7 + 48] : memref<112xf32>
        %439 = affine.load %alloca[%c7 * 8 + %c0 + 48] : memref<112xf32>
        %440 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %441 = affine.load %alloca[%c0 + 40] : memref<112xf32>
        %442 = arith.mulf %440, %441 : f32
        %443 = arith.addf %442, %439 : f32
        affine.store %443, %alloca[%c7 * 8 + %c0 + 48] : memref<112xf32>
        %444 = affine.load %alloca[%c7 * 8 + %c1 + 48] : memref<112xf32>
        %445 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %446 = affine.load %alloca[%c1 + 40] : memref<112xf32>
        %447 = arith.mulf %445, %446 : f32
        %448 = arith.addf %447, %444 : f32
        affine.store %448, %alloca[%c7 * 8 + %c1 + 48] : memref<112xf32>
        %449 = affine.load %alloca[%c7 * 8 + %c2 + 48] : memref<112xf32>
        %450 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %451 = affine.load %alloca[%c2 + 40] : memref<112xf32>
        %452 = arith.mulf %450, %451 : f32
        %453 = arith.addf %452, %449 : f32
        affine.store %453, %alloca[%c7 * 8 + %c2 + 48] : memref<112xf32>
        %454 = affine.load %alloca[%c7 * 8 + %c3 + 48] : memref<112xf32>
        %455 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %456 = affine.load %alloca[%c3 + 40] : memref<112xf32>
        %457 = arith.mulf %455, %456 : f32
        %458 = arith.addf %457, %454 : f32
        affine.store %458, %alloca[%c7 * 8 + %c3 + 48] : memref<112xf32>
        %459 = affine.load %alloca[%c7 * 8 + %c4 + 48] : memref<112xf32>
        %460 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %461 = affine.load %alloca[%c4 + 40] : memref<112xf32>
        %462 = arith.mulf %460, %461 : f32
        %463 = arith.addf %462, %459 : f32
        affine.store %463, %alloca[%c7 * 8 + %c4 + 48] : memref<112xf32>
        %464 = affine.load %alloca[%c7 * 8 + %c5 + 48] : memref<112xf32>
        %465 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %466 = affine.load %alloca[%c5 + 40] : memref<112xf32>
        %467 = arith.mulf %465, %466 : f32
        %468 = arith.addf %467, %464 : f32
        affine.store %468, %alloca[%c7 * 8 + %c5 + 48] : memref<112xf32>
        %469 = affine.load %alloca[%c7 * 8 + %c6 + 48] : memref<112xf32>
        %470 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %471 = affine.load %alloca[%c6 + 40] : memref<112xf32>
        %472 = arith.mulf %470, %471 : f32
        %473 = arith.addf %472, %469 : f32
        affine.store %473, %alloca[%c7 * 8 + %c6 + 48] : memref<112xf32>
        %474 = affine.load %alloca[%c7 * 8 + %c7 + 48] : memref<112xf32>
        %475 = affine.load %alloca[%c7 + 24] : memref<112xf32>
        %476 = affine.load %alloca[%c7 + 40] : memref<112xf32>
        %477 = arith.mulf %475, %476 : f32
        %478 = arith.addf %477, %474 : f32
        affine.store %478, %alloca[%c7 * 8 + %c7 + 48] : memref<112xf32>
        %479 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c0 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c0 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %479, %alloca[%c0 * 4 + %c0 * 2 + 16] : memref<112xf32>, vector<2xf32>
        %480 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c0 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c1 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %480, %alloca[%c0 * 4 + %c1 * 2 + 16] : memref<112xf32>, vector<2xf32>
        %481 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c1 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c0 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %481, %alloca[%c1 * 4 + %c0 * 2 + 16] : memref<112xf32>, vector<2xf32>
        %482 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c1 + ((%thread_id_x mod 64) floordiv 32) floordiv 2) * 32 + (%c1 * 8 + (%thread_id_x mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %482, %alloca[%c1 * 4 + %c1 * 2 + 16] : memref<112xf32>, vector<2xf32>
        %483 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c0 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c0 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %483, %alloca[%c0 * 4 + %c0 * 2 + 32] : memref<112xf32>, vector<2xf32>
        %484 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c0 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c1 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %484, %alloca[%c0 * 4 + %c1 * 2 + 32] : memref<112xf32>, vector<2xf32>
        %485 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c1 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c0 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %485, %alloca[%c1 * 4 + %c0 * 2 + 32] : memref<112xf32>, vector<2xf32>
        %486 = affine.vector_load %2[((%arg3 floordiv 16) mod 2) * 1024 + (%thread_id_x floordiv 64) * 64 + (%c1 * 2 + ((%thread_id_x mod 64) floordiv 32) mod 2) * 16 + (%c1 * 4 + %thread_id_x mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
        affine.vector_store %486, %alloca[%c1 * 4 + %c1 * 2 + 32] : memref<112xf32>, vector<2xf32>
      }
      gpu.barrier
      %15 = affine.vector_load %alloca[(%c0 + %c0 + %c0) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %15, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %16 = affine.vector_load %alloca[(%c0 + %c0 + %c1) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %16, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %17 = affine.vector_load %alloca[(%c0 + %c0 + %c0) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %17, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %18 = affine.vector_load %alloca[(%c0 + %c0 + %c1) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %18, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %19 = affine.vector_load %alloca[(%c0 + %c2 + %c0) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %19, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %20 = affine.vector_load %alloca[(%c0 + %c2 + %c1) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %20, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %21 = affine.vector_load %alloca[(%c0 + %c2 + %c0) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %21, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %22 = affine.vector_load %alloca[(%c0 + %c2 + %c1) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %22, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %23 = affine.vector_load %alloca[(%c0 + %c0 + %c0) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %23, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %24 = affine.vector_load %alloca[(%c0 + %c0 + %c1) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %24, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %25 = affine.vector_load %alloca[(%c0 + %c0 + %c0) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %25, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %26 = affine.vector_load %alloca[(%c0 + %c0 + %c1) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %26, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %27 = affine.vector_load %alloca[(%c0 + %c2 + %c0) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %27, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %28 = affine.vector_load %alloca[(%c0 + %c2 + %c1) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %28, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %29 = affine.vector_load %alloca[(%c0 + %c2 + %c0) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %29, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %30 = affine.vector_load %alloca[(%c0 + %c2 + %c1) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %30, %2[(%thread_id_x floordiv 64) * 4096 + ((%c0 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %31 = affine.vector_load %alloca[(%c4 + %c0 + %c0) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %31, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %32 = affine.vector_load %alloca[(%c4 + %c0 + %c1) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %32, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %33 = affine.vector_load %alloca[(%c4 + %c0 + %c0) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %33, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %34 = affine.vector_load %alloca[(%c4 + %c0 + %c1) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %34, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %35 = affine.vector_load %alloca[(%c4 + %c2 + %c0) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %35, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %36 = affine.vector_load %alloca[(%c4 + %c2 + %c1) * 8 + %c0 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %36, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %37 = affine.vector_load %alloca[(%c4 + %c2 + %c0) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %37, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %38 = affine.vector_load %alloca[(%c4 + %c2 + %c1) * 8 + %c0 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %38, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c0 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %39 = affine.vector_load %alloca[(%c4 + %c0 + %c0) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %39, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %40 = affine.vector_load %alloca[(%c4 + %c0 + %c1) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %40, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %41 = affine.vector_load %alloca[(%c4 + %c0 + %c0) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %41, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %42 = affine.vector_load %alloca[(%c4 + %c0 + %c1) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %42, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c0 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %43 = affine.vector_load %alloca[(%c4 + %c2 + %c0) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %43, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %44 = affine.vector_load %alloca[(%c4 + %c2 + %c1) * 8 + %c4 + %c0 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %44, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c0 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %45 = affine.vector_load %alloca[(%c4 + %c2 + %c0) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %45, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c0) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      %46 = affine.vector_load %alloca[(%c4 + %c2 + %c1) * 8 + %c4 + %c2 + %c0 + 48] : memref<112xf32>, vector<2xf32>
      affine.vector_store %46, %2[(%thread_id_x floordiv 64) * 4096 + ((%c4 + (((%thread_id_x mod 64) floordiv 32) floordiv 2) * 4) * 8 + %c2 * 8 + ((%thread_id_x mod 32) floordiv 4) * 2 + %c1) * 64 + (%c4 * 2 + (((%thread_id_x mod 64) floordiv 32) mod 2) * 4) * 4 + %c2 * 4 + (%thread_id_x mod 4) * 2 + %c0] : memref<8192xf32, 3>, vector<2xf32>
      gpu.barrier
      %47 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %47, %alloca[%c0 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %48 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %48, %alloca[%c1 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %49 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c2 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %49, %alloca[%c2 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %50 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c3 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %50, %alloca[%c3 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %51 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c4 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %51, %alloca[%c4 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %52 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c5 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %52, %alloca[%c5 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %53 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c6 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %53, %alloca[%c6 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %54 = affine.vector_load %2[(%thread_id_x floordiv 16 + %c7 * 8) * 64 + (%thread_id_x mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
      affine.vector_store %54, %alloca[%c7 * 4 + 48] : memref<112xf32>, vector<4xf32>
      %55 = affine.load %alloca[%c0 * 4 + %c0 + 48] : memref<112xf32>
      %56 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %57 = arith.addf %55, %56 : f32
      affine.store %57, %alloca[%c0 * 4 + %c0 + 48] : memref<112xf32>
      %58 = affine.load %alloca[%c0 * 4 + %c1 + 48] : memref<112xf32>
      %59 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %60 = arith.addf %58, %59 : f32
      affine.store %60, %alloca[%c0 * 4 + %c1 + 48] : memref<112xf32>
      %61 = affine.load %alloca[%c0 * 4 + %c2 + 48] : memref<112xf32>
      %62 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %63 = arith.addf %61, %62 : f32
      affine.store %63, %alloca[%c0 * 4 + %c2 + 48] : memref<112xf32>
      %64 = affine.load %alloca[%c0 * 4 + %c3 + 48] : memref<112xf32>
      %65 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c0 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %66 = arith.addf %64, %65 : f32
      affine.store %66, %alloca[%c0 * 4 + %c3 + 48] : memref<112xf32>
      %67 = affine.load %alloca[%c1 * 4 + %c0 + 48] : memref<112xf32>
      %68 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %69 = arith.addf %67, %68 : f32
      affine.store %69, %alloca[%c1 * 4 + %c0 + 48] : memref<112xf32>
      %70 = affine.load %alloca[%c1 * 4 + %c1 + 48] : memref<112xf32>
      %71 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %72 = arith.addf %70, %71 : f32
      affine.store %72, %alloca[%c1 * 4 + %c1 + 48] : memref<112xf32>
      %73 = affine.load %alloca[%c1 * 4 + %c2 + 48] : memref<112xf32>
      %74 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %75 = arith.addf %73, %74 : f32
      affine.store %75, %alloca[%c1 * 4 + %c2 + 48] : memref<112xf32>
      %76 = affine.load %alloca[%c1 * 4 + %c3 + 48] : memref<112xf32>
      %77 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c1 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %78 = arith.addf %76, %77 : f32
      affine.store %78, %alloca[%c1 * 4 + %c3 + 48] : memref<112xf32>
      %79 = affine.load %alloca[%c2 * 4 + %c0 + 48] : memref<112xf32>
      %80 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c2 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %81 = arith.addf %79, %80 : f32
      affine.store %81, %alloca[%c2 * 4 + %c0 + 48] : memref<112xf32>
      %82 = affine.load %alloca[%c2 * 4 + %c1 + 48] : memref<112xf32>
      %83 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c2 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %84 = arith.addf %82, %83 : f32
      affine.store %84, %alloca[%c2 * 4 + %c1 + 48] : memref<112xf32>
      %85 = affine.load %alloca[%c2 * 4 + %c2 + 48] : memref<112xf32>
      %86 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c2 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %87 = arith.addf %85, %86 : f32
      affine.store %87, %alloca[%c2 * 4 + %c2 + 48] : memref<112xf32>
      %88 = affine.load %alloca[%c2 * 4 + %c3 + 48] : memref<112xf32>
      %89 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c2 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %90 = arith.addf %88, %89 : f32
      affine.store %90, %alloca[%c2 * 4 + %c3 + 48] : memref<112xf32>
      %91 = affine.load %alloca[%c3 * 4 + %c0 + 48] : memref<112xf32>
      %92 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c3 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %93 = arith.addf %91, %92 : f32
      affine.store %93, %alloca[%c3 * 4 + %c0 + 48] : memref<112xf32>
      %94 = affine.load %alloca[%c3 * 4 + %c1 + 48] : memref<112xf32>
      %95 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c3 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %96 = arith.addf %94, %95 : f32
      affine.store %96, %alloca[%c3 * 4 + %c1 + 48] : memref<112xf32>
      %97 = affine.load %alloca[%c3 * 4 + %c2 + 48] : memref<112xf32>
      %98 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c3 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %99 = arith.addf %97, %98 : f32
      affine.store %99, %alloca[%c3 * 4 + %c2 + 48] : memref<112xf32>
      %100 = affine.load %alloca[%c3 * 4 + %c3 + 48] : memref<112xf32>
      %101 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c3 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %102 = arith.addf %100, %101 : f32
      affine.store %102, %alloca[%c3 * 4 + %c3 + 48] : memref<112xf32>
      %103 = affine.load %alloca[%c4 * 4 + %c0 + 48] : memref<112xf32>
      %104 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c4 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %105 = arith.addf %103, %104 : f32
      affine.store %105, %alloca[%c4 * 4 + %c0 + 48] : memref<112xf32>
      %106 = affine.load %alloca[%c4 * 4 + %c1 + 48] : memref<112xf32>
      %107 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c4 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %108 = arith.addf %106, %107 : f32
      affine.store %108, %alloca[%c4 * 4 + %c1 + 48] : memref<112xf32>
      %109 = affine.load %alloca[%c4 * 4 + %c2 + 48] : memref<112xf32>
      %110 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c4 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %111 = arith.addf %109, %110 : f32
      affine.store %111, %alloca[%c4 * 4 + %c2 + 48] : memref<112xf32>
      %112 = affine.load %alloca[%c4 * 4 + %c3 + 48] : memref<112xf32>
      %113 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c4 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %114 = arith.addf %112, %113 : f32
      affine.store %114, %alloca[%c4 * 4 + %c3 + 48] : memref<112xf32>
      %115 = affine.load %alloca[%c5 * 4 + %c0 + 48] : memref<112xf32>
      %116 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c5 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %117 = arith.addf %115, %116 : f32
      affine.store %117, %alloca[%c5 * 4 + %c0 + 48] : memref<112xf32>
      %118 = affine.load %alloca[%c5 * 4 + %c1 + 48] : memref<112xf32>
      %119 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c5 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %120 = arith.addf %118, %119 : f32
      affine.store %120, %alloca[%c5 * 4 + %c1 + 48] : memref<112xf32>
      %121 = affine.load %alloca[%c5 * 4 + %c2 + 48] : memref<112xf32>
      %122 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c5 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %123 = arith.addf %121, %122 : f32
      affine.store %123, %alloca[%c5 * 4 + %c2 + 48] : memref<112xf32>
      %124 = affine.load %alloca[%c5 * 4 + %c3 + 48] : memref<112xf32>
      %125 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c5 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %126 = arith.addf %124, %125 : f32
      affine.store %126, %alloca[%c5 * 4 + %c3 + 48] : memref<112xf32>
      %127 = affine.load %alloca[%c6 * 4 + %c0 + 48] : memref<112xf32>
      %128 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c6 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %129 = arith.addf %127, %128 : f32
      affine.store %129, %alloca[%c6 * 4 + %c0 + 48] : memref<112xf32>
      %130 = affine.load %alloca[%c6 * 4 + %c1 + 48] : memref<112xf32>
      %131 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c6 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %132 = arith.addf %130, %131 : f32
      affine.store %132, %alloca[%c6 * 4 + %c1 + 48] : memref<112xf32>
      %133 = affine.load %alloca[%c6 * 4 + %c2 + 48] : memref<112xf32>
      %134 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c6 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %135 = arith.addf %133, %134 : f32
      affine.store %135, %alloca[%c6 * 4 + %c2 + 48] : memref<112xf32>
      %136 = affine.load %alloca[%c6 * 4 + %c3 + 48] : memref<112xf32>
      %137 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c6 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %138 = arith.addf %136, %137 : f32
      affine.store %138, %alloca[%c6 * 4 + %c3 + 48] : memref<112xf32>
      %139 = affine.load %alloca[%c7 * 4 + %c0 + 48] : memref<112xf32>
      %140 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c7 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c0] : memref<8192xf32, 3>
      %141 = arith.addf %139, %140 : f32
      affine.store %141, %alloca[%c7 * 4 + %c0 + 48] : memref<112xf32>
      %142 = affine.load %alloca[%c7 * 4 + %c1 + 48] : memref<112xf32>
      %143 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c7 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c1] : memref<8192xf32, 3>
      %144 = arith.addf %142, %143 : f32
      affine.store %144, %alloca[%c7 * 4 + %c1 + 48] : memref<112xf32>
      %145 = affine.load %alloca[%c7 * 4 + %c2 + 48] : memref<112xf32>
      %146 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c7 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c2] : memref<8192xf32, 3>
      %147 = arith.addf %145, %146 : f32
      affine.store %147, %alloca[%c7 * 4 + %c2 + 48] : memref<112xf32>
      %148 = affine.load %alloca[%c7 * 4 + %c3 + 48] : memref<112xf32>
      %149 = affine.load %2[%c1 * 4096 + (%thread_id_x floordiv 16 + %c7 * 8) * 64 + (%thread_id_x mod 16) * 4 + %c3] : memref<8192xf32, 3>
      %150 = arith.addf %148, %149 : f32
      affine.store %150, %alloca[%c7 * 4 + %c3 + 48] : memref<112xf32>
      %151 = affine.vector_load %alloca[%c0 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %151, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c0 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      %152 = affine.vector_load %alloca[%c1 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %152, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c1 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      %153 = affine.vector_load %alloca[%c2 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %153, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c2 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      %154 = affine.vector_load %alloca[%c3 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %154, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c3 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      %155 = affine.vector_load %alloca[%c4 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %155, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c4 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      %156 = affine.vector_load %alloca[%c5 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %156, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c5 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      %157 = affine.vector_load %alloca[%c6 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %157, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c6 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      %158 = affine.vector_load %alloca[%c7 * 4 + 48] : memref<112xf32>, vector<4xf32>
      affine.vector_store %158, %arg2[%0 * 64 + %thread_id_x floordiv 16 + %c7 * 8, %1 * 64 + (%thread_id_x mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
      gpu.return
    }
  }
}
