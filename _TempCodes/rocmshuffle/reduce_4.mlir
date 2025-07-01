module {
  func.func public @reduce(%arg0: memref<16x4xf32, 1>, %arg1: memref<16x4xf32, 1>) {
    affine.parallel (%arg2) = (0) to (1) {
      affine.parallel (%arg3) = (0) to (64) {
        %0 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4] : memref<16x4xf32, 1>
        // %c4_i32 = arith.constant 4 : i32

        // %c1_i32 = arith.constant 1 : i32
        // %shuffleResult, %valid = gpu.shuffle  down %0, %c1_i32, %c4_i32 : f32
        // %1 = arith.addf %0, %shuffleResult : f32

        // %c2_i32 = arith.constant 2 : i32
        // %shuffleResult_1, %valid_1 = gpu.shuffle  down %1, %c2_i32, %c4_i32 : f32
        // %2 = arith.addf %1, %shuffleResult_1 : f32

        affine.store %0, %arg1[%arg3 floordiv 4, %arg3 mod 4] : memref<16x4xf32, 1>
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}