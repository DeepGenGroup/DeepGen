module {
  func.func public @broadcast(%arg0: memref<32x16xf32, 1>) {
    affine.parallel (%arg1) = (0) to (16) {
      affine.parallel (%arg2) = (0) to (32) {
        %c16_i32 = arith.constant 16 : i32
        %c0_i32 = arith.constant 0 : i32
        %0 = affine.load %arg0[%arg1 + (%arg2 floordiv 16) * 16, %arg2 mod 16] : memref<32x16xf32, 1>
        %shuffleResult, %valid = gpu.shuffle  idx %0, %c0_i32, %c16_i32 : f32
        affine.store %shuffleResult, %arg0[%arg1 + (%arg2 floordiv 16) * 16, %arg2 mod 16] : memref<32x16xf32, 1>
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}