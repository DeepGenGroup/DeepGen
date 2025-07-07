module {
  func.func public @reduce(%arg0: memref<8x16xf32, 1>, %arg1: memref<8x16xf32, 1>) {
    affine.parallel (%bx) = (0) to (2) {
      affine.parallel (%tx) = (0) to (64) {
        %0 = affine.load %arg0[%bx * 4 + %tx floordiv 16, %tx mod 16] : memref<8x16xf32, 1>
        %cgroup_i32 = arith.constant 64 : i32

        %c1_i32 = arith.constant 1 : i32
        %shuffleResult, %valid = gpu.shuffle  down %0, %c1_i32, %cgroup_i32 : f32
        %1 = arith.addf %0, %shuffleResult : f32

        %c2_i32 = arith.constant 2 : i32
        %shuffleResult_1, %valid_1 = gpu.shuffle  down %1, %c2_i32, %cgroup_i32 : f32
        %2 = arith.addf %1, %shuffleResult_1 : f32
        // arith.mulf 

        %c4_i32 = arith.constant 4 : i32
        %shuffleResult_2, %valid_2 = gpu.shuffle  down %2, %c4_i32, %cgroup_i32 : f32
        %3 = arith.addf %2, %shuffleResult_2 : f32

        %c8_i32 = arith.constant 8 : i32
        %shuffleResult_3, %valid_3 = gpu.shuffle  down %3, %c8_i32, %cgroup_i32 : f32
        %4 = arith.addf %3, %shuffleResult_3 : f32

        // %c16_i32 = arith.constant 16 : i32
        // %shuffleResult_4, %valid_4 = gpu.shuffle  down %4, %c16_i32, %cgroup_i32 : f32
        // %5 = arith.addf %4, %shuffleResult_4 : f32
        
        affine.store %4, %arg1[%bx * 4 + %tx floordiv 16, %tx mod 16] : memref<8x16xf32, 1>
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}