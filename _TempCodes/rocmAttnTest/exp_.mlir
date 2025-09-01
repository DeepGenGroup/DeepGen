module {
  func.func public @exp_(%arg0: memref<128x128xf32, 1>, %arg1: memref<128x128xf32, 1>) attributes {arg.tran = [true, false, false], func.op.type = "FlashAttn", func.output.arg.num = 1 : i32, func.state = "gpu", parallel.dim = ["y"]} {
    affine.parallel (%arg4) = (0) to (4) {
      %0 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg4) {apply.desc = "blocky"}
      affine.parallel (%arg7) = (0) to (128) {
        %alloca_15 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tileP"} : memref<4x4xf32>

        affine.for %arg8 = 0 to 128 step 64 {

          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              %1 = affine.vector_load %arg0[%0 + (%arg7 floordiv 64) * 16 + ((%arg7 mod 64) floordiv 16) * 4 + %arg9, %arg8 + ((%arg7 mod 64) mod 16) * 4] : memref<128x128xf32, 1>, vector<4xf32>
              affine.vector_store %1, %alloca_15[%arg9, %arg10] : memref<4x4xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}

          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %alloca_15[%arg9, %arg10] : memref<4x4xf32>
              %4 = math.exp %1 : f32
              affine.store %4, %alloca_15[%arg9, %arg10] : memref<4x4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttilex"}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttiley"}

          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              %1 = affine.vector_load %alloca_15[%arg9, %arg10] : memref<4x4xf32>, vector<4xf32>
              affine.vector_store %1, %arg1[%0 + (%arg7 floordiv 64) * 16 + ((%arg7 mod 64) floordiv 16) * 4 + %arg9, %arg8 + ((%arg7 mod 64) mod 16) * 4] : memref<128x128xf32, 1>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}
          
        } {for.desc = "blockx"}
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}