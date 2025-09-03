module {
  func.func public @exp_(%arg0: memref<128x128xf32, 1>, %arg1: memref<128x128xf32, 1>) attributes {arg.tran = [true, false, false], func.op.type = "FlashAttn", func.output.arg.num = 1 : i32, func.state = "gpu", parallel.dim = ["y"]} {
    affine.parallel (%bx) = (0) to (4) {
      %0 = affine.apply affine_map<(d0) -> (d0 * 32)>(%bx) {apply.desc = "blocky"}
      affine.parallel (%tx) = (0) to (128) {
        %alloca_15 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tileP"} : memref<4x4xf32>

        affine.for %arg8 = 0 to 128 step 64 {

          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              %1 = affine.vector_load %arg0[%0 + (%tx floordiv 64) * 16 + ((%tx mod 64) floordiv 16) * 4 + %arg9, %arg8 + ((%tx mod 64) mod 16) * 4] : memref<128x128xf32, 1>, vector<4xf32>
              affine.vector_store %1, %alloca_15[%arg9, %arg10] : memref<4x4xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}

          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %alloca_15[%arg9, %arg10] : memref<4x4xf32>
              %4 = math.exp %1 : f32  // 替换为泰勒展开   e^x = 1 + sigma{n=1,2,...}(x^n / n!)
              // %c1 = arith.constant 1.0 : f32
              // %res = memref.alloca() : memref<1xf32>  // 结果 
              // %fenmu = memref.alloca() : memref<1xf32>  // n!
              // %fenzi = memref.alloca() : memref<1xf32>  // x^n
              // %counter = memref.alloca() : memref<1xf32> // n
              
              // affine.store %c1, %fenmu[0] : memref<1xf32>  // 初始化为1
              // affine.store %1, %fenzi[0] : memref<1xf32>  // 初始化为x
              // affine.store %c1, %res[0] : memref<1xf32>  // res 初始化为1
              // affine.store %c1, %counter[0] : memref<1xf32>  // counter 初始化为1
              // affine.for %iter = 1 to 10 {
              //   %r = affine.load %res[0] : memref<1xf32>
              //   %n = affine.load %counter[0] : memref<1xf32>
              //   %fz = affine.load %fenzi[0] : memref<1xf32>
              //   %fm = affine.load %fenmu[0] : memref<1xf32>

              //   %temp = arith.divf %fz, %fm : f32  // x^n / n!
              //   %new_fz = arith.mulf %fz, %1 : f32  // x^n * x = x^(n+1)
              //   %add_1 = arith.addf %n, %c1 : f32  // n+1
              //   %new_fm = arith.mulf %fm, %add_1 : f32  // n! * (n+1) = (n+1)!
              //   %new_res = arith.addf %r, %temp : f32

              //   affine.store %new_fz, %fenzi[0] : memref<1xf32>
              //   affine.store %new_fm, %fenmu[0] : memref<1xf32>
              //   affine.store %add_1, %counter[0] : memref<1xf32>
              //   affine.store %new_res, %res[0] : memref<1xf32>

              // }
              // %4 = affine.load %res[0] : memref<1xf32>
              affine.store %4, %alloca_15[%arg9, %arg10] : memref<4x4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttilex"}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttiley"}

          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              %1 = affine.vector_load %alloca_15[%arg9, %arg10] : memref<4x4xf32>, vector<4xf32>
              affine.vector_store %1, %arg1[%0 + (%tx floordiv 64) * 16 + ((%tx mod 64) floordiv 16) * 4 + %arg9, %arg8 + ((%tx mod 64) mod 16) * 4] : memref<128x128xf32, 1>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}
          
        } {for.desc = "blockx"}
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}