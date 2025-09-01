module {
  func.func public @softmax(%arg0: memref<128x128xf32, 1>, %arg1: memref<128x128xf32, 1>) attributes {arg.tran = [true, false, false], func.op.type = "FlashAttn", func.output.arg.num = 1 : i32, func.state = "gpu", parallel.dim = ["y"]} {
    affine.parallel (%arg4) = (0) to (4) {
      %smMax = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smMax"} : memref<32xf32, 3>
      %smSum = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smSum"} : memref<32xf32, 3>
      %by = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg4) {apply.desc = "blocky"}
      affine.parallel (%tid) = (0) to (128) {
        %regMax = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regMax"} : memref<4xf32>
        %regSum = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regSum"} : memref<4xf32>
        %tileP = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tileP"} : memref<4x4xf32>
        
        affine.for %arg8 = 0 to 32 step 128 {
          affine.if affine_set<(d0, d1) : (-d0 - d1 + 31 >= 0)>(%tid, %arg8) {
            %cst = arith.constant 0xFF800000 : f32
            affine.store %cst, %smMax[%arg8 + %tid] : memref<32xf32, 3>
            %cst_18 = arith.constant 0.000000e+00 : f32
            affine.store %cst_18, %smSum[%arg8 + %tid] : memref<32xf32, 3>
          }
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}

        // compute max and sum
        affine.for %bx = 0 to 128 step 64 {
          // init
          affine.for %arg9 = 0 to 4 {
            %cst = arith.constant 0xFF800000 : f32
            affine.store %cst, %regMax[%arg9] : memref<4xf32>
            %cst_18 = arith.constant 0.000000e+00 : f32
            affine.store %cst_18, %regSum[%arg9] : memref<4xf32>
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}
          // load
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              %1 = affine.vector_load %arg0[%by + (%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9, %bx + ((%tid mod 64) mod 16) * 4] : memref<128x128xf32, 1>, vector<4xf32>
              affine.vector_store %1, %tileP[%arg9, %arg10] : memref<4x4xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}
          // thread level
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %tileP[%arg9, %arg10] : memref<4x4xf32>
              %2 = affine.load %regMax[%arg9] : memref<4xf32>
              %3 = affine.load %regSum[%arg9] : memref<4xf32>
              %4 = arith.maxnumf %2, %1 : f32
              %5 = arith.subf %2, %4 : f32
              %6 = math.exp %5 : f32
              %7 = arith.mulf %6, %3 : f32
              %8 = arith.subf %1, %4 : f32
              %9 = math.exp %8 : f32
              %10 = arith.addf %7, %9 : f32
              affine.store %4, %regMax[%arg9] : memref<4xf32>
              affine.store %10, %regSum[%arg9] : memref<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttilexDown"}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttileyDown"}
          // warp level
          affine.for %arg9 = 0 to 4 {
            %c16_i32 = arith.constant 16 : i32
            %c1_i32 = arith.constant 1 : i32
            %1 = affine.load %regMax[%arg9] : memref<4xf32>
            %shuffleResult, %valid = gpu.shuffle  down %1, %c1_i32, %c16_i32 : f32
            %2 = affine.load %regSum[%arg9] : memref<4xf32>
            %shuffleResult_18, %valid_19 = gpu.shuffle  down %2, %c1_i32, %c16_i32 : f32
            %3 = arith.maxnumf %1, %shuffleResult : f32
            %4 = arith.subf %1, %3 : f32
            %5 = math.exp %4 : f32
            %6 = arith.subf %shuffleResult, %3 : f32
            %7 = math.exp %6 : f32
            %8 = arith.mulf %2, %5 : f32
            %9 = arith.mulf %shuffleResult_18, %7 : f32
            %10 = arith.addf %8, %9 : f32
            affine.store %3, %regMax[%arg9] : memref<4xf32>
            affine.store %10, %regSum[%arg9] : memref<4xf32>
            %c2_i32 = arith.constant 2 : i32
            %11 = affine.load %regMax[%arg9] : memref<4xf32>
            %shuffleResult_20, %valid_21 = gpu.shuffle  down %11, %c2_i32, %c16_i32 : f32
            %12 = affine.load %regSum[%arg9] : memref<4xf32>
            %shuffleResult_22, %valid_23 = gpu.shuffle  down %12, %c2_i32, %c16_i32 : f32
            %13 = arith.maxnumf %11, %shuffleResult_20 : f32
            %14 = arith.subf %11, %13 : f32
            %15 = math.exp %14 : f32
            %16 = arith.subf %shuffleResult_20, %13 : f32
            %17 = math.exp %16 : f32
            %18 = arith.mulf %12, %15 : f32
            %19 = arith.mulf %shuffleResult_22, %17 : f32
            %20 = arith.addf %18, %19 : f32
            affine.store %13, %regMax[%arg9] : memref<4xf32>
            affine.store %20, %regSum[%arg9] : memref<4xf32>
            %c4_i32 = arith.constant 4 : i32
            %21 = affine.load %regMax[%arg9] : memref<4xf32>
            %shuffleResult_24, %valid_25 = gpu.shuffle  down %21, %c4_i32, %c16_i32 : f32
            %22 = affine.load %regSum[%arg9] : memref<4xf32>
            %shuffleResult_26, %valid_27 = gpu.shuffle  down %22, %c4_i32, %c16_i32 : f32
            %23 = arith.maxnumf %21, %shuffleResult_24 : f32
            %24 = arith.subf %21, %23 : f32
            %25 = math.exp %24 : f32
            %26 = arith.subf %shuffleResult_24, %23 : f32
            %27 = math.exp %26 : f32
            %28 = arith.mulf %22, %25 : f32
            %29 = arith.mulf %shuffleResult_26, %27 : f32
            %30 = arith.addf %28, %29 : f32
            affine.store %23, %regMax[%arg9] : memref<4xf32>
            affine.store %30, %regSum[%arg9] : memref<4xf32>
            %c8_i32 = arith.constant 8 : i32
            %31 = affine.load %regMax[%arg9] : memref<4xf32>
            %shuffleResult_28, %valid_29 = gpu.shuffle  down %31, %c8_i32, %c16_i32 : f32
            %32 = affine.load %regSum[%arg9] : memref<4xf32>
            %shuffleResult_30, %valid_31 = gpu.shuffle  down %32, %c8_i32, %c16_i32 : f32
            %33 = arith.maxnumf %31, %shuffleResult_28 : f32
            %34 = arith.subf %31, %33 : f32
            %35 = math.exp %34 : f32
            %36 = arith.subf %shuffleResult_28, %33 : f32
            %37 = math.exp %36 : f32
            %38 = arith.mulf %32, %35 : f32
            %39 = arith.mulf %shuffleResult_30, %37 : f32
            %40 = arith.addf %38, %39 : f32
            affine.store %33, %regMax[%arg9] : memref<4xf32>
            affine.store %40, %regSum[%arg9] : memref<4xf32>
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          // block
          affine.if affine_set<(d0) : (d0 mod 16 == 0)>(%tid) {
            affine.for %arg9 = 0 to 4 {
              %1 = affine.load %smMax[(%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9] : memref<32xf32, 3>
              %2 = affine.load %regMax[%arg9] : memref<4xf32>
              %3 = affine.load %smSum[(%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9] : memref<32xf32, 3>
              %4 = affine.load %regSum[%arg9] : memref<4xf32>
              %5 = arith.maxnumf %2, %1 : f32
              %6 = arith.subf %2, %5 : f32
              %7 = math.exp %6 : f32
              %8 = arith.subf %1, %5 : f32
              %9 = math.exp %8 : f32
              %10 = arith.mulf %4, %7 : f32
              %11 = arith.mulf %3, %9 : f32
              %12 = arith.addf %10, %11 : f32
              affine.store %5, %smMax[(%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9] : memref<32xf32, 3>
              affine.store %12, %smSum[(%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9] : memref<32xf32, 3>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          }
        }
        // compute result and store
        affine.for %bx = 0 to 128 step 64 {
          // load
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              %1 = affine.vector_load %arg0[%by + (%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9, %bx + ((%tid mod 64) mod 16) * 4] : memref<128x128xf32, 1>, vector<4xf32>
              affine.vector_store %1, %tileP[%arg9, %arg10] : memref<4x4xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}
          // sm to reg
          affine.for %arg9 = 0 to 4 step 4 {
            %1 = affine.vector_load %smMax[(%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9] : memref<32xf32, 3>, vector<4xf32>
            affine.vector_store %1, %regMax[%arg9] : memref<4xf32>, vector<4xf32>
            %2 = affine.vector_load %smSum[(%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9] : memref<32xf32, 3>, vector<4xf32>
            affine.vector_store %2, %regSum[%arg9] : memref<4xf32>, vector<4xf32>
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          // compute
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %tileP[%arg9, %arg10] : memref<4x4xf32>
              // %2 = affine.load %regMax[%arg9] : memref<4xf32>
              // %5 = affine.load %regSum[%arg9] : memref<4xf32>
              // %5 = arith.constant 1.000000e+00 : f32
              // %3 = arith.subf %1, %2 : f32
              %4 = math.exp %1 : f32
              // %6 = arith.divf %4, %5 : f32
              affine.store %4, %tileP[%arg9, %arg10] : memref<4x4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttilex"}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "ttiley"}
          // store
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              %1 = affine.vector_load %tileP[%arg9, %arg10] : memref<4x4xf32>, vector<4xf32>
              affine.vector_store %1, %arg1[%by + (%tid floordiv 64) * 16 + ((%tid mod 64) floordiv 16) * 4 + %arg9, %bx + ((%tid mod 64) mod 16) * 4] : memref<128x128xf32, 1>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32, for.desc = "initBuf"}
          
        } {for.desc = "blockx"}
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}