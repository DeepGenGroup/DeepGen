module {
  func.func public @kcg_Attention_1_32_2048_128_Br32Bc64Hd128_Sa16Sb8PTr4PTc4OTr4OTc8GLWQ4GLWK4GLWV4BLPY4BLPX1WLPY2WLPX16BSWQ4BSWK2WSWQ4WSWK2BLOY2BLOX2WLOY4WLOX8BSWP4BSWV4WSWP4WSWV4Un8W32LCP1LCO1SPP0RPP0RPO0(%arg0: memref<1x32x128x2048xf32, 1>, %arg1: memref<1x32x128x2048xf32, 1>, %arg2: memref<1x32x2048x128xf32, 1>, %arg3: memref<1x32x2048x128xf32, 1>) attributes {arg.tran = [true, false, false], func.op.type = "FlashAttn", func.output.arg.num = 1 : i32, func.state = "gpu", parallel.dim = ["y"]} {
    affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (64, 32, 1) {
      %alloc = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smQ"} : memref<16x32xf32, 3>
      %alloc_0 = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smK"} : memref<16x64xf32, 3>
      %alloc_1 = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smV"} : memref<8x128xf32, 3>
      %alloc_2 = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smP"} : memref<32x64xf32, 3>
      %alloc_3 = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smFactor"} : memref<32xf32, 3>
      %0 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg4) {apply.desc = "blocky"}
      affine.parallel (%arg7) = (0) to (128) {
        %alloca = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "ORegSum"} : memref<4xf32>
        %alloc_4 = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smMax"} : memref<32xf32, 3>
        %alloc_5 = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smSum"} : memref<32xf32, 3>
        affine.for %arg8 = 0 to 32 step 128 {
          affine.if affine_set<(d0, d1) : (-d0 - d1 + 31 >= 0)>(%arg7, %arg8) {
            %cst = arith.constant 0xFF800000 : f32
            affine.store %cst, %alloc_4[%arg8 + %arg7] : memref<32xf32, 3>
            %cst_7 = arith.constant 0.000000e+00 : f32
            affine.store %cst_7, %alloc_5[%arg8 + %arg7] : memref<32xf32, 3>
          }
        } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "initBuf"}
        %alloca_6 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tileO"} : memref<4x8xf32>
        affine.for %arg8 = 0 to 4 {
          affine.for %arg9 = 0 to 8 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %alloca_6[%arg8, %arg9] : memref<4x8xf32>
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "initBuf"}
        affine.for %arg8 = 0 to 2048 step 64 {
          %alloca_7 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regFactor"} : memref<4xf32>
          %alloca_8 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regMax"} : memref<4xf32>
          %alloca_9 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regSum"} : memref<4xf32>
          affine.for %arg9 = 0 to 4 {
            %cst = arith.constant 0xFF800000 : f32
            affine.store %cst, %alloca_8[%arg9] : memref<4xf32>
            %cst_18 = arith.constant 0.000000e+00 : f32
            affine.store %cst_18, %alloca_9[%arg9] : memref<4xf32>
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "initBuf"}
          %alloca_10 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempQ"} : memref<4xf32>
          %alloca_11 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempK"} : memref<8xf32>
          %alloca_12 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempV"} : memref<8xf32>
          %alloca_13 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regQ"} : memref<4xf32>
          %alloca_14 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regK"} : memref<4xf32>
          %alloca_15 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regP"} : memref<4xf32>
          %alloca_16 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regV"} : memref<8xf32>
          %alloca_17 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tileP"} : memref<4x4xf32>
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %cst = arith.constant 0.000000e+00 : f32
              affine.store %cst, %alloca_17[%arg9, %arg10] : memref<4x4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "initBuf"}
          affine.for %arg9 = 0 to 128 step 16 {
            gpu.barrier
            affine.for %arg10 = 0 to 1 {
              %1 = affine.vector_load %arg0[%arg6, %arg5, %arg9 + %arg10 * 16 + (%arg7 * 4) floordiv 32, %0 + (%arg7 * 4) mod 32] : memref<1x32x128x2048xf32, 1>, vector<4xf32>
              affine.vector_store %1, %alloca_10[%arg10 * 4] : memref<4xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
            affine.for %arg10 = 0 to 2 {
              %1 = affine.vector_load %arg1[%arg6, %arg5, %arg9 + %arg10 * 8 + (%arg7 * 4) floordiv 64, %arg8 + (%arg7 * 4) mod 64] : memref<1x32x128x2048xf32, 1>, vector<4xf32>
              affine.vector_store %1, %alloca_11[%arg10 * 4] : memref<8xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
            affine.for %arg10 = 0 to 1 {
              %1 = affine.vector_load %alloca_10[%arg10 * 4] : memref<4xf32>, vector<4xf32>
              affine.vector_store %1, %alloc[%arg10 * 16 + (%arg7 * 4) floordiv 32, (%arg7 * 4) mod 32] : memref<16x32xf32, 3>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
            affine.for %arg10 = 0 to 2 {
              %1 = affine.vector_load %alloca_11[%arg10 * 4] : memref<8xf32>, vector<4xf32>
              affine.vector_store %1, %alloc_0[%arg10 * 8 + (%arg7 * 4) floordiv 64, (%arg7 * 4) mod 64] : memref<16x64xf32, 3>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
            gpu.barrier
            affine.for %arg10 = 0 to 16 {
              affine.for %arg11 = 0 to 1 {
                affine.for %arg12 = 0 to 1 {
                  %1 = affine.vector_load %alloc[%arg10, (%arg11 * 4 + %arg7 floordiv 32) * 8 + (%arg12 * 2 + (%arg7 mod 32) floordiv 16) * 4] : memref<16x32xf32, 3>, vector<4xf32>
                  affine.vector_store %1, %alloca_13[%arg11 * 4 + %arg12 * 4] : memref<4xf32>, vector<4xf32>
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
              affine.for %arg11 = 0 to 2 {
                affine.for %arg12 = 0 to 1 {
                  %1 = affine.vector_load %alloc_0[%arg10, %arg11 * 32 + (%arg12 * 16 + %arg7 mod 16) * 2] : memref<16x64xf32, 3>, vector<2xf32>
                  affine.vector_store %1, %alloca_14[%arg11 * 2 + %arg12 * 2] : memref<4xf32>, vector<2xf32>
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
              affine.for %arg11 = 0 to 4 {
                affine.for %arg12 = 0 to 4 {
                  %1 = affine.load %alloca_17[%arg11, %arg12] : memref<4x4xf32>
                  %2 = affine.load %alloca_13[%arg11] : memref<4xf32>
                  %3 = affine.load %alloca_14[%arg12] : memref<4xf32>
                  %4 = arith.mulf %2, %3 : f32
                  %5 = arith.addf %4, %1 : f32
                  affine.store %5, %alloca_17[%arg11, %arg12] : memref<4x4xf32>
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttilex"}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttiley"}
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          }
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %alloca_17[%arg9, %arg10] : memref<4x4xf32>
              %2 = affine.load %alloca_8[%arg9] : memref<4xf32>
              %3 = affine.load %alloca_9[%arg9] : memref<4xf32>
              %4 = arith.maxnumf %2, %1 : f32
              %5 = arith.subf %2, %4 : f32
              %6 = math.exp %5 : f32
              %7 = arith.mulf %6, %3 : f32
              %8 = arith.subf %1, %4 : f32
              %9 = math.exp %8 : f32
              %10 = arith.addf %7, %9 : f32
              affine.store %4, %alloca_8[%arg9] : memref<4xf32>
              affine.store %10, %alloca_9[%arg9] : memref<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttilexDown"}
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttileyDown"}
          affine.for %arg9 = 0 to 4 {
            %c16_i32 = arith.constant 16 : i32
            %c1_i32 = arith.constant 1 : i32
            %1 = affine.load %alloca_8[%arg9] : memref<4xf32>
            %shuffleResult, %valid = gpu.shuffle  down %1, %c1_i32, %c16_i32 : f32
            %2 = affine.load %alloca_9[%arg9] : memref<4xf32>
            %shuffleResult_18, %valid_19 = gpu.shuffle  down %2, %c1_i32, %c16_i32 : f32
            %3 = arith.maxnumf %1, %shuffleResult : f32
            %4 = arith.subf %1, %3 : f32
            %5 = math.exp %4 : f32
            %6 = arith.subf %shuffleResult, %3 : f32
            %7 = math.exp %6 : f32
            %8 = arith.mulf %2, %5 : f32
            %9 = arith.mulf %shuffleResult_18, %7 : f32
            %10 = arith.addf %8, %9 : f32
            affine.store %3, %alloca_8[%arg9] : memref<4xf32>
            affine.store %10, %alloca_9[%arg9] : memref<4xf32>
            %c2_i32 = arith.constant 2 : i32
            %11 = affine.load %alloca_8[%arg9] : memref<4xf32>
            %shuffleResult_20, %valid_21 = gpu.shuffle  down %11, %c2_i32, %c16_i32 : f32
            %12 = affine.load %alloca_9[%arg9] : memref<4xf32>
            %shuffleResult_22, %valid_23 = gpu.shuffle  down %12, %c2_i32, %c16_i32 : f32
            %13 = arith.maxnumf %11, %shuffleResult_20 : f32
            %14 = arith.subf %11, %13 : f32
            %15 = math.exp %14 : f32
            %16 = arith.subf %shuffleResult_20, %13 : f32
            %17 = math.exp %16 : f32
            %18 = arith.mulf %12, %15 : f32
            %19 = arith.mulf %shuffleResult_22, %17 : f32
            %20 = arith.addf %18, %19 : f32
            affine.store %13, %alloca_8[%arg9] : memref<4xf32>
            affine.store %20, %alloca_9[%arg9] : memref<4xf32>
            %c4_i32 = arith.constant 4 : i32
            %21 = affine.load %alloca_8[%arg9] : memref<4xf32>
            %shuffleResult_24, %valid_25 = gpu.shuffle  down %21, %c4_i32, %c16_i32 : f32
            %22 = affine.load %alloca_9[%arg9] : memref<4xf32>
            %shuffleResult_26, %valid_27 = gpu.shuffle  down %22, %c4_i32, %c16_i32 : f32
            %23 = arith.maxnumf %21, %shuffleResult_24 : f32
            %24 = arith.subf %21, %23 : f32
            %25 = math.exp %24 : f32
            %26 = arith.subf %shuffleResult_24, %23 : f32
            %27 = math.exp %26 : f32
            %28 = arith.mulf %22, %25 : f32
            %29 = arith.mulf %shuffleResult_26, %27 : f32
            %30 = arith.addf %28, %29 : f32
            affine.store %23, %alloca_8[%arg9] : memref<4xf32>
            affine.store %30, %alloca_9[%arg9] : memref<4xf32>
            %c8_i32 = arith.constant 8 : i32
            %31 = affine.load %alloca_8[%arg9] : memref<4xf32>
            %shuffleResult_28, %valid_29 = gpu.shuffle  down %31, %c8_i32, %c16_i32 : f32
            %32 = affine.load %alloca_9[%arg9] : memref<4xf32>
            %shuffleResult_30, %valid_31 = gpu.shuffle  down %32, %c8_i32, %c16_i32 : f32
            %33 = arith.maxnumf %31, %shuffleResult_28 : f32
            %34 = arith.subf %31, %33 : f32
            %35 = math.exp %34 : f32
            %36 = arith.subf %shuffleResult_28, %33 : f32
            %37 = math.exp %36 : f32
            %38 = arith.mulf %32, %35 : f32
            %39 = arith.mulf %shuffleResult_30, %37 : f32
            %40 = arith.addf %38, %39 : f32
            affine.store %33, %alloca_8[%arg9] : memref<4xf32>
            affine.store %40, %alloca_9[%arg9] : memref<4xf32>
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          affine.if affine_set<(d0) : (d0 mod 16 == 0)>(%arg7) {
            affine.for %arg9 = 0 to 4 step 4 {
              affine.for %arg10 = 0 to 4 step 4 {
                affine.for %arg11 = 0 to 4 {
                  %1 = affine.load %alloc_4[(%arg9 * 4 + (%arg7 floordiv 32) * 4) * 2 + %arg10 * 2 + ((%arg7 mod 32) floordiv 16) * 4 + %arg11] : memref<32xf32, 3>
                  %2 = affine.load %alloca_8[%arg9 + %arg10 + %arg11] : memref<4xf32>
                  %3 = affine.load %alloc_5[(%arg9 * 4 + (%arg7 floordiv 32) * 4) * 2 + %arg10 * 2 + ((%arg7 mod 32) floordiv 16) * 4 + %arg11] : memref<32xf32, 3>
                  %4 = affine.load %alloca_9[%arg9 + %arg10 + %arg11] : memref<4xf32>
                  %5 = arith.maxnumf %2, %1 : f32
                  %6 = arith.subf %2, %5 : f32
                  %7 = math.exp %6 : f32
                  %8 = arith.subf %1, %5 : f32
                  %9 = math.exp %8 : f32
                  %10 = arith.mulf %4, %7 : f32
                  %11 = arith.mulf %3, %9 : f32
                  %12 = arith.addf %10, %11 : f32
                  affine.store %5, %alloc_4[(%arg9 * 4 + (%arg7 floordiv 32) * 4) * 2 + %arg10 * 2 + ((%arg7 mod 32) floordiv 16) * 4 + %arg11] : memref<32xf32, 3>
                  affine.store %12, %alloc_5[(%arg9 * 4 + (%arg7 floordiv 32) * 4) * 2 + %arg10 * 2 + ((%arg7 mod 32) floordiv 16) * 4 + %arg11] : memref<32xf32, 3>
                  affine.store %9, %alloc_3[(%arg9 * 4 + (%arg7 floordiv 32) * 4) * 2 + %arg10 * 2 + ((%arg7 mod 32) floordiv 16) * 4 + %arg11] : memref<32xf32, 3>
                  affine.store %5, %alloca_8[%arg9 + %arg10 + %arg11] : memref<4xf32>
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          }
          affine.for %arg9 = 0 to 4 {
            %1 = affine.load %alloca_8[%arg9] : memref<4xf32>
            %c16_i32 = arith.constant 16 : i32
            %c0_i32 = arith.constant 0 : i32
            %shuffleResult, %valid = gpu.shuffle  idx %1, %c0_i32, %c16_i32 : f32
            affine.store %shuffleResult, %alloca_8[%arg9] : memref<4xf32>
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %alloca_17[%arg9, %arg10] : memref<4x4xf32>
              %2 = affine.load %alloca_8[%arg9] : memref<4xf32>
              %3 = arith.subf %1, %2 : f32
              %4 = math.exp %3 : f32
              affine.store %4, %alloca_17[%arg9, %arg10] : memref<4x4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttilex"}
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttiley"}
          affine.for %arg9 = 0 to 4 step 4 {
            affine.for %arg10 = 0 to 4 step 2 {
              affine.for %arg11 = 0 to 4 step 4 {
                affine.for %arg12 = 0 to 2 step 2 {
                  affine.for %arg13 = 0 to 4 {
                    affine.for %arg14 = 0 to 2 step 2 {
                      %1 = affine.vector_load %alloca_17[%arg9 + %arg11 + %arg13, %arg10 + %arg12 + %arg14] : memref<4x4xf32>, vector<2xf32>
                      affine.vector_store %1, %alloc_2[(%arg9 * 4 + (%arg7 floordiv 32) * 4) * 2 + %arg11 * 2 + ((%arg7 mod 32) floordiv 16) * 4 + %arg13, %arg10 * 16 + %arg12 * 16 + (%arg7 mod 16) * 2 + %arg14] : memref<32x64xf32, 3>, vector<2xf32>
                    } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
                  } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          gpu.barrier
          affine.for %arg9 = 0 to 1 {
            affine.for %arg10 = 0 to 1 {
              %1 = affine.vector_load %alloc_3[(%arg9 * 2 + (%arg7 floordiv 32) floordiv 2) * 16 + (%arg10 * 4 + (%arg7 mod 32) floordiv 8) * 4] : memref<32xf32, 3>, vector<4xf32>
              affine.vector_store %1, %alloca_7[%arg9 * 4 + %arg10 * 4] : memref<4xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 8 {
              %1 = affine.load %alloca_7[%arg9] : memref<4xf32>
              %2 = affine.load %alloca_6[%arg9, %arg10] : memref<4x8xf32>
              %3 = arith.mulf %2, %1 : f32
              affine.store %3, %alloca_6[%arg9, %arg10] : memref<4x8xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          affine.for %arg9 = 0 to 64 step 8 {
            gpu.barrier
            affine.for %arg10 = 0 to 2 {
              %1 = affine.vector_load %arg2[%arg6, %arg5, %arg8 + %arg9 + %arg10 * 4 + (%arg7 * 4) floordiv 128, (%arg7 * 4) mod 128] : memref<1x32x2048x128xf32, 1>, vector<4xf32>
              affine.vector_store %1, %alloca_12[%arg10 * 4] : memref<8xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
            affine.for %arg10 = 0 to 2 {
              %1 = affine.vector_load %alloca_12[%arg10 * 4] : memref<8xf32>, vector<4xf32>
              affine.vector_store %1, %alloc_1[%arg10 * 4 + (%arg7 * 4) floordiv 128, (%arg7 * 4) mod 128] : memref<8x128xf32, 3>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
            gpu.barrier
            affine.for %arg10 = 0 to 8 {
              affine.for %arg11 = 0 to 1 {
                affine.for %arg12 = 0 to 1 {
                  affine.for %arg13 = 0 to 4 {
                    %1 = affine.vector_load %alloc_2[(%arg11 * 2 + (%arg7 floordiv 32) floordiv 2) * 16 + (%arg12 * 4 + (%arg7 mod 32) floordiv 8) * 4 + %arg13, %arg9 + %arg10] : memref<32x64xf32, 3>, vector<1xf32>
                    affine.vector_store %1, %alloca_15[%arg11 * 4 + %arg12 * 4 + %arg13] : memref<4xf32>, vector<1xf32>
                  } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
              affine.for %arg11 = 0 to 2 {
                affine.for %arg12 = 0 to 1 {
                  %1 = affine.vector_load %alloc_1[%arg10, (%arg11 * 2 + (%arg7 floordiv 32) mod 2) * 32 + (%arg12 * 8 + %arg7 mod 8) * 4] : memref<8x128xf32, 3>, vector<4xf32>
                  affine.vector_store %1, %alloca_16[%arg11 * 4 + %arg12 * 4] : memref<8xf32>, vector<4xf32>
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = ""}
              affine.for %arg11 = 0 to 4 {
                affine.for %arg12 = 0 to 8 {
                  %1 = affine.load %alloca_6[%arg11, %arg12] : memref<4x8xf32>
                  %2 = affine.load %alloca_15[%arg11] : memref<4xf32>
                  %3 = affine.load %alloca_16[%arg12] : memref<8xf32>
                  %4 = arith.mulf %2, %3 : f32
                  %5 = arith.addf %4, %1 : f32
                  affine.store %5, %alloca_6[%arg11, %arg12] : memref<4x8xf32>
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttilex"}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32, for.desc = "ttiley"}
            }
          }
        } {for.desc = "blockx"}
        affine.for %arg8 = 0 to 1 {
          affine.for %arg9 = 0 to 1 {
            %1 = affine.vector_load %alloc_5[(%arg8 * 2 + (%arg7 floordiv 32) floordiv 2) * 16 + (%arg9 * 4 + (%arg7 mod 32) floordiv 8) * 4] : memref<32xf32, 3>, vector<4xf32>
            affine.vector_store %1, %alloca[%arg8 * 4 + %arg9 * 4] : memref<4xf32>, vector<4xf32>
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
        affine.for %arg8 = 0 to 4 {
          affine.for %arg9 = 0 to 8 {
            %1 = affine.load %alloca[%arg8] : memref<4xf32>
            %2 = affine.load %alloca_6[%arg8, %arg9] : memref<4x8xf32>
            %3 = arith.divf %2, %1 : f32
            affine.store %3, %alloca_6[%arg8, %arg9] : memref<4x8xf32>
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
        affine.for %arg8 = 0 to 4 step 4 {
          affine.for %arg9 = 0 to 8 step 4 {
            affine.for %arg10 = 0 to 4 step 4 {
              affine.for %arg11 = 0 to 4 step 4 {
                affine.for %arg12 = 0 to 4 {
                  affine.for %arg13 = 0 to 4 step 4 {
                    %1 = affine.vector_load %alloca_6[%arg8 + %arg10 + %arg12, %arg9 + %arg11 + %arg13] : memref<4x8xf32>, vector<4xf32>
                    affine.vector_store %1, %arg3[%arg6, %arg5, %0 + (%arg8 * 2 + ((%arg7 floordiv 32) floordiv 2) * 4) * 4 + %arg10 * 4 + ((%arg7 mod 32) floordiv 8) * 4 + %arg12, (%arg9 * 2 + ((%arg7 floordiv 32) mod 2) * 4) * 8 + %arg11 * 8 + (%arg7 mod 8) * 4 + %arg13] : memref<1x32x2048x128xf32, 1>, vector<4xf32>
                  } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
                } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
            } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 8 : i32}
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}