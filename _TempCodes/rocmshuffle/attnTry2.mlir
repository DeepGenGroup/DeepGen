// ===== moveMemrefDefineAhead =======
module {
  func.func public @kcg_Attention_1_32_2048_128_Br16Bc64Hd128_Sa16Sb8PTr4PTc4OTr4OTc8GLWQ4GLWK4GLWV4BLPY1BLPX1WLPY4WLPX16BSWQ4BSWK2WSWQ4WSWK2BLOY1BLOX1WLOY4WLOX16BSWP4BSWV2WSWP4WSWV2Un1W64LCP1LCO1SPP0RPP0RPO0(%Q: memref<1x32x128x2048xf32, 1>, %K: memref<1x32x128x2048xf32, 1>, %V: memref<1x32x2048x128xf32, 1>, %O: memref<1x32x2048x128xf32, 1> 
  , %P: memref<4x8xf32,1>
  ) attributes {arg.tran = [true, false, false], func.op.type = "FlashAttn", func.output.arg.num = 1 : i32, func.state = "gpu", parallel.dim = ["y"]} {
    affine.parallel (%bx, %by, %bz) = (0, 0, 0) to (128, 32, 1) {
      %smq = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smQ"} : memref<16x16xf32, 3>
      %smk = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smK"} : memref<16x64xf32, 3>
      %smv = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smV"} : memref<8x128xf32, 3>
      %smp = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smP"} : memref<16x64xf32, 3>
      %smFact = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smFactor"} : memref<16xf32, 3>
      %0 = affine.apply affine_map<(d0) -> (d0 * 16)>(%bx) {apply.desc = "blocky"}  // 16 * bx
        %smMax = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smMax"} : memref<16xf32, 3>   // 一个block有一个warp(4*16thread), 一个thread处理 4*4 元素，即warp处理 4*4 = 16行
        %smSum = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smSum"} : memref<16xf32, 3>
      affine.parallel (%tx) = (0) to (64) {
        %rOSum = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "ORegSum"} : memref<4xf32>
        %rtileO = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tileO"} : memref<4x8xf32>
        %rFactor = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regFactor"} : memref<4xf32>
        %rMAX = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regMax"} : memref<4xf32>
        %rSUM = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regSum"} : memref<4xf32>
        %tempQ = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempQ"} : memref<4xf32>
        %tempK = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempK"} : memref<16xf32>
        %tempV = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempV"} : memref<16xf32>
        %regQ = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regQ"} : memref<4xf32>
        %regK = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regK"} : memref<4xf32>
        %regP = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regP"} : memref<4xf32>
        %regV = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regV"} : memref<8xf32>
        %rtileP = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tileP"} : memref<4x4xf32>
        affine.for %arg8 = 0 to 4 {  // debug ： 提前
          affine.for %arg9 = 0 to 8 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %rtileO[%arg8, %arg9] : memref<4x8xf32>
          }
        } {for.desc = "initBuf"}
        // affine.for %arg8 = 0 to 16 step 64 {    
        //   affine.if affine_set<(d0, d1) : (-d0 - d1 + 15 >= 0)>(%tx, %arg8) { // 
        //     %cst = arith.constant 0xFF800000 : f32
        //     affine.store %cst, %smMax[%arg8 + %tx] : memref<16xf32, 3>  // -inf
        //     %cst_18 = arith.constant 0.000000e+00 : f32
        //     affine.store %cst_18, %smSum[%arg8 + %tx] : memref<16xf32, 3>  // 0
        //   }
        // } {for.desc = "initBuf"}

          affine.if affine_set<(d0) : (-d0 + 15 >= 0)>(%tx) { // 
            %cst = arith.constant 0xFF800000 : f32
            affine.store %cst, %smMax[ %tx] : memref<16xf32, 3>  // -inf
            %cst_18 = arith.constant 0.000000e+00 : f32
            affine.store %cst_18, %smSum[ %tx] : memref<16xf32, 3>  // 0
          }

        // debug
        gpu.barrier
        
        affine.for %i_bcReduce = 0 to 2048 step 64 {  // for bc
          affine.for %arg9 = 0 to 4 {
            %cst = arith.constant 0xFF800000 : f32  // -inf 
            affine.store %cst, %rMAX[%arg9] : memref<4xf32>
            %cst_18 = arith.constant 0.000000e+00 : f32
            affine.store %cst_18, %rSUM[%arg9] : memref<4xf32>  // 0
          } {for.desc = "initBuf"}
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %cst = arith.constant 0.000000e+00 : f32
              affine.store %cst, %rtileP[%arg9, %arg10] : memref<4x4xf32>
            }
          } {for.desc = "initBuf"}
          affine.for %i_sa = 0 to 128 step 16 {
            gpu.barrier
            affine.for %arg10 = 0 to 1 {
              %1 = affine.vector_load %Q[%bz, %by, %i_sa + %arg10 * 16 + (%tx * 4) floordiv 16, %0 + (%tx * 4) mod 16] : memref<1x32x128x2048xf32, 1>, vector<4xf32>
              affine.vector_store %1, %tempQ[%arg10 * 4] : memref<4xf32>, vector<4xf32>
            } {for.desc = ""}
            affine.for %arg10 = 0 to 4 {
              %1 = affine.vector_load %K[%bz, %by, %i_sa + %arg10 * 4 + (%tx * 4) floordiv 64, %i_bcReduce + (%tx * 4) mod 64] : memref<1x32x128x2048xf32, 1>, vector<4xf32>
              affine.vector_store %1, %tempK[%arg10 * 4] : memref<16xf32>, vector<4xf32>
            } {for.desc = ""}
            affine.for %arg10 = 0 to 1 {
              %1 = affine.vector_load %tempQ[%arg10 * 4] : memref<4xf32>, vector<4xf32>
              affine.vector_store %1, %smq[%arg10 * 16 + (%tx * 4) floordiv 16, (%tx * 4) mod 16] : memref<16x16xf32, 3>, vector<4xf32>
            } {for.desc = ""}
            affine.for %arg10 = 0 to 4 {
              %1 = affine.vector_load %tempK[%arg10 * 4] : memref<16xf32>, vector<4xf32>
              affine.vector_store %1, %smk[%arg10 * 4 + (%tx * 4) floordiv 64, (%tx * 4) mod 64] : memref<16x64xf32, 3>, vector<4xf32>
            } {for.desc = ""}
            gpu.barrier
            affine.for %arg10 = 0 to 16 {
              affine.for %arg11 = 0 to 1 {
                affine.for %arg12 = 0 to 1 {
                  %1 = affine.vector_load %smq[%arg10, (%arg11 + %tx floordiv 64) * 16 + (%arg12 * 4 + (%tx mod 64) floordiv 16) * 4] : memref<16x16xf32, 3>, vector<4xf32>
                  affine.vector_store %1, %regQ[%arg11 * 4 + %arg12 * 4] : memref<4xf32>, vector<4xf32>
                }
              } {for.desc = ""}
              affine.for %arg11 = 0 to 2 {
                affine.for %arg12 = 0 to 1 {
                  %1 = affine.vector_load %smk[%arg10, %arg11 * 32 + (%arg12 * 16 + %tx mod 16) * 2] : memref<16x64xf32, 3>, vector<2xf32>
                  affine.vector_store %1, %regK[%arg11 * 2 + %arg12 * 2] : memref<4xf32>, vector<2xf32>
                }
              } {for.desc = ""}
              affine.for %arg11 = 0 to 4 {
                affine.for %arg12 = 0 to 4 {
                  %1 = affine.load %rtileP[%arg11, %arg12] : memref<4x4xf32>
                  %2 = affine.load %regQ[%arg11] : memref<4xf32>
                  %3 = affine.load %regK[%arg12] : memref<4xf32>
                  %4 = arith.mulf %2, %3 : f32
                  %5 = arith.addf %4, %1 : f32
                  affine.store %5, %rtileP[%arg11, %arg12] : memref<4x4xf32>  // rtileP += regQ * regK
                } {for.desc = "ttilex"}
              } {for.desc = "ttiley"}
            }
          }  // end for 0 to 128 step 16
          // %i_bcReduce = 0 to 2048 step 64 
          // 计算当前线程的 rmax和 softmax 分母(local)
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %rtileP[%arg9, %arg10] : memref<4x4xf32>
              %2 = affine.load %rMAX[%arg9] : memref<4xf32>  // 每行的max，初始=-inf
              %3 = affine.load %rSUM[%arg9] : memref<4xf32>  // 每行的sum，初始=0  存放 exp(p0-line_max)+exp(p1-line_max)+exp(p2-line_max)+ ... 即该行 softmax的分母
              %4 = arith.maxnumf %2, %1 : f32  // max_j1 = max{ max_j0, rtileP } 
              %5 = arith.subf %2, %4 : f32   // max_j1 - max_j0
              %6 = math.exp %5 : f32  // exp(max_j0 - max_j1) 
              %7 = arith.mulf %6, %3 : f32  // exp(max_j0 - max_j1) * rsum
              %8 = arith.subf %1, %4 : f32  // rtileP - max_j1
              %9 = math.exp %8 : f32  // exp( rtileP - max_j1 )
              %10 = arith.addf %7, %9 : f32  // exp(max_j0 - max_j1)*rsum + exp(rtileP - max_j1)
              affine.store %4, %rMAX[%arg9] : memref<4xf32>  // max_j0 = max_j1
              affine.store %10, %rSUM[%arg9] : memref<4xf32>  // rsum[j+1] = exp(max[j] - max[j+1])*rsum[j] + exp(rtileP - max[j+1])
            } {for.desc = "ttilexDown"}
          } {for.desc = "ttileyDown"}
          gpu.barrier  // debug
          affine.for %arg9 = 0 to 4 {  // 收集当前warp的每一行中的最大值（归并，1，2，4，8 ）
            %c16_i32 = arith.constant 64 : i32
            %c1_i32 = arith.constant 1 : i32
            %1 = affine.load %rMAX[%arg9] : memref<4xf32>  // 线程的最大值 rmax
            %shuffleResult, %valid = gpu.shuffle  down %1, %c1_i32, %c16_i32 : f32  // 读取 tid=selfid+1 的 rMAX  , 每16线程一组
            %2 = affine.load %rSUM[%arg9] : memref<4xf32>  // 线程指数和 rSUM
            %shuffleResult_18, %valid_19 = gpu.shuffle  down %2, %c1_i32, %c16_i32 : f32  // 
            %3 = arith.maxnumf %1, %shuffleResult : f32  // rMAX {t, t+1}
            %4 = arith.subf %1, %3 : f32
            %5 = math.exp %4 : f32
            %6 = arith.subf %shuffleResult, %3 : f32
            %7 = math.exp %6 : f32
            %8 = arith.mulf %2, %5 : f32
            %9 = arith.mulf %shuffleResult_18, %7 : f32
            %10 = arith.addf %8, %9 : f32
            affine.store %3, %rMAX[%arg9] : memref<4xf32>
            affine.store %10, %rSUM[%arg9] : memref<4xf32>
            %c2_i32 = arith.constant 2 : i32
            %11 = affine.load %rMAX[%arg9] : memref<4xf32>
            gpu.barrier // debug
            %shuffleResult_20, %valid_21 = gpu.shuffle  down %11, %c2_i32, %c16_i32 : f32
            %12 = affine.load %rSUM[%arg9] : memref<4xf32>
            %shuffleResult_22, %valid_23 = gpu.shuffle  down %12, %c2_i32, %c16_i32 : f32
            gpu.barrier // debug
            %13 = arith.maxnumf %11, %shuffleResult_20 : f32
            %14 = arith.subf %11, %13 : f32
            %15 = math.exp %14 : f32
            %16 = arith.subf %shuffleResult_20, %13 : f32
            %17 = math.exp %16 : f32
            %18 = arith.mulf %12, %15 : f32
            %19 = arith.mulf %shuffleResult_22, %17 : f32
            %20 = arith.addf %18, %19 : f32
            affine.store %13, %rMAX[%arg9] : memref<4xf32>
            affine.store %20, %rSUM[%arg9] : memref<4xf32>
            %c4_i32 = arith.constant 4 : i32
            %21 = affine.load %rMAX[%arg9] : memref<4xf32>
            gpu.barrier // debug
            %shuffleResult_24, %valid_25 = gpu.shuffle  down %21, %c4_i32, %c16_i32 : f32
            %22 = affine.load %rSUM[%arg9] : memref<4xf32>
            %shuffleResult_26, %valid_27 = gpu.shuffle  down %22, %c4_i32, %c16_i32 : f32
            gpu.barrier // debug
            %23 = arith.maxnumf %21, %shuffleResult_24 : f32
            %24 = arith.subf %21, %23 : f32
            %25 = math.exp %24 : f32
            %26 = arith.subf %shuffleResult_24, %23 : f32
            %27 = math.exp %26 : f32
            %28 = arith.mulf %22, %25 : f32
            %29 = arith.mulf %shuffleResult_26, %27 : f32
            %30 = arith.addf %28, %29 : f32
            affine.store %23, %rMAX[%arg9] : memref<4xf32>
            affine.store %30, %rSUM[%arg9] : memref<4xf32>
            %c8_i32 = arith.constant 8 : i32
            %31 = affine.load %rMAX[%arg9] : memref<4xf32>
            gpu.barrier // debug
            %shuffleResult_28, %valid_29 = gpu.shuffle  down %31, %c8_i32, %c16_i32 : f32  // 语义：shuffleResult_28 = threadid为 selfID+8 的线程的 %31 寄存器值， 边界=16（这里不能跨界，即第二行的最大值不能干扰第一行的）
            %32 = affine.load %rSUM[%arg9] : memref<4xf32>
            %shuffleResult_30, %valid_31 = gpu.shuffle  down %32, %c8_i32, %c16_i32 : f32
            gpu.barrier // debug
            %33 = arith.maxnumf %31, %shuffleResult_28 : f32
            %34 = arith.subf %31, %33 : f32
            %35 = math.exp %34 : f32
            %36 = arith.subf %shuffleResult_28, %33 : f32
            %37 = math.exp %36 : f32
            %38 = arith.mulf %32, %35 : f32
            %39 = arith.mulf %shuffleResult_30, %37 : f32
            %40 = arith.addf %38, %39 : f32
            affine.store %33, %rMAX[%arg9] : memref<4xf32>  // rMAX 村此行的 max
            affine.store %40, %rSUM[%arg9] : memref<4xf32>  // rSUM 存此行的 expsum
          }
          gpu.barrier  // debugAdd
          affine.if affine_set<(d0) : (d0 mod 16 == 0)>(%tx) {  // 对于每行中的0号线程：比较其 rMAX & smMAX, 将大的存进 smMAX， 并更新 rmax
            affine.for %arg9 = 0 to 4 step 4 {
              affine.for %arg10 = 0 to 4 step 4 {
                affine.for %arg11 = 0 to 4 {
                  // smMax : -inf   | rmax    |
                  // smSum : 0      | %rSUM   |
                  // smFact: undef  | e(-inf) |
                  // rMax  : oldmax | newMax  |
                  %1 = affine.load %smMax[(%arg9 + (%tx floordiv 64) * 4) * 4 + %arg10 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg11] : memref<16xf32, 3>  // init=-inf
                  %2 = affine.load %rMAX[%arg9 + %arg10 + %arg11] : memref<4xf32>
                  %3 = affine.load %smSum[(%arg9 + (%tx floordiv 64) * 4) * 4 + %arg10 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg11] : memref<16xf32, 3>  // init=0
                  %4 = affine.load %rSUM[%arg9 + %arg10 + %arg11] : memref<4xf32>
                  %5 = arith.maxnumf %2, %1 : f32  // max{smMax, rmax} 
                  %6 = arith.subf %2, %5 : f32  // rmax - max{smMax, rmax} 
                  %7 = math.exp %6 : f32  // exp(rmax - max{smMax, rmax} ) 
                  %8 = arith.subf %1, %5 : f32  // smMax - max{smMax, rmax} 
                  %9 = math.exp %8 : f32  // exp(smMax - max{smMax, rmax}) 
                  %10 = arith.mulf %4, %7 : f32  // rSUM * exp(rmax - max{smMax, rmax}) 
                  %11 = arith.mulf %3, %9 : f32  // smsum * exp(smMax - max{smMax, rmax}) 
                  %12 = arith.addf %10, %11 : f32  // exp_sum * exp(rMAX - smmax) + smsum * exp(smMax - max{smMax, rmax}), exp_sum * exp(rMAX - smmax)
                  affine.store %5, %smMax[(%arg9 + (%tx floordiv 64) * 4) * 4 + %arg10 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg11] : memref<16xf32, 3>  // 最大值
                  affine.store %12, %smSum[(%arg9 + (%tx floordiv 64) * 4) * 4 + %arg10 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg11] : memref<16xf32, 3>  // 指数和
                  affine.store %9, %smFact[(%arg9 + (%tx floordiv 64) * 4) * 4 + %arg10 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg11] : memref<16xf32, 3>  // 0
                  affine.store %5, %rMAX[%arg9 + %arg10 + %arg11] : memref<4xf32>
                  gpu.barrier // debug
                }
              }
            }
          }
          gpu.barrier  // debug
          affine.for %arg9 = 0 to 4 {
            // debug add
            gpu.barrier
            %1 = affine.load %rMAX[%arg9] : memref<4xf32>
            %c16_i32 = arith.constant 16 : i32
            %c0_i32 = arith.constant 0 : i32
            gpu.barrier // debug
            %shuffleResult, %valid = gpu.shuffle  idx %1, %c0_i32, %c16_i32 : f32  // shuffleResult = "0号"线程的 %1 的值.此处的0号为每16个线程一组得到的编号，不是真正的laneid （即，0-15线程取0号的，16-31线程取16号的, ... 以此类推）
            gpu.barrier // debug
            affine.store %shuffleResult, %rMAX[%arg9] : memref<4xf32>  // 将每个thread的rMAX更新为全局最大值
          }
 
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 4 {
              %1 = affine.load %rtileP[%arg9, %arg10] : memref<4x4xf32>
              %2 = affine.load %rMAX[%arg9] : memref<4xf32>
              %sum = affine.load %smFact[ %arg9 * 4 + %arg10] : memref<16xf32,3>
              %3 = arith.subf %1, %2 : f32  // p - max
              %4 = math.exp %3 : f32  // exp(p-max)
              // affine.store %4, %rtileP[%arg9, %arg10] : memref<4x4xf32> 
              affine.store %2, %rtileP[%arg9, %arg10] : memref<4x4xf32>    // debug
                        // affine.if affine_set<(d0) : (d0 == 0)>(%by) { // 
                        // affine.if affine_set<(d1) : (d1 == 0)>(%bz) { // 
                        // affine.if affine_set<(d2) : (d2 == 127)>(%bx) { // 
                        // affine.if affine_set<(d3) : (d3 == 25)>(%tx) { // 
                        //   affine.store %2, %P[%arg9] : memref<4xf32, 1>  // tx=0, bx=0. 看下 rmax
                        // }}}}
              // affine.store %2, %P[%bz, %by, %bx * 16 + (%tx floordiv 16) * 4 + %arg9 , %i_bcReduce * 64 + (%tx mod 16) * 4 + %arg10 ] : memref<1x32x2048x2048xf32,1> // 将最大值写入P 看看
            } {for.desc = "ttilex"}
          } {for.desc = "ttiley"}

          affine.for %arg9 = 0 to 4 step 4 {
            affine.for %arg10 = 0 to 4 step 2 {
              affine.for %arg11 = 0 to 4 step 4 {
                affine.for %arg12 = 0 to 2 step 2 {
                  affine.for %arg13 = 0 to 4 {
                    affine.for %arg14 = 0 to 2 step 2 {
                      %1 = affine.vector_load %rtileP[%arg9 + %arg11 + %arg13, %arg10 + %arg12 + %arg14] : memref<4x4xf32>, vector<2xf32>
                      affine.vector_store %1, %smp[(%arg9 + (%tx floordiv 64) * 4) * 4 + %arg11 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg13, %arg10 * 16 + %arg12 * 16 + (%tx mod 16) * 2 + %arg14] : memref<16x64xf32, 3>, vector<2xf32>

                      // debug
                      //   affine.if affine_set<(d0) : (d0 == 0)>(%by) { // 
                      //   affine.if affine_set<(d1) : (d1 == 0)>(%bz) { // 
                      //   affine.if affine_set<(d2) : (d2 == 110)>(%bx) { // 
                      // affine.vector_store %1, %P[(%arg9 + (%tx floordiv 64) * 4) * 4 + %arg11 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg13, %arg10 * 16 + %arg12 * 16 + (%tx mod 16) * 2 + %arg14] : memref<16x64xf32, 1>, vector<2xf32>
                      //   }}}
                    }
                  }
                }
              }
            }
          }
          gpu.barrier

          affine.for %arg9 = 0 to 1 {
            affine.for %arg10 = 0 to 1 {
              %1 = affine.vector_load %smFact[(%arg9 + %tx floordiv 64) * 16 + (%arg10 * 4 + (%tx mod 64) floordiv 16) * 4] : memref<16xf32, 3>, vector<4xf32>
              affine.vector_store %1, %rFactor[%arg9 * 4 + %arg10 * 4] : memref<4xf32>, vector<4xf32>
            }
          }
          affine.for %arg9 = 0 to 4 {
            affine.for %arg10 = 0 to 8 {
              %1 = affine.load %rFactor[%arg9] : memref<4xf32>
              %2 = affine.load %rtileO[%arg9, %arg10] : memref<4x8xf32>
              %3 = arith.mulf %2, %1 : f32
              // affine.store %3, %rtileO[%arg9, %arg10] : memref<4x8xf32>
              affine.store %2, %rtileO[%arg9, %arg10] : memref<4x8xf32>
            }
          }
          affine.for %arg9 = 0 to 64 step 8 {
            gpu.barrier
            affine.for %arg10 = 0 to 4 {
              %1 = affine.vector_load %V[%bz, %by, %i_bcReduce + %arg9 + %arg10 * 2 + (%tx * 4) floordiv 128, (%tx * 4) mod 128] : memref<1x32x2048x128xf32, 1>, vector<4xf32>
              affine.vector_store %1, %tempV[%arg10 * 4] : memref<16xf32>, vector<4xf32>
            } {for.desc = ""}
            affine.for %arg10 = 0 to 4 {
              %1 = affine.vector_load %tempV[%arg10 * 4] : memref<16xf32>, vector<4xf32>
              affine.vector_store %1, %smv[%arg10 * 2 + (%tx * 4) floordiv 128, (%tx * 4) mod 128] : memref<8x128xf32, 3>, vector<4xf32>
            } {for.desc = ""}
            gpu.barrier
            affine.for %arg10 = 0 to 8 {
              affine.for %arg11 = 0 to 1 {
                affine.for %arg12 = 0 to 1 {
                  affine.for %arg13 = 0 to 4 {
                    %1 = affine.vector_load %smp[(%arg11 + %tx floordiv 64) * 16 + (%arg12 * 4 + (%tx mod 64) floordiv 16) * 4 + %arg13, %arg9 + %arg10] : memref<16x64xf32, 3>, vector<1xf32>
                    affine.vector_store %1, %regP[%arg11 * 4 + %arg12 * 4 + %arg13] : memref<4xf32>, vector<1xf32>
                  }
                }
              } {for.desc = ""}
              affine.for %arg11 = 0 to 4 {
                affine.for %arg12 = 0 to 1 {
                  %1 = affine.vector_load %smv[%arg10, %arg11 * 32 + (%arg12 * 16 + %tx mod 16) * 2] : memref<8x128xf32, 3>, vector<2xf32>
                  affine.vector_store %1, %regV[%arg11 * 2 + %arg12 * 2] : memref<8xf32>, vector<2xf32>
                }
              } {for.desc = ""}
              gpu.barrier // debugadd
              // regP * regV (QK=P, P=softmax(P) ,PV = O)
              affine.for %arg11 = 0 to 4 {
                affine.for %arg12 = 0 to 8 {
                  %1 = affine.load %rtileO[%arg11, %arg12] : memref<4x8xf32>
                  %2 = affine.load %regP[%arg11] : memref<4xf32>
                  %3 = affine.load %regV[%arg12] : memref<8xf32>
                  %4 = arith.mulf %2, %3 : f32
                  %5 = arith.addf %4, %1 : f32
                  affine.store %5, %rtileO[%arg11, %arg12] : memref<4x8xf32>

                        affine.if affine_set<(d0) : (d0 == 0)>(%by) { // 
                        affine.if affine_set<(d1) : (d1 == 0)>(%bz) { // 
                        affine.if affine_set<(d2) : (d2 == 110)>(%bx) { // 
                        affine.if affine_set<(d2) : (d2 == 34)>(%tx) { // 
                          affine.store %4, %P[%arg11, %arg12] : memref<4x8xf32,1> 
                        }}}}
                } {for.desc = "ttilex"}
              } {for.desc = "ttiley"}
            }
          }
        } {for.desc = "blockx"}
        affine.for %arg8 = 0 to 1 {
          affine.for %arg9 = 0 to 1 {
            %1 = affine.vector_load %smSum[(%arg8 + %tx floordiv 64) * 16 + (%arg9 * 4 + (%tx mod 64) floordiv 16) * 4] : memref<16xf32, 3>, vector<4xf32>
            affine.vector_store %1, %rOSum[%arg8 * 4 + %arg9 * 4] : memref<4xf32>, vector<4xf32>
          }
        }
        // //////////////// rtileO = rtileO / rOSum
        // affine.for %arg8 = 0 to 4 {
        //   affine.for %arg9 = 0 to 8 {
        //     %1 = affine.load %rOSum[%arg8] : memref<4xf32>
        //     %2 = affine.load %rtileO[%arg8, %arg9] : memref<4x8xf32>
        //     %3 = arith.divf %2, %1 : f32
        //     affine.store %3, %rtileO[%arg8, %arg9] : memref<4x8xf32>
        //   }
        // }
        gpu.barrier  // debug
        affine.for %arg8 = 0 to 4 step 4 {
          affine.for %arg9 = 0 to 8 step 2 {
            affine.for %arg10 = 0 to 4 step 4 {
              affine.for %arg11 = 0 to 2 step 2 {
                affine.for %arg12 = 0 to 4 {
                  affine.for %arg13 = 0 to 2 step 2 {
                    %1 = affine.vector_load %rtileO[%arg8 + %arg10 + %arg12, %arg9 + %arg11 + %arg13] : memref<4x8xf32>, vector<2xf32>
                    affine.vector_store %1, %O[%bz, %by, %0 + (%arg8 + (%tx floordiv 64) * 4) * 4 + %arg10 * 4 + ((%tx mod 64) floordiv 16) * 4 + %arg12, %arg9 * 16 + %arg11 * 16 + (%tx mod 16) * 2 + %arg13] : memref<1x32x2048x128xf32, 1>, vector<2xf32>
                  }
                }
              }
            }
          }
        }
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}