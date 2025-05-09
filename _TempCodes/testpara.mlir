// //  === start mlir ===== failed
// module {
//   func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf32, 1>, %arg1: memref<1024x1024xf32, 1>, %arg2: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu"} {
//     affine.parallel (%arg3) = (0) to (256) {
//       %0 = affine.apply affine_map<(d0) -> ((d0 floordiv 128) * 8 + d0 mod 8)>(%arg3)
//       %1 = affine.apply affine_map<(d0) -> ((d0 mod 128) floordiv 8)>(%arg3)
//       %2 = affine.apply affine_map<(d0) -> (d0 floordiv 16)>(%arg3)
//       %3 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg3)
//       %alloc = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smABC"} : memref<8192xf32, 3>
//       %4 = affine.apply affine_map<(d0) -> (d0 * 64)>(%0)
//       %5 = affine.apply affine_map<(d0) -> (d0 * 64)>(%1)
//       affine.parallel (%arg4) = (0) to (128) {
//         %6 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%arg4)
//         %7 = affine.apply affine_map<(d0) -> ((d0 mod 64) floordiv 8)>(%arg4)
//         %8 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg4)
//         %alloca = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempA"} : memref<8xf32>
//         %alloca_0 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempB"} : memref<8xf32>
//         %alloca_1 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regA"} : memref<2x8xf32>
//         %alloca_2 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regB"} : memref<2x8xf32>
//         %9 = affine.apply affine_map<(d0) -> (d0 * 8)>(%7)
//         %10 = affine.apply affine_map<(d0) -> (d0 * 8)>(%8)
//         %alloca_3 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regC"} : memref<64xf32>
//         affine.for %arg5 = 0 to 8 {
//           affine.for %arg6 = 0 to 8 {
//             %cst = arith.constant 0.000000e+00 : f32
//             affine.store %cst, %alloca_3[%arg5 * 8 + %arg6] : memref<64xf32>
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         affine.for %arg5 = 0 to 2 {
//           %11 = affine.vector_load %arg0[%arg4 floordiv 16 + %arg5 * 8, %0 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
//           affine.vector_store %11, %alloc[(%arg4 floordiv 16 + %arg5 * 8) * 64 + (%arg4 mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         affine.for %arg5 = 0 to 2 {
//           %11 = affine.vector_load %arg1[%arg4 floordiv 16 + %arg5 * 8, %1 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
//           affine.vector_store %11, %alloc[(%arg4 floordiv 16 + %arg5 * 8) * 64 + (%arg4 mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         gpu.barrier
//         affine.for %arg5 = 0 to 2 {
//           affine.for %arg6 = 0 to 2 {
//             %11 = affine.vector_load %alloc[(%arg4 floordiv 64) * 64 + (%arg5 + ((%arg4 mod 64) floordiv 32) floordiv 2) * 32 + (%arg6 * 8 + (%arg4 mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
//             affine.vector_store %11, %alloca_1[0, %arg5 * 4 + %arg6 * 2] : memref<2x8xf32>, vector<2xf32>
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         affine.for %arg5 = 0 to 2 {
//           affine.for %arg6 = 0 to 2 {
//             %11 = affine.vector_load %alloc[(%arg4 floordiv 64) * 64 + (%arg5 * 2 + ((%arg4 mod 64) floordiv 32) mod 2) * 16 + (%arg6 * 4 + %arg4 mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
//             affine.vector_store %11, %alloca_2[0, %arg5 * 4 + %arg6 * 2] : memref<2x8xf32>, vector<2xf32>
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         affine.for %arg5 = 16 to 1040 step 16 {
//           affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg5) {
//             affine.for %arg6 = 0 to 2 {
//               %11 = affine.vector_load %arg0[%arg4 floordiv 16 + %arg6 * 8 + %arg5, %0 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
//               affine.vector_store %11, %alloca[%arg6 * 4] : memref<8xf32>, vector<4xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             affine.for %arg6 = 0 to 2 {
//               %11 = affine.vector_load %arg1[%arg4 floordiv 16 + %arg6 * 8 + %arg5, %1 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
//               affine.vector_store %11, %alloca_0[%arg6 * 4] : memref<8xf32>, vector<4xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           }
//           affine.for %arg6 = 0 to 14 step 2 {
//             affine.for %arg7 = 0 to 2 {
//               affine.for %arg8 = 0 to 2 {
//                 %11 = affine.vector_load %alloc[((%arg5 floordiv 16 - 1) mod 2) * 1024 + (%arg6 + %arg4 floordiv 64 + 2) * 64 + (%arg7 + ((%arg4 mod 64) floordiv 32) floordiv 2) * 32 + (%arg8 * 8 + (%arg4 mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
//                 affine.vector_store %11, %alloca_1[(%arg6 floordiv 2 + 1) mod 2, %arg7 * 4 + %arg8 * 2] : memref<2x8xf32>, vector<2xf32>
//               } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             affine.for %arg7 = 0 to 2 {
//               affine.for %arg8 = 0 to 2 {
//                 %11 = affine.vector_load %alloc[((%arg5 floordiv 16 - 1) mod 2) * 1024 + (%arg6 + %arg4 floordiv 64 + 2) * 64 + (%arg7 * 2 + ((%arg4 mod 64) floordiv 32) mod 2) * 16 + (%arg8 * 4 + %arg4 mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
//                 affine.vector_store %11, %alloca_2[(%arg6 floordiv 2 + 1) mod 2, %arg7 * 4 + %arg8 * 2] : memref<2x8xf32>, vector<2xf32>
//               } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             affine.for %arg7 = 0 to 8 {
//               affine.for %arg8 = 0 to 8 {
//                 %11 = affine.load %alloca_3[%arg7 * 8 + %arg8] : memref<64xf32>
//                 %12 = affine.load %alloca_1[(%arg6 floordiv 2) mod 2, %arg7] : memref<2x8xf32>
//                 %13 = affine.load %alloca_2[(%arg6 floordiv 2) mod 2, %arg8] : memref<2x8xf32>
//                 %14 = arith.mulf %12, %13 : f32
//                 %15 = arith.addf %14, %11 : f32
//                 affine.store %15, %alloca_3[%arg7 * 8 + %arg8] : memref<64xf32>
//               } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           }
//           affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg5) {
//             affine.for %arg6 = 0 to 2 {
//               %11 = affine.vector_load %alloca[%arg6 * 4] : memref<8xf32>, vector<4xf32>
//               affine.vector_store %11, %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 16 + %arg6 * 8) * 64 + (%arg4 mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             affine.for %arg6 = 0 to 2 {
//               %11 = affine.vector_load %alloca_0[%arg6 * 4] : memref<8xf32>, vector<4xf32>
//               affine.vector_store %11, %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 16 + %arg6 * 8) * 64 + (%arg4 mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             gpu.barrier
//           }
//           affine.for %arg6 = 0 to 8 {
//             affine.for %arg7 = 0 to 8 {
//               %11 = affine.load %alloca_3[%arg6 * 8 + %arg7] : memref<64xf32>
//               %12 = affine.load %alloca_1[1, %arg6] : memref<2x8xf32>
//               %13 = affine.load %alloca_2[1, %arg7] : memref<2x8xf32>
//               %14 = arith.mulf %12, %13 : f32
//               %15 = arith.addf %14, %11 : f32
//               affine.store %15, %alloca_3[%arg6 * 8 + %arg7] : memref<64xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           affine.for %arg6 = 0 to 2 {
//             affine.for %arg7 = 0 to 2 {
//               %11 = affine.vector_load %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 64) * 64 + (%arg6 + ((%arg4 mod 64) floordiv 32) floordiv 2) * 32 + (%arg7 * 8 + (%arg4 mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
//               affine.vector_store %11, %alloca_1[0, %arg6 * 4 + %arg7 * 2] : memref<2x8xf32>, vector<2xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           affine.for %arg6 = 0 to 2 {
//             affine.for %arg7 = 0 to 2 {
//               %11 = affine.vector_load %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 64) * 64 + (%arg6 * 2 + ((%arg4 mod 64) floordiv 32) mod 2) * 16 + (%arg7 * 4 + %arg4 mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
//               affine.vector_store %11, %alloca_2[0, %arg6 * 4 + %arg7 * 2] : memref<2x8xf32>, vector<2xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         }
//         gpu.barrier
//         affine.for %arg5 = 0 to 8 step 4 {
//           affine.for %arg6 = 0 to 8 step 4 {
//             affine.for %arg7 = 0 to 4 step 2 {
//               affine.for %arg8 = 0 to 4 step 2 {
//                 affine.for %arg9 = 0 to 2 {
//                   affine.for %arg10 = 0 to 2 step 2 {
//                     %11 = affine.vector_load %alloca_3[(%arg5 + %arg7 + %arg9) * 8 + %arg6 + %arg8 + %arg10] : memref<64xf32>, vector<2xf32>
//                     affine.vector_store %11, %alloc[(%arg4 floordiv 64) * 4096 + ((%arg5 + (((%arg4 mod 64) floordiv 32) floordiv 2) * 4) * 8 + %arg7 * 8 + ((%arg4 mod 32) floordiv 4) * 2 + %arg9) * 64 + (%arg6 * 2 + (((%arg4 mod 64) floordiv 32) mod 2) * 4) * 4 + %arg8 * 4 + (%arg4 mod 4) * 2 + %arg10] : memref<8192xf32, 3>, vector<2xf32>
//                   } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//                 } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//               } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         gpu.barrier
//         affine.for %arg5 = 0 to 8 {
//           %11 = affine.vector_load %alloc[(%arg4 floordiv 16 + %arg5 * 8) * 64 + (%arg4 mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
//           affine.vector_store %11, %alloca_3[%arg5 * 4] : memref<64xf32>, vector<4xf32>
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         affine.for %arg5 = 1 to 2 {
//           affine.for %arg6 = 0 to 8 {
//             affine.for %arg7 = 0 to 4 {
//               %11 = affine.load %alloca_3[%arg6 * 4 + %arg7] : memref<64xf32>
//               %12 = affine.load %alloc[%arg5 * 4096 + (%arg4 floordiv 16 + %arg6 * 8) * 64 + (%arg4 mod 16) * 4 + %arg7] : memref<8192xf32, 3>
//               %13 = arith.addf %11, %12 : f32
//               affine.store %13, %alloca_3[%arg6 * 4 + %arg7] : memref<64xf32>
//             } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//           } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//         affine.for %arg5 = 0 to 8 {
//           %11 = affine.vector_load %alloca_3[%arg5 * 4] : memref<64xf32>, vector<4xf32>
//           affine.vector_store %11, %arg2[%0 * 64 + %arg4 floordiv 16 + %arg5 * 8, %1 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
//         } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
//       } {gpu.index = "threadIdx"}
//     } {gpu.index = "blockIdx"}
//     return
//   }
// }

//  === start mlir ===== 修改 %alloc 到内层parallel OK,可生成so，但运行报错 CUDA_ERROR_MISALIGNED_ADDRESS
module {
  func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf32, 1>, %arg1: memref<1024x1024xf32, 1>, %arg2: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu"} {
    affine.parallel (%arg3) = (0) to (256) {
      %0 = affine.apply affine_map<(d0) -> ((d0 floordiv 128) * 8 + d0 mod 8)>(%arg3)
      %1 = affine.apply affine_map<(d0) -> ((d0 mod 128) floordiv 8)>(%arg3)
      %2 = affine.apply affine_map<(d0) -> (d0 floordiv 16)>(%arg3)
      %3 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg3)
      %4 = affine.apply affine_map<(d0) -> (d0 * 64)>(%0)
      %5 = affine.apply affine_map<(d0) -> (d0 * 64)>(%1)
      affine.parallel (%arg4) = (0) to (128) {
        %alloc = memref.alloc() {alignment = 16 : i64, kcg.bufDesc = "smABC"} : memref<8192xf32, 3>
        %6 = affine.apply affine_map<(d0) -> (d0 floordiv 64)>(%arg4)
        %7 = affine.apply affine_map<(d0) -> ((d0 mod 64) floordiv 8)>(%arg4)
        %8 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg4)
        %alloca = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempA"} : memref<8xf32>
        %alloca_0 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "tempB"} : memref<8xf32>
        %alloca_1 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regA"} : memref<2x8xf32>
        %alloca_2 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regB"} : memref<2x8xf32>
        %9 = affine.apply affine_map<(d0) -> (d0 * 8)>(%7)
        %10 = affine.apply affine_map<(d0) -> (d0 * 8)>(%8)
        %alloca_3 = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regC"} : memref<64xf32>
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %alloca_3[%arg5 * 8 + %arg6] : memref<64xf32>
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        affine.for %arg5 = 0 to 2 {
          %11 = affine.vector_load %arg0[%arg4 floordiv 16 + %arg5 * 8, %0 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %11, %alloc[(%arg4 floordiv 16 + %arg5 * 8) * 64 + (%arg4 mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        affine.for %arg5 = 0 to 2 {
          %11 = affine.vector_load %arg1[%arg4 floordiv 16 + %arg5 * 8, %1 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
          affine.vector_store %11, %alloc[(%arg4 floordiv 16 + %arg5 * 8) * 64 + (%arg4 mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        gpu.barrier
        affine.for %arg5 = 0 to 2 {
          affine.for %arg6 = 0 to 2 {
            %11 = affine.vector_load %alloc[(%arg4 floordiv 64) * 64 + (%arg5 + ((%arg4 mod 64) floordiv 32) floordiv 2) * 32 + (%arg6 * 8 + (%arg4 mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
            affine.vector_store %11, %alloca_1[0, %arg5 * 4 + %arg6 * 2] : memref<2x8xf32>, vector<2xf32>
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        affine.for %arg5 = 0 to 2 {
          affine.for %arg6 = 0 to 2 {
            %11 = affine.vector_load %alloc[(%arg4 floordiv 64) * 64 + (%arg5 * 2 + ((%arg4 mod 64) floordiv 32) mod 2) * 16 + (%arg6 * 4 + %arg4 mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
            affine.vector_store %11, %alloca_2[0, %arg5 * 4 + %arg6 * 2] : memref<2x8xf32>, vector<2xf32>
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        affine.for %arg5 = 16 to 1040 step 16 {
          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %11 = affine.vector_load %arg0[%arg4 floordiv 16 + %arg6 * 8 + %arg5, %0 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
              affine.vector_store %11, %alloca[%arg6 * 4] : memref<8xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            affine.for %arg6 = 0 to 2 {
              %11 = affine.vector_load %arg1[%arg4 floordiv 16 + %arg6 * 8 + %arg5, %1 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
              affine.vector_store %11, %alloca_0[%arg6 * 4] : memref<8xf32>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          }
          affine.for %arg6 = 0 to 14 step 2 {
            affine.for %arg7 = 0 to 2 {
              affine.for %arg8 = 0 to 2 {
                %11 = affine.vector_load %alloc[((%arg5 floordiv 16 - 1) mod 2) * 1024 + (%arg6 + %arg4 floordiv 64 + 2) * 64 + (%arg7 + ((%arg4 mod 64) floordiv 32) floordiv 2) * 32 + (%arg8 * 8 + (%arg4 mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
                affine.vector_store %11, %alloca_1[(%arg6 floordiv 2 + 1) mod 2, %arg7 * 4 + %arg8 * 2] : memref<2x8xf32>, vector<2xf32>
              } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            affine.for %arg7 = 0 to 2 {
              affine.for %arg8 = 0 to 2 {
                %11 = affine.vector_load %alloc[((%arg5 floordiv 16 - 1) mod 2) * 1024 + (%arg6 + %arg4 floordiv 64 + 2) * 64 + (%arg7 * 2 + ((%arg4 mod 64) floordiv 32) mod 2) * 16 + (%arg8 * 4 + %arg4 mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
                affine.vector_store %11, %alloca_2[(%arg6 floordiv 2 + 1) mod 2, %arg7 * 4 + %arg8 * 2] : memref<2x8xf32>, vector<2xf32>
              } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            affine.for %arg7 = 0 to 8 {
              affine.for %arg8 = 0 to 8 {
                %11 = affine.load %alloca_3[%arg7 * 8 + %arg8] : memref<64xf32>
                %12 = affine.load %alloca_1[(%arg6 floordiv 2) mod 2, %arg7] : memref<2x8xf32>
                %13 = affine.load %alloca_2[(%arg6 floordiv 2) mod 2, %arg8] : memref<2x8xf32>
                %14 = arith.mulf %12, %13 : f32
                %15 = arith.addf %14, %11 : f32
                affine.store %15, %alloca_3[%arg7 * 8 + %arg8] : memref<64xf32>
              } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          }
          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg5) {
            affine.for %arg6 = 0 to 2 {
              %11 = affine.vector_load %alloca[%arg6 * 4] : memref<8xf32>, vector<4xf32>
              affine.vector_store %11, %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 16 + %arg6 * 8) * 64 + (%arg4 mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            affine.for %arg6 = 0 to 2 {
              %11 = affine.vector_load %alloca_0[%arg6 * 4] : memref<8xf32>, vector<4xf32>
              affine.vector_store %11, %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 16 + %arg6 * 8) * 64 + (%arg4 mod 16) * 4 + 2048] : memref<8192xf32, 3>, vector<4xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            gpu.barrier
          }
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              %11 = affine.load %alloca_3[%arg6 * 8 + %arg7] : memref<64xf32>
              %12 = affine.load %alloca_1[1, %arg6] : memref<2x8xf32>
              %13 = affine.load %alloca_2[1, %arg7] : memref<2x8xf32>
              %14 = arith.mulf %12, %13 : f32
              %15 = arith.addf %14, %11 : f32
              affine.store %15, %alloca_3[%arg6 * 8 + %arg7] : memref<64xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %11 = affine.vector_load %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 64) * 64 + (%arg6 + ((%arg4 mod 64) floordiv 32) floordiv 2) * 32 + (%arg7 * 8 + (%arg4 mod 32) floordiv 4) * 2] : memref<8192xf32, 3>, vector<2xf32>
              affine.vector_store %11, %alloca_1[0, %arg6 * 4 + %arg7 * 2] : memref<2x8xf32>, vector<2xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %11 = affine.vector_load %alloc[((%arg5 floordiv 16) mod 2) * 1024 + (%arg4 floordiv 64) * 64 + (%arg6 * 2 + ((%arg4 mod 64) floordiv 32) mod 2) * 16 + (%arg7 * 4 + %arg4 mod 4) * 2 + 2048] : memref<8192xf32, 3>, vector<2xf32>
              affine.vector_store %11, %alloca_2[0, %arg6 * 4 + %arg7 * 2] : memref<2x8xf32>, vector<2xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        }
        gpu.barrier
        affine.for %arg5 = 0 to 8 step 4 {
          affine.for %arg6 = 0 to 8 step 4 {
            affine.for %arg7 = 0 to 4 step 2 {
              affine.for %arg8 = 0 to 4 step 2 {
                affine.for %arg9 = 0 to 2 {
                  affine.for %arg10 = 0 to 2 step 2 {
                    %11 = affine.vector_load %alloca_3[(%arg5 + %arg7 + %arg9) * 8 + %arg6 + %arg8 + %arg10] : memref<64xf32>, vector<2xf32>
                    affine.vector_store %11, %alloc[(%arg4 floordiv 64) * 4096 + ((%arg5 + (((%arg4 mod 64) floordiv 32) floordiv 2) * 4) * 8 + %arg7 * 8 + ((%arg4 mod 32) floordiv 4) * 2 + %arg9) * 64 + (%arg6 * 2 + (((%arg4 mod 64) floordiv 32) mod 2) * 4) * 4 + %arg8 * 4 + (%arg4 mod 4) * 2 + %arg10] : memref<8192xf32, 3>, vector<2xf32>
                  } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
                } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
              } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        gpu.barrier
        affine.for %arg5 = 0 to 8 {
          %11 = affine.vector_load %alloc[(%arg4 floordiv 16 + %arg5 * 8) * 64 + (%arg4 mod 16) * 4] : memref<8192xf32, 3>, vector<4xf32>
          affine.vector_store %11, %alloca_3[%arg5 * 4] : memref<64xf32>, vector<4xf32>
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        affine.for %arg5 = 1 to 2 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 4 {
              %11 = affine.load %alloca_3[%arg6 * 4 + %arg7] : memref<64xf32>
              %12 = affine.load %alloc[%arg5 * 4096 + (%arg4 floordiv 16 + %arg6 * 8) * 64 + (%arg4 mod 16) * 4 + %arg7] : memref<8192xf32, 3>
              %13 = arith.addf %11, %12 : f32
              affine.store %13, %alloca_3[%arg6 * 4 + %arg7] : memref<64xf32>
            } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
          } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
        affine.for %arg5 = 0 to 8 {
          %11 = affine.vector_load %alloca_3[%arg5 * 4] : memref<64xf32>, vector<4xf32>
          affine.vector_store %11, %arg2[%0 * 64 + %arg4 floordiv 16 + %arg5 * 8, %1 * 64 + (%arg4 mod 16) * 4] : memref<1024x1024xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll", affine.unroll.num = 16 : i32}
      } {gpu.index = "threadIdx"}
    } {gpu.index = "blockIdx"}
    return
  }
}


// ===== after parallel =======  OK 
// module {
//   func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf32, 1>, %arg1: memref<1024x1024xf32, 1>, %arg2: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu"} {
//     affine.parallel (%arg3, %arg4) = (0, 0) to (16, 16) {
//       %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
//       %1 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg4)
//       affine.parallel (%arg5, %arg6) = (0, 0) to (8, 8) {
//         %2 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg5)
//         %3 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg6)
//         affine.for %arg7 = 0 to 8 {
//           affine.for %arg8 = 0 to 8 {
//             %cst = arith.constant 0.000000e+00 : f32
//             %4 = affine.for %arg9 = 0 to 1024 iter_args(%arg10 = %cst) -> (f32) {
//               %5 = affine.load %arg0[%arg9, %0 + %2 + %arg7] : memref<1024x1024xf32, 1>
//               %6 = affine.load %arg1[%arg9, %1 + %3 + %arg8] : memref<1024x1024xf32, 1>
//               %7 = arith.mulf %5, %6 : f32
//               %8 = arith.addf %7, %arg10 : f32
//               affine.yield %8 : f32
//             }
//             affine.store %4, %arg2[%0 + %2 + %arg7, %1 + %3 + %arg8] : memref<1024x1024xf32, 1>
//           }
//         }
//       } {gpu.index = "threadIdx"}
//     } {gpu.index = "blockIdx"}
//     return
//   }
// }


// // ===== after reorder =======
// module {
//   func.func public @GEMM_testKernel(%arg0: memref<1024x1024xf32, 1>, %arg1: memref<1024x1024xf32, 1>, %arg2: memref<1024x1024xf32, 1>) attributes {func.op.name = "Matmul", func.state = "gpu"} {
//     affine.parallel (%arg3, %arg4) = (0, 0) to (16, 16) {
//       %0 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
//       %1 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg4)
//       affine.parallel (%arg5, %arg6, %arg7) = (0, 0, 0) to (2, 8, 8) {
//         %2 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg6)
//         %3 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg7)
//         %alloca = memref.alloca() {alignment = 16 : i64, kcg.bufDesc = "regC"} : memref<8x8xf32>
//         affine.for %arg8 = 0 to 8 {
//           affine.for %arg9 = 0 to 8 {
//             %cst = arith.constant 0.000000e+00 : f32
//             affine.store %cst, %alloca[%arg8, %arg9] : memref<8x8xf32>
//           }
//         }
//         affine.for %arg8 = 0 to 1024 step 16 {
//           affine.for %arg9 = 0 to 16 step 2 {
//             affine.for %arg10 = 0 to 8 {
//               affine.for %arg11 = 0 to 8 {
//                 %4 = affine.load %alloca[%arg10, %arg11] : memref<8x8xf32>
//                 %5 = affine.load %arg0[%arg8 + %arg9 + %arg5, %0 + %2 + %arg10] : memref<1024x1024xf32, 1>
//                 %6 = affine.load %arg1[%arg8 + %arg9 + %arg5, %1 + %3 + %arg11] : memref<1024x1024xf32, 1>
//                 %7 = arith.mulf %5, %6 : f32
//                 %8 = arith.addf %7, %4 : f32
//                 affine.store %8, %alloca[%arg10, %arg11] : memref<8x8xf32>
//               }
//             }
//           }
//         }
//         affine.for %arg8 = 0 to 8 {
//           affine.for %arg9 = 0 to 8 {
//             %4 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf32>
//             affine.store %4, %arg2[%0 + %2 + %arg8, %1 + %3 + %arg9] : memref<1024x1024xf32, 1>
//           }
//         }
//       } {gpu.index = "threadIdx"}
//     } {gpu.index = "blockIdx"}
//     return
//   }
// }