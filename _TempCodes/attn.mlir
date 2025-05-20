onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":1:1) -> builtin.module
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":10:5) -> func.return
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":9:10) -> onnx.MatMul
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":8:10) -> onnx.Softmax
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":7:10) -> onnx.Mul
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":6:10) -> onnx.Constant
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":5:10) -> onnx.MatMul
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":4:10) -> onnx.Transpose
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":2:3) -> func.func
onnxMap : loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":3:13) -> torch.constant.none

======== onnx->torch ===========
module {
  func.func @main_graph(%arg0: !torch.vtensor<[2,8,10,64],f32>, %arg1: !torch.vtensor<[2,8,10,64],f32>, %arg2: !torch.vtensor<[2,8,10,64],f32>) -> !torch.vtensor<[2,8,10,64],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.5.0"} {
    %none = torch.constant.none
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %true = torch.constant.bool true
    %0 = torch.vtensor.literal(dense<1.250000e-01> : tensor<f32>) : !torch.vtensor<[],f32>
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %1 = torch.aten.transpose.int %arg1, %int2, %int3 : !torch.vtensor<[2,8,10,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,8,64,10],f32>
    %2 = torch.aten.matmul %arg0, %1 : !torch.vtensor<[2,8,10,64],f32>, !torch.vtensor<[2,8,64,10],f32> -> !torch.vtensor<[2,8,10,10],f32>
    %3 = torch.aten.mul.Tensor %2, %0 : !torch.vtensor<[2,8,10,10],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[2,8,10,10],f32>
    %values, %indices = torch.aten.max.dim %3, %int3, %true : !torch.vtensor<[2,8,10,10],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,8,10,1],f32>, !torch.vtensor<[2,8,10,1],si64>
    %4 = torch.aten.sub.Tensor %3, %values, %float1.000000e00 : !torch.vtensor<[2,8,10,10],f32>, !torch.vtensor<[2,8,10,1],f32>, !torch.float -> !torch.vtensor<[2,8,10,10],f32>
    %5 = torch.aten.exp %4 : !torch.vtensor<[2,8,10,10],f32> -> !torch.vtensor<[2,8,10,10],f32>
    %6 = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
    %7 = torch.aten.sum.dim_IntList %5, %6, %true, %none : !torch.vtensor<[2,8,10,10],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,8,10,1],f32>
    %8 = torch.aten.div.Tensor %5, %7 : !torch.vtensor<[2,8,10,10],f32>, !torch.vtensor<[2,8,10,1],f32> -> !torch.vtensor<[2,8,10,10],f32>
    %9 = torch.aten.matmul %8, %arg2 : !torch.vtensor<[2,8,10,10],f32>, !torch.vtensor<[2,8,10,64],f32> -> !torch.vtensor<[2,8,10,64],f32>
    return %9 : !torch.vtensor<[2,8,10,64],f32>
  }
}
loc("/home/xushilong/DeepGen/_TempCodes/onnx.mlir":1:1): error: 'builtin.module' op trying to schedule a pass on an unsupported operation

======== torch->stablehlo ===========
module {
  func.func @main_graph(%arg0: tensor<2x8x10x64xf32>, %arg1: tensor<2x8x10x64xf32>, %arg2: tensor<2x8x10x64xf32>) -> tensor<2x8x10x64xf32> {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<2x8x10x64xf32>) -> tensor<2x8x64x10xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<2x8x64x10xf32>) -> tensor<2x8x64x10xf32>
    %2 = stablehlo.dot_general %arg0, %1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x8x10x64xf32>, tensor<2x8x64x10xf32>) -> tensor<2x8x10x10xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2, 3] : (tensor<2x8x10x10xf32>) -> tensor<2x8x10x10xf32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8x10x10xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<2x8x10x10xf32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<2x8x10x10xf32>, tensor<f32>) -> tensor<2x8x10xf32>
    %7 = stablehlo.reshape %6 : (tensor<2x8x10xf32>) -> tensor<2x8x10x1xf32>
    %8 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3] : (tensor<2x8x10x10xf32>) -> tensor<2x8x10x10xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<2x8x10x1xf32>) -> tensor<2x8x10x10xf32>
    %10 = stablehlo.subtract %8, %9 : tensor<2x8x10x10xf32>
    %11 = stablehlo.exponential %10 : tensor<2x8x10x10xf32>
    %12 = stablehlo.reduce(%11 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<2x8x10x10xf32>, tensor<f32>) -> tensor<2x8x10xf32>
    %13 = stablehlo.reshape %12 : (tensor<2x8x10xf32>) -> tensor<2x8x10x1xf32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<2x8x10x10xf32>) -> tensor<2x8x10x10xf32>
    %15 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2, 3] : (tensor<2x8x10x1xf32>) -> tensor<2x8x10x10xf32>
    %16 = stablehlo.divide %14, %15 : tensor<2x8x10x10xf32>
    %17 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<2x8x10x64xf32>) -> tensor<2x8x10x64xf32>
    %18 = stablehlo.dot_general %16, %17, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x8x10x10xf32>, tensor<2x8x10x64xf32>) -> tensor<2x8x10x64xf32>
    return %18 : tensor<2x8x10x64xf32>
  }
}

======== stablehlo -> linalg =========== 
module {
  func.func @main_graph(%arg0: tensor<2x8x10x64xf32>, %arg1: tensor<2x8x10x64xf32>, %arg2: tensor<2x8x10x64xf32>) -> tensor<2x8x10x64xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<1.250000e-01> : tensor<f32>
    %0 = tensor.empty() : tensor<2x8x64x10xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<2x8x10x64xf32>) outs(%0 : tensor<2x8x64x10xf32>) permutation = [0, 1, 3, 2] 
    %1 = tensor.empty() : tensor<2x8x10x10xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<2x8x10x10xf32>) -> tensor<2x8x10x10xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %transposed : tensor<2x8x10x64xf32>, tensor<2x8x64x10xf32>) outs(%2 : tensor<2x8x10x10xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %18 = arith.mulf %in, %in_8 : f32
      %19 = arith.addf %out, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<2x8x10x10xf32>
    %4 = tensor.empty() : tensor<2x8x10x10xf32>
    %broadcasted = linalg.broadcast ins(%cst_1 : tensor<f32>) outs(%4 : tensor<2x8x10x10xf32>) dimensions = [0, 1, 2, 3] 
    %5 = tensor.empty() : tensor<2x8x10x10xf32>
    %mapped = linalg.map { arith.mulf } ins(%3, %broadcasted : tensor<2x8x10x10xf32>, tensor<2x8x10x10xf32>) outs(%5 : tensor<2x8x10x10xf32>)
    %6 = tensor.empty() : tensor<2x8x10xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2x8x10xf32>) -> tensor<2x8x10xf32>
    %reduced = linalg.reduce { arith.maximumf } ins(%mapped : tensor<2x8x10x10xf32>) outs(%7 : tensor<2x8x10xf32>) dimensions = [3] 
    %8 = tensor.empty() : tensor<2x8x10x10xf32>
    %broadcasted_2 = linalg.broadcast ins(%reduced : tensor<2x8x10xf32>) outs(%8 : tensor<2x8x10x10xf32>) dimensions = [3] 
    %9 = tensor.empty() : tensor<2x8x10x10xf32>
    %mapped_3 = linalg.map { arith.subf } ins(%mapped, %broadcasted_2 : tensor<2x8x10x10xf32>, tensor<2x8x10x10xf32>) outs(%9 : tensor<2x8x10x10xf32>)
    %10 = tensor.empty() : tensor<2x8x10x10xf32>
    %mapped_4 = linalg.map { math.exp } ins(%mapped_3 : tensor<2x8x10x10xf32>) outs(%10 : tensor<2x8x10x10xf32>)
    %11 = tensor.empty() : tensor<2x8x10xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<2x8x10xf32>) -> tensor<2x8x10xf32>
    %reduced_5 = linalg.reduce { arith.addf } ins(%mapped_4 : tensor<2x8x10x10xf32>) outs(%12 : tensor<2x8x10xf32>) dimensions = [3] 
    %13 = tensor.empty() : tensor<2x8x10x10xf32>
    %broadcasted_6 = linalg.broadcast ins(%reduced_5 : tensor<2x8x10xf32>) outs(%13 : tensor<2x8x10x10xf32>) dimensions = [3] 
    %14 = tensor.empty() : tensor<2x8x10x10xf32>
    %mapped_7 = linalg.map { arith.divf } ins(%mapped_4, %broadcasted_6 : tensor<2x8x10x10xf32>, tensor<2x8x10x10xf32>) outs(%14 : tensor<2x8x10x10xf32>)
    %15 = tensor.empty() : tensor<2x8x10x64xf32>
    %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<2x8x10x64xf32>) -> tensor<2x8x10x64xf32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%mapped_7, %arg2 : tensor<2x8x10x10xf32>, tensor<2x8x10x64xf32>) outs(%16 : tensor<2x8x10x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %18 = arith.mulf %in, %in_8 : f32
      %19 = arith.addf %out, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<2x8x10x64xf32>
    return %17 : tensor<2x8x10x64xf32>
  }
}

======== linalg->affine (with LICM) =========== 
module {
  func.func @main_graph(%arg0: memref<2x8x10x64xf32>, %arg1: memref<2x8x10x64xf32>, %arg2: memref<2x8x10x64xf32>) -> memref<2x8x10x64xf32> {
    %cst = arith.constant 1.250000e-01 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32

    // 转置
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x8x64x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %arg1[%arg3, %arg4, %arg6, %arg5] : memref<2x8x10x64xf32>
            affine.store %0, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<2x8x64x10xf32>
          }
        }
      }
    }
    // init 0.0f
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            affine.store %cst_1, %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // dot q*k
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            affine.for %arg7 = 0 to 64 {
              %0 = affine.load %arg0[%arg3, %arg4, %arg5, %arg7] : memref<2x8x10x64xf32>
              %1 = affine.load %alloc[%arg3, %arg4, %arg7, %arg6] : memref<2x8x64x10xf32>
              %2 = affine.load %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
              %3 = arith.mulf %0, %1 : f32
              %4 = arith.addf %2, %3 : f32
              affine.store %4, %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            }
          }
        }
      }
    }
    // 根号1/d
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            affine.store %cst, %alloc_3[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // p * 根号1/d
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_3[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %2 = arith.mulf %0, %1 : f32
            affine.store %2, %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // -无穷
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.store %cst_0, %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
        }
      }
    }
    // max reduce
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            %2 = arith.maximumf %1, %0 : f32
            affine.store %2, %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
          }
        }
      }
    }
    // broadcast max值
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            affine.store %0, %alloc_6[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // p - max
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_6[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %2 = arith.subf %0, %1 : f32
            affine.store %2, %alloc_7[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // exp(p - max)
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_7[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = math.exp %0 : f32
            affine.store %1, %alloc_8[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // init sum 0.0f
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.store %cst_1, %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
        }
      }
    }
    // sum reduce sum(exp(p - max))
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_8[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            %2 = arith.addf %1, %0 : f32
            affine.store %2, %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
          }
        }
      }
    }
    // broadcast sum
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            affine.store %0, %alloc_10[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // exp(p - max) / sum(exp(p - max))
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_8[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_10[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %2 = arith.divf %0, %1 : f32
            affine.store %2, %alloc_11[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          }
        }
      }
    }
    // init 0.0f
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x64xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 64 {
            affine.store %cst_1, %alloc_12[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x64xf32>
          }
        }
      }
    }
    // dot
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 64 {
            affine.for %arg7 = 0 to 10 {
              %0 = affine.load %alloc_11[%arg3, %arg4, %arg5, %arg7] : memref<2x8x10x10xf32>
              %1 = affine.load %arg2[%arg3, %arg4, %arg7, %arg6] : memref<2x8x10x64xf32>
              %2 = affine.load %alloc_12[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x64xf32>
              %3 = arith.mulf %0, %1 : f32
              %4 = arith.addf %2, %3 : f32
              affine.store %4, %alloc_12[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x64xf32>
            }
          }
        }
      }
    }
    return %alloc_12 : memref<2x8x10x64xf32>
  }
}
outmostLoop size = 16
module {
  func.func @main_graph(%arg0: memref<2x8x10x64xf32>, %arg1: memref<2x8x10x64xf32>, %arg2: memref<2x8x10x64xf32>) -> memref<2x8x10x64xf32> {
    %cst = arith.constant 1.250000e-01 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x8x64x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %arg1[%arg3, %arg4, %arg6, %arg5] : memref<2x8x10x64xf32>
            affine.store %0, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<2x8x64x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:4:10)", onnx.op = "onnx.Transpose"}
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            affine.store %cst_1, %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:5:10)", onnx.op = "onnx.MatMul"}
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            affine.for %arg7 = 0 to 64 {
              %0 = affine.load %arg0[%arg3, %arg4, %arg5, %arg7] : memref<2x8x10x64xf32>
              %1 = affine.load %alloc[%arg3, %arg4, %arg7, %arg6] : memref<2x8x64x10xf32>
              %2 = affine.load %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
              %3 = arith.mulf %0, %1 : f32
              %4 = arith.addf %2, %3 : f32
              affine.store %4, %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            } {canParallel = "no"}
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:5:10)", onnx.op = "onnx.MatMul"}
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            affine.store %cst, %alloc_3[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:7:10)", onnx.op = "onnx.Mul"}
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_2[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_3[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %2 = arith.mulf %0, %1 : f32
            affine.store %2, %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:7:10)", onnx.op = "onnx.Mul"}
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.store %cst_0, %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
        } {func.desc = "x"}
      } {func.desc = "y"}
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            %2 = arith.maximumf %1, %0 : f32
            affine.store %2, %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
          } {canParallel = "no"}
        } {func.desc = "x"}
      } {func.desc = "y"}
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_5[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            affine.store %0, %alloc_6[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_4[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_6[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %2 = arith.subf %0, %1 : f32
            affine.store %2, %alloc_7[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_7[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = math.exp %0 : f32
            affine.store %1, %alloc_8[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.store %cst_1, %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
        } {func.desc = "x"}
      } {func.desc = "y"}
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_8[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            %2 = arith.addf %1, %0 : f32
            affine.store %2, %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
          } {canParallel = "no"}
        } {func.desc = "x"}
      } {func.desc = "y"}
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_9[%arg3, %arg4, %arg5] : memref<2x8x10xf32>
            affine.store %0, %alloc_10[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x10xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 10 {
            %0 = affine.load %alloc_8[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %1 = affine.load %alloc_10[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
            %2 = arith.divf %0, %1 : f32
            affine.store %2, %alloc_11[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x10xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:8:10)", onnx.op = "onnx.Softmax"}
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<2x8x10x64xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 64 {
            affine.store %cst_1, %alloc_12[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x64xf32>
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:9:10)", onnx.op = "onnx.MatMul"}
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 10 {
          affine.for %arg6 = 0 to 64 {
            affine.for %arg7 = 0 to 10 {
              %0 = affine.load %alloc_11[%arg3, %arg4, %arg5, %arg7] : memref<2x8x10x10xf32>
              %1 = affine.load %arg2[%arg3, %arg4, %arg7, %arg6] : memref<2x8x10x64xf32>
              %2 = affine.load %alloc_12[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x64xf32>
              %3 = arith.mulf %0, %1 : f32
              %4 = arith.addf %2, %3 : f32
              affine.store %4, %alloc_12[%arg3, %arg4, %arg5, %arg6] : memref<2x8x10x64xf32>
            } {canParallel = "no"}
          } {func.desc = "x"}
        } {func.desc = "y"}
      }
    } {loopLevel = "outmostLoop", onnx.loc = "loc(\22/home/xushilong/DeepGen/_TempCodes/onnx.mlir\22:9:10)", onnx.op = "onnx.MatMul"}
    return %alloc_12 : memref<2x8x10x64xf32>
  }
}
