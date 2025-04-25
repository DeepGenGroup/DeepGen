#!/bin/bash
model=/home/xushilong/onnxMLIRLearn/data/self_attention.onnx
# model=/home/xushilong/onnxMLIRLearn/data/fixed_transformer_sim.onnx
# model=/home/xushilong/onnxMLIRLearn/data/mnist.onnx
# model=/home/xushilong/onnxMLIRLearn/data/auto_Opset16_sim.onnx  #  error: failed to legalize operation 'torch.aten.cumsum' that was explicitly marked illegal
# model=/home/xushilong/onnxMLIRLearn/data/model_structure.onnx  

IMPORT_ONNX=~/llvm-install/bin/torch-mlir-import-onnx
TORCH_MLIR_OPT=~/llvm-install/bin/torch-mlir-opt
MLIR_OPT=~/llvm-install/bin/mlir-opt

OUT_DIR=/home/xushilong/onnxMLIRLearn/testTorchMLIR

echo ===Start Import model And lower to onnxDialect
$IMPORT_ONNX  --data-prop $model > ${OUT_DIR}/onnx.mlir 
echo ===Start Lowering to torchMLIR
$TORCH_MLIR_OPT  --torch-onnx-to-torch-backend-pipeline  ${OUT_DIR}/onnx.mlir > 000.mlir
echo ===wating111
$TORCH_MLIR_OPT  --torch-backend-to-stablehlo-backend-pipeline 000.mlir > 111.mlir
echo ===wating222
$TORCH_MLIR_OPT  --pass-pipeline="builtin.module(stablehlo-legalize-to-linalg{enable-primitive-ops}, canonicalize)" 111.mlir > 222Linalg.mlir
echo ===222affine

$MLIR_OPT -one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map " \
    -convert-linalg-to-affine-loops \
    -canonicalize \
    -affine-loop-invariant-code-motion \
    -canonicalize \
    -cse \
    -remove-dead-values \
    222Linalg.mlir > 222affine.mlir
    # --affine-loop-fusion \   两个4层for合为一个4层for,ubs 不变
    # --affine-loop-coalescing \   4层for变1层for，ubs改变


echo ===222affinepara
$MLIR_OPT --affine-parallelize -canonicalize 222affine.mlir > 222affinepara.mlir
echo ===222scfpara
$MLIR_OPT --lower-affine  -canonicalize 222affinepara.mlir > 222scfpara.mlir
# $MLIR_OPT -one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map " \
#     -convert-linalg-to-parallel-loops \
#     -canonicalize \
#     222.mlir > 222para.mlir

echo ===wating333
$MLIR_OPT \
    -gpu-map-parallel-loops \
    -convert-parallel-loops-to-gpu \
    -gpu-kernel-outlining \
    -canonicalize \
    -cse  \
    222scfpara.mlir > 333gpu.mlir

echo ===wating444
$MLIR_OPT --affine-simplify-structures 333gpu.mlir > 444.mlir
echo ===wating555

$MLIR_OPT --pass-pipeline="builtin.module(\
    rocdl-attach-target{chip=gfx906 O=3 triple=amdgcn-amd-amdhsa}, \
    gpu-decompose-memrefs, \
    lower-affine, \
    convert-scf-to-cf, \
    gpu.module(convert-gpu-to-rocdl{use-bare-ptr-memref-call-conv }), \
    convert-index-to-llvm, \
    reconcile-unrealized-casts, \
    canonicalize \
     )"   444.mlir > 555.mlir 


# $MLIR_OPT --pass-pipeline="builtin.module(\
#     convert-func-to-llvm, \
#     gpu-module-to-binary{toolkit=/usr/bin/clang} \
#      )"   555.mlir > 666.mlir 