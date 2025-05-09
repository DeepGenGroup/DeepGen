#!/bin/bash
llvm_install_dir=~/rocm-llvm-install
# model=/home/xushilong/DeepGen/_TempCodes/model/fixed_transformer_sim.onnx
# model=/home/xushilong/DeepGen/_TempCodes/model/softmax_model.onnx
model=/home/xushilong/DeepGen/_TempCodes/model/self_attention.onnx
# model=/home/xushilong/onnxMLIRLearn/data/mnist.onnx
# model=/home/xushilong/onnxMLIRLearn/data/auto_Opset16_sim.onnx  #  error: failed to legalize operation 'torch.aten.cumsum' that was explicitly marked illegal
# model=/home/xushilong/onnxMLIRLearn/data/model_structure.onnx  

OUT_DIR=/home/xushilong/DeepGen/_TempCodes
IMPORT_ONNX=$llvm_install_dir/bin/torch-mlir-import-onnx
TORCH_MLIR_OPT=$llvm_install_dir/bin/torch-mlir-opt
MLIR_OPT=$llvm_install_dir/bin/mlir-opt
MLIR_TRANSLATE=$llvm_install_dir/bin/mlir-translate
MLIR_RUNNER=$llvm_install_dir/bin/mlir-runner


cd $OUT_DIR
echo ===Start Import model And lower to onnxDialect
$IMPORT_ONNX  --data-prop  --opset-version 14 $model > onnx.mlir 
echo ===Start Lowering to torchMLIR

# Equal to : torch-onnx-to-torch-backend-pipeline 
$TORCH_MLIR_OPT  -convert-torch-onnx-to-torch    onnx.mlir >  000t.mlir
echo ===Start Lowering to 000
$TORCH_MLIR_OPT  --pass-pipeline="builtin.module(torch-function-to-torch-backend-pipeline)"     000t.mlir > 000.mlir

# $TORCH_MLIR_OPT  --pass-pipeline="builtin.module(torch-onnx-to-torch-backend-pipeline)"    onnx.mlir >  000.mlir
echo ===wating111
$TORCH_MLIR_OPT  --torch-backend-to-stablehlo-backend-pipeline -stablehlo-aggressive-simplification 000.mlir > 111.mlir   # 删除 in out形状相同的 broadcastOp 
echo ===wating222
$TORCH_MLIR_OPT  --pass-pipeline="builtin.module(stablehlo-legalize-to-linalg{enable-primitive-ops}, canonicalize)" 111.mlir > 222Linalg.mlir
echo ===222affine

# $MLIR_OPT --linalg-fold-into-elementwise  222Linalg.mlir > 222Linalg_opt1.mlir   无效pass，不起作用
# sed -i '1d' 222Linalg_opt1.mlir
# $MLIR_OPT --linalg-fuse-elementwise-ops   222Linalg_opt1.mlir > 222Linalg_opt2.mlir  无效pass，不起作用

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
# $MLIR_OPT --pass-pipeline="builtin.module(\
#     func.func(affine-loop-invariant-code-motion), \
#     func.func(affine-loop-unroll-jam), \
#     canonicalize, \
#     cse \
#     )" \
#     222affine.mlir > 222affine_fuse.mlir

$MLIR_OPT --pass-pipeline="builtin.module(func.func(affine-loop-coalescing), affine-loop-fusion{mode=greedy}, \
    func.func(affine-loop-invariant-code-motion), \
    canonicalize, \
    cse \
    )" \
    222affine.mlir > 222affine_co_fuse.mlir

echo ===222affinepara
$MLIR_OPT --affine-parallelize -canonicalize 222affine_co_fuse.mlir > 222affine_para.mlir
echo ===222scfpara
$MLIR_OPT --lower-affine  -canonicalize 222affine_para.mlir > 222scfpara.mlir
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

# For HIP
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

# For CUDA [测试 转化 fixed_transformer_sim.onnx 时会出错], 应使用 -gpu-lower-to-nvvm-pipeline
# $MLIR_OPT --pass-pipeline="builtin.module(\
#     nvvm-attach-target{chip=sm_80 O=3}, \
#     gpu-decompose-memrefs, \
#     lower-affine, \
#     convert-scf-to-cf, \
#     gpu.module(convert-gpu-to-nvvm), \
#     convert-index-to-llvm, \
#     reconcile-unrealized-casts, \
#     canonicalize \
#      )"   444.mlir > 555.mlir 


# For CUDA Pipeline
# $MLIR_OPT 444.mlir  -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 opt-level=3" > 555.mlir

echo ===wating666
$MLIR_OPT --pass-pipeline="builtin.module(\
    convert-func-to-llvm, \
    gpu-to-llvm, \
    gpu-module-to-binary, \
    canonicalize \
     )"   555.mlir > 666.mlir  # Can be run with mlir-runner : mlir-runner 666.mlir -e main_graph

# echo ===wating Runner
# $MLIR_RUNNER 666.mlir -e main_graph
# $MLIR_TRANSLATE 666.mlir --mlir-to-llvmir -o 777.ll



LLC="$HOME/rocm-llvm-install/bin/llc"

$MLIR_TRANSLATE --mlir-to-llvmir 666.mlir -o testLL.ll  # emit llIR
$LLC -relocation-model=pic -mcpu=native -filetype=obj testLL.ll -o testLL.o  # 使用PIC模式生成目标文件
nvcc  -L/home/xushilong/rocm-llvm-install/lib -lmlir_cuda_runtime -shared testLL.o -o output.so  # 生成so文件

# cd $OUT_DIR/build 
# cmake .. & make 
