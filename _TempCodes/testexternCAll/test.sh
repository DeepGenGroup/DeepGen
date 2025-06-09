llvm_install_dir=~/rocm-llvm-install
MLIR_OPT=$llvm_install_dir/bin/mlir-opt
MLIR_TRANSLATE=$llvm_install_dir/bin/mlir-translate
LLC="$HOME/rocm-llvm-install/bin/llc"

fname=/home/xushilong/DeepGen/_TempCodes/testexternCAll/00TEstExtern.mlir
 
$MLIR_OPT -lower-affine  -convert-func-to-llvm -canonicalize $fname > testLL.mlir
$MLIR_TRANSLATE --mlir-to-llvmir testLL.mlir -o testLL.ll  # emit llIR
$LLC -relocation-model=pic -mcpu=native -filetype=obj testLL.ll -o testLL.o  # 使用PIC模式生成目标文件
g++ testLL.o -o testExe -L/home/xushilong/DeepGen/_TempCodes/testexternCAll -lsayhello

export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
./testExe

