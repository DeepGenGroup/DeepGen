amdgcn=$1

CLANG=/home/xushilong/llvm-install/bin/clang
#  --target=amdgcn-amd-amdhsa -mcpu=gfx906 -x assembler -c $amdgcn -o kernel.hsaco

$CLANG -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx906 -c -o test.o $amdgcn
$CLANG -target amdgcn-amd-amdhsa test.o -o kernel.hsaco