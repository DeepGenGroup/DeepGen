amdgcn=$1

/opt/dtk/bin/clang --target=amdgcn-amd-amdhsa -mcpu=gfx906 -x assembler -c $amdgcn -o kernel.hsaco