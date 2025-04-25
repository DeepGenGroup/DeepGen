# !/bin/bash

pybind11_DIR=/home/xushilong/anaconda3/envs/py310/lib/python3.10/site-packages/pybind11/share/cmake/pybind11
nanobind_DIR=/home/xushilong/anaconda3/envs/py310/lib/python3.10/site-packages/nanobind/cmake 
llvm_install_dir=$HOME/rocm-llvm-install

# Update compilers to support cxx20
update_compiler_support_cxx20(){
    if test -e $llvm_install_dir/gcc
    then
        echo "==== gcc already updated"
    else
        conda install gcc_linux-64=11.2.0
        ln -s  $HOME/anaconda3/envs/py310/bin/x86_64-conda_cos7-linux-gnu-gcc  $llvm_install_dir/gcc
    fi
    if test -e $llvm_install_dir/g++
    then
        echo "==== g++ already updated"
    else
        conda install gxx_linux-64=11.2.0
        ln -s  $HOME/anaconda3/envs/py310/bin/x86_64-conda_cos7-linux-gnu-g++  $llvm_install_dir/g++
    fi
}

# Install rocm-llvm-project
install_rocm_llvm(){
    if test -e $HOME/rocm-llvm-project
    then
        echo "==== rocm-llvm-project exsits"
    else
        echo "==== Installing rocm-llvm-project"
        cd $HOME ; git clone https://github.com/DeepGenGroup/rocm-llvm-project.git ;
        git switch deepgen-dev;
        mkdir build ; cd build;
        cmake -G Ninja ../llvm   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
            -DLLVM_BUILD_EXAMPLES=ON \
            -DLLVM_TARGETS_TO_BUILD="host;Native;NVPTX;AMDGPU" \
            -DCMAKE_BUILD_TYPE=Release \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -Wno-unused-but-set-parameter \
            -Dpybind11_DIR=$pybind11_DIR \
            -Dnanobind_DIR=$nanobind_DIR \
            -DCMAKE_INSTALL_PREFIX=$llvm_install_dir
        ninja -j8; ninja install; 
    fi
}

# Install protobuf v3.21.12
install_protobuf(){
    if test -e $HOME/protobuf
    then
        echo "==== protobuf exsits"
    else
        echo "==== Installing protobuf 3.21.12"
        cd $HOME ; git clone https://github.com/protocolbuffers/protobuf.git ; 
        cd protobuf; git reset --hard v3.21.12; git submodule update --init --recursive;
        mkdir build; cd build; 
        cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/protobuf-3.21.12-install ; make -j8; make install
    fi
}

# Install torch-mlir for rocm
install_torch_mlir(){
    if test -e "$HOME/rocm-torch-mlir"
    then
        echo "==== rocm-torch-mlir exsits"
    else
        echo "==== Installing rocm-torch-mlir "
        cd $HOME ; git clone https://github.com/DeepGenGroup/rocm-torch-mlir.git ; cd rocm-torch-mlir ;
        git switch xsl_self; git submodule init; git submodule update ;
        mkdir build; cd build ;
        cmake -G Ninja .. -DCMAKE_INSTALL_PREFIX=$llvm_install_dir  \
            -DCMAKE_CXX_COMPILER=$llvm_install_dir/g++ \
            -DCMAKE_C_COMPILER=$llvm_install_dir/gcc \
            -Dpybind11_DIR=$pybind11_DIR \
            -Dnanobind_DIR=$nanobind_DIR \
            -DProtobuf_DIR=$HOME/protobuf-3.21.12-install/lib/cmake/protobuf
        ninja -j8; ninja install; 
        ln -s $HOME/rocm-torch-mlir/build/bin/torch-mlir-import-onnx $llvm_install_dir/bin/torch-mlir-import-onnx
    fi
}

update_compiler_support_cxx20 ;
install_rocm_llvm ;
install_protobuf ;
install_torch_mlir ; 

