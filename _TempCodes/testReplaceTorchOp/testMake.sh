nvcc -c ./add.cu -o add_cuda.o -Xcompiler -fPIC -rdc=true
g++ -I/home/xushilong/anaconda3/envs/py310/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
    -I/home/xushilong/anaconda3/envs/py310/lib/python3.10/site-packages/torch/include \
    -I/home/xushilong/anaconda3/envs/py310/include/python3.10 \
    -L/home/xushilong/anaconda3/envs/py310/lib \
    -L/home/xushilong/anaconda3/envs/py310/lib/python3.10/site-packages/torch/lib \
    -fPIC \
    -lpython3.10 \
    -lshm \
    -lc10 \
    -lc10_cuda \
    -lcaffe2_nvrtc \
    -ltorch_cpu \
    -ltorch_cuda \
    -ltorch_python \
    -ltorch_cuda_linalg \
    -ltorch_global_deps \
    -ltorch \
    -c add.cpp  \
    -o add_cpp.o 

x86_64-linux-gnu-g++ -shared \
    add_cuda.o add_cpp.o  \
    -L/home/xushilong/anaconda3/envs/py310/lib \
    -L/home/xushilong/anaconda3/envs/py310/lib/python3.10/site-packages/torch/lib \
    -lpython3.10 \
    -lc10 \
    -lc10_cuda \
    -lshm \
    -lcaffe2_nvrtc \
    -ltorch_cpu \
    -ltorch_cuda \
    -ltorch_python \
    -ltorch_cuda_linalg \
    -ltorch_global_deps \
    -ltorch \
    -o add.cpython-310-x86_64-linux-gnu.so 
    # -o add.cpython-310-x86_64-linux-gnu.so 

export LD_LIBRARY_PATH="/home/xushilong/anaconda3/envs/py310/lib/python3.10/site-packages/torch/lib":$LD_LIBRARY_PATH

python run.py