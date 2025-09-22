# Deepgen User Help Guide
## 1.项目介绍
Deepgen是一个基于MLIR/LLVM的跨平台算子调优器。支持多平台算子调优   
验收 commitid=dbefc4a9

### 1.1 支持平台
HygonDCU Z100/K100, Nvidia A100 & others

### 1.2 支持的算子
Nvidia：Attention、GEMM（需要满足特定形状   
DCU/AMD ： GEMM（需要满足特定形状   

## 2.如何安装和部署
### 2.1 编译&安装 MLIR/LLVM 以及其他第三方依赖
```shell
# 下载指定的MLIR/LLVM 源码
git clone https://github.com/DeepGenGroup/rocm-llvm-project.git -b deepgen-dev

# 安装ninja编译器
pip install ninja
# 安装 pybind11
pip install pybind11

# 编译&安装 mlir/llvm
cd rocm-llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_INSTALL_PREFIX=/path/to/llvm/install

ninja -j16
ninja install

```

### 2.2 编译Deepgen
1. 打开 build_tools/build_deepgen.sh, 修改相关变量
```shell
#! /bin/bash
project_dir="$HOME/DeepGen"   # 当前DeepGen所在path
cd $project_dir
is_as_pymodule='ON'   # 使用'ON'即可
buildType=Release  # 可选择 Debug Release MinSizeRel 。填Release即可
mkdir build
mkdir _dump
cd build  
cmake .. \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \             # g++路径
    -DCMAKE_C_COMPILER=/usr/bin/gcc \               # gcc路径
    -DCOMPILE_AS_PYMODULE=$is_as_pymodule \
    -DCMAKE_BUILD_TYPE=$buildType \
    -DENABLE_GRAPH_OPT=OFF
make -j16

```

2. 打开CMakeLists.txt，修改设置 
```cmake
# project config
###################################################################
cmake_minimum_required(VERSION 3.15.0)
project(KernelCodeGen LANGUAGES CXX C)    
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_C_COMPILER /usr/bin/gcc)
############################ User config #####################
set(LLVM_INSTALL_DIR "/path/to/llvm/install")               # 设置为用户自己的 mlir/llvm 安装路径
set(DEBUG_AMDGCN_OUTPUT_PATH "~/DeepGen/test.amdgcn") 
# set(USER_LLD_PATH "${CMAKE_SOURCE_DIR}/third_party/hip/bin/ld.lld") 
# After change llvm repo, the ld.lld must be compiled from llvm-project. Default ld in dtk is not available
set(USER_LLD_PATH "${LLVM_INSTALL_DIR}/bin/ld.lld") 
set(USER_PTXAS_PATH "/home/xushilong/anaconda3/bin//ptxas")    # ptxas 路径，通过 shell命令 which ptxas 可查看 
set(CUDA_CAP        80)
set(PTXAS_VERSION   82)
set(CUDA_INCLUDE_DIR "/home/xushilong/anaconda3/include")      # cuda安装目录下的include目录
option(ENABLE_GRAPH_OPT "enable graph optimizer" OFF)           # 设置为OFF
option(USE_STABLEHLO_EMBEDDED "Use embedded stableHLO(ON) or external stableHLO(OFF)" ON)

```

3. 修改thirdPartyPath.json ，设置cuda_install_dir 为 cuda 安装的目录(如 /usr/local/cuda)
4. 运行 build_tools/build_deepgen.sh 等待 Deepgen 编译完成

### 2.3 其他运行所需要的第三方依赖
```sh
pip install tqdm
pip install torch

```

## 3.环境变量设置以及其他运行前的准备
设置PYTHONPATH :
```sh
cd DeepGen/Runtime
export PYTHONPATH=$PYTHONPATH:`pwd`

```

## 4.端到端模型算子识别+调优 —— 一个完整示例
下面以 bert_large 为例，演示如何使用 deepgen 对模型进行算子识别+优化   
模型路径 Runtime/kcg/models/bert_large
model.py 是 BERT模型定义文件   
run.py 为运行脚本。其中定义了算子识别与调优过程   

0. 前提
使用deepgen做端到端调优的前提是：模型结构已知（有torch代码）
1. 修改模型源码
我们在 model.py 中做出如下修改 ：
- 开头引入deepgen的类，定义matmul的baseline函数和attention的baseline函数
```py
from kcg.TorchInjector import *
from kcg.ModelUtils import *
g_FmmBaseline = torch.matmul
# g_FmmBaseline = triton_matmul.bmm

def g_FattnBaseline(q, k, v, batch_size, head_num, seq_len, head_dim, mask = None) :
    scores = g_FmmBaseline(q, k.transpose(2, 3)) / math.sqrt(head_dim) # q*k
    if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
    scores = F.softmax(scores.float(), dim=-1).type_as(q)
    output = g_FmmBaseline(scores, v)  # (bs, n_local_heads, seqlen, head_dim)
    return output

```

- 在model.py 中修改torch.matmul算子为 OpProxy.f_matmul, 替换 nn.Linear为 CustomLinear  

```py

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_hidden_dim, isBaseline):
        super(FeedForward, self).__init__()
        if isBaseline:
            f_mm = g_FmmBaseline
        else:
            f_mm = OpProxy.f_matmul
        self.start_linear = CustomLinear(dim, ffn_hidden_dim, bias=False,f_mm=f_mm)
        self.gelu = nn.GELU()
        self.end_linear = CustomLinear(ffn_hidden_dim, dim, bias=False,f_mm=f_mm)
    
    def forward(self, x):
        start = self.start_linear(x)
        gelu = self.gelu(start)
        end = self.end_linear(gelu)
        return end


class Attention(nn.Module):
    def __init__(self, dim, head_num, isBaseline):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num
        if isBaseline :
            f_mm = g_FmmBaseline
            f_attention = g_FattnBaseline
        else:
            f_mm = OpProxy.f_matmul
            f_attention = OpProxy.f_attention
        # f_lin = nn.Linear
        f_lin = CustomLinear
        self.wq = f_lin(dim, head_num * self.head_dim, bias=False, f_mm=f_mm)
        self.wk = f_lin(dim, head_num * self.head_dim, bias=False, f_mm=f_mm)
        self.wv = f_lin(dim, head_num * self.head_dim, bias=False, f_mm=f_mm)
        self.wo = f_lin(head_num * self.head_dim, dim, bias=False, f_mm=f_mm)
        self.f_matmul = f_mm
        self.f_attention = f_attention
        
    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape  # [batch_size, seq_len, hidden_dim]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, self.head_num, self.head_dim) # [batch_size, seq_len, head_num, head_dim]
        xk = xk.view(batch_size, seq_len, self.head_num, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.head_num, self.head_dim)

        query = xq.transpose(1, 2) # [batch_size, head_num, seq_len, head_dim]
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        
        def _f() :
            scores = self.f_matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # q*k
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(query)
            output = self.f_matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
            return output        
        # if mask is None :
        if False :
            output = self.f_attention(query,keys,values,batch_size,self.head_num, seq_len, self.head_dim, mask)
        else:
            output = _f()
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

```


2. 算子扫描
完成模型修改后，运行 run.py。将collectInfoOnly 设置为 True执行算子扫描：
```py
    collectInfoOnly = True
    compile_model(devid, run_model(model_bench,args,input_ids), collectInfoOnly=collectInfoOnly)
    
```
输出如下：
```log
init_cuda devid= [0]
==== set visible device to 0  =====
=== e2e ends :  torch.Size([1, 1024, 1024])
collected mm args =  [[1], 1024, 1024, 1024, torch.float32]
collected mm args =  [[1, 16], 1024, 1024, 64, torch.float32]
collected mm args =  [[1, 16], 1024, 64, 1024, torch.float32]
collected mm args =  [[1], 1024, 4096, 1024, torch.float32]
collected mm args =  [[1], 1024, 1024, 4096, torch.float32]

```
可以看到，model中涉及到的所有matmul算子的 MNK以及batch参数  `[[batch], M,N,K, torch.dtype]`
 
3. 算子调优
针对上述matmul算子，分别执行调优。在 Runtime/kcg/SimpleLocalTester.py 中   
设置硬件信息（架构信息，nvgpu的sm号，amdgpu或dcu的gfx号），设置算子类型为 kcg_mm.MatmulOp 
```py
def main():
    cfgFile,result_json_path,start,maxCount,checktflops, checkAcc = getInputs()
    # cfgFile = "/home/xushilong/DeepGen/TuningConfigs/GEMM_cfg_32.json"
    opty = kcg_mm.MatmulOp
    # opty = kcg_att.AttentionOp
    devId = 7

    if is_hip():
        backend = EnumBackendType.HIP
        arch = "906"
    else:
        backend = EnumBackendType.CUDA
        arch = "80"
    
```
根据机器性能，配置进程池数目。数量越多，单次编译的kernel会更多，但是CPU负载会更大
```py
    print("=== checktflops, checkAcc",checktflops, checkAcc)
    ts = get_tuning_space(opty, cfgFile)
    bc = BenchmarkConfig()
    bc.keepTopNum = 10   # topK 结果保留
    bc.max_kernel_per_iter = 80  # 用于kernel编译的进程池大小
    bc.result_json_path = result_json_path  
    bc.maxCount = maxCount
    st = time.time()
    print(f"=====  start at : {st}")
    
```

根据MNK和batch，构建相关配置文件。相关文件已在 TuningConfigs/modelTest 中，和形状参数一一对应：
```log
mm_1._1024_1024_1024.json
mm_1._1024_1024_4096.json
mm_1._1024_4096_1024.json
mm_1.16._1024_64_1024.json
mm_1.16._1024_1024_64.json

```   
注意，Nvgpu和amdgpu的warp大小不同。需在配置文件内修改（nvidia应为32， amd或dcu=64）：   
```json
    "WARP_SIZE": [
        64  
    ],
```

配置文件可自行增删参数的取值规模。需注意，MNK和batch需要固定不变. 以 collected mm args =  [[1, 16], 1024, 64, 1024, torch.float32] 为例：   
```json
    // ...
    "M_SIZE": [
        1024
    ],
    "N_SIZE": [
        64
    ],
    "K_SIZE": [
        1024
    ],
    "BATCH_SIZE": [
        1,
        16
    ],
    // ...
```

**注意：为保证结果准确 请将gpu锁定频率。**
对于大部分GPU设备，其存在自动调节时钟频率的功能，在负载情况不同时时钟频率也不同。这可能使最终性能的测定不准确，因此需要锁定频率后再测试：
对于nvidia：   
```shell
   # 以设置7号卡的频率举例 (-i 7即可)
   sudo nvidia-smi -pm 1 -i 7  # 设置persistence mode, 防止驱动卸载后设置失效
   nvidia-smi -q -d CLOCK # 查看当前时钟状态
   nvidia-smi -q -d SUPPORTED_CLOCKS # 查看可用频率
   sudo nvidia-smi -lgc 1410,1410 -i 7  # 锁定上下限
   nvidia-smi -q -d CLOCK # 再次查看当前时钟状态
   ```

对于amdgpu：   
   ```shell
   cat /sys/class/drm/card0/device/pp_dpm_sclk  # 查看核心频率级别
   cat /sys/class/drm/card0/device/pp_dpm_mclk  # 查看显存频率级别
   echo "manual" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level # /sys/class/drm下gpu卡不一定叫card0，可能叫renderXXX之类的。根据需要自己改。下述同理
   # set clock level
   echo "4" | sudo tee /sys/class/drm/card0/device/pp_dpm_sclk
   echo "2" | sudo tee /sys/class/drm/card0/device/pp_dpm_mclk
   # 如果想撤销修改
   echo "auto" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level

   ```


构建配置文件完成后，确保gpu锁频后，可根据配置文件执行命令：
```sh
cd Runtime/kcg
python SimpleLocalTester.py ../../TuningConfigs/modelTest/mm_1._1024_1024_1024.json  res_mm_1._1024_1024_1024.json 0 0 0 0 > log_1._1024_1024_1024.log 2>&1 
python SimpleLocalTester.py ../../TuningConfigs/modelTest/mm_1._1024_1024_4096.json  res_mm_1._1024_1024_4096.json 0 0 0 0 > log_1._1024_1024_4096.log 2>&1 
python SimpleLocalTester.py ../../TuningConfigs/modelTest/mm_1._1024_4096_1024.json  res_mm_1._1024_4096_1024.json 0 0 0 0 > log_1._1024_4096_1024.log 2>&1 
python SimpleLocalTester.py ../../TuningConfigs/modelTest/mm_1.16._1024_64_1024.json  res_mm_1.16._1024_64_1024.json 0 0 0 0 > log_1.16._1024_64_1024.log 2>&1 
python SimpleLocalTester.py ../../TuningConfigs/modelTest/mm_1.16._1024_1024_64.json  res_mm_1.16._1024_1024_64.json 0 0 0 0 > log_1.16._1024_1024_64.log 2>&1 

```
调优结果将存放在各自 `res_mm_*.json`文件中
调优时间根据参数空间大小、机器性能会有差别。一般在数小时到数十个小时不等

4. 结果注册&模型运行
调优完成后，结果格式如下：
```json
{
  "testResult": [
    {
      "name": "kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN2WSWM1WSWN2LSU1Map4GSW0UN8RP0SP0LC1RC0",
      "speedup": 1.2757928203038418,
      "time": 0.3799990117549896,
      "time_base": 0.4848000109195709
    },
    {
      "name": "kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN4WSWM1WSWN2LSU1Map4GSW0UN8RP1SP0LC1RC0",
      "speedup": 1.275789518428972,
      "time": 0.3799999952316284,
      "time_base": 0.4848000109195709
    },
    {
      "name": "kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN4WSWM1WSWN2LSU1Map4GSW0UN8RP0SP0LC1RC0",
      "speedup": 1.2752525408917632,
      "time": 0.38016000390052795,
      "time_base": 0.4848000109195709
    },
    // ...
  ]
}
```
上述展示了 top3的最佳结果。注册时，一般只用注册 top1的name字段即可。 在 precompiled.json 中注册：
```json
{
    "kernels" : [
        {
            "type" : "matmul",
            "kernelName" : "kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN2WSWM1WSWN2LSU1Map4GSW0UN8RP0SP0LC1RC0",
            "pklpath" : ""
        },
        // ...
    ]
}
```

我们根据dcu-z100上的运行结果提前注册好了一些kernel在precompiled.json 中   

使用该注册文件运行model ：
```py
# Runtime/kcg/models/bert_large/run.py
    
    # 手动注册已经调好的kernl
    registerPreCompiledKernelByJson('/home/xushilong/DeepGen/precompiled.json',7)  # 7表示gpu卡号
    # 没有调好的kernel，首次执行：
    collectInfoOnly = False
    # compile_model(devid, run_model(model_bench,args,input_ids), collectInfoOnly=collectInfoOnly)
    
```
进行上述修改，运行 run.py 执行模型（会根据注册的name先做一遍算子编译，之后运行model）
