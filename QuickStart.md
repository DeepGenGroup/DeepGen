# 快速上手
## 0.docker 启动
```shell

docker run -it \
    --network=host \
    --ipc=host \
    --shm-size=16G \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal \
    --group-add video \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    [镜像名字] \
    /bin/bash
```

## 1.算子测试
### dcu z100 matmul
docker环境内，执行如下指令即可：
```shell
cd ./Runtime/
export PYTHONPATH=`pwd`
cd kcg
python SimpleLocalTester.py /home/DeepGen/TuningConfigs/GEMM_cfg_32.json ./result_matmul.json 0 1 0 1.5
# 含义 SimpleLocalTester.py [配置文件路径] [存放结果的json] [从哪个index开始测（0=从头）] [最多测试几个case] [是否检测tflops，选0就可] [预期的 speedup达到多少就停止搜索。0表示不限制，任意浮点数表示达到了就停]

```
若想测试其他形状的matmul算子，需要修改 /home/DeepGen/TuningConfigs/GEMM_cfg_32.json 中的M,N,K,batch字段：(如 TuningConfigs/mm-b2-2048-2048-1024.json)
TuningConfigs/mm-*.json 中已经给出了部分matmul形状。可以直接用于测试

## 2.端到端测模型

### 2.1 模型入口
dcu z100下的模型在 DeepGen/Runtime/kcg/models ，提供了bert,gpt2 和 llama2 为参考
model.py 定义了模型结构， run.py 为模型执行

以bert_large 为例，run.py 的入口说明如下：
```python
if __name__ == "__main__":
    devid = 7  # 设备号
    PathManager.init(clearPkl=True, clearCache=True, clearTmp=True, clearDump=True)  # 路径管理器初始化。执行缓存目录清理等
    DeviceInfo.init_cuda([devid])  # 设备初始化
    # 模型构建以及参数定义（注意：需要对模型定义做修改以实现算子替换）
    args = ModelArgs()
    model = BERT(True).to(devid)
    model_bench = BERT(False).to(devid)
    batch = 2
    max_seq_len = 1024
    input_ids = torch.randint(1, args.vocab_size, size=(batch, max_seq_len)).to(devid)

    # 如果存在调好的高性能kernel，想要使用，需要根据配置文件手动注册之。 precompiled.json 中已经注册了一些性能好的kernel（之前根据bert、gpt2和llama2中的matmul形状调过一次）
    registerPreCompiledKernelByJson('/home/xushilong/DeepGen/precompiled.json',7)
    # 没有kernel，想通过优化matmul算子优化模型性能，则执行下列函数，会获取所有matmul的形状后开始调kernel ：
    # compile_model(7, run_model(model_bench,args,input_ids))

    def f_benchmark():
        print("========= eval bench time =======",flush=True)
        return model_bench(input_ids)
    def f_base():
        print("========= eval base time =======",flush=True)
        return model(input_ids)
    
    # 
    
    out0,t0 = evaluate_model_time(f_base)
    out1,t1 = evaluate_model_time(f_benchmark)
    
    print(f"=== model run time : ours ={t1}, base = {t0}, speedup : {t0/t1}")
    opCallCounter = OpProxy.GetOpCallCounts()
    print("==== call ops :",opCallCounter)
    # mmCallCount = opCallCounter[matmul.MatmulOp.__name__]
    
    if torch.allclose(out0,out1,atol=1e-3,rtol=1e-3):
        print("===== model test correct ")
    else:
        diff, maxerr = compare_with_error(out0,out1)
        print(f"===== model test error ! diff, maxerr = {diff, maxerr}")
        print("baseline = ",out0)
        print("user = ", out1)
```

上述代码可实现模型调优/调优后的性能验证。前提是：
1.precompiled.json中必须有形状和模型中算子形状适配的matmul算子    
2.模型的代码按照Deepgen的规则进行了修改，可做算子替换


### 2.2 模型修改
下面讲解如何修改模型以实现Deepgen的算子替换。以bert_large/model.py 举例，介绍其中的关键修改：

开头需要导入Deepgen需要的模块
```python
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from kcg.TorchInjector import *   # 导入Injector注入器
from kcg.ModelUtils import *   # 导入模型基础设施

g_FmmBaseline=triton_matmul.bmm  # baseline的matmul定义。如果选baseline为triton，选择triton_matmul.bmm； 若选torch，则为 torch.matmul
```

为了验证模型结果正确性，需要把`nn.Embedding` 替换为 `create_fixed_embedding`（nn.Embedding 会随机权重，导致结果不同）

```python
class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_embeddings, type_vocab_size):
        super(BertEmbedding, self).__init__()
        f_embedding = create_fixed_embedding # nn.Embedding 替换为 create_fixed_embedding
        
        self.token_embedding = f_embedding(vocab_size, embedding_dim)  # 
        self.position_embedding = f_embedding(max_position_embeddings, embedding_dim)
        self.type_embedding = f_embedding(type_vocab_size, embedding_dim)
        self.layer_norm = LayerNorm(embedding_dim)

```

前馈层：`nn.Linear` 替换为 `CustomLinear`, 并将构造`CustomLinear`所需的matmul算子设置为 `OpProxy.f_matmul`

```python
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

```

attention类，将 torch.matmul 替换为 `OpProxy.f_matmul`. `nn.Linear` 替换为 `CustomLinear`
```python

class Attention(nn.Module):
    def __init__(self, dim, head_num, isBaseline):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num
        if isBaseline :
            f_mm = g_FmmBaseline
        else:
            f_mm = OpProxy.f_matmul
        # f_lin = nn.Linear
        f_lin = CustomLinear
        self.wq = f_lin(dim, head_num * self.head_dim, bias=False, f_mm=f_mm)
        self.wk = f_lin(dim, head_num * self.head_dim, bias=False, f_mm=f_mm)
        self.wv = f_lin(dim, head_num * self.head_dim, bias=False, f_mm=f_mm)
        self.wo = f_lin(head_num * self.head_dim, dim, bias=False, f_mm=f_mm)
        self.f_matmul = f_mm
        
```

### 2.3 模型运行
完成模型修改后，通过执行命令可运行模型：
```shell
cd Runtime
export PYTHONPATH=`pwd`
cd ./kcg/models/bert_large
python run.py
```
如果 run.py 中启用 registerPreCompiledKernelByJson， 则直接进行模型性能验证（直接替换matmul为已有的高性能算子，测试性能收益，跳过算子调优阶段）
如果 run.py 中启用 compile_model 则进行完整流程（Deepgen会进行：matmul形状收集 -> 每个形状的matmul依次调优 -> 自动注册最优matmul算子 -> 性能测试 ）
需要注意的是，目前 compile_model 函数并不完善，缺少收集最优kernel写入json的机制，所有结果都在内存里, 一旦进程崩溃结果就无了。而调优空间一般很大，故运行时间会很长（>24h）因此不推荐直接运行 compile_model 函数

- 建议做法：
将最后的 collectInfoOnly 设为 True，只收集算子形状信息。之后根据形状新建类似于 TuningConfigs/GEMM_cfg_32.json 的配置文件，设置 MNKbatch为对应形状的值，之后结合 SimpleLocalTester.py 进行逐个形状调优
```python
compile_model(7, run_model(model_bench,args,input_ids), True)

```

### 2.4 算子注册
参考 precompiled.json ，按如下格式，填入算子的kernelName、type 即可。 DCU下目前仅支持 matmul
```json
{
    "kernels" : [
        {
            "type" : "matmul",
            "kernelName" : "kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN2WSWM1WSWN2LSU1Map4GSW0UN8RP0SP0LC1RC0",
            "pklpath" : ""
        }
    ]
}
```
