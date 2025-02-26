# KCG 项目介绍
## 1.项目&目录结构
**_cache** : 缓存目录，存放程序运行时的编译器缓存文件（kernel loader&launcher的so及stubcode、benchmarkTorchEps记录   
**_dump** : 临时目录，存放MLIR生成kernel过程里·生成的bc和o文件   
**_override** : 暂无作用   
**_pkls** : 运行时存放生成的kernel信息序列化后的pkl文件。perftester会根据自己的卡号周期   性检测对应文件夹下的pkl，反序列化之并做perftest   
**_TempCodes** : 其他code，不是项目本体代码   
**_tmp** : 其他缓存目录   
**.vscode/c_cpp_properties.json :** intellisense 使用的头文件目录、宏定义   
**.vscode/launch.json** : debug配置    
**.vscode/settings.json** : 文件后缀名关联以及颜色主题   
**bin : DeepGen编译后的**库/可执行文件存放位置   
**build** : 构建目录   
**cmake** : MLIR使用的cmake   
**doc** : 项目文档   
**include** : MLIR后端的头文件   
**Runtime** : python后端   
**scripts/Benchmark**.sh ：从前端启动DeepGen   
**scripts/ClearTmpKernels.py** ：删除/tmp目录下的kernel文件   
**scripts/GetCudaInfo**.py ：获取cuda的计算力和ptxas信息，用于填入CMakeLists的对应变量   
**scripts/StopBenchmark**.sh ：杀死所有DeepGen运行的进程   
**src** : MLIR后端源代码目录   
**src/lib** : MLIR后端源码   
**src/CMakeLists.txt** : MLIR后端CMakeLists   
**src/main.cc :** MLIR后端源码，定义了exe以及python module的接口   
**third_party** : cuda和hip的第三方头文件、bitcode、其他所需程序等   
**TuningCombs** : 存放生成好的调优空间文件   
**TuningConfigs** : 调优参数配置文件   
**ClearTemp**.sh : 清理临时目录   
**CMakeLists.txt** : 根CMakeLists文件。用户变量在这里赋值   
**Compile.sh** : 编译MLIR后端的脚本   
**config.h.in** : 用户变量模板文件   



## 2.构建&运行方法
### 2.1 构建
使用Compile.sh脚本编译。其中`is_as_pymodule`表示将MLIR后端编译为库（ON）或调试用exe文件（OFF）

### 2.2 参数配置&运行
1. exe模式   
参数配置：debug用，只能用固定参数配置，在 src/main.cc 的 `main()`函数中修改。只用于测试MLIR后端的代码生成过程，不进行kernel的执行
运行：
```sh
${project_folder}/bin/kcg_compiler > log.txt 2>&1
```
   
调试：f5进入调式模式。配置文件在 .vscode/launch.json 注意配置选择   
<p align = 'center'>
<img src="./doc/image.png" width=50%>
</p>

2. lib模式   

启动脚本为 ${project_dir}/scripts/Benchmark.sh   
其调用 testGetKernels.py ,开启进程池处理编译和测试任务。可以将该进程设置为会话分离的（nohup），即ssh链接断开后也不会停止，用于长时间跑测试   
需要查看总体运行时间，执行 ： 
```shell
ps -eo pid,etime,cmd | grep testGetKernels
```


## 3. 使用说明
#### 3.1 运行机制   
1. DeepGen首先读取用户的调优参数文件，生成并剪枝调优空间，存储到json文件。如果检测到调优空间json已存在，则跳过这步
2. 随后DeepGen根据参数空间json开始编译和benchmark。编译的进程池大小由用户决定。benchmark过程由守护进程（ perfmonitor ）和 工作进程（perftester）构成。perftester 执行测试，并将结果存入 `perfPAth` 为前缀指定的json中。
perfmonitor 检测到 perftester 意外退出时，会重启perftester进程. perftester会根据用户输入的 `perfPAth` 路径重新读取历史最佳纪录，继续统计并benchmark，直到正常结束


#### 3.2 脚本参数说明

Benchmark.sh   

```shell   
#! /bin/bash
mydir="/home/xushilong/DeepGen"  # 设置用户当前项目的目录
export PYTHONPATH=$mydir/Runtime
cd ${mydir}/Runtime/kcg

tuning_param_file=$mydir/TuningConfigs/GEMM_configs_1024.json # 指定调优参数配置
cacheTuningSPaceFile=$mydir/TuningCombs/tuingspace_gemm_1024x1024.json # 指定调优空间文件名字（不存在会创建，存在则直接使用）
onlyGenerateCfg=0 # 是否只进行调优空间生成并存入 cacheTuningSPaceFile，不执行编译和benchmark

# 启动指令1 ：使用Benchmark脚本参数启动，会话进程分离，用于长期执行
nohup python testGetKernels.py $tuning_param_file $cacheTuningSPaceFile $onlyGenerateCfg  > ${mydir}/log.log 2>&1 &
# 启动指令2 ： 使用python内的参数启动， 会话进程不分离
# python testGetKernels.py > ${mydir}/log.log 2>&1 &
# hipprof测试指令
# hipprof --pmc python testGetKernels.py > log.log 2>&1 &

```

testGetKernels.py ：参数含义见代码注释

## 4.项目协同文档

周报记录  https://www.notion.so/dbe373c194d844748f693751460dad4a


缺陷：
python直接寻找ptxas等不能满足所有要求。改为用户自行配置路径 （cuda安装路径）
