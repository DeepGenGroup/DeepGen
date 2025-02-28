## 原始最佳数据 ：   
doc/测试结果&数据/20250228/NV-v100/perflogtmp__card0.json

## 原始最佳数据构建出的空间：
doc/测试结果&数据/20250228/NV-v100/perflogtmp__card0_space.json  (nolsu)

## 运行完后的三次复测 （使用上述空间）：   
doc/测试结果&数据/20250228/NV-v100/perf_20250228_noLSU_check_card0-1.json
doc/测试结果&数据/20250228/NV-v100/perf_20250228_noLSU_check_card0-2.json
doc/测试结果&数据/20250228/NV-v100/perf_20250228_noLSU_check_card0-3.json

## 结果：   
原始大批量测试中，torch性能测定偏低(0.26)，导致acc偏高（1.305）
对比复测和原始数据，可看出DeepGen表现稳定, 即复测eps和原始eps基本相同 (0.22); 复测中，torcheps=0.23左右波动
实际最佳 acc 估计为 1.05 ~ 1.08 


