## 原始最佳数据 ：   
doc/测试结果&数据/20250228/NV-a100/__tttt.json (nolsu)
## 原始最佳数据构建出的空间：
doc/测试结果&数据/20250228/NV-a100/__tttt_s.json (nolsu)

## 运行完后的三次复测 （使用上述空间）：   
doc/测试结果&数据/20250228/NV-a100/_PerfCHeck_card6-1.json
doc/测试结果&数据/20250228/NV-a100/_PerfCHeck_card6-2.json
doc/测试结果&数据/20250228/NV-a100/_PerfCHeck_card6-3.json

## 结果：   
对比复测和原始数据，可看出DeepGen表现不稳定。大批量测试中kcg明显更优，而torch表现稳定; 原始测试中kcg的最佳性能 < 0.16ms kcg稳定在0.181
预计需要锁频
