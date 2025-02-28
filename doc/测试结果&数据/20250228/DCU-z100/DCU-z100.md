## 原始最佳数据 ：   
doc/测试结果&数据/20250228/perfRecordlog__card7.json   

## 原始最佳数据构建出的空间：
doc/测试结果&数据/20250228/DCU-z100/bestPerfs-dcu-z100.json  （不含LSU）

## 运行完后的三次复测 （使用上述空间）：   
doc/测试结果&数据/20250228/perfRecordlog_check_card7-1.json
doc/测试结果&数据/20250228/perfRecordlog_check_card7-2.json
doc/测试结果&数据/20250228/perfRecordlog_check_card7-3.json

## 结果：   
原始大批量测试中，torch性能测定偏低，导致acc偏高（1.385）
对比复测和原始数据，可看出DeepGen表现稳定, 即复测eps和原始eps基本相同。
实际最佳 acc=1.11，以复测为准
