from KCGTask import *
import json

kpm = KernelArgMatmul(0,0,0,1,1,1)
ktr = KernelTestResult(kpm)

path = '/home/xushilong/CodeGenDemo/perfRecordlog_7_0.json'
ktr_ = None
with open(path) as f :
    ktr_ = json.load(f)

ktr.parseFromJson(ktr_['results'][0])
print(ktr)