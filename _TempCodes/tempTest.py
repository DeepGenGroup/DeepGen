# json_lists = [
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_0.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_1.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_2.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_3.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_4.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_5.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_6.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_7.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_8.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_9.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_10.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_11.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_12.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_13.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_14.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_15.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_16.json",
# "/home/xushilong/CodeGenDemo/perfRecordlog_5_17.json",
# ]

# mergedPath = '/home/xushilong/CodeGenDemo/combs_ALL.json'

# import json
# merged = None
# combs = {'cfgs' : []}
# for jsonPath in json_lists :
#     with open(jsonPath) as f :
#         obj = json.load(f)
#         for res in obj['results'] :
#             combs['cfgs'].append(res['config'])
#         if merged is None :
#             merged = obj
#         else:
#             merged['results'] += obj['results']

# for config in combs['cfgs']:
#     config = {key: int(value) for key, value in config.items()}

# with open(mergedPath,'w') as f :
#     json.dump(combs,f,indent=4)
    
import json    
jsonPath = "/home/xushilong/CodeGenDemo/combs_ALL.json"
obj = None
with open(jsonPath) as f :
    obj = json.load(f)
itemList = obj['cfgs']
for i in range(0,len(itemList)) :
    itemList[i] = {key: int(value) for key, value in zip(itemList[i].keys(), map(int, itemList[i].values()))}
with open(jsonPath,'w') as f :
    json.dump(obj,f,indent=4)
    