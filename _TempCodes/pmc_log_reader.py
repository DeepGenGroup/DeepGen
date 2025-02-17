pmc_log_path = '/home/xushilong/CodeGenDemo/Runtime/python/kcg/pmc_results_239084.txt'
lines = []
with open(pmc_log_path) as f:
    lines = f.readlines()
KEY_KCG = "GEMM_MNK"
KEY_TORCH = "Cij" 
KEY_EPS = "kernel time"
KEY_FLOP = "performance"
eps_torch = []
eps_kcg = []
collectKcg = False
collectTorch = False
for line in lines :
    if line.find(KEY_KCG) >= 0:
        collectKcg = True
        continue
    if line.find(KEY_TORCH) >= 0:
        collectTorch = True
        continue
    st = line.find(KEY_FLOP)
    if st >= 0:
        if collectKcg :
            eps_kcg.append(line.split(' ')[-1][0:-1])
            collectKcg = False
        if collectTorch :
            eps_torch.append(line.split(' ')[-1][0:-1])
            collectTorch = False
        continue

print('eps_torch : ',eps_torch)
print('eps_kcg : ',eps_kcg)

# eps_torch :  ['5710.062503(Gflops)', '5641.267935(Gflops)', '5650.672854(Gflops)', '5636.140996(Gflops)', '5644.990242(Gflops)', '5630.978712(Gflops)', '5657.583740(Gflops)', '5639.207084(Gflops)', '5658.507971(Gflops)', '5626.494543(Gflops)', '5650.223220(Gflops)', '5630.375946(Gflops)', '5650.920184(Gflops)', '5629.728673(Gflops)', '5658.417789(Gflops)', '5636.342315(Gflops)', '5650.740306(Gflops)', '5644.384472(Gflops)', '5650.672854(Gflops)', '5634.553313(Gflops)', '5652.089672(Gflops)']
# eps_kcg :  ['6166.351190(Gflops)', '6111.142333(Gflops)', '6116.151749(Gflops)', '6131.293492(Gflops)', '6123.508649(Gflops)', '6111.690037(Gflops)', '6126.067270(Gflops)', '6116.164506(Gflops)', '6126.681657(Gflops)', '6129.550427(Gflops)', '6153.423992(Gflops)', '6098.445341(Gflops)', '6117.887200(Gflops)', '6114.009315(Gflops)', '6162.813142(Gflops)', '6087.784675(Gflops)', '6145.995077(Gflops)', '6091.186464(Gflops)', '6135.578463(Gflops)', '6114.901813(Gflops)', '6161.803016(Gflops)']
