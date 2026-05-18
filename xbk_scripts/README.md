# /home/xushilong/DeepGen/xbk_scripts/testGemm_best.sh : matmul 最佳kernel测试（指定配置）
json文件：
- /home/xushilong/DeepGen/TuningConfigs/xbk/gemm_best.json : matmul top1 kenrel
- /home/xushilong/DeepGen/TuningConfigs/xbk/gemm.json : matmul topN kernel

```sh
python SimpleLocalTester.py $cfg ./xbk_best_gemm2048_${SEQLEN}.json 0 $max 0 0 matmul 4 float32 --seqlen 2048 notune
# 最后一个参数 tune : 执行调优空间调优（通常模式）  notune: 单独验证某几个config。（如最好的几个kernel方案）
```

# /home/xushilong/DeepGen/xbk_scripts/testGemm_tuning.sh ： matmul 调优空间执行演示（生成tunespace，剪枝后遍历）

# /home/xushilong/DeepGen/xbk_scripts/testAttnV1_best.sh ： attention_v1 测试最佳kernel（指定配置）

# /home/xushilong/DeepGen/xbk_scripts/testAttnV1_tuning.sh : attention_v1 tune 演示

