import torch
import torch.nn as nn
from kcg.TorchInjector import *
from kcg.ModelUtils import *


class SimpleTwoLinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f_linear = nn.Linear):
        """
        一个只包含两个线性层的简单神经网络
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出层维度
        """
        super(SimpleTwoLinearNet, self).__init__()
        
        # 第一个线性层 (输入层 -> 隐藏层)
        self.linear1 = f_linear(input_size, hidden_size)
        
        # 第二个线性层 (隐藏层 -> 输出层)
        self.linear2 = f_linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量, 形状为 [batch_size, input_size]
        
        返回:
            输出张量, 形状为 [batch_size, output_size]
        """
        # 通过第一个线性层
        x = self.linear1(x)
        print("----after linear1 : ",x)
        # 通过第二个线性层
        x = self.linear2(x)
        print("----after linear2 : ",x)
        
        return x

# 测试网络
if __name__ == "__main__":
    
    # 创建网络实例
    input_size = 1024    # 输入特征维度
    hidden_size = 1024  # 隐藏层维度
    output_size = 1024    # 输出层维度
    
    
    devid = 7
    PathManager.init(clearPkl=True, clearCache=True, clearTmp=True, clearDump=True)
    DeviceInfo.init_cuda(devid)
    model = SimpleTwoLinearNet(input_size, hidden_size, output_size).to(devid)
    # model = SimpleTwoLinearNet(input_size, hidden_size, output_size, CustomLinear).to(devid)
    
    # batchs = [1, 2, 4]
    # print("seq_len: {}".format(max_seq_len))
    # for batch in batchs:
    #     input = torch.randint(low=1, high=vocab_size, size=(batch, max_seq_len), dtype=torch.long, device=device)
    #     model = model.to(device)
    #     cost = test_model(model, input)
    #     print("batch: {} time cost: {}".format(batch, cost))
    input = torch.rand((1024,1024),dtype = torch.float32).to(devid)
    # optimizedModel = model
    optimizedModel = get_op_optimized_model(model).to(devid)
    
    print("======== compare weight ========") 
    t0 = model.linear1.weight
    t1 = optimizedModel.model.linear1.weight
    print('weight : ',t0,t1)
    if torch.allclose(t0,t1) :
        print("linear1 equal")
    else:
        print("linear1 not equal")
    
    # 手动注册已经调好的kernl
    registerPreCompiledKernelByJson('/home/xushilong/DeepGen/precompiled.json',7)
    # 没有调好的kernel，首次执行：
    # compile_model(7, run_model(optimizedModel,args,input_ids))

    def f_benchmark():
        return optimizedModel(input)
    def f_base():
        return model(input)
    
    # 
    print('--------------- Base ----------------')
    out0,t0 = evaluate_model_time(f_base)
    print('--------------- Ours ----------------')
    out1,t1 = evaluate_model_time(f_benchmark)
    
    print(f"=== model run time : ours ={t1}, base = {t0}, speedup : {t1/t0}")
    opCallCounter = OpProxy.GetOpCallCounts()
    print("==== call ops :",opCallCounter)
    # mmCallCount = opCallCounter[matmul.MatmulOp.__name__]
    
    if torch.allclose(out0,out1,atol=1e-1,rtol=1e-1):
        print("===== model test correct ")
    else:
        diff, maxerr = compare_with_error(out0,out1)
        print(f"===== model test error ! diff, maxerr = {diff, maxerr}")
        print("baseline = ",out0)
        print("user = ", out1)