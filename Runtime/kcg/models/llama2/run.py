from model import *
import torch, json
import torch.nn.functional as F


def test_model(model, input):
    times = []
    repeat = 10
    for i in range(repeat):
        start_event = torch_ns.Event(enable_timing=True)
        end_event = torch_ns.Event(enable_timing=True)
        start_event.record()

        output = model(input)

        end_event.record()
        torch_ns.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    times = sorted(times)
    mid_time = times[int(repeat/2)]
    return mid_time
    

def run(max_seq_len=2048, max_batch_size=16, vocab_size=32000):
    device = torch.device("cuda")
    model = LLAMA2()
    batchs = [1]
    print("seq_len: {}".format(max_seq_len))
    for batch in batchs:
        input = torch.randint(low=1, high=vocab_size, size=(batch, max_seq_len), dtype=torch.long, device=device)
        model = model.to(device)
        cost = test_model(model, input)
        print("batch: {} time cost: {}".format(batch, cost))


# if __name__ == "__main__":
#     DeviceInfo.init_cuda(7)
#     model = LLAMA2()
#     optimizedModel = get_op_optimized_model(model).to(7)
#     compile_model(optimizedModel,ModelArgs(), 7)
#     batchs = [1]
#     max_seq_len=2048, max_batch_size=16, vocab_size=32000
    
#     print("seq_len: {}".format(max_seq_len))
#     for batch in batchs:
#         input = torch.randint(low=1, high=vocab_size, size=(batch, max_seq_len), dtype=torch.long, device='cuda:7')
#         model = model.to(7)
#         cost = test_model(model, input)
#         print("batch: {} time cost: {}".format(batch, cost))
    
    
# 如何运行模型
def run_model(model, args : ModelArgs, input_ids : torch.Tensor) :
    # input_ids = torch.randint(0, args.vocab_size, (1, args.max_seq_len)).to(7)
    def _f() :
        out = model(input_ids)
        return out
    return _f
    
if __name__ == "__main__":
    isBase = sys.argv[1] == 'base'
    devid = 7
    
    PathManager.init(clearPkl=True, clearCache=True, clearTmp=True, clearDump=True)
    DeviceInfo.init_cuda([devid])

    args = ModelArgs()
    model = LLAMA2(isBase).to(device=devid)
    # model_bench = LLAMA2(False).to(devid)
    # 复制权重（关键步骤）
    # model_bench.load_state_dict(model.state_dict())

    # 验证权重是否相同
    # for (name_a, param_a), (name_b, param_b) in zip(model.named_parameters(), model_bench.named_parameters()):
    #     assert name_a == name_b, "参数名称不一致"
    #     assert torch.equal(param_a, param_b), f"参数 {name_a} 不相同"
  
    input_ids = torch.randint(0, args.vocab_size, size=(1, args.max_seq_len)).to(devid)
    # input_ids_0 = input_ids.to(6)
    # optimizedModel = model_bench
    # optimizedModel = get_op_optimized_model(model).to(devid)
    
    # 手动注册已经调好的kernl
    # registerPreCompiledKernelByJson('/home/xushilong/DeepGen/precompiled.json',devid)
    # 没有调好的kernel，首次执行：
    compile_model(devid, run_model(model, args, input_ids), collectInfoOnly=True, invokeDeepgenGraph=True)
    print("collected info : ")
    for (ty, args) in OpProxy.collector.getInfo() :
        print(f"{ty}, {args}")
    import run_kernel
    # compile_model(devid, run_model(model, args, input_ids), collectInfoOnly=False)
    
    # if isBase :
    def f_base():
        if isBase:
            print("========= eval base time =======",flush=True)
        else :
            print("========= eval bench time =======",flush=True)
        return model(input_ids)
    # else:
        # def f_benchmark():
        #     print("========= eval bench time =======",flush=True)
        #     return optimizedModel(input_ids)
    # 

    out0,t0 = evaluate_model_time(f_base)
    # out1,t1 = evaluate_model_time(f_benchmark)
    
    # print(f"=== model run time : {t0}, ")
    # opCallCounter = OpProxy.GetOpCallCounts()
    # print("==== call ops :",opCallCounter)
    # mmCallCount = opCallCounter[matmul.MatmulOp.__name__]
    
    print("===== model test done! ")
    # if torch.allclose(out0,out1,atol=1e-1,rtol=1e-1):
    #     print("===== model test correct ")
    # else:
    #     diff, maxerr = compare_with_error(out0,out1)
    #     print(f"===== model test error ! diff, maxerr = {diff, maxerr}")
    #     print("baseline = ",out0)
    #     print("user = ", out1)