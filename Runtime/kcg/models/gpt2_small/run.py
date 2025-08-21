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


def run(max_seq_len=1024, max_batch_size=32, vocab_size=32000):
    device = torch.device("cuda")
    args = ModelArgs()
    model = GPT2()
    model_bench = GPT2(False)

    batchs = [1, 2, 4, 8]
    print("seq_len: {}".format(max_seq_len))
    for batch in batchs:
        input = torch.randint(low=1, high=vocab_size, size=(batch, max_seq_len), dtype=torch.long, device=device)
        model = model.to(device)
        cost = test_model(model, input)
        print("batch: {} time cost: {}".format(batch, cost))


# if __name__ == "__main__":
#   run()


    
# 如何运行模型
def run_model(model, args : ModelArgs, input_ids : torch.Tensor) :
    # input_ids = torch.randint(0, args.vocab_size, (1, args.max_seq_len)).to(7)
    def _f() :
        out = model(input_ids)
        return out
    return _f
    

if __name__ == "__main__":
    devid = 7
    
    PathManager.init(clearPkl=True, clearCache=True, clearTmp=True, clearDump=True)
    DeviceInfo.init_cuda([devid])
    
    args = ModelArgs()
    
    # embedding = Embedding(args.vocab_size, args.embedding_dim, args.max_position_embeddings, args.type_vocab_size)
    # embedding.to(devid)
    input_ids = torch.randint(1, args.vocab_size, size=(1, 1024)).to(devid)

    # h = embedding(input_ids)
    # hh = h.clone()
    model = GPT2().to(devid)
    model_bench = GPT2(False).to(devid)
    # 复制权重（关键步骤）
    model_bench.load_state_dict(model.state_dict())

    # 验证权重是否相同
    # for (name_a, param_a), (name_b, param_b) in zip(model.named_parameters(), model_bench.named_parameters()):
    #     assert name_a == name_b, "参数名称不一致"
    #     assert torch.equal(param_a, param_b), f"参数 {name_a} 不相同"
    
    # batchs = [1, 2, 4]
    # print("seq_len: {}".format(max_seq_len))
    # for batch in batchs:
    #     input = torch.randint(low=1, high=vocab_size, size=(batch, max_seq_len), dtype=torch.long, device=device)
    #     model = model.to(device)
    #     cost = test_model(model, input)
    #     print("batch: {} time cost: {}".format(batch, cost))
    optimizedModel = model_bench
    # optimizedModel = get_op_optimized_model(model).to(devid)
    
    # 手动注册已经调好的kernl
    registerPreCompiledKernelByJson('/home/xushilong/DeepGen/precompiled.json',7)
    # 没有调好的kernel，首次执行：
    # compile_model(7, run_model(optimizedModel,args,input_ids))

    def f_benchmark():
        print("========= eval bench time =======",flush=True)
        return optimizedModel(input_ids)
    def f_base():
        print("========= eval base time =======",flush=True)
        return model(input_ids)
    
    # 

    out0,t0 = evaluate_model_time(f_base)
    out1,t1 = evaluate_model_time(f_benchmark)
    
    print(f"=== model run time : ours ={t1}, base = {t0}, speedup : {t0/t1}")
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