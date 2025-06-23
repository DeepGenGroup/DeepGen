from model import *
import torch, json
import torch.nn.functional as F


def test_model(model, input):
    times = []
    repeat = 10
    for i in range(repeat):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        output = model(input)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    times = sorted(times)
    mid_time = times[int(repeat/2)]
    return mid_time


def run(max_seq_len=1024, max_batch_size=32, vocab_size=32000):
    device = torch.device("cuda")
    args = ModelArgs()
    model = GPT2()
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
    DeviceInfo.init_cuda(devid)

    args = ModelArgs()
    model = GPT2()
    input = torch.randint(low=1, high=args.vocab_size, size=(1, 1024), dtype=torch.long, device=f'cuda:{devid}')
    
    baselineModel = get_baseline_model(model).to(devid)
    # optimizedModel = model
    optimizedModel = get_op_optimized_model(model).to(devid)
    
    # 手动注册已经调好的kernl
    registerPreCompiledKernelByJson('/home/xushilong/DeepGen/precompiled.json',7)
    # 没有调好的kernel，首次执行：
    # compile_model(7, run_model(optimizedModel,args,input_ids))

    def f_benchmark():
        global g_testBaseline
        g_testBaseline = False
        return optimizedModel(input)
    def f_base():
        global g_testBaseline
        g_testBaseline = True
        return baselineModel(input)
    
    # warmup 
    f_base()
    f_benchmark()
    print("======= evaluating ...",flush=True)
    out0,t0 = evaluate_model_time(f_base)
    out1,t1 = evaluate_model_time(f_benchmark)
    
    print(f"=== model run time : ours ={t1}, base = {t0}, speedup : {t1/t0}")
    opCallCounter = OpProxy.GetOpCallCounts()
    print("==== call ops :",opCallCounter)
    # mmCallCount = opCallCounter[matmul.MatmulOp.__name__]
    
    if torch.allclose(out0,out1,atol=1e-3,rtol=1e-3):
        print("===== model test correct ")
    else:
        diff, maxerr = compare_with_error(out0,out1)
        print(f"===== model test error ! diff, maxerr = {diff, maxerr}")
        print("baseline = ",out0)
        print("user = ", out1)