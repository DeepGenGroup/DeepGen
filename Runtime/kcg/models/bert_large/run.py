from model import *
import torch.nn as nn
import torch

def test_model(model, input):
    times = []
    for i in range(20):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        output = model(input)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    times = sorted(times)
    mid_time = times[10]
    return mid_time


def run(max_seq_len=1024, max_batch_size=64, vocab_size=32000):
    device = torch.device("cuda")
    model = BERT()
    batchs = [1, 2, 4]
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
    PathManager.init(clearPkl=True, clearCache=True, clearTmp=True, clearDump=True)
    DeviceInfo.init_cuda(7)

    args = ModelArgs()
    model = BERT()
    # batchs = [1, 2, 4]
    # print("seq_len: {}".format(max_seq_len))
    # for batch in batchs:
    #     input = torch.randint(low=1, high=vocab_size, size=(batch, max_seq_len), dtype=torch.long, device=device)
    #     model = model.to(device)
    #     cost = test_model(model, input)
    #     print("batch: {} time cost: {}".format(batch, cost))
    input_ids = torch.randint(1, args.vocab_size, size=(1, 1024)).to(7)
    # optimizedModel = model
    optimizedModel = get_op_optimized_model(model).to(7)
    compile_model(7, run_model(optimizedModel,args,input_ids))
    
    # 手动注册已经调好的kernl
    #       "name": "kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN2WSWM1WSWN2LSU1Map4GSW0UN8RP0SP0LC1RC0",
    #   "speedup": 1.2757928203038418,
    #   "time": 0.3799990117549896,
    #   "time_base": 0.4848000109195709
    r = TuneResult()
    r.OpTy = matmul.MatmulOp
    r.bestSpeedup = 1.2757928203038418
    cfg = KernelConfigs()
    cfg.kernelFuncName = 'kcg_MM_bM1024N1024K1024isAT1W64_BM32BN32BK8TM4TN4BLY1BLX1WLY8WLX8GLWA4GLWB4BSWM2BSWN2WSWM1WSWN2LSU1Map4GSW0UN8RP0SP0LC1RC0'
    cfg.backend = EnumBackendType.HIP
    cfg.dtypes = 
    r.be
    OpProxy.registKernel()
    
    # def f_benchmark():
    #     return optimizedModel(input_ids)
    # def f_base():
    #     return model(input_ids)
    
    # out0,t0 = evaluate_model_time(f_base)
    # out1,t1 = evaluate_model_time(f_benchmark)
    
    # print(f"=== model run time : ours ={t1}, base = {t0}, speedup : {(t0-t1)/t0}")
    # opCallCounter = OpProxy.GetOpCallCounts()
    # print("==== call ops :",opCallCounter)
    # mmCallCount = opCallCounter[matmul.MatmulOp.__name__]
    
    # if torch.allclose(out0,out1,atol=1e-1,rtol=1e-1):
    #     print("===== model test correct ")
    # else:
    #     diff, maxerr = compare_with_error(out0,out1)
    #     print(f"===== model test error ! diff, maxerr = {diff, maxerr}")
    #     print("baseline = ",out0)
    #     print("user = ", out1)