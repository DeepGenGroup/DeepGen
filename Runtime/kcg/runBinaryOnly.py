from kcg.Kernel import *
import kcg.Operators.testop as testOp
import kcg.Operators.attention as attnOp
import torch.nn.functional as F
import math
import numpy as np

def perf(fn) :
    st = torch.cuda.Event(enable_timing=True)
    et = torch.cuda.Event(enable_timing=True)
    st.record()
    fn()
    et.record()
    torch.cuda.synchronize()
    t = st.elapsed_time(et)
    return t

def test_attention() :
    if len(sys.argv) < 2 :
        print("usage : kernelBinPath")
        sys.exit()
    binpath = sys.argv[1]
    op = attnOp.AttentionOp()
    devid = 7
    DeviceInfo.init_cuda([devid])
    funcName = 'kcg_Attention_1_32_2048_128_Br32Bc64Hd128_Sa16Sb8PTr4PTc4OTr4OTc8GLWQ4GLWK4GLWV4BLPY4BLPX1WLPY2WLPX16BSWQ4BSWK2WSWQ4WSWK2BLOY2BLOX2WLOY4WLOX8BSWP4BSWV4WSWP4WSWV4Un8W32LCP1LCO1SPP0RPP0RPO0'
    dtypes = [torch.float32]
    
    backend = EnumBackendType.HIP
    def testTorch(inputs):
        def attnFunc(Q, K, V, O):
            d = Q.shape[1] * Q.shape[3]
            P = torch.matmul(Q, K) # / math.sqrt(d)
            S = F.softmax(P, dim=-1)
            O = torch.matmul(S, V)
            return O
        # t = perf(attnFunc, inputs)
        return attnFunc(*inputs)
    kc = KernelConfigs(binpath, funcName, dtypes, backend)
    # func.block.dim = array<i32: 128>, func.grid.dim = array<i32: 64, 32, 1>
    kc.m_gridDims = [64,32,1]
    kc.m_blockDims = [128,1,1]
    kc.shmBytes = 18816
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16

    bs,hn,sl,hd = [1,32,2048, 128]
    factor = 1
    Q = torch.randn((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}') * factor
    K = torch.randn((bs, hn, hd, sl), dtype=torch.float32, device=f'cuda:{devid}') * factor
    V = torch.randn((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}') * factor
    O = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')
    O_ = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')

    inputs = [Q.transpose(-1, -2).contiguous(), K, V, O]
    inputs_ = [Q, K, V, O_]
    
    # print(f"torch time cost: {t} ms")
    def _f_ours() :
        kernel.run(inputs[0],inputs[1],inputs[2],inputs[3])
    def _f_baseline() :
        testTorch(inputs=inputs_)
    
    eps_ours = []
    eps_base = []
    # warmup
    O_ = testTorch(inputs=inputs_)
    kernel.run(*inputs)
    
    # benchmark
    for i in range(7):
        t0 = perf(_f_baseline)
        t = perf(_f_ours)
        eps_ours.append(t)
        eps_base.append(t0)
    eps = np.median(eps_ours)
    eps_0 = np.median(eps_base)
    # kernel.run(*inputs)
    
    print('torchOut=',O_)
    print('out=',O)
    if torch.allclose(O,O_,1e-3,1e-3):
        print("test correct!")
    else:
        print("test error!")
    if torch.isnan(O).any() :
        print("detect NaN!")
    else:
        print("No NaN!")
        
    print(f'eps_ours = {eps}, eps_baseline = {eps_0}, acc = {eps_0/eps}')
    

def test_broadcastOp() :
    if len(sys.argv) < 2 :
        print("usage : kernelBinPath")
        sys.exit()
    binpath = sys.argv[1]
    devid = 7
    op = testOp.BroadcastOp()
    
    func_name = "broadcast"
    dtypes = [torch.float32]
    backend = EnumBackendType.HIP
    
    kc = KernelConfigs(binpath, func_name, dtypes, backend)
    kc.m_gridDims = [2,1,1]
    kc.m_blockDims = [64,1,1]
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16
    DeviceInfo.init_cuda([devid])
    
    tensorSHape = (32,4)
    a = torch.ones(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    for i in range(0,tensorSHape[0]) :
        a[i,0] = i+10
    # b = torch.empty(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    print("original a = ",a)
    kernel.run(a)
    print("result a = ",a)
    # print("b = ",b)
    

def test_expOp() :
    if len(sys.argv) < 2 :
        print("usage : kernelBinPath")
        sys.exit()
    binpath = sys.argv[1]
    devid = 7
    op = testOp.ExpOp()
    
    func_name = "my_exp"
    dtypes = [torch.float32]
    backend = EnumBackendType.HIP
    
    kc = KernelConfigs(binpath, func_name, dtypes, backend)
    kc.m_gridDims = [2,1,1]
    kc.m_blockDims = [64,1,1]
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16
    DeviceInfo.init_cuda([devid])
    
    tensorSHape = (32,4)
    a = torch.ones(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    for i in range(0,tensorSHape[0]) :
        a[i,0] = i+10
    # b = torch.empty(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    print("original a = ",a)
    kernel.run(a)
    print("result a = ",a)
    # print("b = ",b)
    
def test_reduceOp() :
    if len(sys.argv) < 2 :
        print("usage : kernelBinPath")
        sys.exit()
    binpath = sys.argv[1]
    devid = 7
    op = testOp.ReduceOp()
    
    func_name = "reduce"
    dtypes = [torch.float32]
    backend = EnumBackendType.HIP
    
    kc = KernelConfigs(binpath, func_name, dtypes, backend)
    kc.m_gridDims = [2,1,1]
    kc.m_blockDims = [64,1,1]
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16
    DeviceInfo.init_cuda([devid])
    
    tensorSHape = (8,16)
    a = torch.ones(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    b = torch.empty(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    for i in range(0,tensorSHape[0]) :
        a[i,0] = i+10
        a[i,8] = 99
    # b = torch.empty(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    print("original a = ",a)
    kernel.run(a,b)
    print("result b = ",b)
    # print("b = ",b)
    

if __name__ == "__main__" :
    # test_broadcastOp()
    # test_reduceOp()
    test_attention()
    # test_expOp()