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

def test_attention(devid) :

    op = attnOp.AttentionOp()
    # binpath = "/home/xushilong/DeepGen/_tmp/test.hsaco"
    binpath = "/home/xushilong/DeepGen/_tmp/kernelatt.hsaco"
    func_name = "kcg_Attention_1_32_2048_128_Br32Bc32Hd128_Sa8Sb8PTr4PTc4OTr8OTc8GLWQ4GLWK4GLWV4BLPY1BLPX1WLPY8WLPX8BSWQ4BSWK4WSWQ2WSWK4BLOY1BLOX1WLOY4WLOX16BSWP8BSWV2WSWP2WSWV1Un16W64LCP1LCO1SPP0RPP0RPO0"
    dtypes = [torch.float32]
    backend = EnumBackendType.HIP
        
    def testTorch(inputs):
        def attnFunc(Q, K, V, O):
            d = Q.shape[1] * Q.shape[3]
            P = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d)
            S = F.softmax(P, dim=-1)
            O = torch.matmul(S, V)
            return O
        # t = perf(attnFunc, inputs)
        return attnFunc(*inputs)
  
    kc = KernelConfigs(binpath, func_name, dtypes, backend)
    kc.m_gridDims = [64,32,1]
    kc.m_blockDims = [64,1,1]
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16

    bs,hn,sl,hd = [1,32,2048, 128]
    
    Q = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device=f'cuda:{devid}')
    K = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device=f'cuda:{devid}')
    V = torch.randn(bs, hn, sl, hd, dtype=torch.float32, device=f'cuda:{devid}')
    O = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')
    O_ = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')

    inputs = [Q.transpose(2, 3).contiguous(), K.transpose(2, 3).contiguous(), V, O]
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
    print('out=',O)
    if torch.allclose(O,O_,1e-3,1e-3):
        print("test correct!")
    else:
        print("test error!")
    print(f'eps_ours = {eps}, eps_baseline = {eps_0}, acc = {eps_0/eps}')
    
# if __name__ == "__main__" :
#     devid = 7
#     DeviceInfo.init_cuda([devid])
#     test_attention(devid)
    
if __name__ == "__main__" :
    binpath = sys.argv[1]
    devid = 7
    op = testOp.ReduceOp()
    # binpath = "/tmp/kcg_kernel-336c5b.hsaco"
    
    func_name = "reduce"
    dtypes = [torch.float32]
    backend = EnumBackendType.HIP
    
    kc = KernelConfigs(binpath, func_name, dtypes, backend)
    kc.m_gridDims = [2,1,1]
    kc.m_blockDims = [64,1,1]
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16
    DeviceInfo.init_cuda([devid])
    a = torch.ones((8,16),dtype=torch.float32,device=f'cuda:{devid}')
    b = torch.empty((8,16),dtype=torch.float32,device=f'cuda:{devid}')
    kernel.run(a,b)
    print("a = ",a)
    print("b = ",b)
    