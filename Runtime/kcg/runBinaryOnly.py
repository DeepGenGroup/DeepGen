from kcg.Kernel import *
import kcg.Operators.testop as testOp
import kcg.Operators.attention as attnOp
import torch.nn.functional as F
import math
import numpy as np

def row_wise_softmax(matrix, dim=-1):
    """
    手动实现按行 softmax
    :param matrix: 输入矩阵 (m, n)
    :return: 按行 softmax 后的矩阵 (m, n)
    """
    # 步骤1: 减去每行的最大值（数值稳定性）
    max_vals, _ = torch.max(matrix, dim= dim, keepdim=True)
    shifted = matrix - max_vals
    
    # 步骤2: 计算指数
    exp_vals = torch.exp(shifted)
    
    # 步骤3: 计算每行的指数和
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)
    
    # 步骤4: 计算 softmax
    softmax_result = exp_vals
    # softmax_result = exp_vals / sum_exp
    # print(f"exp_vals = {exp_vals}, sum_exp = {sum_exp}")
    return [ softmax_result, sum_exp]


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
    funcName = 'kcg_Attention_1_32_2048_128_Br16Bc64Hd128_Sa16Sb8PTr4PTc4OTr4OTc8GLWQ4GLWK4GLWV4BLPY1BLPX1WLPY4WLPX16BSWQ4BSWK2WSWQ4WSWK2BLOY1BLOX1WLOY4WLOX16BSWP4BSWV2WSWP4WSWV2Un1W64LCP1LCO1SPP0RPP0RPO0'
    dtypes = [torch.float32]
    
    backend = EnumBackendType.HIP
    def testTorch(inputs, needSoftmax = True):
        def attnFunc(Q, K, V, O):
            d = Q.shape[1] * Q.shape[3]
            sum_exp = None
            P = torch.matmul(Q, K) # / math.sqrt(d)
            temp = P[:,:,0,:].max()
            # print("torchM = ",temp)
            if needSoftmax :
                # S = F.softmax(P, dim=-1)
                S,sum_exp = row_wise_softmax(P)
            else:
                S = P
            O = torch.matmul(S, V)
            return [O, sum_exp]
        # t = perf(attnFunc, inputs)
        return attnFunc(*inputs)
    kc = KernelConfigs(binpath, funcName, dtypes, backend)
    # func.block.dim = array<i32: 128>, func.grid.dim = array<i32: 64, 32, 1>
    #     blockDims = [64, 1, 1]
    # gridDims = [128, 32, 1]
    kc.m_gridDims = [128,32,1]
    kc.m_blockDims = [64,1,1]
    kc.shmBytes = 18816
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16

    bs,hn,sl,hd = [1,32,2048, 128]
    factor = 1
    Q = torch.ones((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}') * factor
    K = torch.ones((bs, hn, hd, sl), dtype=torch.float32, device=f'cuda:{devid}') * factor
    V = torch.ones((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}') * factor
    O = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')
    O_ = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')
    
    # for i in range(0,2048) :
    # Q[:,:,0,0] = 2
        # Q[:,:,i,64] = 12
    K[:,:,0,0] = 2
    K[:,:,0,64] = 3
    V = torch.zeros((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}') 
    
    V[:,:,0,:] = 1
    # [exp(0), exp(-1), ... , exp(-1) ]
    # O = [exp(0) + 2047 * exp(-1)]
    
    import math
    checkNum = math.exp(-2) * 2046 + math.exp(0) + math.exp(-1)

    # Q[:,:,:,::2] *= 2
    # Q[:,:,:,::3] *= 3
    # Q*K = error, 检查相关IR
    
    inputs = [Q.transpose(-1, -2).contiguous(), K, V, O]
    inputs_ = [Q, K, V, O_]
    
    # print(f"torch time cost: {t} ms")
    def _f_ours() :
        kernel.run(inputs[0],inputs[1],inputs[2],inputs[3])
    def _f_baseline() :
        testTorch(inputs=inputs_, needSoftmax=True)
    
    eps_ours = []
    eps_base = []
    # warmup
    [O_,sumexp] = testTorch(inputs=inputs_)
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
    print('ours=',O)
    
    # debug check

    # 检查张量中的所有元素是否等于 2048
    all_elements_equal_to_checknum = torch.all(O == checkNum).item()
    if all_elements_equal_to_checknum:
        print(f"All elements in the tensor are equal to {checkNum }.")
    else:
        print(f"Not all elements in the tensor are equal to {checkNum }.")
        
    if torch.allclose(O,O_,atol=1e-3,rtol=1e-3):
        print("test correct!")
    else:
        print("test error!")
        # diff = O / sumexp
        # (diff < 1).count_nonzero()
        # print(f"O_ /O = {diff},  lessThanOneRatio = {(diff < 1).count_nonzero() / (32*2048*128)} ")
        # torchMax = O[:,:,0,:].max()
        # print("max = ",torchMax)
        
    if torch.isnan(O).any() :
        print("detect NaN!")
    else:
        print("No NaN!")
        
    print(f'eps_ours = {eps}, eps_baseline = {eps_0}, acc = {eps_0/eps}')
    
def test_debug_attention() :
    if len(sys.argv) < 2 :
        print("usage : kernelBinPath")
        sys.exit()
    binpath = sys.argv[1]
    op = testOp.DebugAttnOp()
    devid = 7
    DeviceInfo.init_cuda([devid])
    funcName = 'kcg_Attention_1_32_2048_128_Br16Bc64Hd128_Sa16Sb8PTr4PTc4OTr4OTc8GLWQ4GLWK4GLWV4BLPY1BLPX1WLPY4WLPX16BSWQ4BSWK2WSWQ4WSWK2BLOY1BLOX1WLOY4WLOX16BSWP4BSWV2WSWP4WSWV2Un1W64LCP1LCO1SPP0RPP0RPO0'
    dtypes = [torch.float32]
    
    backend = EnumBackendType.HIP
    def testTorch(inputs, needSoftmax = True):
        def attnFunc(Q, K, V, O):
            d = Q.shape[1] * Q.shape[3]
            sum_exp = None
            P = torch.matmul(Q, K) # / math.sqrt(d)
            temp = P[:,:,0,:].max()
            # print("torchM = ",temp)
            if needSoftmax :
                # S = F.softmax(P, dim=-1)
                S,sum_exp = row_wise_softmax(P)
            else:
                S = P
            O = torch.matmul(S, V)
            return [O, sum_exp]
        # t = perf(attnFunc, inputs)
        return attnFunc(*inputs)
    kc = KernelConfigs(binpath, funcName, dtypes, backend)
    # func.block.dim = array<i32: 128>, func.grid.dim = array<i32: 64, 32, 1>
    #     blockDims = [64, 1, 1]
    # gridDims = [128, 32, 1]
    kc.m_gridDims = [128,32,1]
    kc.m_blockDims = [64,1,1]
    kc.shmBytes = 18816
    kernel = op.GetCompiledKernel(kc,devid)
    # test 8x16

    bs,hn,sl,hd = [1,32,2048, 128]
    factor = 1
    Q = torch.ones((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}') * factor
    K = torch.ones((bs, hn, hd, sl), dtype=torch.float32, device=f'cuda:{devid}') * factor
    V = torch.ones((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}') * factor
    O = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')
    O_ = torch.empty((bs, hn, sl, hd), dtype=torch.float32, device=f'cuda:{devid}')
    
    P = torch.zeros((4,8), dtype=torch.float32, device=f'cuda:{devid}')
    # for i in range(0,2048) :
    # Q[:,:,0,0] = 2
        # Q[:,:,i,64] = 12
    K[:,:,0,63] = 3
    K[:,:,0,-1] = 4
    # [exp(0), exp(-1), ... , exp(-1) ]
    # O = [exp(0) + 2047 * exp(-1)]
    
    import math
    checkNum = math.exp(-2) * 2046 + math.exp(0) + math.exp(-1)

    # Q[:,:,:,::2] *= 2
    # Q[:,:,:,::3] *= 3
    # Q*K = error, 检查相关IR
    
    inputs = [Q.transpose(-1, -2).contiguous(), K, V, O, P]
    inputs_ = [Q, K, V, O_]
    
    # print(f"torch time cost: {t} ms")
    def _f_ours() :
        kernel.run(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4])
    def _f_baseline() :
        testTorch(inputs=inputs_, needSoftmax=True)
    
    eps_ours = []
    eps_base = []
    # warmup
    [O_,sumexp] = testTorch(inputs=inputs_)
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
    print('ours=',O)
    
    # debug check
    print("P=",P)
    checkp = 1
    if torch.all(P == checkp).item() :
        print(f"P all {checkp}")
    else:
        print(f"P not all {checkp}")
    
    # 检查张量中的所有元素是否等于 2048
    checkNum = 0
    all_elements_equal_to_checknum = torch.all(O == checkNum).item()
    if all_elements_equal_to_checknum:
        print(f"All elements in the tensor are equal to {checkNum }.")
    else:
        print(f"Not all elements in the tensor are equal to {checkNum }.")
        
    if torch.allclose(O,O_,atol=1e-3,rtol=1e-3):
        print("test correct!")
    else:
        print("test error!")
        # diff = O / sumexp
        # (diff < 1).count_nonzero()
        # print(f"O_ /O = {diff},  lessThanOneRatio = {(diff < 1).count_nonzero() / (32*2048*128)} ")
        # torchMax = O[:,:,0,:].max()
        # print("max = ",torchMax)
        
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
        for j in range(0,tensorSHape[1]) :
            a[i,j] = i*10 + j
    # b = torch.empty(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}')
    print("original a = ",a)
    kernel.run(a)
    print("result a = ",a)
    # print("b = ",b)
    

def test_infOp() :
    if len(sys.argv) < 2 :
        print("usage : kernelBinPath")
        sys.exit()
    binpath = sys.argv[1]
    devid = 7
    op = testOp.BroadcastOp()
    
    func_name = "inftest"
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
    a = torch.ones(tensorSHape,dtype=torch.float32,device=f'cuda:{devid}') * -1
    # for i in range(0,tensorSHape[0]) :
    #     a[i,0] = i+10
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
    # test_attention()
    test_debug_attention()
    # test_expOp()
    # test_infOp()
    