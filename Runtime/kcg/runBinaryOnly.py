from kcg.Kernel import *
import kcg.Operators.testop as testOp

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
    