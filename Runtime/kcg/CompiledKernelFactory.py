#####  测试用。接口不完善，只是提供了相对便利的构造 CompiledKernel 的方式

from kcg.CompiledKernel import *
from kcg.Operators.matmul import *


class UserInputs:
    def __init__(self,binary_path:str,kernel_func_name:str,kernelParam : KernelArgMatmul, backend : EnumBackendType):
        self.operatorKind = EnumOperator.Matmul
        self.binaryPath = binary_path
        self.kernelFuncName = kernel_func_name
        self.kernelParam = kernelParam
        self.m_gridDims = [1,1,1]
        self.m_blockDims = [1,1,1]
        self.backend = backend
        self.shmBytes = 0
        
    def gridDims(self):  # 行优先矩阵，行方向为x方向，尺寸为n
        return self.m_gridDims
    
    def blockDims(self):
        return self.m_blockDims
        
    def sharedMem(self):
        return self.shmBytes
        # 假设 ABC类型相同
        # # 还需要考虑 doublebuffer的情况
        # kp = self.kernelParam
        # sizeA = kp.BLOCK_SIZE_M*kp.BLOCK_SIZE_K*sizeof(kp.dtype('A'))
        # sizeB = kp.BLOCK_SIZE_N*kp.BLOCK_SIZE_K*sizeof(kp.dtype('B'))
        # sizeAB = sizeA + sizeB
        # if kp.SHARED_PREFETCH > 0 :
        #     sizeAB *= 2
        # sizeC = -1
        # if kp.LOCAL_SPLIT_U > 1 :
        #     sizeC = kp.BLOCK_SIZE_M * kp.BLOCK_SIZE_N * kp.LOCAL_SPLIT_U * sizeof(kp.dtype('A'))
        # if sizeAB > sizeC :
        #     return sizeAB
        # return sizeC
    
    def numCTA(self) : 
        ret = 1
        m = self.kernelParam.M
        n = self.kernelParam.N
        k = self.kernelParam.K
        for dim in self.gridDims(m,n,k):
            ret *= dim
        return ret
    
# 用户输入：hsacopath，kernel名字(通过amdgcn获取)，
class CompiledKernelFactory :
    @staticmethod
    def getKernel(info : UserInputs, deviceId : int) -> CompiledKernel:
        if info.operatorKind is EnumOperator.Matmul :
            signature = getMatmulSignature(info.kernelParam.dtypeTorch('A'),info.kernelParam.dtypeTorch('B'),info.kernelParam.dtypeTorch('C'))
            return CompiledKernel(
                info.backend,
                info.binaryPath,
                info.kernelFuncName,
                info.sharedMem(),
                signature,
                info.gridDims(),
                info.blockDims(),
                deviceId
            )
        if info.operatorKind is EnumOperator.Convolution :
            return None
        if info.operatorKind is EnumOperator.Poll:
            return None