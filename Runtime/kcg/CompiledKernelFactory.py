#####  测试用。接口不完善，只是提供了相对便利的构造 CompiledKernel 的方式

from kcg.CompiledKernel import *
from kcg.Operators.matmul import *

class EnumOperator(Enum):
    Invalid = 0
    Matmul = 1
    Convolution = 2
    Poll = 3
    def __str__(self):
        return f'{self.name}'


class UserInputs:
    def __init__(self,hsaco_path:str,kernel_func_name:str,kernelParam : KernelArgMatmul):
        self.operatorKind = EnumOperator.Matmul
        self.hsacoPath = hsaco_path
        self.kernelFuncName = kernel_func_name
        self.kernelParam = kernelParam
        self.m_gridDims = [1,1,1]
        self.m_blockDims = [1,1,1]

    def gridDims(self):  # 行优先矩阵，行方向为x方向，尺寸为n
        return self.m_gridDims
    
    def blockDims(self):
        return self.m_blockDims
        
    def sharedMem(self):
        # 假设 ABC类型相同
        # 还需要考虑 doublebuffer的情况
        kp = self.kernelParam
        sizeA = kp.BLOCK_SIZE_M*kp.BLOCK_SIZE_K*sizeof(kp.dtype('A'))
        sizeB = kp.BLOCK_SIZE_N*kp.BLOCK_SIZE_K*sizeof(kp.dtype('B'))
        sizeAB = sizeA + sizeB
        if kp.SHARED_PREFETCH > 0 :
            sizeAB *= 2
        sizeC = -1
        if kp.LOCAL_SPLIT_U > 1 :
            sizeC = kp.BLOCK_SIZE_M * kp.BLOCK_SIZE_N * kp.LOCAL_SPLIT_U * sizeof(kp.dtype('A'))
        if sizeAB > sizeC :
            return sizeAB
        return sizeC
    
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
        if info.operatorKind==EnumOperator.Matmul :
            signature = getMatmulSignature(info.kernelParam.dtypeTorch('A'),info.kernelParam.dtypeTorch('B'),info.kernelParam.dtypeTorch('C'))
            return CompiledKernel(
                info.hsacoPath,
                info.kernelFuncName,
                info.sharedMem(),
                signature,
                info.gridDims(),
                info.blockDims(),
                deviceId
            )
        if info.operatorKind==EnumOperator.Convolution :
            return None
        if info.operatorKind==EnumOperator.Poll:
            return None