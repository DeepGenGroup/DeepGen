# #####  测试用。接口不完善，只是提供了相对便利的构造 CompiledKernel 的方式


# # from kcg.Kernel import *
# # from kcg.CompiledKernel import *
# from kcg.Operators.matmul import *


    
# # 用户输入：hsacopath，kernel名字(通过amdgcn获取)，

# class CompiledKernelFactory :
#     @staticmethod
#     def getKernel(info : KernelConfigs, deviceId : int) -> CompiledKernel:
#         if info.operatorKind is EnumOperator.Matmul :
#             # signature = getMatmulSignature(info.kernelParam.dtypeTorch('A'),info.kernelParam.dtypeTorch('B'),info.kernelParam.dtypeTorch('C'))
#             signature = MatmulOp.GetSignature(info.dtypes)
#             return CompiledKernel(
#                 info.backend,
#                 info.binaryPath,
#                 info.kernelFuncName,
#                 info.sharedMem(),
#                 signature,
#                 info.gridDims(),
#                 info.blockDims(),
#                 deviceId
#             )
#         if info.operatorKind is EnumOperator.Convolution :
#             return None
#         if info.operatorKind is EnumOperator.Poll:
#             return None