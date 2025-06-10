from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add",
    include_dirs=["/home/xushilong/DeepGen/_TempCodes/testReplaceTorchOp"],
    ext_modules=[
        CUDAExtension(
            "add",
            ["add.cpp", "add.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
