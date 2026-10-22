from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_flash_attention',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            'my_fa._C',
            ['my_fa/csrc/fa1.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                # -Xptxas -v: 编译时打印每个 kernel 的寄存器用量和 spill 情况
                'nvcc': ['-O3', '-lineinfo', '-Xptxas', '-v'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
