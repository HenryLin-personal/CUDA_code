from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
# from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='myhello',
    packages=find_packages(),
    ext_modules=[
        # CppExtension('myhello._C', ['myhello/csrc/myadd.cpp'])
        Pybind11Extension('myhello._C', ['myhello/csrc/myadd.cpp'])
    ],
    cmdclass={'build_ext': build_ext},
    # cmdclass={'build_ext': BuildExtension},
)