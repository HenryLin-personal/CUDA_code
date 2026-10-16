cmake_minimum_required(VERSION 3.18)
project(flash_attention LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 导出一份编译命令
set(CMAKE_CUDA_STANDARD 17) # 用于指定CUDA编译器应使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 17) # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

find_package(CUDAToolkit REQUIRED)

add_executable(self_attn_cpu self_attn.cc)
add_executable(self_attn self_attn.cu)