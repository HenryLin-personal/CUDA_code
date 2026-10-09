# 最低版本要求
cmake_minimum_required(VERSION 3.20)
# 项目信息
project(global_memory LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 14) # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

add_executable(reductionv0 reduction_v0.cu)
add_executable(reductionv1 reduction_v1.cu)
add_executable(reductionv2 reduction_v2.cu)
add_executable(reductionv3 reduction_v3.cu)
add_executable(reductionv4 reduction_v4.cu)
