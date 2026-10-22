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

add_executable(attn_cpu attn_cpu.cc)
add_executable(fa0 fa0.cu)
add_executable(fa1 fa1.cu)
add_executable(fa2 fa2.cu)
add_executable(fa3 fa3.cu)

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")

message(STATUS "CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG = ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "CMAKE_CUDA_FLAGS_RELEASE = ${CMAKE_CUDA_FLAGS_RELEASE}")
