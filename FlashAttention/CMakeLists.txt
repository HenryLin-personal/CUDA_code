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
target_include_directories(attn_cpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

# fa0.cu ~ fa9.cu: 文件存在才生成对应 target, 新增版本后重新 cmake 即可
foreach(v RANGE 0 9)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fa${v}.cu)
        add_executable(fa${v} fa${v}.cu)
        target_include_directories(fa${v} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    endif()
endforeach()

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")

message(STATUS "CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG = ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "CMAKE_CUDA_FLAGS_RELEASE = ${CMAKE_CUDA_FLAGS_RELEASE}")

# Release构建命令行：
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# 该语句默认开启-O3优化，并不生成gdb调试符号

# Debug构建命令行：
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
# 该语句默认开启-O0优化，生成gdb调试符号
