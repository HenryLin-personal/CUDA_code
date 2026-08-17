cmake_minimum_required(VERSION 3.18)
project(matmul LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 
set(CMAKE_CUDA_STANDARD 17) # 用于指定CUDA编译器应使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 17) # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDAToolkit_TARGET_DIR = ${CUDAToolkit_TARGET_DIR}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS = ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIR = ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDA_cublas_LIBRARY = ${CUDA_cublas_LIBRARY}")

add_executable(matmulv0 matmul0.cu)
target_link_libraries(matmulv0 PRIVATE CUDA::cublas)
add_executable(matmulv1 matmul1.cu)
target_link_libraries(matmulv1 PRIVATE CUDA::cublas)