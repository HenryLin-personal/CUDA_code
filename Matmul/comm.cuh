#ifndef __COMM_CUH__
#define __COMM_CUH__
#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <random>
#include <memory>

/**
 * @brief 错误检查函数
 */
inline void check_cuda(cudaError_t error, const char* expression)
{
    if (error == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string{"CUDA call failed: "} + expression + " -> " +
        cudaGetErrorString(error));
}
/**
 * @brief 初始化设备
 */
void setDevice()
{
    int iDeviceCount = 0;
    check_cuda(cudaGetDeviceCount(&iDeviceCount), "CUDA GET DEVICE FAIL!");
    std::printf("GPU count is %d\n", iDeviceCount);

    int iDeviceindex = 0;
    check_cuda(cudaSetDevice(iDeviceindex), "CUDA SET DEVICE 0 FAIL!");
    std::printf("Set GPU 0 success!\n");
}
/**
 * @brief 初始化主机设备
 */
template <typename T>
void initialize(std::vector<T> &t, std::mt19937 &rng)
{
    // std::uniform_real_distribution dist(0.f, 5.f);
    // std::generate(t.begin(), t.end(), [&] { return dist(rng); });
    std::generate(t.begin(), t.end(), [&] { return 1.f; });
}

/**
 * @brief cuda设备数据结构封装为智能指针
 */
template <typename T>
struct cudaDeleter
{
    void operator()(T *ptr) noexcept
    {
        if(ptr) cudaFree(ptr);
    }
};
template <typename T>
using DevicePtr = std::unique_ptr<T, cudaDeleter<T>>;
template <typename T>
DevicePtr<T> make_device_buffer(std::size_t count)
{
    T *raw{nullptr};
    check_cuda(
        cudaMalloc((void**)&raw, count * sizeof(T)),
        "cudaMalloc fail!"
    );
    return DevicePtr<T>(raw);
}


#endif