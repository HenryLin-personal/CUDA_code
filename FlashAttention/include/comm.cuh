#ifndef __COMM_CUH__
#define __COMM_CUH__
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace comm {
/**
 * @brief 错误检查函数
 */
inline void check_cuda(cudaError_t error, const char *expression)
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
inline void setDevice()
{
    int iDeviceCount = 0;
    check_cuda(cudaGetDeviceCount(&iDeviceCount), "CUDA GET DEVICE FAIL!");
    std::printf("GPU count is %d\n", iDeviceCount);

    int iDeviceindex = 0;
    check_cuda(cudaSetDevice(iDeviceindex), "CUDA SET DEVICE 0 FAIL!");
    std::printf("Set GPU 0 success!\n");
}

/**
 * @brief 初始化主机数据
 * 注意: 不能全填 1, 否则每行 softmax 都是均匀分布, 索引写错也能"算对"
 */
template <typename T>
void initialize(std::vector<T> &t, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::generate(t.begin(), t.end(), [&] { return static_cast<T>(dist(rng)); });
}

/**
 * @brief cuda设备数据结构封装为智能指针
 */
template <typename T>
struct cudaDeleter
{
    void operator()(T *ptr) noexcept
    {
        if (ptr) cudaFree(ptr);
    }
};
template <typename T>
using DevicePtr = std::unique_ptr<T, cudaDeleter<T>>;
template <typename T>
DevicePtr<T> make_device_buffer(std::size_t count)
{
    T *raw{nullptr};
    check_cuda(
        cudaMalloc((void **)&raw, count * sizeof(T)),
        "cudaMalloc fail!"
    );
    return DevicePtr<T>(raw);
}

/**
 * @brief 计时对象封装
 */
class CudaEvent
{
private:
    cudaEvent_t event_{nullptr};
public:
    CudaEvent()
    {
        check_cuda(cudaEventCreate(&event_), "cuda create event");
    }
    ~CudaEvent()
    {
        if (event_) cudaEventDestroy(event_);
    }

    cudaEvent_t get() const
    {
        return event_;
    }

    // 禁用拷贝构造函数
    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;
};

inline int ceil_div(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

/**
 * @brief 先 warmup, 再取 iters 次的平均耗时 (ms)
 */
template <typename Operation>
float benchmark_gpu(Operation &&operation, int warmup = 10, int iters = 50)
{
    for (int i = 0; i < warmup; ++i) {
        operation();
    }
    check_cuda(cudaGetLastError(), "GPU warmup launch");
    check_cuda(cudaDeviceSynchronize(), "GPU warmup synchronize");

    CudaEvent start;
    CudaEvent stop;

    check_cuda(cudaEventRecord(start.get()), "cudaEventRecord start");
    for (int i = 0; i < iters; ++i) {
        operation();
    }
    check_cuda(cudaGetLastError(), "GPU benchmark launch");
    check_cuda(cudaEventRecord(stop.get()), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");

    float total_ms = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&total_ms, start.get(), stop.get()),
        "cudaEventElapsedTime"
    );
    return total_ms / static_cast<float>(iters);
}

} // namespace comm
#endif
