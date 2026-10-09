#ifndef __COMM_CUH__
#define __COMM_CUH__
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <random>
#include <memory>

namespace comm {
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
        if(!event_) check_cuda(cudaEventCreate(&event_), "cuda create event");
    }
    ~CudaEvent()
    {
        if(event_) check_cuda(cudaEventDestroy(event_), "cuda destroy event");
    }

    cudaEvent_t get()
    {
        return event_;
    }

    // 禁用拷贝构造函数
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};  

/*
 * @brief 误差比较函数
*/
void verify_results(
    const std::vector<float> &expected,
    const std::vector<float> &actual, int columns
) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("Result sizes do not match");
    }
    const float AbsoluteTolerance = 1.0e-4f;
    const float RelativeTolerance = 1.0e-4f;
    bool valid = true;
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float reference = expected[index];
        const float value = actual[index];
        const bool finite = std::isfinite(reference) && std::isfinite(value);
        const float absolute_error = finite
            ? std::fabs(reference - value)
            : std::numeric_limits<float>::infinity();
        const float relative_error = finite
            ? absolute_error / std::max(std::fabs(reference), AbsoluteTolerance)
            : std::numeric_limits<float>::infinity();

        const float allowed_error = AbsoluteTolerance + RelativeTolerance * std::fabs(reference);
        if (!(finite && absolute_error <= allowed_error)) {
            valid = false;  // 只能置 false, 不能被后面正确的元素覆盖回 true
            if (++mismatches > 10) continue;  // 只打印前 10 个
            const std::size_t row =
                index / static_cast<std::size_t>(columns);
            const std::size_t col =
                index % static_cast<std::size_t>(columns);
            std::cerr << "Mismatch at (" << row << ", " << col << ")"
                        << ": expected=" << reference
                        << ", actual=" << value
                        << ", abs_error=" << absolute_error << '\n';
        }
    }
    if(valid)
        std::cout << "results correct!" << '\n';
    else
        std::cerr << "total mismatches: " << mismatches << '\n';
}
__device__ __inline__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__device__ __inline__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

} // namespace comm
#endif