#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "setDevice.cuh"

namespace {

constexpr int kElementCount = 512;
constexpr int kThreadsPerBlock = 256;
constexpr float kTolerance = 1.0e-6f;

inline void check_cuda(cudaError_t error, const char* expression)
{
    if (error == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string{"CUDA call failed: "} + expression + " -> " +
        cudaGetErrorString(error));
}

template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const noexcept
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
using DevicePtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
DevicePtr<T> make_device_buffer(std::size_t count)
{
    T* raw_ptr = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&raw_ptr), count * sizeof(T)),
               "cudaMalloc");
    return DevicePtr<T>(raw_ptr);
}

__device__ inline float device_add(float lhs, float rhs)
{
    return lhs + rhs;
}

__global__ void matrix_add(const float* a,
                           const float* b,
                           float* c,
                           int element_count)
{
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_index >= element_count) {
        return;
    }

    c[global_index] = device_add(a[global_index], b[global_index]);
}

void initialize(std::vector<float>& values, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(0.0f, 25.5f);

    std::generate(values.begin(), values.end(), [&]() { return dist(rng); });
}

std::vector<float> host_matrix_add(const std::vector<float>& lhs,
                                   const std::vector<float>& rhs)
{
    std::vector<float> result(lhs.size(), 0.0f);
    std::transform(lhs.begin(),
                   lhs.end(),
                   rhs.begin(),
                   result.begin(),
                   [](float a, float b) { return a + b; });
    return result;
}

bool verify_result(const std::vector<float>& expected,
                   const std::vector<float>& actual)
{
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (std::fabs(expected[index] - actual[index]) > kTolerance) {
            std::printf("Mismatch at index %zu: expected=%f, actual=%f\n",
                        index,
                        expected[index],
                        actual[index]);
            return false;
        }
    }

    return true;
}

}  // namespace

int main()
{
    try {
        setDevice();

        std::vector<float> host_a(kElementCount, 0.0f);
        std::vector<float> host_b(kElementCount, 0.0f);
        std::vector<float> host_c(kElementCount, 0.0f);

        std::mt19937 rng(666);
        initialize(host_a, rng);
        initialize(host_b, rng);

        auto device_a = make_device_buffer<float>(host_a.size());
        auto device_b = make_device_buffer<float>(host_b.size());
        auto device_c = make_device_buffer<float>(host_c.size());

        const std::size_t bytes = host_a.size() * sizeof(float);
        check_cuda(cudaMemcpy(device_a.get(),
                              host_a.data(),
                              bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(host_a -> device_a)");
        check_cuda(cudaMemcpy(device_b.get(),
                              host_b.data(),
                              bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(host_b -> device_b)");

        const int blocks =
            static_cast<int>((kElementCount + kThreadsPerBlock - 1) /
                             kThreadsPerBlock);
        matrix_add<<<blocks, kThreadsPerBlock>>>(
            device_a.get(),
            device_b.get(),
            device_c.get(),
            static_cast<int>(kElementCount));
        check_cuda(cudaGetLastError(), "matrix_add launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        check_cuda(cudaMemcpy(host_c.data(),
                              device_c.get(),
                              bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(device_c -> host_c)");

        const auto expected = host_matrix_add(host_a, host_b);
        const bool ok = verify_result(expected, host_c);

        std::printf("matrix_add verification: %s\n", ok ? "PASS" : "FAIL");
        for (int index = 0; index < 5; ++index) {
            std::printf("host_a[%d]=%.3f, host_b[%d]=%.3f, host_c[%d]=%.3f\n",
                        index,
                        host_a[index],
                        index,
                        host_b[index],
                        index,
                        host_c[index]);
        }

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return EXIT_FAILURE;
    }
}
