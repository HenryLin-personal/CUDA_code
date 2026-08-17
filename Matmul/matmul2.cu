#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "comm.cuh"

namespace {

// Row-major matrix layout:
// A: M x K, B: K x N, C: M x N.
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int BlockX = 16;
constexpr int BlockY = 16;
constexpr int Warmup = 10;
constexpr int Iters = 50;

constexpr float Alpha = 1.0f;
constexpr float Beta = 0.0f;
constexpr float AbsoluteTolerance = 1.0e-4f;
constexpr float RelativeTolerance = 1.0e-4f;

// benchmark_gpu repeatedly launches the same operation without restoring C.
// Keeping beta at zero makes every timed launch independent of the previous one.
static_assert(Beta == 0.0f, "Repeated benchmark launches require beta == 0");

int ceil_div(int numerator, int denominator)
{
    if (numerator < 0 || denominator <= 0) {
        throw std::invalid_argument("ceil_div expects non-negative numerator and positive denominator");
    }
    return numerator / denominator + (numerator % denominator != 0);
}

void check_cublas(cublasStatus_t status, const char *expression)
{
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }

    throw std::runtime_error(
        std::string{"cuBLAS call failed: "} + expression +
        " -> status " + std::to_string(static_cast<int>(status))
    );
}

class CublasHandle {
public:
    CublasHandle()
    {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle()
    {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle &) = delete;
    CublasHandle &operator=(const CublasHandle &) = delete;

    cublasHandle_t get() const
    {
        return handle_;
    }

private:
    cublasHandle_t handle_{nullptr};
};

class CudaEvent {
public:
    CudaEvent()
    {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    ~CudaEvent()
    {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    cudaEvent_t get() const
    {
        return event_;
    }

private:
    cudaEvent_t event_{nullptr};
};

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_v2(
    int m,
    int n,
    int k,
    float alpha,
    const float *a,
    const float *b,
    float beta,
    float *c)
{
    // const int bx = blockIdx.x;
    // const int by = blockIdx.y;
    // const int tx = threadIdx.x * TN;
    // const int ty = threadIdx.y * TM;

    // __shared__ float As[BM * BK];
    // __shared__ float Bs[BK * BN];

    // __shared__ float As_2d[BM][BK];
    // __shared__ float Bs_2d[BK][BN];

    // a = &a[by * BM * k];
    // b = &b[bx * BN];
    // c = &c[by * BM * n + bx * BN];

    // const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // const int thread_num = blockDim.x * blockDim.y;

    // // 协作加载A tile
    // int a_tile_row = tid / BK;  // 思考：若改为threadIdx.x/theadIdx.y, a_tile_row对每个[TM, TN]的微块起始地址一样吗？
    // int a_tile_col = tid % BK;
    // int a_tile_stride = thread_num / BK;

    // // 协作加载B tile
    // int b_tile_row = tid / BN;
    // int b_tile_col = tid % BN;
    // int b_tile_stride = thread_num / BN;

    // float tmp[TM][TN] = {0.};
    // for(int k_i = 0; k_i < k; k_i += BK)
    // {
    //     // 1. 线程复制数据到共享内存
    //     for (int i = 0; i < BM; i += a_tile_stride) {
    //         // As[(a_tile_row + i) * BK + a_tile_col] =
    //         //     a[(a_tile_row + i) * k + a_tile_col];
    //         As_2d[a_tile_row + i][a_tile_col] = a[(a_tile_row + i) * k + a_tile_col];
    //     }
    //     for (int i = 0; i < BK; i += b_tile_stride) {
    //         // Bs[(b_tile_row + i) * BN + b_tile_col] =
    //         //     b[(b_tile_row + i) * n + b_tile_col];
    //         Bs_2d[b_tile_row + i][b_tile_col] = b[(b_tile_row + i) * n + b_tile_col];
    //     }
       
    //     __syncthreads();

    //     // 2. 复用matmul0中的矩阵乘法逻辑
    //     for (int i = 0; i < BK; i++) {
    //         for (int j = 0; j < TM; j++) {
    //             for (int l = 0; l < TN; l++) {
    //                 // tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + i * BN + l];
    //                 tmp[j][l] += As_2d[ty + j][i] * Bs_2d[i][tx + l];
    //             }
    //         }
    //     }
    //     __syncthreads();

    //     // 3. 更新A和B
    //     a += BK; b += n * BK;
    // }
    // // 4. 将结果传回c矩阵
    // for (int j = 0; j < TM; j++) {
    //     for (int l = 0; l < TN; l++) {
    //         c[(ty + j) * N + tx + l] =  
    //             alpha * tmp[j][l] + beta * c[(ty + j) * N + tx + l];
    //     }
    // }

    // 已有BM, BN, BK, TM, TN
    // 4096^2个数据项，
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_num = blockDim.x * blockDim.y;

    a = &a[by * BM * k];
    b = &b[bx * BN];
    c = &c[by * BM * n + bx * BN];

    __shared__ float As_2d[BM][BK];
    __shared__ float Bs_2d[BK][BN];

    const int tx = threadIdx.x * TN;
    const int ty = threadIdx.y * TM;
    const int a_tile_row = tid / BK;
    const int a_tile_col = tid % BK;
    const int a_tile_stride = thread_num / BK;

    const int b_tile_row = tid / BN;
    const int b_tile_col = tid % BN;
    const int b_tile_stride = thread_num / BN;
    float tmp[TM][TN] = {0.f};
    for(int k_i = 0; k_i < k; k_i += BK, a += BK, b += n * BK)
    {
        // 1. 搬运数据至共享显存
        for(int i = 0; i < BM; i += a_tile_stride) {
            As_2d[a_tile_row + i][a_tile_col] = a[(a_tile_row + i) * k + a_tile_col];
        }
        for(int i = 0; i < BK; i += b_tile_stride) {
            Bs_2d[b_tile_row + i][b_tile_col] = b[(b_tile_row + i) * n + b_tile_col];
        }
        __syncthreads();

        // 2. 共享显存求乘法
        for(int i = 0; i < BK; i++)
        {
            for(int j = 0; j < TM; j++)
            {
                for(int l = 0; l < TN; l++)
                {
                    tmp[j][l] += As_2d[ty + j][i] * Bs_2d[i][tx + l];
                }
            }
        }
        __syncthreads();

        // 3. 更新至下一个循环
    }
    // 4. tmp写回C矩阵
    for(int j = 0; j < TM; ++j)
    {
        for(int l = 0; l < TN; ++l)
        {
            float &cur = c[(ty + j) * n + tx + l];
            cur = alpha * tmp[j][l] + beta * cur;
        }
    }
}

void run_cublas_row_major(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    float alpha,
    const float *device_a,
    const float *device_b,
    float beta,
    float *device_c)
{
    // cuBLAS is column-major by default. A row-major C = A * B is represented
    // by the column-major identity C^T = B^T * A^T, so A and B are swapped.
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            device_b,
            n,
            device_a,
            k,
            &beta,
            device_c,
            n
        ),
        "cublasSgemm"
    );
}

template <typename Operation>
float benchmark_gpu(Operation &&operation)
{
    for (int i = 0; i < Warmup; ++i) {
        operation();
    }
    check_cuda(cudaGetLastError(), "GPU warmup launch");
    check_cuda(cudaDeviceSynchronize(), "GPU warmup synchronize");

    CudaEvent start;
    CudaEvent stop;

    check_cuda(cudaEventRecord(start.get()), "cudaEventRecord start");
    for (int i = 0; i < Iters; ++i) {
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
    return total_ms / static_cast<float>(Iters);
}

struct VerificationResult {
    bool matched{true};
    std::size_t mismatch_count{0};
    float max_absolute_error{0.0f};
    float max_relative_error{0.0f};
};

VerificationResult verify_result(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    int columns)
{
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("Result sizes do not match");
    }
    if (columns <= 0) {
        throw std::invalid_argument("Result column count must be positive");
    }

    VerificationResult result;
    constexpr std::size_t MaxReportedMismatches = 5;

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

        result.max_absolute_error =
            std::max(result.max_absolute_error, absolute_error);
        result.max_relative_error =
            std::max(result.max_relative_error, relative_error);

        const float allowed_error =
            AbsoluteTolerance + RelativeTolerance * std::fabs(reference);
        const bool valid = finite && absolute_error <= allowed_error;

        if (!valid) {
            if (result.mismatch_count < MaxReportedMismatches) {
                const std::size_t row =
                    index / static_cast<std::size_t>(columns);
                const std::size_t col =
                    index % static_cast<std::size_t>(columns);
                std::cerr << "Mismatch at (" << row << ", " << col << ")"
                          << ": expected=" << reference
                          << ", actual=" << value
                          << ", abs_error=" << absolute_error << '\n';
            }
            ++result.mismatch_count;
        }
    }

    result.matched = result.mismatch_count == 0;
    return result;
}

double calculate_gflops(int m, int n, int k, float average_ms)
{
    const double operations =
        2.0 * static_cast<double>(m) * static_cast<double>(n) *
        static_cast<double>(k);
    return operations / (static_cast<double>(average_ms) * 1.0e6);
}

} // namespace

int main()
{
    try {
        setDevice();

        const std::size_t a_count =
            static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
        const std::size_t b_count =
            static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
        const std::size_t c_count =
            static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

        const std::size_t a_bytes = a_count * sizeof(float);
        const std::size_t b_bytes = b_count * sizeof(float);
        const std::size_t c_bytes = c_count * sizeof(float);

        std::vector<float> host_a(a_count);
        std::vector<float> host_b(b_count);
        std::vector<float> host_cublas(c_count, 0.0f);
        std::vector<float> host_v2(c_count, 0.0f);

        // Small random integers are exactly representable as float and expose
        // row/column indexing mistakes that all-one inputs can hide.
        std::mt19937 rng(666);
        std::uniform_int_distribution<int> distribution(-2, 2);
        std::generate(host_a.begin(), host_a.end(), [&] {
            return static_cast<float>(distribution(rng));
        });
        std::generate(host_b.begin(), host_b.end(), [&] {
            return static_cast<float>(distribution(rng));
        });

        auto device_a = make_device_buffer<float>(a_count);
        auto device_b = make_device_buffer<float>(b_count);
        auto device_cublas = make_device_buffer<float>(c_count);
        auto device_v2 = make_device_buffer<float>(c_count);

        check_cuda(
            cudaMemcpy(device_a.get(), host_a.data(), a_bytes, cudaMemcpyHostToDevice),
            "copy A to device"
        );
        check_cuda(
            cudaMemcpy(device_b.get(), host_b.data(), b_bytes, cudaMemcpyHostToDevice),
            "copy B to device"
        );
        check_cuda(cudaMemset(device_cublas.get(), 0, c_bytes), "clear cuBLAS C");
        check_cuda(cudaMemset(device_v2.get(), 0, c_bytes), "clear v2 C");

        CublasHandle cublas;
        const dim3 block(16, 16);
        const dim3 grid(ceil_div(N, 128), ceil_div(M, 128));

        const float cublas_ms = benchmark_gpu([&] {
            run_cublas_row_major(
                cublas.get(),
                M,
                N,
                K,
                Alpha,
                device_a.get(),
                device_b.get(),
                Beta,
                device_cublas.get()
            );
        });

        const float v2_ms = benchmark_gpu([&] {
            matmul_v2<128, 128, 8, 8, 8><<<grid, block>>>(
                M,
                N,
                K,
                Alpha,
                device_a.get(),
                device_b.get(),
                Beta,
                device_v2.get()
            );
        });

        check_cuda(
            cudaMemcpy(
                host_cublas.data(),
                device_cublas.get(),
                c_bytes,
                cudaMemcpyDeviceToHost
            ),
            "copy cuBLAS result to host"
        );
        check_cuda(
            cudaMemcpy(
                host_v2.data(),
                device_v2.get(),
                c_bytes,
                cudaMemcpyDeviceToHost
            ),
            "copy v2 result to host"
        );

        const VerificationResult verification =
            verify_result(host_cublas, host_v2, N);
        const double cublas_gflops = calculate_gflops(M, N, K, cublas_ms);
        const double v2_gflops = calculate_gflops(M, N, K, v2_ms);
        const double cublas_ratio = 100.0 * v2_gflops / cublas_gflops;

        std::cout << "Matrix: A[" << M << " x " << K << "] * B["
                  << K << " x " << N << "] -> C[" << M << " x " << N
                  << "]\n";
        std::cout << "Block: (" << block.x << ", " << block.y
                  << "), Grid: (" << grid.x << ", " << grid.y << ")\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "| Implementation | Average time (ms) | GFLOPS | cuBLAS ratio | Correct |\n";
        std::cout << "|---|---:|---:|---:|---|\n";
        std::cout << "| cuBLAS | " << cublas_ms << " | " << cublas_gflops
                  << " | 100.00% | reference |\n";
        std::cout << "| matmul_v2 | " << v2_ms << " | " << v2_gflops
                  << " | " << cublas_ratio << "% | "
                  << (verification.matched ? "YES" : "NO") << " |\n";
        std::cout << "max absolute error: " << verification.max_absolute_error
                  << ", max relative error: " << verification.max_relative_error
                  << ", mismatches: " << verification.mismatch_count << '\n';

        return verification.matched ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
