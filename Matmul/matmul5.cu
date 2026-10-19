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
#define LOAD_FLOAT4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])
#define STORE_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])

// Row-major matrix layout:
// A: M x K, B: K x N, C: M x N.
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

constexpr int BlockX = BN / TN;
constexpr int BlockY = BM / TM;
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

// V5_MIN_BLOCKS > 0 时加 __launch_bounds__, 要求编译器把寄存器压到能让每个 SM 至少放这么多个 block
#ifndef V5_MIN_BLOCKS
#define V5_MIN_BLOCKS 0
#endif
#if V5_MIN_BLOCKS > 0
#define V5_BOUNDS __launch_bounds__((BM / TM) * (BN / TN), V5_MIN_BLOCKS)
#else
#define V5_BOUNDS
#endif

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void V5_BOUNDS matmul_v5(
    int m,
    int n,
    int k,
    float alpha,
    const float *a,
    const float *b,
    float beta,
    float *c)
{
    static_assert(BK % 2 == 0, "寄存器双缓冲按 inner 奇偶交替, 要求 BK 为偶数");
    const float *a_start = a + BM * blockIdx.y * k;
    const float *b_start = b + BN * blockIdx.x;
    const int tid = threadIdx.x + blockDim.x * threadIdx.y; // 展平为一维线程
    constexpr int threads = (BM / TM) * (BN / TN); // 线程数量
    const int a_tile_row = tid / (BK / 4);
    const int a_tile_col = tid % (BK / 4) * 4;
    constexpr int a_tile_stride = threads / (BK / 4);
    const int b_tile_row = tid / (BN / 4);
    const int b_tile_col = tid % (BN / 4) * 4;
    constexpr int b_tile_stride = threads / (BN / 4);
    constexpr int a_ldg_num = BM / a_tile_stride; // 每个线程每个 tile 搬 A 的 float4 个数
    constexpr int b_ldg_num = BK / b_tile_stride; // 每个线程每个 tile 搬 B 的 float4 个数

    // shared 双缓冲 (同 v4)
    __shared__ float a_shm_T[2][BK][BM];
    __shared__ float b_shm[2][BK][BN];

    float sum[TM][TN]{0.f};
    // 寄存器双缓冲: [inner % 2] 在算, [(inner + 1) % 2] 在装下一个 inner
    float reg_a[2][TM];
    float reg_b[2][TN];
    float4 a_ldg[a_ldg_num];
    float4 b_ldg[b_ldg_num];

    // 0. 预取 tile 0 到 shared buffer 0
    #pragma unroll
    for(int i = 0; i < a_ldg_num; ++i)
    {
        const int row = a_tile_row + i * a_tile_stride;
        float4 va = LOAD_FLOAT4(a_start[row * k + a_tile_col]);
        a_shm_T[0][a_tile_col][row] = va.x;
        a_shm_T[0][a_tile_col + 1][row] = va.y;
        a_shm_T[0][a_tile_col + 2][row] = va.z;
        a_shm_T[0][a_tile_col + 3][row] = va.w;
    }
    #pragma unroll
    for(int j = 0; j < b_ldg_num; ++j)
    {
        const int row = b_tile_row + j * b_tile_stride;
        STORE_FLOAT4(b_shm[0][row][b_tile_col]) = LOAD_FLOAT4(b_start[row * n + b_tile_col]);
    }
    __syncthreads();

    // 再预取 tile 0 的 inner 0 到 reg[0]
    #pragma unroll
    for(int i = 0; i < TM; i += 4) STORE_FLOAT4(reg_a[0][i]) = LOAD_FLOAT4(a_shm_T[0][0][threadIdx.y * 4 + (BM / TM) * i]);
    #pragma unroll
    for(int j = 0; j < TN; j += 4) STORE_FLOAT4(reg_b[0][j]) = LOAD_FLOAT4(b_shm[0][0][threadIdx.x * 4 + (BN / TN) * j]);

    int read_idx = 0;
    for(int s = 0; s < k; s += BK)
    {
        const int next = s + BK;
        const bool has_next = next < k;

        // 1. 发出下一个 tile 的 global 读取 (同 v4)
        if(has_next)
        {
            #pragma unroll
            for(int i = 0; i < a_ldg_num; ++i)
            {
                a_ldg[i] = LOAD_FLOAT4(a_start[(a_tile_row + i * a_tile_stride) * k + a_tile_col + next]);
            }
            #pragma unroll
            for(int j = 0; j < b_ldg_num; ++j)
            {
                b_ldg[j] = LOAD_FLOAT4(b_start[(b_tile_row + j * b_tile_stride + next) * n + b_tile_col]);
            }
        }

        // 2. inner 0 .. BK-2: 先发出 inner+1 的 LDS, 再用 reg[inner % 2] 算 inner
        #pragma unroll
        for(int inner = 0; inner < BK - 1; ++inner)
        {
            const int cur = inner % 2, nxt = (inner + 1) % 2;
            #pragma unroll
            for(int i = 0; i < TM; i += 4) STORE_FLOAT4(reg_a[nxt][i]) = LOAD_FLOAT4(a_shm_T[read_idx][inner + 1][threadIdx.y * 4 + (BM / TM) * i]);
            #pragma unroll
            for(int j = 0; j < TN; j += 4) STORE_FLOAT4(reg_b[nxt][j]) = LOAD_FLOAT4(b_shm[read_idx][inner + 1][threadIdx.x * 4 + (BN / TN) * j]);
            #pragma unroll
            for(int i = 0; i < TM; ++i)
            {
                #pragma unroll
                for(int j = 0; j < TN; ++j)
                {
                    sum[i][j] += reg_a[cur][i] * reg_b[cur][j];
                }
            }
        }

        // 3. 下一个 tile 写进另一份 shared, 同步, 然后立刻发出它 inner 0 的 LDS
        if(has_next)
        {
            const int write_idx = read_idx ^ 1;
            #pragma unroll
            for(int i = 0; i < a_ldg_num; ++i)
            {
                const int row = a_tile_row + i * a_tile_stride;
                a_shm_T[write_idx][a_tile_col][row] = a_ldg[i].x;
                a_shm_T[write_idx][a_tile_col + 1][row] = a_ldg[i].y;
                a_shm_T[write_idx][a_tile_col + 2][row] = a_ldg[i].z;
                a_shm_T[write_idx][a_tile_col + 3][row] = a_ldg[i].w;
            }
            #pragma unroll
            for(int j = 0; j < b_ldg_num; ++j)
            {
                STORE_FLOAT4(b_shm[write_idx][b_tile_row + j * b_tile_stride][b_tile_col]) = b_ldg[j];
            }
            __syncthreads();
            read_idx = write_idx;

            // BK 为偶数, 最后一个 inner (BK-1) 用的是 reg[1], 所以下一个 tile 的 inner 0 可以装进 reg[0]
            #pragma unroll
            for(int i = 0; i < TM; i += 4) STORE_FLOAT4(reg_a[0][i]) = LOAD_FLOAT4(a_shm_T[read_idx][0][threadIdx.y * 4 + (BM / TM) * i]);
            #pragma unroll
            for(int j = 0; j < TN; j += 4) STORE_FLOAT4(reg_b[0][j]) = LOAD_FLOAT4(b_shm[read_idx][0][threadIdx.x * 4 + (BN / TN) * j]);
        }

        // 4. 最后一个 inner 放到同步之后算: 它用的寄存器早就读好了, 正好垫在下一个 tile inner 0 的 LDS 延迟里
        #pragma unroll
        for(int i = 0; i < TM; ++i)
        {
            #pragma unroll
            for(int j = 0; j < TN; ++j)
            {
                sum[i][j] += reg_a[(BK - 1) % 2][i] * reg_b[(BK - 1) % 2][j];
            }
        }
    }

    // 5. 写回, 和 v3 / v4 相同
    float *c_start = c + BM * blockIdx.y * n + BN * blockIdx.x;
    #pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        const int cy = threadIdx.y * 4 + (BM / TM) * 4 * (i / 4) + i % 4;
        #pragma unroll
        for (int j = 0; j < TN; j += 4)
        {
            const int cx = threadIdx.x * 4 + (BN / TN) * 4 * (j / 4);
            float4 cv = LOAD_FLOAT4(c_start[cy * n + cx]);
            cv.x = alpha * sum[i][j + 0] + beta * cv.x;
            cv.y = alpha * sum[i][j + 1] + beta * cv.y;
            cv.z = alpha * sum[i][j + 2] + beta * cv.z;
            cv.w = alpha * sum[i][j + 3] + beta * cv.w;
            STORE_FLOAT4(c_start[cy * n + cx]) = cv;
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
        std::vector<float> host_v5(c_count, 0.0f);

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
        auto device_v5 = make_device_buffer<float>(c_count);

        check_cuda(
            cudaMemcpy(device_a.get(), host_a.data(), a_bytes, cudaMemcpyHostToDevice),
            "copy A to device"
        );
        check_cuda(
            cudaMemcpy(device_b.get(), host_b.data(), b_bytes, cudaMemcpyHostToDevice),
            "copy B to device"
        );
        check_cuda(cudaMemset(device_cublas.get(), 0, c_bytes), "clear cuBLAS C");
        check_cuda(cudaMemset(device_v5.get(), 0, c_bytes), "clear v5 C");

        CublasHandle cublas;
        const dim3 block(BlockX, BlockY);
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

        const float v5_ms = benchmark_gpu([&] {
            matmul_v5<BM, BN, BK, TM, TN><<<grid, block>>>(
                M,
                N,
                K,
                Alpha,
                device_a.get(),
                device_b.get(),
                Beta,
                device_v5.get()
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
                host_v5.data(),
                device_v5.get(),
                c_bytes,
                cudaMemcpyDeviceToHost
            ),
            "copy v5 result to host"
        );

        const VerificationResult verification =
            verify_result(host_cublas, host_v5, N);
        const double cublas_gflops = calculate_gflops(M, N, K, cublas_ms);
        const double v5_gflops = calculate_gflops(M, N, K, v5_ms);
        const double cublas_ratio = 100.0 * v5_gflops / cublas_gflops;

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
        std::cout << "| matmul_v5 | " << v5_ms << " | " << v5_gflops
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
