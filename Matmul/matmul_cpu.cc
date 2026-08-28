#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

// Keep the matrix dimensions and row-major layout consistent with matmul2.cu.
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

void matmul_cpu(const std::vector<float> &a,
                const std::vector<float> &b,
                std::vector<float> &c)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[static_cast<std::size_t>(i) * K + k] *
                       b[static_cast<std::size_t>(k) * N + j];
            }
            c[static_cast<std::size_t>(i) * N + j] = sum;
        }
    }
}

} // namespace

int main()
{
    const std::size_t a_count =
        static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t b_count =
        static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
    const std::size_t c_count =
        static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> c(c_count, 0.0f);

    // Use the same deterministic input distribution as matmul2.cu.
    std::mt19937 rng(666);
    std::uniform_int_distribution<int> distribution(-2, 2);
    for (float &value : a) {
        value = static_cast<float>(distribution(rng));
    }
    for (float &value : b) {
        value = static_cast<float>(distribution(rng));
    }

    const auto start = std::chrono::steady_clock::now();
    matmul_cpu(a, b, c);
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    const double gflops =
        2.0 * static_cast<double>(M) * static_cast<double>(N) *
        static_cast<double>(K) / (elapsed_ms * 1.0e6);

    // Consume the result so the computation remains observable to the compiler.
    double checksum = 0.0;
    for (float value : c) {
        checksum += static_cast<double>(value);
    }

    std::cout << "Matrix: A[" << M << " x " << K << "] * B["
              << K << " x " << N << "] -> C[" << M << " x " << N
              << "]\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU triple-for time: " << elapsed_ms << " ms\n";
    std::cout << "GFLOPS: " << gflops << "\n";
    std::cout << "C checksum: " << checksum << '\n';

    return EXIT_SUCCESS;
}
