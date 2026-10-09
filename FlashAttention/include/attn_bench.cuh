#ifndef __ATTN_BENCH_CUH__
#define __ATTN_BENCH_CUH__
/**
 * 各版本 faN.cu 共用的测试框架: 生成输入 -> 跑 kernel -> 与 CPU 参考对比 -> 计时
 *
 * 用法 (faN.cu 里只需写 kernel、launcher 和一个很短的 main):
 *
 *   int main(int argc, char **argv)
 *   {
 *       const AttnShape s = parse_shape(argc, argv);
 *       // 需要的 workspace 在这里提前分配, 不要放进 launcher (会被计时)
 *       return run_attention_test("fa0", s,
 *           [&](const float *q, const float *k, const float *v, float *o) {
 *               my_kernel<<<grid, block>>>(q, k, v, o, ...);
 *           });
 *   }
 *
 * 命令行: ./faN [B H N d [causal]], 例如调试时用 ./fa0 1 1 8 4
 */
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "attn_ref.h"
#include "comm.cuh"

constexpr float AbsoluteTolerance = 1.0e-4f;
constexpr float RelativeTolerance = 1.0e-4f;

/**
 * @brief 从命令行解析形状, 未给出时使用 AttnShape 的默认值
 */
inline AttnShape parse_shape(int argc, char **argv)
{
    AttnShape s;
    if (argc >= 5) {
        s.B = std::atoi(argv[1]);
        s.H = std::atoi(argv[2]);
        s.N = std::atoi(argv[3]);
        s.d = std::atoi(argv[4]);
    }
    if (argc >= 6) {
        s.causal = std::atoi(argv[5]) != 0;
    }
    if (s.B <= 0 || s.H <= 0 || s.N <= 0 || s.d <= 0) {
        throw std::invalid_argument("usage: ./faN [B H N d [causal]], all positive");
    }
    return s;
}

struct VerificationResult {
    bool matched{true};
    std::size_t mismatch_count{0};
    float max_absolute_error{0.0f};
};

/**
 * @brief 逐元素比较, 出错时打印 (b, h, row, col) 方便定位
 */
inline VerificationResult verify_attention(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const AttnShape &s)
{
    VerificationResult result;
    constexpr std::size_t MaxReportedMismatches = 5;

    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float reference = expected[index];
        const float value = actual[index];
        const bool finite = std::isfinite(reference) && std::isfinite(value);
        const float absolute_error = finite
            ? std::fabs(reference - value)
            : std::numeric_limits<float>::infinity();
        result.max_absolute_error =
            std::max(result.max_absolute_error, absolute_error);

        const float allowed_error =
            AbsoluteTolerance + RelativeTolerance * std::fabs(reference);
        if (!(finite && absolute_error <= allowed_error)) {
            if (result.mismatch_count < MaxReportedMismatches) {
                const std::size_t col = index % s.d;
                const std::size_t row = index / s.d % s.N;
                const std::size_t h = index / s.d / s.N % s.H;
                const std::size_t b = index / s.d / s.N / s.H;
                std::cerr << "Mismatch at (b=" << b << ", h=" << h
                          << ", row=" << row << ", col=" << col << ")"
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

/**
 * @brief launch(q, k, v, o): 只负责发射 kernel, 不要在里面 cudaMalloc / 同步
 * @return EXIT_SUCCESS 表示结果正确
 */
template <typename Launcher>
int run_attention_test(const char *name, const AttnShape &s, Launcher &&launch)
{
    try {
        comm::setDevice();

        const std::size_t count = s.elements();
        const std::size_t bytes = count * sizeof(float);

        std::vector<float> host_q(count), host_k(count), host_v(count);
        std::vector<float> host_ref(count), host_o(count);

        std::mt19937 rng(666);
        comm::initialize(host_q, rng);
        comm::initialize(host_k, rng);
        comm::initialize(host_v, rng);

        auto device_q = comm::make_device_buffer<float>(count);
        auto device_k = comm::make_device_buffer<float>(count);
        auto device_v = comm::make_device_buffer<float>(count);
        auto device_o = comm::make_device_buffer<float>(count);

        comm::check_cuda(cudaMemcpy(device_q.get(), host_q.data(), bytes, cudaMemcpyHostToDevice), "copy Q to device");
        comm::check_cuda(cudaMemcpy(device_k.get(), host_k.data(), bytes, cudaMemcpyHostToDevice), "copy K to device");
        comm::check_cuda(cudaMemcpy(device_v.get(), host_v.data(), bytes, cudaMemcpyHostToDevice), "copy V to device");
        // 填成 NaN: kernel 漏写的位置会直接被检查出来
        comm::check_cuda(cudaMemset(device_o.get(), 0xFF, bytes), "fill O with NaN");

        auto run = [&] {
            launch(device_q.get(), device_k.get(), device_v.get(), device_o.get());
        };

        // 1. 正确性: 先单独跑一次
        run();
        comm::check_cuda(cudaGetLastError(), "kernel launch");
        comm::check_cuda(cudaDeviceSynchronize(), "kernel execute");
        comm::check_cuda(cudaMemcpy(host_o.data(), device_o.get(), bytes, cudaMemcpyDeviceToHost), "copy O to host");

        std::cout << "Computing CPU reference..." << std::endl;
        attention_cpu(host_q, host_k, host_v, host_ref, s);
        const VerificationResult verification = verify_attention(host_ref, host_o, s);

        // 2. 性能
        const float ms = comm::benchmark_gpu(run);
        const double tflops = attention_flops(s) / (static_cast<double>(ms) * 1.0e9);

        std::cout << "Shape: B=" << s.B << ", H=" << s.H << ", N=" << s.N
                  << ", d=" << s.d << ", causal=" << (s.causal ? "true" : "false") << '\n';
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "| Implementation | Average time (ms) | TFLOPS | Correct |\n";
        std::cout << "|---|---:|---:|---|\n";
        std::cout << "| " << name << " | " << ms << " | " << tflops << " | "
                  << (verification.matched ? "YES" : "NO") << " |\n";
        std::cout << std::scientific << std::setprecision(3)
                  << "max absolute error: " << verification.max_absolute_error
                  << ", mismatches: " << verification.mismatch_count << '\n';

        return verification.matched ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

#endif
