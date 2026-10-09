#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "attn_ref.h"

int main(int argc, char **argv)
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

    const std::size_t count = s.elements();
    std::vector<float> q(count), k(count), v(count), o(count);

    // 与 comm::initialize 相同的输入分布和种子
    std::mt19937 rng(666);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto *t : {&q, &k, &v}) {
        for (float &value : *t) {
            value = dist(rng);
        }
    }

    const auto start = std::chrono::steady_clock::now();
    attention_cpu(q, k, v, o, s);
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    const double gflops = attention_flops(s) / (elapsed_ms * 1.0e6);

    // Consume the result so the computation remains observable to the compiler.
    double checksum = 0.0;
    for (float value : o) {
        checksum += static_cast<double>(value);
    }

    std::cout << "Shape: B=" << s.B << ", H=" << s.H << ", N=" << s.N
              << ", d=" << s.d << ", causal=" << (s.causal ? "true" : "false") << '\n';
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU reference time: " << elapsed_ms << " ms\n";
    std::cout << "GFLOPS: " << gflops << "\n";
    std::cout << "O checksum: " << checksum << '\n';

    return EXIT_SUCCESS;
}
