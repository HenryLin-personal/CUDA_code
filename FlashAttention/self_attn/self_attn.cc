#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <vector>

namespace
{
    // 单头 self-attention, Q / K / V / O 均为 [N, d] 行主序
    // 形状必须与 self_attn.cu 保持一致
    constexpr int N = 1024;    // seqlen
    constexpr int d = 128;   // head dim
    constexpr size_t count = static_cast<size_t>(N) * d;
}

/**
 * @brief CPU 参考实现: O = softmax(Q Kᵀ / sqrt(d)) V
 * 全程用 double 计算, 作为 self_attn.cu 的标准答案
 */
void self_attention_cpu(const float *Q, const float *K, const float *V, float *O)
{
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    std::vector<double> S(static_cast<size_t>(N) * N);  // 中间矩阵 S / P: [N, N]

    // 1. S = Q Kᵀ * scale
    //    Kᵀ 不需要真的转置: S[i][j] = dot(Q 的第 i 行, K 的第 j 行), 两者都沿 d 连续
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double dot = 0.0;
            for (int x = 0; x < d; ++x) {
                dot += static_cast<double>(Q[i * d + x]) * K[j * d + x];
            }
            S[i * N + j] = dot * scale;
        }
    }

    // 2. 逐行 safe softmax: P = exp(S - max) / sum
    for (int i = 0; i < N; ++i) {
        double *row = &S[i * N];
        double row_max = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < N; ++j) {
            row_max = std::max(row_max, row[j]);
        }
        double denom = 0.0;
        for (int j = 0; j < N; ++j) {
            row[j] = std::exp(row[j] - row_max);
            denom += row[j];
        }
        for (int j = 0; j < N; ++j) {
            row[j] /= denom;
        }
    }

    // 3. O = P V, 规约维是 seqlen N
    for (int i = 0; i < N; ++i) {
        for (int x = 0; x < d; ++x) {
            double acc = 0.0;
            for (int j = 0; j < N; ++j) {
                acc += S[i * N + j] * V[j * d + x];
            }
            O[i * d + x] = static_cast<float>(acc);
        }
    }
}

int main()
{
    std::vector<float> h_q(count), h_k(count), h_v(count), h_o(count);

    // 与 comm::initialize 相同的输入分布、种子和调用顺序 (q -> k -> v)
    std::mt19937 rng(666);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto *t : {&h_q, &h_k, &h_v}) {
        for (float &value : *t) value = dist(rng);
    }

    auto start = std::chrono::high_resolution_clock::now();
    self_attention_cpu(h_q.data(), h_k.data(), h_v.data(), h_o.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto elapse = std::chrono::duration<double, std::milli>(end - start).count();

    printf("self_attn_cpu N=%d d=%d\n", N, d);
    printf("cpu time: %.4fms\n", elapse);

    std::ofstream ofs("self_attn_ref.txt");
    if (!ofs) {
        fprintf(stderr, "cannot open self_attn_ref.txt for writing\n");
        return 1;
    }
    ofs << std::setprecision(9);  // float 需要 9 位有效数字才能无损往返
    for (float val : h_o) ofs << val << '\n';
    printf("wrote %zu values to self_attn_ref.txt\n", h_o.size());

    return 0;
}
