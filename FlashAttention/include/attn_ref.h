#ifndef __ATTN_REF_H__
#define __ATTN_REF_H__
// 纯 C++ 头文件, attn_cpu.cc 和各个 faN.cu 共用
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

/**
 * @brief attention 形状
 * Q, K, V, O 均为 [B, H, N, d] 行主序连续存储
 * 第 (b, h) 个 head 的起始偏移为 (b * H + h) * N * d
 */
struct AttnShape
{
    int B = 1;      // batch
    int H = 8;      // head 数
    int N = 1024;   // seqlen
    int d = 64;     // head dim
    bool causal = false;

    std::size_t heads() const { return static_cast<std::size_t>(B) * H; }
    std::size_t elements() const { return heads() * N * d; }
    float scale() const { return 1.0f / std::sqrt(static_cast<float>(d)); }
};

/**
 * @brief CPU 参考实现: O = softmax(Q K^T * scale [+ causal mask]) V
 * 用 double 累加, 作为各版本 GPU kernel 的标准答案
 */
inline void attention_cpu(
    const std::vector<float> &q,
    const std::vector<float> &k,
    const std::vector<float> &v,
    std::vector<float> &o,
    const AttnShape &s)
{
    const int N = s.N;
    const int d = s.d;
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    std::vector<double> row(N);
    std::vector<double> acc(d);

    for (std::size_t bh = 0; bh < s.heads(); ++bh) {
        const float *Q = q.data() + bh * N * d;
        const float *K = k.data() + bh * N * d;
        const float *V = v.data() + bh * N * d;
        float *O = o.data() + bh * N * d;

        for (int i = 0; i < N; ++i) {
            // causal: 第 i 行只看 [0, i]
            const int kv_len = s.causal ? i + 1 : N;

            // 1. S = Q K^T * scale, 并求行最大值
            double row_max = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < kv_len; ++j) {
                double dot = 0.0;
                for (int x = 0; x < d; ++x) {
                    dot += static_cast<double>(Q[i * d + x]) * K[j * d + x];
                }
                row[j] = dot * scale;
                row_max = std::max(row_max, row[j]);
            }

            // 2. P = exp(S - max), 求分母
            double denom = 0.0;
            for (int j = 0; j < kv_len; ++j) {
                row[j] = std::exp(row[j] - row_max);
                denom += row[j];
            }

            // 3. O = P V / denom
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < kv_len; ++j) {
                for (int x = 0; x < d; ++x) {
                    acc[x] += row[j] * V[j * d + x];
                }
            }
            for (int x = 0; x < d; ++x) {
                O[i * d + x] = static_cast<float>(acc[x] / denom);
            }
        }
    }
}

/**
 * @brief attention 的浮点运算量: 两个 GEMM 各 2*N*N*d, causal 时约减半
 * (softmax 的运算量相对 GEMM 可忽略, 与 FlashAttention 论文的统计口径一致)
 */
inline double attention_flops(const AttnShape &s)
{
    const double flops = 4.0 * static_cast<double>(s.heads()) *
                         s.N * static_cast<double>(s.N) * s.d;
    return s.causal ? flops / 2.0 : flops;
}

#endif
