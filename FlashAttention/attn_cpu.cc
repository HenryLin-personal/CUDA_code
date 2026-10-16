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
    // Q, K, V, O 均为 [B, H, N, d] 行主序, 第 (b, h) 个 head 的偏移为 (b * H + h) * N * d
    // 形状必须与 faN.cu 保持一致
    constexpr int B = 1;      // batch
    constexpr int H = 8;      // head 数
    constexpr int N = 1024;   // seqlen
    constexpr int d = 64;     // head dim
    constexpr bool causal = false; // 因果掩码
    constexpr int BH = B * H;
    constexpr size_t count = static_cast<size_t>(BH) * N * d;
}

/**
 * @brief CPU 参考实现: O = softmax(Q K^T * scale [+ causal mask]) V
 * 用 double 累加, 作为各版本 GPU kernel 的标准答案
 */
void attention_cpu(const float *q, const float *k, const float *v, float *o)
{
    // const double scale = 1.0 / std::sqrt(static_cast<double>(d)); // 根号下d_k
    // std::vector<double> row(N);
    // std::vector<double> acc(d);

    // for (int bh = 0; bh < BH; ++bh) {
    //     const float *Q = q + static_cast<size_t>(bh) * N * d;
    //     const float *K = k + static_cast<size_t>(bh) * N * d;
    //     const float *V = v + static_cast<size_t>(bh) * N * d;
    //     float *O = o + static_cast<size_t>(bh) * N * d;

    //     for (int i = 0; i < N; ++i) {
    //         // causal: 第 i 行只看 [0, i]
    //         const int kv_len = causal ? i + 1 : N;

    //         // 1. S = Q K^T * scale, 并求行最大值
    //         double row_max = -std::numeric_limits<double>::infinity();
    //         for (int j = 0; j < kv_len; ++j) {
    //             // 用局部变量累加: row 在每行、每个 head 之间复用, 直接 row[j] += 会带上上一行的残留值
    //             double dot = 0.0;
    //             for (int x = 0; x < d; ++x) {
    //                 dot += static_cast<double>(Q[i * d + x]) * K[j * d + x]; // Q @ K.T
    //             }
    //             row[j] = dot * scale;
    //             row_max = std::max(row_max, row[j]);
    //         }

    //         // 2. P = exp(S - max), 求分母
    //         double denom = 0.0;
    //         for (int j = 0; j < kv_len; ++j) {
    //             row[j] = std::exp(row[j] - row_max);
    //             denom += row[j];
    //         }

    //         // 3. O = P V / denom
    //         std::fill(acc.begin(), acc.end(), 0.0);
    //         for (int j = 0; j < kv_len; ++j) {
    //             for (int x = 0; x < d; ++x) {
    //                 acc[x] += row[j] * V[j * d + x];
    //             }
    //         }
    //         for (int x = 0; x < d; ++x) {
    //             O[i * d + x] = static_cast<float>(acc[x] / denom);
    //         }
    //     }
    // }

    // [B, H, N, d]
    const double scale = 1.0 / std::sqrt(static_cast<double>(d)); // 根号下d_k
    // [B, H, :]
    for(int bh = 0; bh < BH; ++bh)
    {
        const float *Q = q + bh * N * d;
        const float *K = k + bh * N * d;
        const float *V = v + bh * N * d;
        float *O = o + bh * N * d;

        std::vector<double> row(N, 0.f);
        std::vector<double> acc(d, 0.f);
        // 1. Q @ K.T
        for(int i = 0; i < N; ++i)
        {
            int kv_len = causal ? i + 1 : N;
            double row_max = -std::numeric_limits<double>::infinity();
            for(int j = 0; j < kv_len; ++j)
            {
                float dot = 0.f;
                for(int z = 0; z < d; ++z)
                {
                    dot += Q[i * d + z] * K[j * d + z];
                }
                row[j] = dot * scale;
                row_max = std::max(row_max, row[j]);
            }
            
            // 2. softmax求分母
            float denom = 0.f;
            for(int j = 0; j < kv_len; ++j)
            {
                row[j] = std::exp(row[j] - row_max);
                denom += row[j];
            }

            // 3. A @ V / denom
            std::fill(acc.begin(), acc.end(), 0.f);
            for(int j = 0; j < kv_len; ++j)
            {
                for(int z = 0; z < d; ++z)
                {
                    acc[z] += row[j] * V[j * d + z];
                }
            }

            // 4. 写回O
            for(int z = 0; z < d; ++z)
            {
                O[i * d + z] = acc[z] / denom;
            }
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
    attention_cpu(h_q.data(), h_k.data(), h_v.data(), h_o.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto elapse = std::chrono::duration<double, std::milli>(end - start).count();

    // 两个 GEMM 各 2*N*N*d, causal 时约减半
    const double flops = 4.0 * BH * N * static_cast<double>(N) * d / (causal ? 2.0 : 1.0);
    printf("attn_cpu B=%d H=%d N=%d d=%d causal=%d\n", B, H, N, d, causal);
    printf("avg_ms :%.2fms, GFLOPS: %.4f\n", elapse, flops / (elapse * 1.0e6));

    std::ofstream ofs("attn_ref.txt");
    ofs << std::setprecision(9);
    for (float val : h_o) ofs << val << '\n';

    return 0;
}
