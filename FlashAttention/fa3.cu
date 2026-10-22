#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "comm.cuh"

namespace
{
    constexpr int B = 1;      // batch
    constexpr int H = 8;      // head 数
    constexpr int N = 1024;   // seqlen
    constexpr int d = 64;     // head dim
    constexpr bool causal = false; // 因果掩码
    constexpr int BH = B * H;
    constexpr size_t count = BH * N * d;       // Q/K/V/O 元素数, 不再需要 N x N 的 S
    constexpr size_t bytes = count * sizeof(float);
    constexpr int loops = 10;

    constexpr int Br = 64;    // 一维线程模型, 每个 block 负责的 Q 行数 = block 内线程数
    constexpr int Bc = 32;    // 每次从 K/V 搬进 smem 的行数
}

// 融合版: S = Q·Kᵀ·scale, online softmax, O = P·V 全部在一个 kernel 里完成, S/P 只存在于寄存器
// grid(N/Br, BH) = (16, 8) = 128个线程块, 每个线程块 Br=64 个线程 = 2个warp
// 一个线程负责 Q 的一整行: 线程内部执行的就是 CPU 上推导的单遍 online softmax, 不需要跨线程规约
// Br, Bc, D 必须是编译期常量: q_reg[D], acc[D], s[Bc] 的下标在 #pragma unroll 展开后全是常量, 才能放进寄存器,
// 否则会被放到 local memory (显存). 可以用 nvcc -Xptxas -v 检查 "0 bytes stack frame" 
template <int Br, int Bc, int D, int N>
__global__ void flash_attn_v3(const float *q, const float *k, const float *v, float *o,
                              float scale, bool causal)
{
//     __shared__ float sK[Bc][D];
//     __shared__ float sV[Bc][D];

//     const int tid = threadIdx.x;
//     const int bh = blockIdx.y;
//     const int row = blockIdx.x * Br + tid;  // 本线程负责的 Q 行
//     // 越界的行不能提前 return: 它们仍要参与搬运 K/V 和 __syncthreads, 只是最后不写 O
//     const bool valid = row < N;

//     const float *Q = q + static_cast<size_t>(bh) * N * D;
//     const float *K = k + static_cast<size_t>(bh) * N * D;
//     const float *V = v + static_cast<size_t>(bh) * N * D;
//     float *O = o + static_cast<size_t>(bh) * N * D;

//     // Q 的一行放寄存器 (加载时直接乘上 scale), 整个 K/V 循环中复用
//     // 不放 smem 的原因: sQ[tid][x] 相邻线程相隔 D=64 个 float, 正好落在同一个 bank, 32 路 bank conflict
//     float q_reg[D];
//     float acc[D];             // 未归一化的 O 行
//     float m = -INFINITY;      // 当前行最大值
//     float l = 0.f;            // 以 m 为基准的指数和
// #pragma unroll
//     for (int x = 0; x < D; ++x) {
//         q_reg[x] = valid ? Q[row * D + x] * scale : 0.f;
//         acc[x] = 0.f;
//     }

//     // causal: 本 block 最后一行是 (blockIdx.x + 1) * Br - 1, 之后的 K/V tile 对本 block 所有行都被 mask, 直接跳过
//     const int kv_end = causal ? min(N, (blockIdx.x + 1) * Br) : N;

//     for (int kv_start = 0; kv_start < kv_end; kv_start += Bc) {
//         // 1. 搬运 K/V tile 到 smem
//         __syncthreads(); // 等所有线程用完上一个 tile, 才能覆盖 sK/sV
//         // Br=64 个线程合作搬 Bc*D = 32*64 = 2048 个 float, 每个线程搬 32 个
//         // 例: 第一次循环 idx = tid, tid 0~63 搬 K[kv_start+0][0~63]; warp 0 读 K 第 kv_start 行的 0~31 列, 128B 连续 = 4个sector
//         for (int idx = tid; idx < Bc * D; idx += Br) {
//             const int r = idx / D;
//             const int c = idx % D;
//             const int kv_row = kv_start + r;
//             sK[r][c] = kv_row < N ? K[kv_row * D + c] : 0.f;
//             sV[r][c] = kv_row < N ? V[kv_row * D + c] : 0.f;
//         }
//         __syncthreads(); // 等搬运完成

//         // 2. S 的一行 tile: s[c] = q·K[kv_start+c], 并求 tile 内最大值
//         // 读 sK[c][x]: 同一时刻 warp 内所有线程读同一地址 -> 广播, 没有 bank conflict
//         float s[Bc];
//         float tile_max = -INFINITY;
// #pragma unroll
//         for (int c = 0; c < Bc; ++c) {
//             float dot = 0.f;
// #pragma unroll
//             for (int x = 0; x < D; ++x) {
//                 dot += q_reg[x] * sK[c][x];
//             }
//             const int j = kv_start + c;
//             // 越界列 (N 不是 Bc 的倍数) 和 causal 的上三角都设为 -inf, exp 后为 0
//             s[c] = (j >= N || (causal && j > row)) ? -INFINITY : dot;
//             tile_max = fmaxf(tile_max, s[c]);
//         }

//         // 3. online softmax: 更新 m, l, 并求修正系数 alpha
//         // 第一个 tile 时 m = -inf, alpha = exp(-inf) = 0, 旧值被自然清掉
//         // 第一个 tile 一定包含 j = 0, 之后 m 都是有限值, 即使整个 tile 被 mask 也不会出现 -inf - (-inf) = NaN
//         const float m_new = fmaxf(m, tile_max);
//         const float alpha = expf(m - m_new);
//         float tile_sum = 0.f;
// #pragma unroll
//         for (int c = 0; c < Bc; ++c) {
//             s[c] = expf(s[c] - m_new); // 未归一化的 P
//             tile_sum += s[c];
//         }
//         l = alpha * l + tile_sum;

//         // 4. O = alpha * O + P·V_tile
//         // 读 sV[c][x]: 同样是 warp 内所有线程读同一地址 -> 广播
// #pragma unroll
//         for (int x = 0; x < D; ++x) {
//             float t = acc[x] * alpha;
// #pragma unroll
//             for (int c = 0; c < Bc; ++c) {
//                 t += s[c] * sV[c][x];
//             }
//             acc[x] = t;
//         }
//         m = m_new;
//     }

//     // 5. 最后统一归一化一次
//     if (valid) {
//         const float inv = 1.f / l;
// #pragma unroll
//         for (int x = 0; x < D; ++x) {
//             O[row * D + x] = acc[x] * inv;
//         }
//     }

    const int bh = blockIdx.y;
    const float *Q = q + static_cast<size_t>(bh) * N * D;
    const float *K = k + static_cast<size_t>(bh) * N * D;
    const float *V = v + static_cast<size_t>(bh) * N * D;
    float *O = o + static_cast<size_t>(bh) * N * D;

    float q_reg[D];
    float acc[D];
    const int row = blockIdx.x * Br + threadIdx.x;
    const bool valid = row < N;
    for (int col = 0; col < D; ++col) {
        q_reg[col] = valid ? Q[row * D + col] * scale : 0.f;
        acc[col] = 0.f;
    }

    float m = -INFINITY;
    float l = 0.f;
    __shared__ float sK[Bc][D];   // [32][64]
    __shared__ float sV[Bc][D];   // [32][64]

    for(int k_row = 0; k_row < N; k_row += Bc)
    {
        __syncthreads();
        for(int i = threadIdx.x; i < Bc * D; i += Br)
        {
            const int r = i / D;
            const int c = i % D;
            const int g_row = k_row + r;
            sK[r][c] = g_row < N ? K[g_row * D + c] : 0.f;
            sV[r][c] = g_row < N ? V[g_row * D + c] : 0.f;
        }
        __syncthreads();

        float s[Bc];
        float tile_max = -INFINITY;
        #pragma unroll
        for(int k = 0; k < Bc; ++k) 
        {        
            float dot = 0.f;
            for(int col = 0; col < D; ++col)
            {
                dot += q_reg[col] * sK[k][col];
            }
            s[k] = (k_row + k < N) ? dot : -INFINITY;
            tile_max = fmaxf(s[k], tile_max);
        }
        float m_new = fmaxf(m, tile_max);
        float alpha = expf(m - m_new);
        float tile_sum = 0.f;
        #pragma unroll
        for (int k = 0; k < Bc; ++k) 
        {
            s[k] = expf(s[k] - m_new);  // s 原地变成 p
            tile_sum += s[k];
        }
        l = l * alpha + tile_sum;
        #pragma unroll
        for (int col = 0; col < D; ++col)
        {
            float t = acc[col] * alpha;
            for (int k = 0; k < Bc; ++k)
                t += s[k] * sV[k][col];
            acc[col] = t;
        }
        m = m_new;
    }
    if(valid)
    {
        #pragma unroll
        for(int col = 0; col < D; ++col)
        {
            O[row * D + col] = acc[col] / l;
        }
    }
}

void call_attention(const float *q, const float *k, const float *v, float *o)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    dim3 block(Br);
    dim3 grid(comm::ceil_div(N, Br), BH);
    flash_attn_v3<Br, Bc, d, N><<<grid, block>>>(q, k, v, o, scale, causal);
}

int main()
{
    std::vector<float> h_q(count), h_k(count), h_v(count), h_o(count);

    // 与 attn_cpu.cc 相同的输入分布、种子和调用顺序 (q -> k -> v)
    std::mt19937 rng(666);
    comm::initialize(h_q, rng);
    comm::initialize(h_k, rng);
    comm::initialize(h_v, rng);

    auto d_q = comm::make_device_buffer<float>(count);
    auto d_k = comm::make_device_buffer<float>(count);
    auto d_v = comm::make_device_buffer<float>(count);
    auto d_o = comm::make_device_buffer<float>(count);

    comm::check_cuda(cudaMemcpy(d_q.get(), h_q.data(), bytes, cudaMemcpyHostToDevice), "h_q to d_q fail!");
    comm::check_cuda(cudaMemcpy(d_k.get(), h_k.data(), bytes, cudaMemcpyHostToDevice), "h_k to d_k fail!");
    comm::check_cuda(cudaMemcpy(d_v.get(), h_v.data(), bytes, cudaMemcpyHostToDevice), "h_v to d_v fail!");
    // 填成 NaN: kernel 漏写的位置会直接被检查出来
    comm::check_cuda(cudaMemset(d_o.get(), 0xFF, bytes), "fill d_o with NaN fail!");

    // warmup
    call_attention(d_q.get(), d_k.get(), d_v.get(), d_o.get());
    comm::check_cuda(cudaGetLastError(), "kernel launch fail!");
    comm::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize() fail!");

    comm::CudaEvent start, end;
    comm::check_cuda(cudaEventRecord(start.get()), "cudaEventRecord(start) fail!");
    for(int i = 0; i < loops; ++i)
    {
        call_attention(d_q.get(), d_k.get(), d_v.get(), d_o.get());
    }
    comm::check_cuda(cudaEventRecord(end.get()), "cudaEventRecord(end) fail!");
    comm::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize() fail!");

    float total_ms = 0.f;
    comm::check_cuda(cudaEventElapsedTime(&total_ms, start.get(), end.get()), "cudaEventElapsedTime");
    const float avg_ms = total_ms / loops;
    // 两个 GEMM 各 2*N*N*d, causal 时约减半
    const double flops = 4.0 * BH * N * static_cast<double>(N) * d / (causal ? 2.0 : 1.0);
    printf("fa1 B=%d H=%d N=%d d=%d causal=%d Br=%d Bc=%d\n", B, H, N, d, causal, Br, Bc);
    printf("fa1 avg_ms :%.4fms, TFLOPS: %.4f\n", avg_ms, flops / (avg_ms * 1.0e9));

    comm::check_cuda(cudaMemcpy(h_o.data(), d_o.get(), bytes, cudaMemcpyDeviceToHost), "d_o to h_o fail!");
    std::ifstream ifs("./attn_ref.txt");
    if (!ifs)
    {
        std::cerr << "attn_ref.txt not found, run ./build/attn_cpu first\n";
        return 1;
    }
    std::vector<float> ref(count);
    for(size_t i = 0; i < count; ++i) ifs >> ref[i];

    // 出错时打印的 (row, col): row = bh * N + i, col = d 维下标
    comm::verify_results(ref, h_o, d);
    return 0;
}
