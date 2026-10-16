#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "comm.cuh"

namespace
{
    constexpr int N = 1024;                  // seqlen
    constexpr int d = 128;                 // head dim
    constexpr size_t count =  N * d;       // Q/K/V/O 元素数
    constexpr size_t s_count = N * N;      // S/P 元素数
    constexpr size_t bytes = count * sizeof(float);
    constexpr int loops = 10;
}

// S = Q·Kᵀ·scale        Q, K: [N, d]  S: [N, N]
__global__ void qk_kernel(const float *Q, const float *K, float *S,
                          int N, 
                          int d, 
                          float scale
                        )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // 步长 = 网格内总线程数 (grid-stride loop), 不要写死成 N / 2:
    // 写死后只要 launch 的线程数 != N / 2, 就会有多个线程处理同一行
    int stride = blockDim.x * gridDim.x;

    // 1个线程负责2行
    // tid = 0:
    //  Q[0][0~127] Q[32][0~127]
    // tid = 1:
    //  Q[1][0~127] Q[33][0~127]
    // ...
    // tid = 31:
    //  Q[31][0~127] Q[63][0~127]
    for(int i = tid; i < N; i += stride)
    {
        for(int j = 0; j < N; ++j)
        {
            float sum = 0.f;
            for(int k = 0; k < d; ++k)
            {
                // Kᵀ 不需要真的转置: S[i][j] = dot(Q 的第 i 行, K 的第 j 行)
                // K 是 [N, d] 行主序, 第 j 行第 k 列是 K[j * d + k]
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }
}

__global__ void softmax_kernel(float *S, int rows, int N)
{
    // 0. 1个线程负责2行
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tid; i < rows; i += stride)
    {
        // 1. find max
        float row_max = -INFINITY;
        float *row = &S[i * N];
        for(int j = 0; j < N; ++j)
        {
            row_max = fmaxf(row_max, row[j]);  // std::max 是 host constexpr 函数, device 里不能直接调用
        }

        // 2. exp(s - max), 同时累加分母
        float denom = 0.f;
        for(int j = 0; j < N; ++j)
        {
            row[j] = expf(row[j] - row_max);
            denom += row[j];
        }

        // 3. 归一化
        for(int j = 0; j < N; ++j)
        {
            row[j] /= denom;
        }
    }
}

// O = P·V               P: [N, N]  V: [N, d]  O: [N, d]
// O[i][x] = sum_j P[i][j] * V[j][x], 规约维是 seqlen N
__global__ void pv_kernel(const float *P, const float *V, float *O,
                          int N, int d)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = tid; i < N; i += stride)
    {
        for(int x = 0; x < d; ++x)          // 输出列: 共 d 列
        {
            float acc = 0.f;
            for(int j = 0; j < N; ++j)      // 规约: 共 N 项
            {
                acc += P[i * N + j] * V[j * d + x];  // 必须累加, 不能用 =
            }
            O[i * d + x] = acc;
        }
    }
}

void call_attention(const float *q, const float *k, const float *v, float *o, float *S)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const int thread_num = N / 2; // 32个线程, 每个线程负责2行数据, 2*128=256个数据

    // kernel 只用 x 维, 而且只要 32 个线程: grid 就是 1 个 block
    // (多开 block 会让多个线程处理同一行: qk/pv 重复计算, softmax 原地写会产生数据竞争)
    dim3 block(thread_num);
    dim3 grid(1);
    qk_kernel<<<grid, block>>>(q, k, S, N, d, scale);
    softmax_kernel<<<grid, block>>>(S, N, N);
    pv_kernel<<<grid, block>>>(S, v, o, N, d);
}

int main()
{
    std::vector<float> h_q(count), h_k(count), h_v(count), h_o(count);

    std::mt19937 rng(666);
    comm::initialize(h_q, rng);
    comm::initialize(h_k, rng);
    comm::initialize(h_v, rng);

    auto d_q = comm::make_device_buffer<float>(count);
    auto d_k = comm::make_device_buffer<float>(count);
    auto d_v = comm::make_device_buffer<float>(count);
    auto d_o = comm::make_device_buffer<float>(count);
    auto d_s = comm::make_device_buffer<float>(s_count); // N x N 中间矩阵, fa0 的显存瓶颈所在

    comm::check_cuda(cudaMemcpy(d_q.get(), h_q.data(), bytes, cudaMemcpyHostToDevice), "h_q to d_q fail!");
    comm::check_cuda(cudaMemcpy(d_k.get(), h_k.data(), bytes, cudaMemcpyHostToDevice), "h_k to d_k fail!");
    comm::check_cuda(cudaMemcpy(d_v.get(), h_v.data(), bytes, cudaMemcpyHostToDevice), "h_v to d_v fail!");
    comm::check_cuda(cudaMemset(d_o.get(), 0xFF, bytes), "fill d_o with NaN fail!");

    // warmup
    call_attention(d_q.get(), d_k.get(), d_v.get(), d_o.get(), d_s.get());
    comm::check_cuda(cudaGetLastError(), "kernel launch fail!");
    comm::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize() fail!");

    comm::CudaEvent start, end;
    comm::check_cuda(cudaEventRecord(start.get()), "cudaEventRecord(start) fail!");
    for(int i = 0; i < loops; ++i)
    {
        call_attention(d_q.get(), d_k.get(), d_v.get(), d_o.get(), d_s.get());
    }
    comm::check_cuda(cudaEventRecord(end.get()), "cudaEventRecord(end) fail!");
    comm::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize() fail!");

    float total_ms = 0.f;
    comm::check_cuda(cudaEventElapsedTime(&total_ms, start.get(), end.get()), "cudaEventElapsedTime");
    const float avg_ms = total_ms / loops;

    printf("self_attn N=%d d=%d\n", N, d);
    printf("self_attn avg_ms :%.4fms\n", avg_ms);

    comm::check_cuda(cudaMemcpy(h_o.data(), d_o.get(), bytes, cudaMemcpyDeviceToHost), "d_o to h_o fail!");
    std::ifstream ifs("./self_attn_ref.txt");
    if (!ifs)
    {
        std::cerr << "self_attn_ref.txt not found, run ./build/self_attn_cpu first\n";
        return 1;
    }
    std::vector<float> ref(count);
    for(size_t i = 0; i < count; ++i) ifs >> ref[i];
    if (!ifs)
    {
        // 参考文件比 count 短 (例如形状改了但没重新生成) 时, 不能拿残缺数据去比
        std::cerr << "self_attn_ref.txt has fewer than " << count << " values, regenerate it\n";
        return 1;
    }

    // 出错时打印的 (row, col): row = seqlen 下标 i, col = d 维下标
    comm::verify_results(ref, h_o, d);
    return 0;
}
