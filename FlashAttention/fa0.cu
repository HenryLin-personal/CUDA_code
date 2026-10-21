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
    constexpr int BlockX = 16;
    constexpr int BlockY = 16;

    constexpr bool causal = false; // 因果掩码
    constexpr int BH = B * H;
    constexpr size_t count = BH * N * d;       // Q/K/V/O 元素数
    constexpr size_t s_count = BH * N * N;     // S/P 元素数
    constexpr size_t bytes = count * sizeof(float);
    constexpr int loops = 10;
}

__global__ void qk_kernel(const float *q, const float *k, float *S,
                          int N, int d, float scale, bool causal)
{
    // grid(64, 64, 8) = 64*64*8 = 32768个线程块, 每个线程块16*16个线程, 每个线程块共16*16/32=8个warp
    // 一个warp = 2行(threadIdx.y) × 16列(threadIdx.x), 例: warp 0 中 lane 0~15 为 threadIdx=(0~15, 0), lane 16~31 为 threadIdx=(0~15, 1)
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh = blockIdx.z;
    if (i >= N || j >= N) return;

    const float *Q = q + bh * N * d;
    const float *K = k + bh * N * d;
    float *S_bh = S + bh * N * N;

    if (causal && j > i) {
        S_bh[i * N + j] = -INFINITY;
        return;
    }

    float acc = 0.f;
    for (int x = 0; x < d; ++x) {
        acc += 
        Q[i * d + x] *      
        // 读Q时, 第i个y方向上的线程读取Q的第i行, 每个y方向上的线程物理上间隔16个线程/半个warp. 因此一个warp读取Q的两行, 对应2个sector
        // 例: block(0,0,0)的warp 0, x=5时: lane 0~15 (i=0) 都读Q[0*64+5]=Q[5] (字节20, sector 0), lane 16~31 (i=1) 都读Q[1*64+5]=Q[69] (字节276, sector 8) → 2个sector
        K[j * d + x];       
        // 读K时, 第j个x方向上的线程读取K的第j行, 每个x方向上的线程物理上连续排布, x方向上只有16个线程/半个warp, 这半个warp的线程读取K的16行, 对应16个sector, 另半个warp的threadIdx.x同样是0~15, 读取同样的16行, 因此一个warp只读取16个sector
        // 例: x=5时: lane 0~15 (j=0~15) 读K[0*64+5], K[1*64+5], ..., K[15*64+5] = K[5], K[69], ..., K[965], 相邻线程间隔64个float=256B (sector 0, 8, ..., 120); lane 16~31 (j=0~15) 读同样的16个地址 → 16个sector
    }

    S_bh[i * N + j] = acc * scale; 
}


__global__ void softmax_kernel(float *S, int rows, int N)
{
    // safe softmax
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float *p = S + row * N;

    float row_max = -INFINITY;
    for (int j = 0; j < N; ++j) {
        row_max = fmaxf(row_max, p[j]);
    }

    float denom = 0.f;
    for (int j = 0; j < N; ++j) {
        const float e = expf(p[j] - row_max);
        p[j] = e;
        denom += e;
    }

    const float inv = 1.f / denom;
    for (int j = 0; j < N; ++j) {
        p[j] *= inv;
    }
}

__global__ void pv_kernel(const float *P, const float *v, float *o,
                          int N, int d)
{
    // grid(4, 64, 8) = 4*64*8 = 2048个线程块, 每个线程块16*16个线程, 每个线程块共16*16/32=8个warp
    // 一个warp = 2行(threadIdx.y) × 16列(threadIdx.x), 例: warp 0 中 lane 0~15 为 threadIdx=(0~15, 0), lane 16~31 为 threadIdx=(0~15, 1)
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh = blockIdx.z;
    if (y >= N || x >= d) return;

    const float *P_bh = P + bh * N * N;
    const float *V = v + bh * N * d;
    float *O = o + bh * N * d;

    float acc = 0.f;
    for (int inner = 0; inner < N; ++inner) {
        acc += 
        P_bh[y * N + inner] *             
        // 读P时, 第i个y方向上的线程读取P的第i行, 每个y方向上的线程间隔16个线程/半个warp. 因此一个warp读取P的两行, 对应2个sector
        // 例: block(0,0,0)的warp 0, j=5时: lane 0~15 (i=0) 都读P[0*1024+5]=P[5] (字节20, sector 0), lane 16~31 (i=1) 都读P[1*1024+5]=P[1029] (字节4116, sector 128) → 2个sector

        V[inner * d + x];          
        // 读V时, 单次循环中j对整个warp相同, 第x个x方向上的线程读取V的第j行的第x个元素, 每个x方向上的线程连续排布, 单次循环一个线程读取4B, 半个warp读取16*4B=64B=2个sector, 另半个warp的threadIdx.x同样是0~15, 读取同样的64B, 因此一个warp读取2个sector
        // 例: j=5时: lane 0~15 (x=0~15) 读V[5*64+0~15]=V[320~335] (字节1280~1343, sector 40、41), lane 16~31 (x=0~15) 读同样的V[320~335] → 2个sector
    }
    O[y * d + x] = acc;
}

void call_attention(const float *q, const float *k, const float *v, float *o, float *S)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    dim3 qk_block(BlockX, BlockY);
    dim3 qk_grid(comm::ceil_div(N, BlockX), comm::ceil_div(N, BlockY), BH);
    qk_kernel<<<qk_grid, qk_block>>>(q, k, S, N, d, scale, causal);

    const int rows = BH * N;
    dim3 sm_block(128);
    dim3 sm_grid(comm::ceil_div(rows, sm_block.x));
    softmax_kernel<<<sm_grid, sm_block>>>(S, rows, N);

    dim3 pv_block(BlockX, BlockY);
    dim3 pv_grid(comm::ceil_div(d, BlockX), comm::ceil_div(N, BlockY), BH);
    pv_kernel<<<pv_grid, pv_block>>>(S, v, o, N, d);
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
    auto d_s = comm::make_device_buffer<float>(s_count); // N x N 中间矩阵, fa0 的显存瓶颈所在

    comm::check_cuda(cudaMemcpy(d_q.get(), h_q.data(), bytes, cudaMemcpyHostToDevice), "h_q to d_q fail!");
    comm::check_cuda(cudaMemcpy(d_k.get(), h_k.data(), bytes, cudaMemcpyHostToDevice), "h_k to d_k fail!");
    comm::check_cuda(cudaMemcpy(d_v.get(), h_v.data(), bytes, cudaMemcpyHostToDevice), "h_v to d_v fail!");
    // 填成 NaN: kernel 漏写的位置会直接被检查出来
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
    // 两个 GEMM 各 2*N*N*d, causal 时约减半
    const double flops = 4.0 * BH * N * static_cast<double>(N) * d / (causal ? 2.0 : 1.0);
    printf("fa0 B=%d H=%d N=%d d=%d causal=%d\n", B, H, N, d, causal);
    printf("fa0 avg_ms :%.4fms, TFLOPS: %.4f\n", avg_ms, flops / (avg_ms * 1.0e9));

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
