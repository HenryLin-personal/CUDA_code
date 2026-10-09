#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include "comm.cuh"

namespace
{
    constexpr int batch = 1024;
    constexpr int size = 4096;
    constexpr int count = batch * size;
    constexpr float eps = 1.0f;
    constexpr int loops = 10;

    constexpr int block_size = 256;
    constexpr int grid_size = batch;

    using comm::operator*;
}

#define FULL_MASK 0xffffffff
__device__ float warpReduce(float v) {
  v += __shfl_down_sync(FULL_MASK, v, 16);
  v += __shfl_down_sync(FULL_MASK, v, 8);
  v += __shfl_down_sync(FULL_MASK, v, 4);
  v += __shfl_down_sync(FULL_MASK, v, 2);
  v += __shfl_down_sync(FULL_MASK, v, 1);
  return v;
}

__device__ float reductionv4(float acc, const int tid)
{
    const int warp_size = 32;
    const int lane_id = tid % warp_size;
    const int warp_id = tid / warp_size;
    const int warp_count = (block_size + warp_size - 1) / warp_size;

    // 1. 每个warp内执行一次warpReduce, 保存在共享内存中
    __shared__ float warpSum[warp_count];
    acc = warpReduce(acc);
    if(lane_id == 0) warpSum[warp_id] = acc; // 让每个线程束的第0个线程去写当前线程束内的规约结果.
    __syncthreads();

    // 2. warp0对共享内存进行规约
    if(warp_id == 0) {
        acc = lane_id < warp_count ? warpSum[lane_id] : 0.f; // lane_id∈[0,31], warp_count的最大值可能小于31, 比如block_size<1024时, warp_count<31. 但这里block_size=1024
        acc = warpReduce(acc);
    }
    return acc;
}

static_assert(size % 4 == 0, "float4向量化要求size是4的倍数");
__global__ void rmsnormv4(float* in, float* weight, float* out, int batch, int size, float eps)
{
    const float4 *in_ptr = reinterpret_cast<const float4*>(in + blockIdx.x * size);
    const float4 *w_ptr = reinterpret_cast<const float4*>(weight);
    float4 *out_ptr = reinterpret_cast<float4*>(out + blockIdx.x * size);
    const int tid = threadIdx.x;

    // 1. 
    // 每个block负责一个batch, block内有256个线程, 每个线程用N=4次float4读4个连续数据
    // v[N]: v[i]表示线程i存储的第i个float4, i∈[0, 3]
    // threadIdx.x = 0:
    //   v[0] = in[0] * weight[0], v[1] = in[256] * weight[256], v[2] = in[512] * weight[512], v[3] = in[768] * weight[768]
    // ...
    // threadIdx.x = 255:
    //   v[0] = in[255] * weight[255], v[1] = in[511] * weight[511], v[2] = in[767] * weight[767], v[3] = in[] * weight[768]

    constexpr int N = ::size / block_size / 4; // 每个线程处理多少个float4, N=4
    float4 v[N]; // 取in_ptr中的数据
    float acc = 0.f;
    #pragma unroll
    for(int i = 0; i < N; ++i)
    {
        v[i] = in_ptr[tid + i * block_size];
        acc += comm::dot(v[i], v[i]);
    }

    // 2. sum规约
    float sum = reductionv4(acc, tid); // 此时只有warp0拿到了正确结果
    __shared__ float s_rms;
    if(tid == 0) s_rms = rsqrtf(sum / size + eps);
    __syncthreads();
    float rms = s_rms; // 广播, 不存在bank conflict!!!

    // 3. 计算x_i * gamma_i
    #pragma unroll
    for(int i = 0; i < N; ++i)
    {
        float4 V = v[i];
        float4 W = w_ptr[tid + i * block_size];
        out_ptr[tid + i * block_size] = make_float4(V.x * W.x * rms, V.y * W.y *rms, V.z * W.z * rms, V.w * W.w * rms);
    }
}

void call_rmsnorm(float* in, float* weight, float* out, int batch, int size, float eps)
{
    dim3 grid(grid_size);
    dim3 block(block_size);
    rmsnormv4<<<grid, block>>>(in, weight, out, batch, size, eps);
}

int main()
{
    std::vector<float> input(count);
    std::vector<float> weights(size, 1.0f);
    std::vector<float> out(count);
    for(int i = 0; i < count; ++i)
    {
        input[i] = i % 3; // data
    }

    auto d_in = comm::make_device_buffer<float>(count);
    auto d_weights = comm::make_device_buffer<float>(size);
    auto d_out = comm::make_device_buffer<float>(count);

    comm::check_cuda(cudaMemcpy(d_in.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice), "data to d_in fail!");
    comm::check_cuda(cudaMemcpy(d_weights.get(), weights.data(), size * sizeof(float), cudaMemcpyHostToDevice), "weights to d_weights fail!");

    call_rmsnorm(d_in.get(), d_weights.get(), d_out.get(), batch, size, eps);

    comm::CudaEvent start, end;
    comm::check_cuda(cudaEventRecord(start.get()), "cudaEventRecord(start) fail!");
    for(int i = 0; i < loops; ++i)
    {
        call_rmsnorm(d_in.get(), d_weights.get(), d_out.get(), batch, size, eps);
    }
    comm::check_cuda(cudaEventRecord(end.get()), "cudaEventRecord(end) fail!");
    comm::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize() fail!");

    float avg_ms = 0.f;
    comm::check_cuda(cudaEventElapsedTime(&avg_ms, start.get(), end.get()), "cudaEventElapsedTime");
    printf("rmsnorm3 avg_ms :%.2fus\n", avg_ms * 1000 / loops);

    
    comm::check_cuda(cudaMemcpy(out.data(), d_out.get(), count * sizeof(float), cudaMemcpyDeviceToHost), "d_out to out fail!");
    std::ifstream ifs("./rms_ref.txt");
    std::vector<float> ref(count);
    for(int i = 0; i < count; ++i) ifs >> ref[i];

    comm::verify_results(ref, out, size);
    return 0;
}