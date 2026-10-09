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

    constexpr int block_size = 1024;
    constexpr int grid_size = batch;
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

__global__ void rmsnormv3(float* in, float* weight, float* out, int batch, int size, float eps)
{
    // 每个block负责一个batch, block内有1024个线程, 每个线程用一次float4读4个连续数据
    // threadIdx.x = 0:
    //   in[0..3] * weight[0..3]
    // ...
    // threadIdx.x = 1023:
    //   in[4092..4095] * weight[4092..4095]
    // size % 4 == 0 且 cudaMalloc 256B对齐, 所以每行起始地址都满足float4的16B对齐要求
    const float4 *in_ptr = reinterpret_cast<const float4*>(in + blockIdx.x * size);
    const float4 *w_ptr = reinterpret_cast<const float4*>(weight);
    float4 *out_ptr = reinterpret_cast<float4*>(out + blockIdx.x * size);
    const int tid = threadIdx.x;
    const int vec_size = size / 4;

    // 1. 每个线程维护一个寄存器变量
    float acc = 0.f;
    for(int i = tid; i < vec_size; i += blockDim.x)
    {
        float4 v = in_ptr[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // 2. sum规约
    float sum = reductionv4(acc, tid); // 此时只有warp0拿到了正确结果
    __shared__ float s_rms;
    if(tid == 0) s_rms = rsqrtf(sum / size + eps);
    __syncthreads();
    float rms = s_rms;
    // 3. 计算x_i * gamma_i
    for(int i = tid; i < vec_size; i += blockDim.x)
    {
        float4 v = in_ptr[i];
        float4 w = w_ptr[i];
        out_ptr[i] = make_float4(v.x * w.x * rms, v.y * w.y * rms, v.z * w.z * rms, v.w * w.w * rms);
    }
}

void call_rmsnorm(float* in, float* weight, float* out, int batch, int size, float eps)
{
    dim3 grid(grid_size);
    dim3 block(block_size);
    rmsnormv3<<<grid, block>>>(in, weight, out, batch, size, eps);
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