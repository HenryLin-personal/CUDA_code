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

__device__ void reductionv2(float *cache, const int tid)
{
    for(int offset = block_size / 2; offset > 0; offset >>= 1)
    {
        if(tid < offset) cache[tid] += cache[tid + offset];
        __syncthreads();
    }
}

__global__ void rmsnormv0(float* in, float* weight, float* out, int batch, int size, float eps)
{
    // 每个block负责一个batch, block内有1024个线程, 每个线程负责
    // threadIdx.x = 0:
    //   in[0] * weight[0]       in[1024] * weight[1024] in[2048] * weight[2048] in[3072] * weight[3072] 
    // ...
    // threadIdx.x = 1023:
    //   in[1023] * weight[1023] in[2047] * weight[2047] in[3071] * weight[3071] in[4095] * weight[4095]
    float *in_ptr = in + blockIdx.x * size;
    float *out_ptr = out + blockIdx.x * size;
    const int tid = threadIdx.x;

    // 1. 每个线程写入共享内存
    __shared__ float sum[block_size];
    float acc = 0.f;
    for(int i = tid; i < size; i += blockDim.x)
    {
        acc += in_ptr[i] * in_ptr[i];
    }
    sum[tid] = acc;
    __syncthreads();

    // 2. sum规约 
    reductionv2(sum, tid);

    float rms = 1.0f / sqrt(sum[0] / size + eps);
    // 3. 计算x_i * gamma_i
    for(int i = tid; i < size; i += blockDim.x)
    {
        out_ptr[i] = in_ptr[i] * weight[i] * rms;
    }
}

void call_rmsnorm(float* in, float* weight, float* out, int batch, int size, float eps)
{
    dim3 grid(grid_size);
    dim3 block(block_size);
    rmsnormv0<<<grid, block>>>(in, weight, out, batch, size, eps);
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
    printf("rmsnorm0 avg_ms :%.2fus\n", avg_ms * 1000 / loops);

    
    comm::check_cuda(cudaMemcpy(out.data(), d_out.get(), count * sizeof(float), cudaMemcpyDeviceToHost), "d_out to out fail!");
    std::ifstream ifs("./rms_ref.txt");
    std::vector<float> ref(count);
    for(int i = 0; i < count; ++i) ifs >> ref[i];

    comm::verify_results(ref, out, size);
    return 0;
}