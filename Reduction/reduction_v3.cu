#include <iostream>
#include <cmath>
#include <chrono>
#include "comm.cuh"

namespace {
constexpr int BLOCK_SIZE = 1024;
constexpr int N = 1024 * 1024;
constexpr float Tolerance = 1.0e-4f;

float reduction_forward_cpu(const std::vector<float> &data)
{
    // 以sum算子举例
    float sum = 0.0f;
    for (float val : data) 
    {
        sum += val;
    }
    return sum;
}

#define FULL_MASK 0xffffffff
__device__ void warpReduce(float *cache, unsigned int tid) {
  float v = cache[tid] + cache[tid + 32];
  v += __shfl_down_sync(FULL_MASK, v, 16);
  v += __shfl_down_sync(FULL_MASK, v, 8);
  v += __shfl_down_sync(FULL_MASK, v, 4);
  v += __shfl_down_sync(FULL_MASK, v, 2);
  v += __shfl_down_sync(FULL_MASK, v, 1);
  cache[tid] = v;
}

__global__ void reduction_v3(float *out, const float *inp)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    sdata[tid] = inp[gid] + inp[gid + blockDim.x];
    __syncthreads();

    // 1. 除法归约
    for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 2. 使用线程束源语
    if (tid < 32) warpReduce(sdata, tid);
    // 3. 由每个线程块的第一个线程将sdata[0]写到结果
    if(tid == 0) out[blockIdx.x] = sdata[0];
}

bool verify_result(const float &expected, const float &actual)
{
    if(std::fabs(expected - actual) > Tolerance)
    {
        std::printf("Mismatch: expected=%f, actual=%f\n",
            expected, actual
        );
        return false;
    }
    return true;
}

} // namespace

int main()
{
    try 
    {
        setDevice();

        std::size_t count = N;
        const std::size_t bytes = count * sizeof(float);

        std::vector<float> host_tensor(count, 0.f);
        std::vector<float> gpu_results(1, 0.f);
        std::mt19937 rng(666);
        initialize(host_tensor, rng);

        auto device_in = make_device_buffer<float>(count); // 1024 * 1024
        auto device_block_out = make_device_buffer<float>(count / 1024 / 2); // 512
        auto device_final_out = make_device_buffer<float>(1);
        check_cuda(
            cudaMemcpy(device_in.get(), host_tensor.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpyHostToDevice fail!"
        );

        // 1. CPU函数计时
        auto begin = std::chrono::steady_clock::now();
        float cpu_result = reduction_forward_cpu(host_tensor);
        auto stop = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(stop - begin).count();

        // 2. GPU warmup + 平均计时
        cudaEvent_t start, end;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start fail!");
        check_cuda(cudaEventCreate(&end), "cudaEventCreate end fail!");

        constexpr int Warmup = 20;
        constexpr int Iters = 200;

        const int ThreadsPerBlock3 = N / 1024;
        const int BlocksPerGrid3 = N / 1024 / 2;
        const int SecondStageThreads = BlocksPerGrid3 / 2; // 256

        std::cout << "线程块数量:" << BlocksPerGrid3 << ", 线程块内线程数量:" << ThreadsPerBlock3 << "\n";

        auto benchmark_v3 = [&]() {
            for (int i = 0; i < Warmup; ++i) {
                // 1. 
                reduction_v3<<<BlocksPerGrid3, ThreadsPerBlock3>>>(
                    device_block_out.get(), device_in.get()
                );
                // 2. 
                reduction_v3<<<1, SecondStageThreads>>>(
                    device_final_out.get(), device_block_out.get()
                );  
            }
            check_cuda(cudaGetLastError(), "reduction_forward_v0 warmup launch fail!");
            check_cuda(cudaDeviceSynchronize(), "reduction_forward_v0 warmup sync fail!");

            check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start fail!");
            for (int i = 0; i < Iters; ++i) {
                reduction_v3<<<BlocksPerGrid3, ThreadsPerBlock3>>>(
                    device_block_out.get(), device_in.get()
                );
                reduction_v3<<<1, SecondStageThreads>>>(
                    device_final_out.get(), device_block_out.get()
                );  
            }
            check_cuda(cudaGetLastError(), "reduction_forward_v0 launch fail!");
            check_cuda(cudaEventRecord(end, 0), "cudaEventRecord end fail!");
            check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end fail!");

            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime fail!");
            return ms / Iters;
        };

        const float v3_ms = benchmark_v3();
        check_cuda(
            cudaMemcpy(gpu_results.data(), device_final_out.get(), sizeof(float), cudaMemcpyDeviceToHost), 
            "cudaMemcpyDeviceToHost v3 fail!"
        );
        const bool v3_ok = verify_result(cpu_result, gpu_results[0]);

        check_cuda(cudaEventDestroy(start), "cudaEventDestroy start fail!");
        check_cuda(cudaEventDestroy(end), "cudaEventDestroy end fail!");

        std::cout << "| 实现 | CPU耗时(ms) | GPU平均耗时(ms) | 加速比 | 正确性 |\n";
        std::cout << "|---|---:|---:|---:|---|\n";
        std::cout << "| gpu2 shared block reduce | " << cpu_ms << " | " << v3_ms
                  << " | " << cpu_ms / v3_ms << "x | " << (v3_ok ? "YES" : "NO") << " |\n";
    }
    catch(const std::exception & e)
    {
        std::cout << e.what() << "\n";
    }

    return 0;
}
