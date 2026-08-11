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

__global__ void reduction_v1(float *out, const float *inp)
{
    // 使用相比于reduction_v0一半的线程块数就可以完成规约
    // 思考：是否可以实现更少的线程块数就可以完成规约？
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    // block = 0:
    //   thread0    -> gid = 0
    //   thread1    -> gid = 1
    //   ...
    //   thread1023 -> gid = 1023
    // block = 1:
    //   thread0    -> gid = 1 * 2 * 1024 = 2048
    //   thread1    -> gid = 2049
    //   ...
    //   thread1023 -> gid = 3071
    // ...
    // block = 511:
    //   thread0    -> gid = 511 * 2 * 1024 = 1046528
    //   thread1    -> gid = 1046529
    //   ...
    //   thread1023 -> gid = 1047511
    unsigned int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // block = 0:
    //  thread0 读取inp[0] + inp[1024]
    //  thread1 读取inp[1] + inp[1025]
    //  ...
    //  thread1023 读取inp[1023] + inp[2047]
    // block = 1:
    //  thread0 读取inp[2048] + inp[3072]
    //  thread1 读取inp[2049] + inp[3073]
    //  ...
    //  thread1023 读取inp[3071] + inp[4095]
    // ...
    // block = 511:
    //  thread0 读取inp[1046528] + inp[1047522]
    //  thread1 读取inp[1046529] + inp[1047523]
    //  ...
    //  thread1023 读取inp[1047511] + inp[1048535]
    sdata[tid] = inp[gid] + inp[gid + blockDim.x];
    __syncthreads();

    // 1. 乘法归约 : 缺点是仅偶数编号的线程进行计算，奇数编号的线程空闲，导致线程束分化！！！
    for(unsigned int s = 1; s < blockDim.x; s <<= 1)
    {
        if(tid % (2 * s) == 0 && tid + s < blockDim.x)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 2. 由每个线程块的第一个线程将sdata[0]写到结果
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

        const int ThreadsPerBlock1 = N / 1024;
        const int BlocksPerGrid1 = N / 1024 / 2;
        const int SecondStageThreads = BlocksPerGrid1 / 2; // 256

        std::cout << "线程块数量:" << BlocksPerGrid1 << ", 线程块内线程数量:" << ThreadsPerBlock1 << "\n";

        auto benchmark_v1 = [&]() {
            for (int i = 0; i < Warmup; ++i) {
                // 1. 
                reduction_v1<<<BlocksPerGrid1, ThreadsPerBlock1>>>(
                    device_block_out.get(), device_in.get()
                );
                // 2. 
                reduction_v1<<<1, SecondStageThreads>>>(
                    device_final_out.get(), device_block_out.get()
                );  
            }
            check_cuda(cudaGetLastError(), "reduction_forward_v0 warmup launch fail!");
            check_cuda(cudaDeviceSynchronize(), "reduction_forward_v0 warmup sync fail!");

            check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start fail!");
            for (int i = 0; i < Iters; ++i) {
                reduction_v1<<<BlocksPerGrid1, ThreadsPerBlock1>>>(
                    device_block_out.get(), device_in.get()
                );
                reduction_v1<<<1, SecondStageThreads>>>(
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

        const float v1_ms = benchmark_v1();
        check_cuda(
            cudaMemcpy(gpu_results.data(), device_final_out.get(), sizeof(float), cudaMemcpyDeviceToHost), 
            "cudaMemcpyDeviceToHost v1 fail!"
        );
        const bool v1_ok = verify_result(cpu_result, gpu_results[0]);

        check_cuda(cudaEventDestroy(start), "cudaEventDestroy start fail!");
        check_cuda(cudaEventDestroy(end), "cudaEventDestroy end fail!");

        std::cout << "| 实现 | CPU耗时(ms) | GPU平均耗时(ms) | 加速比 | 正确性 |\n";
        std::cout << "|---|---:|---:|---:|---|\n";
        // std::cout << "| gpu2 shared block reduce | " << cpu_ms << " | " << v0_ms
        //           << " | " << cpu_ms / v0_ms << "x | " << (v0_ok ? "YES" : "NO") << " |\n";

        std::cout << "| gpu2 shared block reduce | " << cpu_ms << " | " << v1_ms
                  << " | " << cpu_ms / v1_ms << "x | " << (v1_ok ? "YES" : "NO") << " |\n";
    }
    catch(const std::exception & e)
    {
        std::cout << e.what() << "\n";
    }

    return 0;
}
