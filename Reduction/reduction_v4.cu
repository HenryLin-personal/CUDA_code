#include <iostream>
#include <cmath>
#include <chrono>
#include "comm.cuh"

namespace {
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

__device__ float warp_reduce(float v)
{
    #define FULL_MASK 0xffffffff
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    return v;
}

__inline__ __device__ float block_reduce(float val) {
    const int tid = threadIdx.x;
    const int warpSize = 32;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    __shared__ float warpSum[warpSize];
    // 1. 每个线程束内执行一次warp_reduce
    val = warp_reduce(val);
    if(lane == 0) warpSum[warp_id] = val;
    // 必须等待所有warp规约完毕
    __syncthreads();
    // 此时warpSum[i]保存了线程块内第i个线程束的局部规约结果

    // 2. 线程束之间执行一次warp_reduce, 仅需一个warp执行即可，这里指定warp0去执行，或者指定warp1/warp31都可以
    // 但是要注意，若指定warp_id == 1，返回的val保存在threadIdx.x == 32的线程的寄存器中
    // 在reduction_v4中if(threadIdx.x == 0) out[blockIdx.x] = sum;就必须改为
    // if(threadIdx.x == 32) out[blockIdx.x] = sum
    if(warp_id == 0) {
        // 2.1 先让warp内所有线程的值更新为warpSum[0~warp_count]
        // val = warpSum[lane];
        val = lane < warp_count
            ? warpSum[lane]
            : 0.0f; // 此时warp0的lane0~lane warp_count都读取了warp_count个线程束的局部规约结果
        // 2.2 然后执行warp_reduce
        val = warp_reduce(val);
    }
    return val;
}

__global__ void reduction_v4(float *out, const float *inp, const int n)
{
    // 假设传入1024个线程块，每个线程块1024个线程，即gridDim.x = 1024, blockDim.x = 1024
    // block = 0:
    //  thread0 : sum += inp[0]
    //  thread1 : sum += inp[1]
    //  ...
    //  thread1023 : sum += inp[1023]
    // block = 1:
    //  thread0 : sum += inp[1024]
    //  thread1 : sum += inp[1025]
    //  ...
    //  thread1023 : sum += inp[2047]
    // ...
    // block = 1023:
    //  thread0 : sum += inp[1047552]
    //  thread1 : sum += inp[1047553]
    //  ...
    //  thread1023 : sum += inp[1048575]
    // 此时每个线程仅处理一个数据，若改为512个线程块，每个线程块1024个线程，那么每个线程就会处理2个数据：
    // block = 0:
    //  thread0 : sum += inp[0] + inp[512 * 1024] = inp[0] + inp[52488]
    //  ...
    // coarsening的通用写法！！！

    // 1. 1024^2个线程每个各自维护一个寄存器变量sum
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) 
    {
        sum += inp[i];
    }

    // 2. blockReduce
    sum = block_reduce(sum);
    // 3. 保存线程块内的局部规约结果
    if(threadIdx.x == 0) out[blockIdx.x] = sum;
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

        std::size_t count = N;                      // 1024^2
        std::size_t BLOCK_SIZE = N / 1024 / 2;      // 512
        const std::size_t bytes = count * sizeof(float);

        std::vector<float> host_tensor(count, 0.f);
        std::vector<float> gpu_results(1, 0.f);
        std::mt19937 rng(666);
        initialize(host_tensor, rng);

        auto device_in = make_device_buffer<float>(count);
        auto device_block_out = make_device_buffer<float>(BLOCK_SIZE);
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

        const int ThreadsPerBlock4 = N / 1024;
        const int BlocksPerGrid4 = BLOCK_SIZE;
        const int SecondStageThreads = BlocksPerGrid4 / 2; 

        std::cout << "线程块数量:" << BlocksPerGrid4 << ", 线程块内线程数量:" << ThreadsPerBlock4 << "\n";

        auto benchmark_v4 = [&]() {
            // 1. warmup
            for (int i = 0; i < Warmup; ++i) {
                reduction_v4<<<BlocksPerGrid4, ThreadsPerBlock4>>>(
                    device_block_out.get(), device_in.get(), count
                );
                reduction_v4<<<1, SecondStageThreads>>>(
                    device_final_out.get(), device_block_out.get(), BLOCK_SIZE
                );  
            }
            check_cuda(cudaGetLastError(), "reduction_forward_v0 warmup launch fail!");
            check_cuda(cudaDeviceSynchronize(), "reduction_forward_v0 warmup sync fail!");

            check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start fail!");
            // 2. 正式
            for (int i = 0; i < Iters; ++i) {
                reduction_v4<<<BlocksPerGrid4, ThreadsPerBlock4>>>(
                    device_block_out.get(), device_in.get(), count // 第一次规约1024^2个元素
                );
                reduction_v4<<<1, SecondStageThreads>>>(
                    device_final_out.get(), device_block_out.get(), BLOCK_SIZE // 第二次规约512个元素
                );  
            }
            check_cuda(cudaGetLastError(), "reduction_forward_v0 launch fail!");
            check_cuda(cudaEventRecord(end, 0), "cudaEventRecord end fail!");
            check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end fail!");

            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime fail!");
            return ms / Iters;
        };

        const float v4_ms = benchmark_v4();
        check_cuda(
            cudaMemcpy(gpu_results.data(), device_final_out.get(), sizeof(float), cudaMemcpyDeviceToHost), 
            "cudaMemcpyDeviceToHost v4 fail!"
        );
        const bool v4_ok = verify_result(cpu_result, gpu_results[0]);

        check_cuda(cudaEventDestroy(start), "cudaEventDestroy start fail!");
        check_cuda(cudaEventDestroy(end), "cudaEventDestroy end fail!");

        std::cout << "| 实现 | CPU耗时(ms) | GPU平均耗时(ms) | 加速比 | 正确性 |\n";
        std::cout << "|---|---:|---:|---:|---|\n";
        std::cout << "| gpu2 shared block reduce | " << cpu_ms << " | " << v4_ms
                  << " | " << cpu_ms / v4_ms << "x | " << (v4_ok ? "YES" : "NO") << " |\n";
    }
    catch(const std::exception & e)
    {
        std::cout << e.what() << "\n";
    }

    return 0;
}
