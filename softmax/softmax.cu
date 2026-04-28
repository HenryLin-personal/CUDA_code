#include <iostream>
#include <cmath>
#include <chrono>
#include "comm.cuh"

namespace {
constexpr int N = 32;
constexpr int C = 4096;
constexpr float Tolerance = 1.0e-6f;

void softmax_forward_cpu(float *out, const float *inp, int N, int C) 
{
    // input: tensor, tensor's shape is [N, C]
    for(int i = 0; i < N; ++i)
    {
        // 1. 定位每行起始元素
        const float *row_base = inp + i * C;
        float *out_row = out + i * C;

        // 2. 找每行元素的最大值
        float maxval = -INFINITY;
        for(int j = 0; j < C; ++j)
        {
            maxval = std::max(maxval, row_base[j]);
        }

        // 3. 求分母
        float sum = 0.f;
        for(int j = 0; j < C; ++j)
        {
            out_row[j] = std::exp(row_base[j] - maxval);
            sum += out_row[j];
        }

        // 4. 分子 / 分母
        float norm = 1.f / sum;
        for(int j = 0; j < C; ++j)
        {
            out_row[j] *= norm;
        }
    }
}

constexpr int ThreadsPerBlock1 = 1;
__global__ void softmax_forward_gpu1(float *out, const float *inp, int N, int C)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    // if(id == 0)
    // {
    //     std::printf("线程块数量:%d\n", gridDim.x);
    //     std::printf("线程数量:%d\n", blockDim.x);
    // }
    // 每行分配一个线程
    if(id < N)
    {
        // 1. 定位当前行的起始元素
        const float *inp_row = inp + id * C;
        float *out_row = out + id * C;

        // 2. 按行找最大值
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) 
        {
            if (inp_row[j] > maxval)    
            {
                maxval = inp_row[j];
            }
        }

        // 3. 分母求和 + 指数减去maxval
        float sum = 0.f;
        for (int j = 0; j < C; j++) 
        {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        
        // 4. 分子 / 分母得到每个元素的softmax结果
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) 
        {
            out_row[j] *= norm;
        }
    }
}

constexpr int PixelPerThreads = 32;
constexpr int ThreadsPerBlock2 = C / PixelPerThreads;
__global__ void softmax_forward_gpu2(float *out, const float *inp, int N, int C)
{
    // 为每个线程块/每行声明一个大小为线程数 * sizeof(float)的共享内存区
    // 共享内存需要使用 void __syncthreads();进行线程同步
    extern __shared__ float shared[];
    int idx = blockIdx.x;                   // 当前block对应第几行
    int tid = threadIdx.x;                  // 当前线程在block内的编号
    int block_size = blockDim.x;            // blockDim.x == 128
    const float *row_base = inp + idx * C;  // 当前行的起始元素地址

    // 1. 每个线程找自己负责的最大值
    // thread0: x[0], x[128], x[256], ..., x[3968]
    // thread1: x[1], x[129], x[257], ..., x[3969]
    // ...
    // thread127: x[127], x[255], x[383], ..., x[4095]
    // 以上分配方式保证了访存的连续性(竖着看，线程是并行的)
    float maxval = -INFINITY;
    for(int i = tid; i < C; i += block_size)
    {
        maxval = fmaxf(maxval, row_base[i]);
    }

    // 2. 把局部max放入shared memory
    shared[tid] = maxval; // shared[tid]
    __syncthreads();

    // 3. 由局部max计算全局max
    // // 法一: 每个线程块的第一个线程使用for循环遍历共享内存
    // if(tid == 0)
    // {   
    //     float maxval = shared[0];
    //     for(int i = 1; i < block_size; ++i)
    //     {
    //         maxval = std::max(maxval, shared[i]);
    //     }
    //     shared[0] = maxval; // 将maxval存入shared[0]
    // }
    // __syncthreads();

    // 法二: 并行的层次化策略
    for (int stride = block_size / 2; stride >= 1; stride /= 2) 
    {
        __syncthreads();
        if (tid < stride) 
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();

    // 4. 线程计算自己负责的分子，分子是全局的
    float offset = shared[0]; // 此时shared[0]存的是当前行所有元素的最大值
    float sum = 0.f;
    float *out_base = out + idx * C;
    for(int i = tid; i < C; i += block_size)
    {
        out_base[i] = expf(row_base[i] - offset);
        // out[idx * C + i] = expf(row_base[i] - offset);
        // 5. 线程计算自己负责的分母，分母是局部的
        sum += out_base[i];
    }

    // 6. 由局部求和统计全体求和
    shared[tid] = sum;
    __syncthreads();
    for(int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if(tid < stride)
            shared[tid] += shared[tid + stride];
    }

    // 7. 分子 / 分母
    sum = shared[0];
    for(int i = tid; i < C; i += block_size)
    {
        out[idx * C + i] /= sum;
    }
}

constexpr int ThreadsPerBlock3 = 32;
__global__ void softmax_forward_gpu3(float *out, const float *inp, int N, int C)
{
    extern __shared__ float shared[];   // 32 * sizeof(float)
    int idx = blockIdx.x;               // 每行分配一个线程块，一共有4096 / 32 = 128个线程块
    int tid = threadIdx.x;              // tid∈[0, 31]
    const float *x = inp + idx * C;     // 每行元素的起始地址

    // 1. 找局部max
    // thread0: x[0], x[128], x[256], ...
    // thread1: x[1], x[129], x[257], ...
    // ...
    // thread31: x[31], x[159], x[287], ...
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
    {
        maxval = fmaxf(maxval, x[i]);
    }

    // 2. __shfl_down_sync归约局部最大值
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maxval = fmaxf(maxval, __shfl_down_sync(0xFFFFFFFF, maxval, offset));
    }
    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);
    // maxval = 32个线程统计的最大值

    // 3. 并行计算分子 + 局部求和
    float sum = 0.f;
    float *out_base = out + idx * C;
    for(int i = tid; i < C; i += blockDim.x)
    {
        out_base[i] = expf(inp[idx * C + i] - maxval);
        sum += out_base[i];
    }

    // 4. 归约求和
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);

    // 5. 分子 / 分母
    float norm = 1.f / sum;
    for(int i = tid; i < C; i += blockDim.x)
    {
        out_base[i] *= norm;
    }
}

constexpr int ThreadsPerBlock4 = 128; // 每行分配一个线程块：4个线程束，128个线程
__global__ void softmax_forward_gpu4(float *out, const float *inp, int N, int C)
{
    extern __shared__ float shared[];   // shared[4]
    int idx = blockIdx.x;               // idx ∈ [0, N-1]
    int tid = threadIdx.x;              // tid ∈ [0, 127]
    int lane_id = tid & 31;             // lane_id ∈ [0, 31]
    int lane = tid / 32;                // lane ∈ [0, 3]
    const float *x = inp + idx * C;     // 每行的起始元素地址

    // 1. 每个线程找自己负责的最大值
    // thread0: x[0], x[128], x[256], ..., x[3968]
    // thread1: x[1], x[129], x[257], ..., x[3969]
    // ...
    // thread127: x[127], x[255], x[383], ..., x[4095]
    float maxval = -INFINITY;
    for(int i = tid; i < C; i += blockDim.x)
    {
        maxval = fmaxf(maxval, x[i]);
    }

    // 2. 线程束内使用__shfl_down_sync() + __shfl_sync()
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        maxval = fmaxf(maxval, __shfl_down_sync(0xFFFFFFFF, maxval, offset));
    }
    // __shfl_sync(0xFFFFFFF, maxval, 0);
    
    // 3. 由每个线程束的第一个线程将线程束统计的局部最大值写入共享内存
    if(lane_id == 0)
    {
        shared[lane] = maxval;
    }
    __syncthreads();

    // 4. 共享内存规约
    for(int stride = 2; stride > 0; stride >>= 1)
    {
        // if(lane < stride)
        //     shared[lane] = fmaxf(shared[lane], shared[lane + stride]);
        
        // 上面那种写法会让一个线程束内的32个线程都去写共享内存，破坏了互斥性
        // 只需要让每个线程束的第一个线程去写即可
        if(lane_id == 0 && lane < stride)
            shared[lane] = fmaxf(shared[lane], shared[lane + stride]);
        __syncthreads();
    }
    
    // 5. shared[0]存储128个线程的全局最大值，使用__shfl_sync进行广播即可
    maxval = shared[0];
    __shfl_sync(0xFFFFFFFF, maxval, 0);

    // 6. 全局分子 + 局部求和
    float sum = 0.f;
    for(int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = expf(x[i] - maxval);
        sum += out[idx * C + i];
    }

    // 7. 线程束内规约局部求和
    for(int offset = 16; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shfl_sync(0xFFFFFFFF, sum, 0);

    // 8. 由每个线程束内的第一个线程将线程束统计的局部求和写入共享内存
    if(lane_id == 0)
    {
        shared[lane] = sum;
    }
    __syncthreads();

    // 9. 共享内存规约局部求和
    for(int stride = 2; stride > 0; stride >>= 1)
    {
        if(lane_id == 0 && lane < stride)
            shared[lane] += shared[lane + stride];
        __syncthreads();
    }

    // // 10. shared[0]存储128个线程的全局求和，使用__shfl_sync进行广播即可
    // __shfl_sync(0xFFFFFFFF, sum, 0);

    // 11. 分子 / 分母
    sum = shared[0];
    float norm = 1.f / sum;
    for(int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] *= norm;
    }
}


bool verify_result(const std::vector<float> &expected, const std::vector<float> &actual)
{
    for (std::size_t index = 0; index < expected.size(); ++index) 
    {
        if (std::fabs(expected[index] - actual[index]) > Tolerance) 
        {
            std::printf("Mismatch at index %zu: expected=%f, actual=%f\n",
                        index,
                        expected[index],
                        actual[index]);
            return false;
        }
    }
    return true;
}
} // namespace


int main()
{
    try 
    {
        setDevice();

        std::size_t count = N * C;
        const std::size_t bytes = count * sizeof(float);

        std::vector<float> host_tensor(count, 0.f);
        std::vector<float> gpu_results(count, 0.f);
        std::vector<float> cpu_results(count, 0.f);
        std::mt19937 rng(666);
        initialize(host_tensor, rng);

        auto device_in = make_device_buffer<float>(count);
        auto device_out = make_device_buffer<float>(count);
        check_cuda(
            cudaMemcpy(device_in.get(), host_tensor.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpyHostToDevice fail!"
        );
        
        const int BlocksPerGrid2 = N; // 每行分配一个线程块，每个线程块内有128个线程
        const int BlocksPerGrid3 = N; // 每行分配一个线程块，每个线程块内有32个线程
        const int BlocksPerGrid4 = N; // 每行分配一个线程块，每个线程块内有128个线程

        // 1. CPU函数计时
        auto begin = std::chrono::steady_clock::now();
        softmax_forward_cpu(cpu_results.data(), host_tensor.data(), N, C);
        auto stop = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(stop - begin).count();

        // 2. GPU warmup + 平均计时
        cudaEvent_t start, end;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start fail!");
        check_cuda(cudaEventCreate(&end), "cudaEventCreate end fail!");

        constexpr int Warmup = 20;
        constexpr int Iters = 200;

        auto benchmark_gpu2 = [&]() {
            for (int i = 0; i < Warmup; ++i) {
                softmax_forward_gpu2<<<BlocksPerGrid2, ThreadsPerBlock2, ThreadsPerBlock2 * sizeof(float)>>>(
                    device_out.get(), device_in.get(), N, C
                );
            }
            check_cuda(cudaGetLastError(), "softmax_forward_gpu2 warmup launch fail!");
            check_cuda(cudaDeviceSynchronize(), "softmax_forward_gpu2 warmup sync fail!");

            check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start fail!");
            for (int i = 0; i < Iters; ++i) {
                softmax_forward_gpu2<<<BlocksPerGrid2, ThreadsPerBlock2, ThreadsPerBlock2 * sizeof(float)>>>(
                    device_out.get(), device_in.get(), N, C
                );
            }
            check_cuda(cudaGetLastError(), "softmax_forward_gpu2 launch fail!");
            check_cuda(cudaEventRecord(end, 0), "cudaEventRecord end fail!");
            check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end fail!");

            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime fail!");
            return ms / Iters;
        };

        auto benchmark_gpu3 = [&]() {
            for (int i = 0; i < Warmup; ++i) {
                softmax_forward_gpu3<<<BlocksPerGrid3, ThreadsPerBlock3, 0>>>(
                    device_out.get(), device_in.get(), N, C
                );
            }
            check_cuda(cudaGetLastError(), "softmax_forward_gpu3 warmup launch fail!");
            check_cuda(cudaDeviceSynchronize(), "softmax_forward_gpu3 warmup sync fail!");

            check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start fail!");
            for (int i = 0; i < Iters; ++i) {
                softmax_forward_gpu3<<<BlocksPerGrid3, ThreadsPerBlock3, 0>>>(
                    device_out.get(), device_in.get(), N, C
                );
            }
            check_cuda(cudaGetLastError(), "softmax_forward_gpu3 launch fail!");
            check_cuda(cudaEventRecord(end, 0), "cudaEventRecord end fail!");
            check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end fail!");

            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime fail!");
            return ms / Iters;
        };

        auto benchmark_gpu4 = [&]() {
            const int shared_bytes = 4 * sizeof(float);
            for (int i = 0; i < Warmup; ++i) {
                softmax_forward_gpu4<<<BlocksPerGrid4, ThreadsPerBlock4, shared_bytes>>>(
                    device_out.get(), device_in.get(), N, C
                );
            }
            check_cuda(cudaGetLastError(), "softmax_forward_gpu4 warmup launch fail!");
            check_cuda(cudaDeviceSynchronize(), "softmax_forward_gpu4 warmup sync fail!");

            check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start fail!");
            for (int i = 0; i < Iters; ++i) {
                softmax_forward_gpu4<<<BlocksPerGrid4, ThreadsPerBlock4, shared_bytes>>>(
                    device_out.get(), device_in.get(), N, C
                );
            }
            check_cuda(cudaGetLastError(), "softmax_forward_gpu4 launch fail!");
            check_cuda(cudaEventRecord(end, 0), "cudaEventRecord end fail!");
            check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end fail!");

            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime fail!");
            return ms / Iters;
        };

        const float gpu2_ms = benchmark_gpu2();
        check_cuda(
            cudaMemcpy(gpu_results.data(), device_out.get(), bytes, cudaMemcpyDeviceToHost), 
            "cudaMemcpyDeviceToHost gpu2 fail!"
        );
        const bool gpu2_ok = verify_result(cpu_results, gpu_results);

        const float gpu3_ms = benchmark_gpu3();
        check_cuda(
            cudaMemcpy(gpu_results.data(), device_out.get(), bytes, cudaMemcpyDeviceToHost), 
            "cudaMemcpyDeviceToHost gpu3 fail!"
        );
        const bool gpu3_ok = verify_result(cpu_results, gpu_results);

        const float gpu4_ms = benchmark_gpu4();
        check_cuda(
            cudaMemcpy(gpu_results.data(), device_out.get(), bytes, cudaMemcpyDeviceToHost), 
            "cudaMemcpyDeviceToHost gpu3 fail!"
        );
        const bool gpu4_ok = verify_result(cpu_results, gpu_results);

        check_cuda(cudaEventDestroy(start), "cudaEventDestroy start fail!");
        check_cuda(cudaEventDestroy(end), "cudaEventDestroy end fail!");

        std::cout << "| 实现 | CPU耗时(ms) | GPU平均耗时(ms) | 加速比 | 正确性 |\n";
        std::cout << "|---|---:|---:|---:|---|\n";
        std::cout << "| gpu2 shared block reduce | " << cpu_ms << " | " << gpu2_ms
                  << " | " << cpu_ms / gpu2_ms << "x | " << (gpu2_ok ? "YES" : "NO") << " |\n";
        std::cout << "| gpu3 warp shuffle | " << cpu_ms << " | " << gpu3_ms
                  << " | " << cpu_ms / gpu3_ms << "x | " << (gpu3_ok ? "YES" : "NO") << " |\n";
        std::cout << "| gpu4 warp shuffle | " << cpu_ms << " | " << gpu4_ms
                  << " | " << cpu_ms / gpu4_ms << "x | " << (gpu4_ok ? "YES" : "NO") << " |\n";
    }
    catch(const std::exception & e)
    {
        std::cout << e.what() << "\n";
    }

    return 0;
}
