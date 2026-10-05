#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
/*
    操作	函数名	                说明
    加法	atomicAdd(addr, val)	 加法
    减法	atomicSub(addr, val)	 减法
    最大值	atomicMax(addr, val)	 返回两个值的最大值
    最小值	atomicMin(addr, val)	 返回两个值的最小值
    与	    atomicAnd(addr, val)	     按位与
    或	    atomicOr(addr, val)	         按位或
    异或	atomicXor(addr, val)	 按位异或
    交换	atomicExch(addr, val)	 设置新值并返回旧值
    比较交换	atomicCAS(addr, compare, val)	如果当前值等于 compare，则设置为 val
*/

// __global__ void add(int *host)
// {
//     // int tmp = *host;
//     // tmp += 1;
//     // *host = tmp;
//     atomicAdd(host, 1);
// }

// int main()
// {
//     int *host = nullptr;
//     int h_data = 0;
//     cudaMalloc(&host, sizeof(int));
//     cudaMemcpy(host, &h_data, sizeof(int), cudaMemcpyHostToDevice);
//     add<<<2, 8>>>(host);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&h_data, host, sizeof(int), cudaMemcpyDeviceToHost);
//     printf("Result: %d\n", h_data);
//     cudaFree(host);
//     return 0;
// }

// 案例：多线程统计图像像素值写入直方图
namespace
{
    constexpr int HIST_SIZE = 256;
    constexpr int block_per_grid = 256;
    constexpr int thread_per_block = 256;
    constexpr int height = 4096, width = 4096;
    constexpr int size = height * width;
}// namespace


__global__ void hist(u_int8_t *input, int *hist, int n)
{
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = gid; i < n; i += gridDim.x * blockDim.x)
    {
        atomicAdd(&hist[input[i]], 1);
    }
}

__global__ void sh_hist(u_int8_t *input, int *hist, int n)
{
    __shared__ int hist_private[HIST_SIZE];
    // 1. 多线程初始化线程块共享内存
    // 当前共4个线程块，每个线程块64个线程
    for(int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x)
    {
        hist_private[i] = 0;
    }
    __syncthreads();

    // 2. 多线程统计像素值写入共享内存直方图
    // tid∈[0, 63], gid∈[0, 255], gridDim.x * blockDim.x = 256
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = gid; i < n; i += gridDim.x * blockDim.x)
    {
        atomicAdd(&hist_private[input[i]], 1);
    }
    __syncthreads();

    // 3. 共享内存写入全局内存
    for(int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x)
    {
        atomicAdd(&hist[i], hist_private[i]);
    }
}


int main()
{
    int *host_hist = new int[HIST_SIZE]{0};
    u_int8_t *image = new u_int8_t[size];
    for (int i = 0; i < size; ++i) {
        image[i] = rand() % 256;
    }
    int *device_hist = nullptr;
    u_int8_t *d_image = nullptr;
    cudaMalloc(&device_hist, sizeof(int) * HIST_SIZE);
    cudaMalloc(&d_image, sizeof(u_int8_t) * size);
    cudaMemcpy(device_hist, host_hist, sizeof(int) * HIST_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image, sizeof(u_int8_t) * size, cudaMemcpyHostToDevice);

    dim3 grid(block_per_grid);
    dim3 block(thread_per_block);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    hist<<<grid, block>>>(d_image, device_hist, size);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float total_ms = 0.f;
    cudaEventElapsedTime(&total_ms, start, stop);
    printf("total_ms: %fms\n", total_ms);

    cudaMemcpy(host_hist, device_hist, sizeof(int) * HIST_SIZE, cudaMemcpyDeviceToHost);
    for(int i = 0; i < HIST_SIZE; ++i)
    {
        if(host_hist[i] != 0)
            printf("Histogram[%d]: %d\n", i, host_hist[i]);
    }
    cudaFree(device_hist);
    cudaFree(d_image);

    delete[] host_hist;

    return 0;
}