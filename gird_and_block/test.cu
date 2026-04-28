#include <cstdio>
#include <cuda_runtime.h>

__global__ void print()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + blockDim.x * blockIdx.x;
    printf("hello world from block:%d, thread:%d, global id:%d\n", bid, tid, id);
}

__global__ void two_dim_block_and_thread()
{
    const int blockId = blockIdx.x + gridDim.x * blockIdx.y; // x优先
    const int threadId = threadIdx.x + blockDim.x * threadIdx.y; // x优先
    const int globalId = threadId + blockId * (blockDim.x * blockDim.y);
    if(globalId == 0){
        printf("这个函数内的网格尺寸为:%d x %d\n", gridDim.x, gridDim.y);
        printf("这个函数内的线程块尺寸为:%d x %d\n", blockDim.x, blockDim.y);
    }

}

__global__ void three_dim_block_and_thread()
{
    const int blockId = blockIdx.x + gridDim.x * blockIdx.y + (gridDim.x * gridDim.y) * blockIdx.z; // x优先
    const int threadId = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z; // x优先
    const int globalId = threadId + blockId * (blockDim.x * blockDim.y * blockDim.z);
    if(globalId == 0){
        printf("这个函数内的网格尺寸为:%d x %d x %d\n", gridDim.x, gridDim.y, gridDim.z);
        printf("这个函数内的线程块尺寸为:%d x %d x %d\n", blockDim.x, blockDim.y, blockDim.z);
    }
}

int main()
{
    // print<<<2, 4>>>();

    // dim3 grid_size(2, 2);
    // dim3 block_size(3, 4);
    // two_dim_block_and_thread<<<grid_size, block_size>>>();

    dim3 grid_size(2, 3, 2);
    dim3 block_size(3, 4, 2);
    three_dim_block_and_thread<<<grid_size, block_size>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
