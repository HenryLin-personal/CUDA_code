#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include "comm.cuh"

namespace
{
    constexpr int nx = 4096;
    constexpr int ny = 4096;
    constexpr size_t mat_elem_num = nx * ny;
    constexpr size_t mat_bytes = mat_elem_num * sizeof(float);
    constexpr int thread_x = 32;
    constexpr int thread_y = 16;
}// namespace

void call_transpose(float *out, float *in, int nx, int ny);
__global__ void transpose0(float *out, float *in, int nx, int ny)
{
    const int ix = threadIdx.x + blockDim.x * blockIdx.x;
    const int iy = threadIdx.y + blockDim.y * blockIdx.y;
    if(ix < nx && iy < ny) // 仅限于线程数 >= 数据元素数
    {
        out[iy * ny + ix] = in[ix * nx + iy];
    }
}

void call_transpose(float *out, float *in, int nx, int ny)
{
    dim3 block(thread_x, thread_y);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    transpose0<<<grid, block>>>(out, in, nx, ny);
}

int main()
{
    std::vector<float> h_in(mat_elem_num);
    std::vector<float> h_in_T(mat_elem_num);
    std::vector<float> h_out(mat_elem_num); 
    for (int i = 0; i < mat_elem_num; i++) {
        h_in[i] = i;
    }
    int a = 0;
    for(int j = 0; j < ny; ++j) {
        for(int i = 0; i < nx; ++i)
        {
            h_in_T[i * nx + j] = a++;
        }
    }


    auto d_in = comm::make_device_buffer<float>(mat_elem_num);
    auto d_out = comm::make_device_buffer<float>(mat_elem_num);
    comm::check_cuda(cudaMemcpy(d_in.get(), h_in.data(), mat_bytes, cudaMemcpyHostToDevice), "copy h_in to d_in");
    comm::check_cuda(cudaMemcpy(d_out.get(), h_out.data(), mat_bytes, cudaMemcpyHostToDevice), "copy h_out to d_out");
    
    comm::CudaEvent start, end;
    // comm::check_cuda(cudaEventRecord(start.get()), "cudaEventRecord(start)");
    // warmup
    call_transpose(d_out.get(), d_in.get(), nx, ny);
    // comm::check_cuda(cudaEventRecord(end.get()), "cudaEventRecord(end)");
    // comm::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // loop 
    float total_ms = 0.f;
    comm::check_cuda(cudaEventRecord(start.get()), "cudaEventRecord(start)");
    for(int i = 0; i < 100; i++)
    {  
        call_transpose(d_out.get(), d_in.get(), nx, ny);
    }
    comm::check_cuda(cudaEventRecord(end.get()), "cudaEventRecord(end)");
    comm::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    comm::check_cuda(cudaEventElapsedTime(&total_ms, start.get(), end.get()), "cudaEventElapsedTime");
    printf("avg_ms :%fms\n", total_ms / 100.f);

    comm::check_cuda(cudaMemcpy(h_out.data(), d_out.get(), mat_bytes, cudaMemcpyDeviceToHost), "copy d_out to h_out");
    comm::verify_results(h_in_T, h_out, nx);
    return 0;
}