#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void workEfficientScan(float *g_out, float *g_in, int n) {
  // Dynamic shared memory allocation
  // Size should be n * sizeof(float) where n = 2 * blockDim.x
  extern __shared__ float T[];

  int t = threadIdx.x;
  int BLOCK_SIZE = blockDim.x; // Number of threads
  int m = 2 * BLOCK_SIZE;      // Number of elements processed by this block

  // Load input into shared memory (each thread loads 2 elements)
  if (2 * t < n)
    T[2 * t] = g_in[2 * t];
  else
    T[2 * t] = 0.0f;

  if (2 * t + 1 < n)
    T[2 * t + 1] = g_in[2 * t + 1];
  else
    T[2 * t + 1] = 0.0f;

  // Reduction Step (Up-sweep)
  int stride = 1;
  while (stride < m) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index < m && (index - stride) >= 0) {
      T[index] += T[index - stride];
    }
    stride *= 2;
  }

  // Post Scan Step (Down-sweep / Distribution Tree)
  // This part converts the partial sums into an inclusive scan
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < m) {
      T[index + stride] += T[index];
    }
    stride /= 2;
  }

  __syncthreads();
  // Write results back to global memory
  if (2 * t < n)
    g_out[2 * t] = T[2 * t];
  if (2 * t + 1 < n)
    g_out[2 * t + 1] = T[2 * t + 1];
}

int main() {
  const int n = 1024;
  const int threadsPerBlock = n / 2;
  const size_t size = n * sizeof(float);

  std::vector<float> h_in(n, 1.0f); // Initialize with 1.0 for easy verification
  std::vector<float> h_out(n);
  std::vector<float> h_ref(n);

  // CPU Reference
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += h_in[i];
    h_ref[i] = sum;
  }

  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);
  cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

  // Launch kernel: 1 block, n/2 threads, n*sizeof(float) shared memory
  workEfficientScan<<<1, threadsPerBlock, size>>>(d_out, d_in, n);

  cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

  bool success = true;
  for (int i = 0; i < n; i++) {
    if (std::abs(h_out[i] - h_ref[i]) > 1e-5) {
      printf("Error at index %d: Expected %f, Got %f\n", i, h_ref[i], h_out[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("Success! Work-efficient inclusive scan completed correctly for "
           "n=%d.\n",
           n);
    for (int i = 0; i < 10; ++i)
      printf("%.1f ", h_out[i]);
    printf("...\n");
  }

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}