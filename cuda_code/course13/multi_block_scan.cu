#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Kernel 1: Local scan and collect block sums
// Each block processes 2 * blockDim.x elements using Brent-Kung logic
__global__ void scan_and_collect_sums(float *g_out, float *g_in, float *g_sums,
                                      int n) {
  extern __shared__ float T[];
  int t = threadIdx.x;
  int b = blockIdx.x;
  int BLOCK_SIZE = blockDim.x;
  int m = 2 * BLOCK_SIZE;
  int global_idx_base = b * m;

  // Load input into shared memory
  if (global_idx_base + 2 * t < n)
    T[2 * t] = g_in[global_idx_base + 2 * t];
  else
    T[2 * t] = 0.0f;

  if (global_idx_base + 2 * t + 1 < n)
    T[2 * t + 1] = g_in[global_idx_base + 2 * t + 1];
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
  if (global_idx_base + 2 * t < n)
    g_out[global_idx_base + 2 * t] = T[2 * t];
  if (global_idx_base + 2 * t + 1 < n)
    g_out[global_idx_base + 2 * t + 1] = T[2 * t + 1];

  // Write the total sum of this block to the sums array
  if (t == BLOCK_SIZE - 1) {
    g_sums[b] = T[m - 1];
  }
}

// Kernel 2: Add scanned sums to elements of corresponding sections
__global__ void add_scanned_sums(float *g_out, float *g_sums, int n) {
  int b = blockIdx.x;
  if (b == 0)
    return; // First block is already correct

  int t = threadIdx.x;
  int m = 2 * blockDim.x;
  int global_idx_base = b * m;
  float add_val = g_sums[b - 1];

  if (global_idx_base + 2 * t < n)
    g_out[global_idx_base + 2 * t] += add_val;
  if (global_idx_base + 2 * t + 1 < n)
    g_out[global_idx_base + 2 * t + 1] += add_val;
}

// Reusing the single-block work-efficient scan for the sums array
__global__ void workEfficientScan(float *g_out, float *g_in, int n) {
  extern __shared__ float T[];
  int t = threadIdx.x;
  int BLOCK_SIZE = blockDim.x;
  int m = 2 * BLOCK_SIZE;
  if (2 * t < n)
    T[2 * t] = g_in[2 * t];
  else
    T[2 * t] = 0.0f;
  if (2 * t + 1 < n)
    T[2 * t + 1] = g_in[2 * t + 1];
  else
    T[2 * t + 1] = 0.0f;
  int stride = 1;
  while (stride < m) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index < m && (index - stride) >= 0)
      T[index] += T[index - stride];
    stride *= 2;
  }
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index + stride < m)
      T[index + stride] += T[index];
    stride /= 2;
  }
  __syncthreads();
  if (2 * t < n)
    g_out[2 * t] = T[2 * t];
  if (2 * t + 1 < n)
    g_out[2 * t + 1] = T[2 * t + 1];
}

int main() {
  const int n = 4096;
  const int threadsPerBlock = 512;
  const int elementsPerBlock = 2 * threadsPerBlock;
  const int numBlocks = (n + elementsPerBlock - 1) / elementsPerBlock;

  std::vector<float> h_in(n, 1.0f);
  std::vector<float> h_out(n);
  std::vector<float> h_ref(n);
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += h_in[i];
    h_ref[i] = sum;
  }

  float *d_in, *d_out, *d_sums;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMalloc(&d_sums, numBlocks * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  scan_and_collect_sums<<<numBlocks, threadsPerBlock,
                          elementsPerBlock * sizeof(float)>>>(d_out, d_in,
                                                              d_sums, n);
  workEfficientScan<<<1, threadsPerBlock, elementsPerBlock * sizeof(float)>>>(
      d_sums, d_sums, numBlocks);
  add_scanned_sums<<<numBlocks, threadsPerBlock>>>(d_out, d_sums, n);

  cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  bool success = true;
  for (int i = 0; i < n; i++) {
    if (std::abs(h_out[i] - h_ref[i]) > 1e-5) {
      printf("Error at index %d: Expected %f, Got %f\n", i, h_ref[i], h_out[i]);
      success = false;
      break;
    }
  }
  if (success)
    printf("Success! Multi-block scan completed correctly for n=%d.\n", n);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_sums);
  return 0;
}