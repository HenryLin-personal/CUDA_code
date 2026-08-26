#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void koggeStoneScan(float *g_out, float *g_in, int n) {
  // Dynamic shared memory allocation
  extern __shared__ float T[];
  float *source = T;
  float *destination = T + n;

  int j = threadIdx.x;

  if (j < n) {
    source[j] = g_in[j];
  } else {
    source[j] = 0.0f;
  }

  for (int stride = 1; stride < n; stride *= 2) {
    __syncthreads(); // Ensure input from previous step (or initial load) is
                     // ready
    if (j >= stride) {
      destination[j] = source[j] + source[j - stride];
    } else {
      destination[j] = source[j];
    }
    // Swap pointers for the next iteration
    float *temp = source;
    source = destination;
    destination = temp;
  }

  if (j < n) {
    g_out[j] = source[j];
  }
}

int main() {
  const int n = 1024; // Must be power of 2
  const size_t size = n * sizeof(float);

  // Host memory
  std::vector<float> h_in(n);
  std::vector<float> h_out(n);
  std::vector<float> h_ref(n);

  // Initialize input and compute reference (CPU sequential scan)
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    h_in[i] = 1.0f; // Using 1.0 for easy verification
    sum += h_in[i];
    h_ref[i] = sum;
  }

  // Device memory
  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // Launch kernel with n threads in 1 block
  // We use dynamic shared memory: 2 * n * sizeof(float) for double buffering
  koggeStoneScan<<<1, n, 2 * size>>>(d_out, d_in, n);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

  printf("Kernel execution time: %f ms\n", milliseconds);
  printf("First 10 elements: ");
  for (int i = 0; i < 10 && i < n; i++) {
    printf("%.1f ", h_out[i]);
  }
  printf("\n");

  // Verify results
  bool success = true;
  for (int i = 0; i < n; i++) {
    if (std::abs(h_out[i] - h_ref[i]) > 1e-5) {
      printf("Error at index %d: Expected %f, Got %f\n", i, h_ref[i], h_out[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("Success! Kogge-Stone scan completed correctly for n=%d.\n", n);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}
