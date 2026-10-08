#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>

namespace
{
    constexpr int batch = 1024;
    constexpr int size = 4096;
    constexpr int count = batch * size;
    constexpr float eps = 1.0f;
    constexpr int loops = 10;
}
void row_rmsnorm_f32_dim_cpu(float* in, float* weight, float* out, int batch, int size, float eps) {
  for (int i = 0; i < batch; ++i) {
    float* in_ptr = in + i * size;
    float* out_ptr = out + i * size;

    float sum = 0.0f;
    for (int j = 0; j < size; ++j) {
      float val = in_ptr[j];
      sum += val * val;
    }
    float rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);

    for (int j = 0; j < size; ++j) {
      float x = in_ptr[j] * weight[j];
      out_ptr[j] = x * rms;
    }
  }
}

int main()
{
    std::vector<float> input(count);
    std::vector<float> weights(size, 1.0f);
    std::vector<float> out(count);
    for(int i = 0; i < batch * size; ++i)
    {
        input[i] = i % 3; // data
    }

    row_rmsnorm_f32_dim_cpu(input.data(), weights.data(), out.data(), batch, size, eps);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < loops; ++i) row_rmsnorm_f32_dim_cpu(input.data(), weights.data(), out.data(), batch, size, eps);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapse = std::chrono::duration<double, std::milli>(end - start).count();
    printf("avg_ms :%.2fus\n", elapse * 1000 / loops);

    std::ofstream ofs("rms_ref.txt");
    for (float v : out) ofs << v << '\n';

    return 0;
}