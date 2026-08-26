#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define TILE_WIDTH 16

// 卷积核函数
__global__ void ConvLayerForward_Kernel(int C, int H_in, int W_in, int K, int H_out, int W_out, int W_grid, float* X, float* W, float* Y) {
    // blockIdx.x 对应输出通道 m
    int m = blockIdx.x;
    // blockIdx.y 是线性化的 Tile 索引，还原出输出图中的起始坐标 (h, w)
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;

    // 边界检查：确保线程在输出图像范围内
    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) { // 遍历输入通道
            for (int p = 0; p < K; p++) { // 卷积核高度
                for (int q = 0; q < K; q++) { // 卷积核宽度
                    // 计算输入 X 的一维索引: [c, h+p, w+q]
                    int x_idx = c * (H_in * W_in) + (h + p) * W_in + (w + q);
                    // 计算权重 W 的一维索引: [m, c, p, q]
                    int w_idx = m * (C * K * K) + c * (K * K) + p * K + q;
                    acc += X[x_idx] * W[w_idx];
                }
            }
        }
        // 计算输出 Y 的一维索引: [m, h, w]
        int y_idx = m * (H_out * W_out) + h * W_out + w;
        Y[y_idx] = acc;
    }
}

// 主机端验证函数
void verify_result(float* h_X, float* h_W, float* h_Y, int M, int C, int H_in, int W_in, int K, int H_out, int W_out) {
    for (int m = 0; m < M; m++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                float acc = 0.0f;
                for (int c = 0; c < C; c++) {
                    for (int p = 0; p < K; p++) {
                        for (int q = 0; q < K; q++) {
                            acc += h_X[c * (H_in * W_in) + (h + p) * W_in + (w + q)] * 
                                   h_W[m * (C * K * K) + c * (K * K) + p * K + q];
                        }
                    }
                }
                if (std::abs(h_Y[m * (H_out * W_out) + h * W_out + w] - acc) > 1e-3) {
                    std::cout << "验证失败 at m=" << m << ", h=" << h << ", w=" << w << std::endl;
                    return;
                }
            }
        }
    }
    std::cout << "结果验证通过！" << std::endl;
}

int main() {
    // 定义维度
    int M = 16;      // 输出通道数
    int C = 3;       // 输入通道数
    int H_in = 32, W_in = 32; // 输入图像尺寸
    int K = 3;       // 卷积核尺寸
    int H_out = H_in - K + 1; // 输出高度 (无 padding, stride=1)
    int W_out = W_in - K + 1; // 输出宽度

    size_t size_X = C * H_in * W_in * sizeof(float);
    size_t size_W = M * C * K * K * sizeof(float);
    size_t size_Y = M * H_out * W_out * sizeof(float);

    // 1. 分配主机内存并初始化
    std::vector<float> h_X(C * H_in * W_in);
    std::vector<float> h_W(M * C * K * K);
    std::vector<float> h_Y(M * H_out * W_out);

    for (auto& val : h_X) val = static_cast<float>(rand()) / RAND_MAX;
    for (auto& val : h_W) val = static_cast<float>(rand()) / RAND_MAX;

    // 2. 分配设备内存
    float *d_X, *d_W, *d_Y;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_Y, size_Y);

    // 3. 拷贝数据到设备
    cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W.data(), size_W, cudaMemcpyHostToDevice);

    // 4. 配置执行参数
    // 计算网格
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y_grid_total = W_grid * H_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    // grid.x 映射到输出通道，grid.y 映射到图像内的分块
    dim3 gridDim(M, Y_grid_total);

    // 5. 启动核函数
    ConvLayerForward_Kernel<<<gridDim, blockDim>>>(C, H_in, W_in, K, H_out, W_out, W_grid, d_X, d_W, d_Y);

    // 6. 拷贝结果回主机
    cudaMemcpy(h_Y.data(), d_Y, size_Y, cudaMemcpyDeviceToHost);

    // 7. 验证结果
    verify_result(h_X.data(), h_W.data(), h_Y.data(), M, C, H_in, W_in, K, H_out, W_out);

    // 8. 释放资源
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);

    return 0;
}