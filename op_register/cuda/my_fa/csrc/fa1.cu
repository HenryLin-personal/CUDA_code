#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <climits>
#include <cmath>
#include <optional>

// 空的 Python 模块: import my_fa._C 时加载这个 .so, 触发下面 TORCH_LIBRARY 的静态注册
extern "C" PyObject* PyInit__C(void) {
  static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL};
  return PyModule_Create(&def);
}

namespace my_fa {

constexpr int Br = 64;  // 每个 block 负责的 Q 行数 = block 内线程数

// 移植自 FlashAttention/fa1.cu 中当前生效的版本 (一个线程负责 Q 的一整行, K/V 直接读全局内存)
// 与原版的区别:
//   1. N 从模板参数改为运行时参数: 序列长度由输入张量决定, 不能再是编译期常量
//   2. D 仍是模板参数: q_reg[D], acc[D] 要放进寄存器, 由 host 端 switch 分发到 32/64/128
//   3. 支持 causal: 第 row 行只需要看 K 的前 row+1 行
//   4. 循环加了 #pragma unroll, 保证 q_reg/acc 的下标是常量, 不会落到 local memory
template <int Br, int D>
__global__ void flash_attn_v1(const float* __restrict__ q, const float* __restrict__ k,
                              const float* __restrict__ v, float* __restrict__ o,
                              int N, float scale, bool causal)
{
    const int bh = blockIdx.y;
    const float *Q = q + static_cast<size_t>(bh) * N * D;
    const float *K = k + static_cast<size_t>(bh) * N * D;
    const float *V = v + static_cast<size_t>(bh) * N * D;
    float *O = o + static_cast<size_t>(bh) * N * D;

    float q_reg[D];
    float acc[D];

    const int row = blockIdx.x * Br + threadIdx.x;  // 本线程负责的 Q 行, Br=blockDim.x
    if (row >= N) return;  // 这个 kernel 没有 __syncthreads, 越界线程可以直接返回
#pragma unroll
    for (int col = 0; col < D; ++col) {
        q_reg[col] = Q[static_cast<size_t>(row) * D + col] * scale;
        acc[col] = 0.f;
    }

    float m = -INFINITY;  // 到目前为止的最大值
    float l = 0.f;        // 以目前 m 为基准的指数和
    const int kv_end = causal ? row + 1 : N;
    for (int k_row = 0; k_row < kv_end; ++k_row)
    {
        float s = 0.f;  // S[row][k_row]
#pragma unroll
        for (int col = 0; col < D; ++col)
        {
            s += q_reg[col] * K[static_cast<size_t>(k_row) * D + col];
        }

        // online softmax + P * V
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float p = expf(s - m_new);
        l = l * alpha + p;
#pragma unroll
        for (int x = 0; x < D; ++x)
            acc[x] = alpha * acc[x] + p * V[static_cast<size_t>(k_row) * D + x];
        m = m_new;
    }
    const float inv = 1.f / l;
#pragma unroll
    for (int x = 0; x < D; ++x)
    {
        O[static_cast<size_t>(row) * D + x] = acc[x] * inv;
    }
}

// q/k/v: [B, H, N, d], 与 torch.nn.functional.scaled_dot_product_attention 的布局一致
at::Tensor fa1_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                    bool is_causal, std::optional<double> scale)
{
    TORCH_CHECK(q.dim() == 4, "fa1: q must be [B, H, N, d], got ", q.sizes());
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(),
                "fa1: q/k/v must have the same shape, got ", q.sizes(), ", ", k.sizes(), ", ", v.sizes());
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                v.scalar_type() == at::kFloat, "fa1: only float32 is supported");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "fa1: q/k/v must be on the same device");
    TORCH_INTERNAL_ASSERT(q.is_cuda());

    // 多卡时切到输入所在的卡, 否则 kernel 会在当前默认卡上启动
    const c10::cuda::CUDAGuard device_guard(q.device());

    // kernel 按 [BH, N, d] 连续布局计算地址, 转置/切片过的输入要先变连续
    const at::Tensor q_contig = q.contiguous();
    const at::Tensor k_contig = k.contiguous();
    const at::Tensor v_contig = v.contiguous();
    at::Tensor o = at::empty(q_contig.sizes(), q_contig.options());
    if (o.numel() == 0) return o;

    const int64_t BH = q.size(0) * q.size(1);
    const int64_t N = q.size(2);
    const int64_t D = q.size(3);
    TORCH_CHECK(BH <= 65535, "fa1: B * H must be <= 65535 (gridDim.y limit), got ", BH);
    TORCH_CHECK(N <= INT_MAX, "fa1: seqlen too large");

    const float s = scale.has_value() ? static_cast<float>(*scale)
                                      : 1.f / std::sqrt(static_cast<float>(D));
    const dim3 block(Br);
    const dim3 grid((N + Br - 1) / Br, BH);
    // 用 PyTorch 当前的 stream, 才能和前后的 torch 算子正确排序 (torch.compile / CUDA graph 下尤其重要)
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float* q_ptr = q_contig.data_ptr<float>();
    const float* k_ptr = k_contig.data_ptr<float>();
    const float* v_ptr = v_contig.data_ptr<float>();
    float* o_ptr = o.data_ptr<float>();
    switch (D) {
        case 32:
            flash_attn_v1<Br, 32><<<grid, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, N, s, is_causal);
            break;
        case 64:
            flash_attn_v1<Br, 64><<<grid, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, N, s, is_causal);
            break;
        case 128:
            flash_attn_v1<Br, 128><<<grid, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, N, s, is_causal);
            break;
        default:
            TORCH_CHECK(false, "fa1: head dim must be 32, 64 or 128, got ", D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

// schema 的参数名和默认值与 F.scaled_dot_product_attention 保持一致
// bool -> bool, float? -> std::optional<double>
TORCH_LIBRARY(my_fa, m) {
    m.def("fa1(Tensor q, Tensor k, Tensor v, bool is_causal=False, float? scale=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_fa, CUDA, m) {
    m.impl("fa1", &fa1_cuda);
}

} // namespace my_fa
