"""my_fa.fa1 与 torch scaled_dot_product_attention 的正确性 / 性能对比

torch 的 FLASH_ATTENTION 后端只支持 fp16/bf16, 而 fa1 是 fp32, 所以对比了几种后端:
  - fa1        fp32  本仓库的实现
  - MATH       fp32  显式算出 N x N 的 S 矩阵 (matmul + softmax + matmul), 最朴素的参考
  - EFFICIENT  fp32  xformers 的 memory-efficient attention, 同样是分块 + online softmax, 支持 fp32
  - FLASH      fp16  FlashAttention-2, 精度不同, 只能看速度和误差量级
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.library import opcheck

import my_fa

WARMUP, ITERS = 5, 20


def bench_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def reference(q, k, v, causal):
    """fp64 的朴素实现, 作为所有后端的误差基准"""
    q, k, v = q.double(), k.double(), v.double()
    s = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        n = q.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def candidates(q, k, v, causal):
    """(名字, 可调用对象) 列表; 每个可调用对象在各自的 sdpa_kernel 上下文里执行"""
    q16, k16, v16 = q.half(), k.half(), v.half()

    def sdpa(backend, *args):
        def run():
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(*args, is_causal=causal)
        return run

    return [
        ("fa1 fp32", lambda: my_fa.fa1(q, k, v, is_causal=causal)),
        ("MATH fp32", sdpa(SDPBackend.MATH, q, k, v)),
        ("EFFICIENT fp32", sdpa(SDPBackend.EFFICIENT_ATTENTION, q, k, v)),
        ("FLASH fp16", sdpa(SDPBackend.FLASH_ATTENTION, q16, k16, v16)),
    ]


def check_registration():
    """opcheck: 检查 schema / fake kernel / 注册是否写对 (不含 autograd, fa1 只有前向)"""
    q = torch.randn(1, 2, 100, 64, device="cuda")  # N=100 不是 Br=64 的倍数
    for causal in (False, True):
        opcheck(torch.ops.my_fa.fa1.default, (q, q.clone(), q.clone()), {"is_causal": causal})
    # 非连续输入: transpose 之后 stride 改变, C++ 里的 .contiguous() 要能处理
    qt = torch.randn(1, 100, 2, 64, device="cuda").transpose(1, 2)
    torch.testing.assert_close(my_fa.fa1(qt, qt, qt), reference(qt, qt, qt, False).float(),
                               atol=1e-4, rtol=1e-4)
    # torch.compile: 依赖 register_fake 才能 fullgraph 追踪
    compiled = torch.compile(lambda a: my_fa.fa1(a, a, a) * 2, fullgraph=True)
    torch.testing.assert_close(compiled(q), my_fa.fa1(q, q, q) * 2)
    print("opcheck / non-contiguous / torch.compile: OK\n")


def main():
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
    check_registration()

    B, H, d = 1, 8, 64
    header = f"{'N':>5} {'causal':>6} | {'backend':<15} {'ms':>9} {'TFLOPS':>7} {'max_err':>9} {'vs fa1':>7}"
    print(header)
    print("-" * len(header))
    for causal in (False, True):
        for N in (512, 1024, 2048, 4096):
            q, k, v = (torch.rand(B, H, N, d, device="cuda") * 2 - 1 for _ in range(3))
            ref = reference(q, k, v, causal)
            flops = 4 * B * H * N * N * d / (2 if causal else 1)
            fa1_ms = None
            for name, fn in candidates(q, k, v, causal):
                try:
                    err = (fn().double() - ref).abs().max().item()
                    ms = bench_ms(fn)
                except RuntimeError as e:
                    print(f"{N:>5} {causal!s:>6} | {name:<15} unsupported: {str(e).splitlines()[0][:60]}")
                    continue
                fa1_ms = fa1_ms or ms
                print(f"{N:>5} {causal!s:>6} | {name:<15} {ms:>9.4f} {flops / ms / 1e9:>7.3f} "
                      f"{err:>9.2e} {fa1_ms / ms:>6.1f}x")
            print()


if __name__ == "__main__":
    main()
