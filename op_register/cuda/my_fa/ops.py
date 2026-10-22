from typing import Optional

import torch
from torch import Tensor

__all__ = ["fa1"]


def fa1(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False,
        scale: Optional[float] = None) -> Tensor:
    """FlashAttention v1 前向, q/k/v: [B, H, N, d] float32 CUDA 张量"""
    return torch.ops.my_fa.fa1.default(q, k, v, is_causal, scale)


# FakeTensor kernel: 只描述输出的 shape/dtype/device, torch.compile 追踪时使用
# C++ 实现返回的是连续张量, 所以这里也要返回连续张量 (empty_like 会保留输入的 stride, opcheck 会报错)
@torch.library.register_fake("my_fa::fa1")
def _(q, k, v, is_causal=False, scale=None):
    torch._check(q.dim() == 4)
    torch._check(q.shape == k.shape and q.shape == v.shape)
    torch._check(q.dtype == torch.float32)
    return q.new_empty(q.shape)
