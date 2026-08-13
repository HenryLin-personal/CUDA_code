# CUDA Reduction 优化实践：从 v0 到 v4

本目录用多个 CUDA kernel 实现同一个目标：对 `N = 1024^2` 个 `float` 求和。版本从最直接的共享内存归约开始，逐步引入线程粗化、连续寻址、warp shuffle、grid-stride loop 和两级 warp 归约。

这里的重点不是让每个版本都成为最终实现，而是观察每一步解决了什么瓶颈，以及优化之间为什么不能简单叠加。

## 版本总览

| 版本 | 一阶段启动配置 | 每线程初始处理量 | 块内归约核心 | 本阶段的核心优化 |
|---|---:|---:|---|---|
| `v0` | `<<<1024, 1024>>>` | 1 个元素 | 交错线程、`tid % (2*s)` | 正确性基线 |
| `v1` | `<<<512, 1024>>>` | 2 个元素 | 与 v0 相同 | 首次加载时在寄存器中合并两个元素 |
| `v1_coarsening` | `<<<256, 1024>>>` | 4 个元素 | 与 v0 相同 | 将固定线程粗化扩展到 4 个元素 |
| `v2` | `<<<512, 1024>>>` | 2 个元素 | 连续线程、`tid < s` | 去掉取模并基本消除广泛的 warp 内分化 |
| `v2_coarsening` | `<<<256, 1024>>>` | 4 个元素 | 与 v2 相同 | v2 与 4 元素线程粗化组合 |
| `v3` | `<<<512, 1024>>>` | 2 个元素 | shared memory 归约到 64，再用 shuffle | 省去归约尾段的块级同步和共享内存往返 |
| `v3_coarsening` | `<<<256, 1024>>>` | 4 个元素 | 与 v3 相同 | v3 与 4 元素线程粗化组合 |
| `v4` | `<<<512, 1024>>>` | 当前为 2 个，运行时可变 | warp 内归约，再归约各 warp 的结果 | grid-stride loop + 两级 warp reduction |

表中的线程粗化只描述当前 `N` 和启动配置。v4 的 grid-stride loop 不把每线程处理量写死，因此可以自然适应不同的 `n`、block 数量和多轮跨步读取。

## v0：共享内存交错归约基线

v0 让每个线程先读取一个全局内存元素，再执行：

```cpp
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

它的问题主要有三个：

- `tid % (2*s)` 在循环中反复计算；在未充分优化或 Debug 构建中可能生成代价很高的动态取模代码。Release 构建是否将其化简，应以实际 PTX/SASS 为准。
- 活跃线程以交错方式分布。以 `s=1` 为例，一个 warp 只有偶数 lane 工作，因此所有 warp 都存在源代码层面的分歧条件和低 lane 利用率；编译器可能将短分支谓词化，是否生成真实分支应以 SASS 为准。
- 对 1024 线程的 block，需要 1 次加载后同步和 10 次归约同步。

这个访问模式在 32 个 bank、4 字节 `float` 的常见配置下没有 shared-memory bank conflict。bank conflict 必须按“同一 warp 的同一条共享内存指令”判断；不同 warp 同时命中 bank 0 不构成冲突。

## v1：每线程先合并两个元素

v1 在写入共享内存前先做一次局部求和：

```cpp
unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
sdata[tid] = inp[i] + inp[i + blockDim.x];
```

这次加法的中间值通常保存在寄存器中。一个 1024 线程 block 因而消费 2048 个输入，一阶段 block 数量和 partial sum 数量都从 1024 降到 512，第二阶段的输入也随之减半。

它没有减少读取全部输入所需的总字节数，也没有改变块内的 v0 交错归约，所以取模、广泛的 warp 内分化和 10 轮归约同步仍然存在。它优化的是 block 调度、partial sum 数量和第二阶段工作量。

## v1_coarsening：每线程处理四个元素

这一版继续把固定粗化因子提高到 4：

```cpp
sdata[tid] = inp[i]
           + inp[i + blockDim.x]
           + inp[i + 2 * blockDim.x]
           + inp[i + 3 * blockDim.x];
```

一阶段只需 256 个 block，partial sum 数量变为 v0 的四分之一。连续 warp 的每一轮读取仍然是连续地址，因此可以保持合并访问。

粗化并非越大越好。它不减少总输入流量和必要的加法次数；粗化过度会减少并行 block、降低隐藏内存延迟的能力，并可能增加寄存器压力。

## v2：连续寻址归约

v2 将归约方向反转，让前半段连续线程工作：

```cpp
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

相较 v1，它带来两点关键变化：

- 去掉循环中的 `%` 取模。
- 当 `s >= 32` 时，warp 以完整 warp 为单位启用或停用，不再让每个 warp 都只执行交错 lane。

当 `s < 32` 时，warp 0 仍只有部分 lane 活跃，所以不能说它彻底消除了分化；更准确的说法是，它基本消除了 v0/v1 那种遍布所有 warp 的交错分化。共享内存访问也保持无 bank conflict。

`v2_coarsening` 则把同样的连续寻址归约与每线程 4 个输入组合起来。

## v3：使用 warp shuffle 完成归约尾段

v3 的共享内存循环只执行到还剩 64 个 partial sum。下面用概念等价代码展示该过程；当前源码将尾段封装为 `warpReduce(sdata, tid)`：

```cpp
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

if (tid < 32) {
    float val = sdata[tid] + sdata[tid + 32];
    val = warp_reduce(val);
}
```

最后一个 warp 先把 64 个值合并成 32 个，再通过 `__shfl_down_sync` 直接交换 lane 的寄存器值。相较 v2，它省去了 `s=32,16,8,4,2,1` 六轮块级同步及对应的共享内存读写。

这里的局部变量必须是 `float`；如果写成 `int`，非整数输入会在转换时被截断，而全 1 输入可能把这个错误掩盖掉。

`v3_coarsening` 再叠加每线程 4 个输入。不过实测中它与 v3 基本持平，说明当固定开销已经较低时，继续减少 block 未必能带来收益。

## v4：grid-stride loop 与两级 warp reduction

v4 首先用 grid-stride loop 累加每个线程负责的输入：

```cpp
float val = 0.0f;
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n;
     i += gridDim.x * blockDim.x) {
    val += inp[i];
}
```

它可以看作固定 coarsening 的通用写法。当前一阶段为 `<<<512, 1024>>>`，总线程数是 524288，所以每个线程恰好处理两个元素；如果 `n` 或 grid 大小改变，同一段代码仍可通过更多或更少的循环迭代覆盖输入。

随后块内采用两级归约：

1. 每个 warp 用 shuffle 得到一个局部和，由 lane 0 写入 `warpSum[warp_id]`。
2. `__syncthreads()` 保证所有 warp 的结果已经写入。
3. warp 0 的 `lane < warp_count` 线程读取 `warpSum[lane]`，其余 lane 补 0，再执行一次 warp reduction。
4. block 的 thread 0 写出最终 partial sum。

一个 CUDA block 最多有 32 个 warp，所以只需 `warpSum[32]`，即 128 字节共享内存。完整块内归约只需要 1 次 `__syncthreads()`。当前第一阶段产生 512 个 partial sum，第二阶段用 `<<<1, 256>>>` 处理这 512 个值，两个阶段的映射是匹配的。

当前 `warp_reduce` 使用 `FULL_MASK`，因此现有的 1024 和 256 线程配置是安全的；若要支持非 32 倍数的 `blockDim.x`，还需要为尾部不完整 warp 正确构造参与掩码，不能直接宣称支持任意 block 大小。

## 性能对比

测试条件：RTX 3060 Laptop GPU，`N = 1,048,576`，统一使用 `nvcc -O3 -arch=sm_86 -lineinfo` 编译；GPU 计时包含两阶段 kernel。8 个版本交错运行 7 轮，表中取各程序所报告 GPU 平均耗时的中位数。

| 版本 | GPU 中位耗时 (ms) | 相对 v0 加速比 | 相对 v0 耗时降低 | 相对对应前序版本 |
|---|---:|---:|---:|---:|
| v0 | 0.165371 | 1.000x | 0.0% | 基线 |
| v1 | 0.092652 | 1.785x | 44.0% | 比 v0 降低 44.0% |
| v1 coarsening | 0.054522 | 3.033x | 67.0% | 比 v1 降低 41.2% |
| v2 | 0.061296 | 2.698x | 62.9% | 比 v1 降低 33.8% |
| v2 coarsening | 0.046422 | 3.562x | 71.9% | 比 v2 降低 24.3% |
| v3 | 0.045373 | 3.645x | 72.6% | 比 v2 降低 26.0% |
| v3 coarsening | 0.045568 | 3.629x | 72.4% | 比 v3 慢约 0.4%，基本持平 |
| v4 | 0.040637 | 4.069x | 75.4% | 比 v3 降低 10.4% |

这些数字用于比较本机当前实现，不应直接推广到其他 GPU。归约是内存与同步都很敏感的操作，GPU 架构、频率状态、编译选项、输入规模和 launch 配置都可能改变版本排序。粗化版本尤其容易受 block 并行度影响，因此应优先比较多轮交错测试的中位数，而不是单次结果。

## 同步次数对比

以一阶段的完整 1024 线程 block 为例：

| 版本族 | 每个 block 的 `__syncthreads()` 次数 | 原因 |
|---|---:|---|
| v0 / v1 / v1 coarsening | 11 | 加载后 1 次，加上 10 轮归约 |
| v2 / v2 coarsening | 11 | 改善寻址和分化，但仍保留 10 轮归约同步 |
| v3 / v3 coarsening | 5 | 加载后 1 次，共享内存只执行 `s=512,256,128,64` |
| v4 | 1 | 只同步一次各 warp 写出的局部和 |

同步次数能解释部分趋势，但不能直接换算为运行时间；全局内存流量、指令数、occupancy、寄存器压力和 block 数量也同时参与决定性能。

## 构建与运行

当前 `Makefile` 使用 `-O3`。修改源码或编译选项后可用 `make -B` 强制重建，避免误跑旧二进制：

```bash
cd /home/lhl/code/test_CUDA/Reduction
make -B

./reduction_v0
./reduction_v1
./reduction_v1_coarsening
./reduction_v2
./reduction_v2_coarsening
./reduction_v3
./reduction_v3_coarsening
./reduction_v4
```

严格复现上表时，还需要把 Makefile 改为统一的 Release 配置，或逐个目标使用相同参数重新编译，例如：

```bash
nvcc -O3 -arch=sm_86 -lineinfo ...
```

`sm_86` 对应当前测试 GPU；换到其他 GPU 时应设置匹配的架构。

## 正确性与适用范围

- 当前固定 `N = 1024^2` 和表中 launch 配置下，v0～v4 以及 coarsening 版本均能得到正确结果。
- v0～v3 及其 coarsening 版本的 kernel 没有通用的 `n` 边界保护，当前正确依赖输入规模恰好整除。改变 `N` 或启动配置时必须重新计算 block 数、partial sum 数和第二阶段线程数。
- v2 要求参与共享内存树形归约的 block 大小符合其二次幂假设；v3 还要求 block 足够大，才能按当前方式留下 64 个 partial sum。
- v4 的全局读取已通过 `i < n` 支持尾部输入，但当前 `FULL_MASK` 归约仍要求完整 warp。
- `comm.cuh` 当前把输入全部设为 `1.0f`，适合冒烟测试，却可能掩盖漏读、重复读取、整数截断等错误。更强的测试可以先使用固定种子的 `[0,3]` 随机小整数做精确比较，再用随机小数、CPU `double` 参考值和绝对/相对混合容差测试数值误差。
- 验证函数还应先检查结果是否为有限值，避免 `NaN` 绕过单纯的 `fabs(error) > tolerance` 判断。

## 结论

这组实验展示了一条清晰的优化路径：

```text
v0 交错归约
  -> v1/v1_coarsening：在线程内先合并更多输入
  -> v2：连续寻址，去掉取模并减少广泛分化
  -> v3：用 warp shuffle 消除归约尾段的块级同步
  -> v4：用 grid-stride loop 通用化粗化，并用两级 warp reduction 缩小共享内存和同步开销
```

本机测试中 v4 相对 v0 达到约 `4.07x` 加速、耗时降低约 `75.4%`。更重要的是，每一阶段都针对一个可识别的瓶颈；是否继续粗化或调整 grid 大小，应以目标 GPU 上的 Release 基准和 profiler 数据为准。
