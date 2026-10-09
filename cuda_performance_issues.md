# CUDA 常见性能问题总结

本文按问题类别整理，每类包括：**怎么产生的**、**典型场景**、**怎么排查**、**怎么修复**。

排查工具默认使用 Nsight Compute（`ncu`），命令示例：

```bash
ncu --set full -o report ./your_program
```

---

## 1. 访存不合并（Uncoalesced Global Memory Access）

### 怎么产生的
一个 warp 执行一条访存指令时，硬件会把 32 个线程的地址合并成若干个 **32B sector** 事务。地址越分散，需要的事务越多，实际搬运的字节数远大于有用的字节数，带宽被浪费。

理想情况：32 个线程 × 4B = 128B，正好 4 个 sector。

### 典型场景
- **跨步访问**：`a[tid * stride]`。比如每个线程处理一段连续数据，写成 `a[tid * 4 + k]`，同一时刻相邻线程的地址相差 16B。
- **按列访问行主序矩阵**：`A[tid * N + col]`，相邻线程的地址相差一整行。
- **地址不对齐**：起始地址不是 32B 的整数倍，一个 warp 的访问要多跨一个 sector。
- **AoS 结构体数组**：`struct { float x, y, z; } p[N];`，只读 `p[tid].x` 时有 2/3 的带宽是浪费的。
- **间接访问 / 随机访问**：`a[idx[tid]]`，地址完全由数据决定。

### 怎么排查
- `Memory Workload Analysis` → `sectors/request`：理想值是 4（float），明显偏大说明访问不合并。
- `L1/TEX Global Load/Store Efficiency`。
- Source 页面里的 `Uncoalesced Global Accesses` 提示，可以定位到具体代码行（编译时加 `-lineinfo`）。

### 怎么修复
- 让**相邻线程访问相邻地址**，用 grid-stride 循环：`for (i = tid; i < n; i += blockDim.x * gridDim.x)`。
- 必须按列访问时，先合并读入共享内存，在共享内存中转置后再使用。
- AoS 改为 SoA：`float x[N], y[N], z[N];`。
- 保证起始地址对齐（`cudaMalloc` 返回的地址是 256B 对齐的，注意偏移量）。

---

## 2. 共享内存存储体冲突（Shared Memory Bank Conflict）

### 怎么产生的
共享内存分成 **32 个 bank**，每个 bank 宽 4B，地址所在的 bank 是 `(addr / 4) % 32`。同一个 warp 中，多个线程访问**同一个 bank 的不同地址**时，这些访问会串行执行。n 路冲突，耗时就是原来的 n 倍。

- 多个线程访问**同一个地址**不算冲突，硬件会广播。

### 典型场景
- **按列访问二维共享数组**：`__shared__ float tile[32][32]; tile[tid][k]`。每行正好 32 个 float，同一列的元素全部落在同一个 bank，形成 32 路冲突。
- **跨步访问**：`smem[tid * 2]` 是 2 路冲突，`smem[tid * 32]` 是 32 路冲突。
- **跨步随迭代增大**：循环中访问 `smem[tid * 2 * s]`，s 越大冲突越严重。
- **64 位类型**：`double` 或 `float2` 等每个元素占两个 bank，跨步访问时更容易冲突。

### 怎么排查
- `Shared Memory` 一栏的 bank conflict 统计。
- 指标：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` / `..._op_st.sum`。
- Source 页面中的 `Shared Memory Bank Conflicts` 提示。

### 怎么修复
- **Padding**：`__shared__ float tile[32][33];`，每行多一个元素，让同一列错开到不同 bank。
- **Swizzle**：用 `col ^ row` 之类的 XOR 方式重排索引，不浪费空间。
- 改写索引，让**连续线程访问连续地址**。

---

## 3. 线程束分化（Warp Divergence）

### 怎么产生的
一个 warp 中的线程走进了不同的分支。硬件会依次执行每条分支，执行某条分支时，不走这条分支的线程被屏蔽，只是空等。分支越多、越不均衡，有效算力越低。

- 分化只发生在 **warp 内部**，不同 warp 走不同分支没有额外代价。

### 典型场景
- **分支条件依赖 `tid`，且在 warp 内取值不一致**：`if (tid % 2 == 0)`，每个 warp 都有一半线程空转。
- **活跃线程越来越稀疏**：`if (tid % (2 * s) == 0)`，随着 s 增大，每个 warp 里干活的线程越来越少，但每个 warp 仍要被调度。
- **数据相关的分支或循环次数**：`while (data[tid] > 0)`，同一个 warp 内的线程循环次数不同，要等最慢的那个线程。
- **边界判断**：`if (i < n)`，通常只有最后一个 warp 会分化，影响很小，一般可以忽略。

### 怎么排查
- `Warp State Statistics` / `Source Counters` 中的 `Branch Efficiency`、`Divergent Branches`。
- 指标：`smsp__thread_inst_executed_per_inst_executed.ratio`（平均每条指令的活跃线程数，理想值为 32）。

### 怎么修复
- 让分支条件**以 warp 为单位一致**：例如 `if (tid < s)`，在 s ≥ 32 时整个 warp 要么全部进入，要么全部跳过。
- 重新映射线程与数据，让需要干活的线程集中在前面几个 warp。
- 分支体很短时，用 `?:` 或算术方式替代（编译器会生成 select / predication）。
- 数据相关的工作量不均时，考虑先按工作量排序或分桶。

---

## 4. 同步开销过大（Excessive Synchronization）

### 怎么产生的
每次 `__syncthreads()` 都要让整个 block 等待最慢的那个 warp，期间 SM 上可调度的 warp 变少，延迟难以被掩盖。同步次数越多，block 越大，开销越明显。

### 典型场景
- 多轮迭代、每轮都需要 block 内同步的算法，后几轮只有很少线程在工作，仍然每轮全 block 同步。
- 一轮迭代中有多余的 `__syncthreads()`，前后其实没有跨线程的数据依赖（例如只读写了寄存器）。
- 全局层面频繁拆分 kernel、依赖 host 同步（`cudaDeviceSynchronize`）。

### 怎么排查
- `Warp State Statistics` 中 `Stall Barrier` 占比高。
- 对照代码，检查每次 `__syncthreads()` 前后是否真的有跨线程的共享内存读写。

### 怎么修复
- 当参与线程数 ≤ 32 时，改用 **warp 级原语**（`__shfl_down_sync`、`__shfl_xor_sync`、`__reduce_add_sync` 等），warp 内无需 `__syncthreads()`。
- 删除没有跨线程数据依赖的同步。
- 只需部分线程协作时，使用 Cooperative Groups 的细粒度分组同步。
- 减少不必要的 kernel 拆分和 host 端同步，必要时使用 CUDA Graph 降低启动开销。

---

## 5. 数据竞争（Race Condition，正确性问题）

### 怎么产生的
一个线程写共享内存或全局内存后，其他线程在**没有同步**的情况下读取，读到的可能是旧值或中间值。warp 内线程虽然通常同步执行，但从 Volta 开始支持独立线程调度，不能依赖隐式的 warp 同步。

### 典型场景
- 某个线程（或某个 warp）计算出结果并写入共享内存，其他 warp 立即读取，中间缺少 `__syncthreads()`。
- 循环中复用同一块共享内存，下一轮写入之前缺少同步，覆盖了其他线程还没读完的数据。
- warp 内通过共享内存交换数据，却没有 `__syncwarp()`。
- 多个 block 写同一个全局地址却没有使用原子操作。

### 怎么排查
```bash
compute-sanitizer --tool racecheck ./your_program
compute-sanitizer --tool synccheck ./your_program
```
- 现象：结果不稳定、部分元素错误、换 block 大小后结果变化。

### 怎么修复
- 跨线程读写共享内存之间加 `__syncthreads()`；仅在 warp 内交换时用 `__syncwarp()`。
- 复用共享内存缓冲区时，在覆盖写入之前也要同步一次。
- 同一线程先写后读自己的数据不需要同步。
- 跨 block 的数据依赖：使用原子操作，或拆成多个 kernel。

---

## 6. 占用率不足（Low Occupancy）

### 怎么产生的
占用率 = SM 上实际驻留的 warp 数 / SM 支持的最大 warp 数。GPU 依靠在 warp 之间快速切换来掩盖访存延迟，驻留的 warp 太少时，所有 warp 都在等数据，SM 处于空闲状态。限制因素有：
- 每个线程使用的**寄存器**太多；
- 每个 block 使用的**共享内存**太多；
- **block 太小**，触及每个 SM 最大驻留 block 数的上限；
- **grid 太小**，block 总数不足以填满所有 SM。

### 典型场景
- kernel 中大量局部变量、循环展开过度，寄存器用量很高。
- 为了分块计算开了很大的共享内存。
- block 只有 32 或 64 个线程。
- 问题规模小，launch 的 block 数少于 SM 数量。

### 怎么排查
- `Occupancy` 页面：`Theoretical Occupancy`、`Achieved Occupancy`，以及具体的限制因素（Block Limit Registers / Shared Mem / Warps）。
- 编译时查看资源用量：`nvcc -Xptxas -v`。
- CUDA Occupancy Calculator 或 `cudaOccupancyMaxPotentialBlockSize`。

### 怎么修复
- 用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` 或 `-maxrregcount` 限制寄存器数量（注意可能引起寄存器溢出）。
- 减少每个 block 的共享内存用量，或调整分块大小。
- block 大小一般选 128～512，并且是 32 的整数倍。
- 增大 grid，或让每个线程处理的数据更少以产生更多 block。
- 注意：占用率并非越高越好，一些依靠寄存器复用的 kernel 在较低占用率下反而更快，以实测为准。

---

## 7. 寄存器溢出（Register Spilling）

### 怎么产生的
寄存器不够用时，编译器会把部分变量放到 **local memory**。local memory 物理上位于 DRAM（经过 L1/L2 缓存），访问延迟远高于寄存器。

### 典型场景
- 局部数组使用**运行时下标**：`float buf[8]; buf[i] = ...;`（i 不是编译期常量），数组无法放进寄存器。
- 同时存活的变量太多，或循环展开过度。
- 用 `__launch_bounds__` / `-maxrregcount` 把寄存器限制得过紧。

### 怎么排查
- `nvcc -Xptxas -v` 输出中的 `spill stores` / `spill loads` 和 `stack frame`。
- Nsight Compute 中 `Memory Workload Analysis` 的 local memory 流量。

### 怎么修复
- 局部数组的下标改为编译期常量（配合 `#pragma unroll` 完全展开循环）。
- 缩短变量的生命周期，减少同时存活的变量。
- 适当放宽寄存器限制。
- 大的临时数据改放到共享内存。

---

## 8. 访存指令过多 / 未向量化（Too Many Memory Instructions）

### 怎么产生的
每次只读写 4B，搬运同样的数据量需要更多的访存指令，指令发射与 LSU 成为瓶颈，带宽跑不满。

### 典型场景
- 逐元素 `float` 读写的带宽受限 kernel。
- 数据本身是连续的，但每个线程只处理 1 个元素。

### 怎么排查
- `Instruction Statistics` 中 LD/ST 指令占比高。
- `Memory Workload Analysis` 中 DRAM 带宽利用率明显低于峰值，但 `Stall LG Throttle` / `Stall MIO Throttle` 较高。

### 怎么修复
- 使用 `float2` / `float4`（或 `int4`、`half2` 等）向量化访存，一条指令搬 8B / 16B。
- 前提：元素数量是向量宽度的整数倍（否则单独处理尾部元素），且地址按 8B / 16B 对齐。
- 适当增加每个线程处理的数据量（线程粗化，thread coarsening），减少索引计算和指令开销。

---

## 9. 重复读取全局内存（Redundant Global Memory Access）

### 怎么产生的
同一份数据被同一个线程或同一个 block 内的多个线程从全局内存反复读取。对带宽受限的 kernel，DRAM 流量直接决定耗时。

### 典型场景
- 多遍算法：第一遍读数据计算统计量，第二遍再读同一份数据计算结果。
- 相邻线程需要重叠的数据（如卷积、模板计算），每个线程各自从全局内存读取。
- 分块计算时，同一块数据被多个线程重复加载。

### 怎么排查
- 对比 `DRAM Bytes Read` 与理论最少读取量。
- `L2 Hit Rate`：命中率很高但 DRAM 流量仍大，或者 L1 / L2 请求数远多于数据量。

### 怎么修复
- 能放进寄存器的，读一次后留在寄存器里复用。
- block 内共享的数据先协作加载到**共享内存**再复用。
- 只读数据使用 `const __restrict__` 或 `__ldg()`，走只读缓存路径。
- 合并多个 kernel（kernel fusion），避免中间结果写回全局内存再读出。

---

## 10. 原子操作竞争（Atomic Contention）

### 怎么产生的
大量线程对**同一个地址**执行原子操作，这些操作会在 L2 上串行化，吞吐急剧下降。

### 典型场景
- 每个线程直接 `atomicAdd(&global_sum, val)`。
- 直方图统计中少数 bin 非常集中。

### 怎么排查
- `Memory Workload Analysis` 中原子操作吞吐、`Stall LG Throttle` 偏高。
- 对比：去掉原子操作后耗时变化很大。

### 怎么修复
- **分层规约**：先在 warp 内（shuffle）、再在 block 内（共享内存）规约，最后每个 block 只做一次全局原子操作。
- 共享内存中的私有化副本（例如每个 block 一份直方图），最后再合并。
- 对热点地址做分散（多个副本，最后合并）。

---

## 11. 尾部效应 / 负载不均衡（Tail Effect / Load Imbalance）

### 怎么产生的
- grid 的 block 数不是「SM 数 × 每 SM 驻留 block 数」的整数倍时，最后一波（wave）只有少数 SM 在工作，其他 SM 空闲。
- 不同 block 或不同线程的工作量差异大时，整体耗时由最慢的那个决定。

### 典型场景
- block 数量略多于一波能容纳的数量，例如一波能放 132 个 block，实际 launch 了 140 个。
- 稀疏数据、变长序列，每个 block 的工作量不同。

### 怎么排查
- `Launch Statistics` 中的 `Waves Per SM`：小数部分很小说明最后一波利用率低。
- Nsight Systems 时间线中 kernel 末尾的长尾。

### 怎么修复
- 调整 block 大小或每个线程的工作量，使 waves 接近整数，或者让波数足够多以摊薄尾部。
- 使用 **persistent kernel**：只 launch 刚好占满 GPU 的 block 数，在 kernel 内部循环取任务。
- 工作量不均时，按工作量排序或动态分配任务（原子计数器取任务）。

---

## 12. 主机-设备传输与启动开销（Host-Device Overhead）

### 怎么产生的
- PCIe 带宽远低于显存带宽，频繁的 `cudaMemcpy` 会成为瓶颈。
- 可分页内存（pageable memory）的拷贝需要额外的中转。
- 每次 kernel 启动有几微秒的固定开销，大量小 kernel 时尤为明显。

### 典型场景
- 循环中每次迭代都在 host 和 device 之间来回拷贝数据。
- 计算和拷贝串行执行，没有重叠。
- 一连串很小的 kernel。

### 怎么排查
- **Nsight Systems**（`nsys profile ./your_program`）时间线：观察 memcpy、kernel 之间的空隙和是否重叠。

### 怎么修复
- 数据尽量常驻显存，减少来回拷贝。
- 使用锁页内存（`cudaMallocHost` / `cudaHostAlloc`）。
- 使用多个 **stream** 配合 `cudaMemcpyAsync`，让拷贝与计算重叠。
- 合并小 kernel（kernel fusion），或使用 **CUDA Graph** 降低启动开销。

---

## 13. 计算相关问题（Compute Bound 时）

### 怎么产生的
kernel 受计算限制时，指令本身的开销成为瓶颈。

### 典型场景
- 不经意间使用了**双精度**：常量写成 `1.0` 而不是 `1.0f`，或调用 `sqrt()`、`exp()` 时参数被提升为 `double`。消费级 GPU 的 FP64 吞吐通常只有 FP32 的 1/32 甚至 1/64。
- 大量整数除法和取模：`/`、`%` 的除数不是 2 的幂时开销较高。
- 使用了慢速的精确数学函数，而实际精度要求不高。

### 怎么排查
- `Compute Workload Analysis` 中各管线利用率；`Instruction Statistics` 中是否出现 FP64 指令（`DADD`、`DMUL`、`DFMA`）。
- 查看 SASS（`cuobjdump -sass`），确认是否有意外的双精度转换。

### 怎么修复
- float 常量统一加 `f` 后缀，使用 `sqrtf`、`expf` 等单精度版本。
- 除数是 2 的幂时用移位和位运算；除数固定时预先计算倒数。
- 精度允许时使用快速内建函数：`__expf`、`__fdividef`、`rsqrtf`，或者编译选项 `--use_fast_math`。
- 矩阵类计算使用 Tensor Core（WMMA / MMA / cuBLAS / CUTLASS）。

---

## 附：通用排查流程

1. **先确定瓶颈类型**：Nsight Compute 的 `GPU Speed Of Light` 查看 Memory 与 Compute 吞吐占峰值的百分比，判断是访存受限、计算受限还是延迟受限（两者都低）。
2. **访存受限**：依次检查访存合并（第 1 节）、重复读取（第 9 节）、向量化（第 8 节）。
3. **计算受限**：检查指令组成（第 13 节）、分化（第 3 节）。
4. **延迟受限**：检查占用率（第 6 节）、同步与 stall 原因（第 4 节）、寄存器溢出（第 7 节）。
5. **结果不对时**：先用 `compute-sanitizer`（memcheck / racecheck / synccheck）排除越界和数据竞争（第 5 节）。
6. **整体流程慢**：用 Nsight Systems 看时间线，检查传输与启动开销（第 12 节）。

| 工具 / 选项 | 用途 |
|---|---|
| `ncu --set full` | 单个 kernel 的详细性能分析 |
| `nsys profile` | 整体时间线：kernel、memcpy、stream 重叠 |
| `compute-sanitizer` | 越界访问、数据竞争、同步错误 |
| `nvcc -Xptxas -v` | 寄存器、共享内存用量，寄存器溢出 |
| `nvcc -lineinfo` | 让 ncu 把指标对应到源代码行 |
| `cuobjdump -sass` | 查看实际生成的机器指令 |
