# Kimi-K3 Triton Ascend A/B reproducer

这个目录把两个容易混淆的问题拆开验证：

1. `forced-autotune` 直接调用 `cumsum` 和 KDA inter/intra 的
   `triton.autotune` kernel，判断 TA 3.2.1 与 3.2.2 在相同算子源码下，
   首次编译、候选 benchmark 或 `torch_npu.synchronize()` 是否有差异。
2. `production` 调用 SGLang 公开 wrapper，判断 NPU 固定 config 的改动是否
   避开候选 benchmark，同时保持数值正确。

四节点模式按 K3 的 64 rank 拆出 16 个跨节点 DP4 group：

```text
local slot 0:  [0, 16, 32, 48]
local slot 1:  [1, 17, 33, 49]
...
local slot 15: [15, 31, 47, 63]
```

因此某个 rank 在 Triton JIT/autotune 落后时，同组其他 rank 会在
`all_gather` 等待。日志里的 phase marker 和定时 faulthandler stack 可以区分
“算子仍在 autotune”和“collective 自己发生错误”。

## 必须先对齐的变量

- 两个容器挂载完全相同的 SGLang 源码；不要分别 `git pull`。
- 四台机器的 `cumsum.py`、`kda.py` SHA256 必须相同。
- 模型不参与这个用例，因此不会受权重加载和显存占用影响。
- 冷缓存 A/B 使用不同目录，但两个目录都必须从不存在开始。
- warm 阶段分别复用各自 cold 阶段生成的缓存，不允许跨 TA 版本共用缓存。
- TA 3.2.1/3.2.2 的 Python ABI 必须与容器匹配。现有 cp312 的 3.2.2 wheel
  不能直接装入 Python 3.11 的 3.2.1 镜像做纯 TA 单变量实验。

`run_case.sh` 会记录源码哈希、commit、dirty status、Python/Torch/TA 版本和
关键环境变量。`summarize_ab.py` 在源码哈希不一致时以退出码 2 拒绝比较。

## 1. 单卡、单算子 TA 对比

先将同一份源码复制或只读挂载到两个镜像的同一路径，例如：

```text
/home/wzy/sglang-ta-ab
```

TA 3.2.1 容器中执行 cold：

```bash
NODE_RANK=0 \
NNODES=1 \
NPROC_PER_NODE=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=30221 \
CASE_NAME=ta321-same-source \
PHASE=cold \
SGLANG_SOURCE=/home/wzy/sglang-ta-ab \
CACHE_ROOT=/tmp/k3-ta-ab/ta321-same-source \
RESULT_ROOT=/home/wzy/k3-ta-ab-results \
LAUNCH_PATH=forced-autotune \
OP=both \
LAYOUT=varlen \
bash /home/wzy/kernel-regression-repros/k3_ta_autotune_dp/run_case.sh
```

使用完全相同的命令将 `CASE_NAME` 和 `CACHE_ROOT` 改为
`ta322-same-source`，在 TA 3.2.2 容器运行。重点比较：

- `cumsum_varlen_first`；
- `kda_varlen_first`；
- `candidate_count`，预期 cumsum 为 18、KDA inter 为 24、intra 为 4；
- 日志是否停在 `triton.autotuner._bench/do_bench/torch_npu.synchronize`；
- 数值误差和 `correct`。

然后把 `PHASE=warm` 原样重启。warm 复用已编译 kernel，但新 Python 进程的
autotune 内存 cache 为空，因此仍能验证“二进制已缓存、候选 benchmark 重跑”
是否会挂。

### 单机 16-rank 冷缓存编译变慢 A/B

要复现 TA 3.2.2 相对 3.2.1 的 KDA 首次编译变慢，推荐在同一个 CANN 容器里
创建两个 Python ABI 相同的 venv，只切换 TA 安装。然后直接运行：

```bash
# 两个 venv 必须使用相同 Python、torch 和 torch-npu。右侧安装待测的修复版
# wheel，不要用 PyPI/镜像里另一份同名 3.2.2 代替。
TA321_VENV=/path/to/ta321-venv
TA322_VENV=/path/to/ta322-fixed-venv
${TA321_VENV}/bin/python3 -m pip install --force-reinstall /path/to/ta-3.2.1.whl
${TA322_VENV}/bin/python3 -m pip install --force-reinstall \
  /path/to/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

本次已验证的修复版 wheel SHA256 为
`bb02d9eb2181172f783aefdb2b4505fb7c52bf93c506d43290bac58b2b3f4e7e`；
运行其他构建时应把实际 wheel 和 `compiler.py` 哈希一起保存在结果中。

```bash
SGLANG_SOURCE=/home/wzy/sglang-main-5f216fc-cann91-ta322 \
TA321_VENV=/home/wzy/venvs/ta321-official-3e7b638e-py312-20260829a \
TA322_VENV=/home/wzy/venvs/ta-fixed-bb02d9eb-py312-20260829a \
RUNS=3 \
RUN_ID=k3-ta-compile-$(date +%Y%m%d-%H%M%S) \
RESULT_ROOT=/home/wzy/k3-ta-compile-ab-results \
CACHE_ROOT=/tmp/k3-ta-compile-ab \
bash k3_ta_autotune_dp/run_compile_slowdown_ab_16rank.sh
```

脚本每轮启动单机 16 rank，使用 `forced-autotune`、`OP=both`、`varlen` 和
`CACHE_LAYOUT=per-node`；每个 TA/轮次生成全新的 cache root，冷跑目录已存在时
直接拒绝运行。奇数轮先跑 3.2.1，偶数轮先跑 3.2.2，降低固定运行顺序带来的
温度和系统负载偏差。不要设置 `ASCEND_LAUNCH_BLOCKING=1`，也不要复用另一版本
或上一轮的 `/tmp` cache。脚本会拒绝 Python ABI、torch 或 torch-npu 不一致的
A/B；CANN 一致性由“在同一个容器中运行两个 venv”保证。

每轮结束会在以下目录生成机器可读的 A/B 汇总：

```text
${RESULT_ROOT}/ab-summaries/${RUN_ID}/roundN.json
```

重点看：

- `right_vs_left_median_slowdown_percent.first_call_total_seconds`：cumsum 与 KDA
  合计冷启动回退；实测三轮通常约 `+15%`；
- `right_vs_left_median_slowdown_percent.kda_first_seconds`：KDA inter/intra
  合计回退；实测约 `+23%`；
- `autotune_log_seconds._chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`：
  24 个 inter candidate 的总 autotune 时间；3.2.1 约 70 秒，3.2.2 修复版
  约 90 秒，是主要差异；
- `_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra` 和
  `_chunk_local_cumsum_vector_kernel` 通常接近；
- `dp_all_gather_seconds` 两边应接近，说明 collective 是等待者而不是回退源头；
- `sources_aligned` 必须是 `true`，16/16 rank 必须完整且 `all_correct=true`。

这个用例证明的是同节点 16 个 TA 编译进程竞争时的冷缓存编译性能回退，不证明
kernel 热态执行变慢，也不等同于通用 `alloc_extend` 的 BiShengIR UB overflow。

## 2. 四节点整网模拟

四台机器分别设置 `NODE_RANK=0/1/2/3`，其余参数完全相同，并发启动：

```bash
NODE_RANK=<0|1|2|3> \
NNODES=4 \
NPROC_PER_NODE=16 \
MASTER_ADDR=192.168.25.209 \
MASTER_PORT=30221 \
CASE_NAME=ta322-network-cold \
PHASE=cold \
SGLANG_SOURCE=/home/wzy/sglang-ta-ab \
CACHE_ROOT=/tmp/k3-ta-ab/ta322-network-cold \
RESULT_ROOT=/home/wzy/k3-ta-ab-results \
CACHE_LAYOUT=per-node \
LAUNCH_PATH=forced-autotune \
OP=both \
LAYOUT=varlen \
TIMEOUT_SECONDS=1200 \
bash /home/wzy/kernel-regression-repros/k3_ta_autotune_dp/run_case.sh
```

`CACHE_LAYOUT=per-node` 很重要：同一节点的 16 个 worker 共用一个 Triton
缓存目录，模拟 SGLang 服务的真实继承方式；四台机器之间仍是四份独立缓存。

再跑一轮 `CACHE_LAYOUT=per-rank`。如果 per-node 明显容易挂、per-rank 稳定，
问题更偏向同节点 16 进程的 cache lock/launcher build 竞争；如果两者都在相同
`do_bench/synchronize` 位置挂，更偏向 TA benchmark/异步 launch 本身。

## 3. 验证“all-gather 是受害者”的控制用例

以下参数人为让 node 2 的 16 个 rank 在 collective 前慢 30 秒：

```bash
INJECT_DELAY_NODE_RANK=2 \
INJECT_DELAY_SECONDS=30 \
...其余四节点参数... \
bash /home/wzy/kernel-regression-repros/k3_ta_autotune_dp/run_case.sh
```

预期结果：node 0/1/3 的所有 DP group 都在 `dp_all_gather` 多等待约 30 秒，
但 node 2 的日志显示它仍处于 injected delay。这一控制只用于证明堆栈解释，
不能作为 TA 回归证据。真实 TA A/B 必须不设置任何 `INJECT_DELAY_*`。

## 4. 验证生产修改

在未修改与固定 config 的源码上分别使用：

```bash
LAUNCH_PATH=production
```

预期固定 config 版本：

- 数值仍通过；
- production 首次调用不再遍历 18/24/4 个候选；
- `forced-autotune` 仍能直接暴露 TA 版本自己的候选 benchmark 行为；
- 四节点 first-call skew 和 DP all-gather 等待显著下降。

## 5. 汇总结果

先将四台机器的 `node0` 到 `node3` 结果目录收集到同一个目录，再执行：

```bash
python3 /home/wzy/kernel-regression-repros/k3_ta_autotune_dp/summarize_ab.py \
  /home/wzy/k3-ta-ab-collected/ta321-same-source/cold \
  /home/wzy/k3-ta-ab-collected/ta322-same-source/cold
```

退出码含义：

- `0`：源码一致、rank 完整、两边数值均正确；性能和 skew 见 JSON；
- `2`：两边或各 rank 的算子源码哈希不一致，拒绝比较；
- `3`：缺少 rank 结果，通常是 timeout/crash；
- `4`：至少一边发生数值错误。

主 A/B 不应设置 `ASCEND_LAUNCH_BLOCKING=1`，因为它会改变 benchmark 时序。
只有发生 timeout 后，才单独用 blocking 重跑单卡用例，结合 180 秒一次的
faulthandler stack 定位具体停在 cumsum、KDA inter、KDA intra 还是 collective。

## 6. `alloc_extend` 64-rank 编译定位

`alloc_extend` 是普通 `@triton.jit` kernel，不存在 autotune candidate。它与
KDA/cumsum 的 autotune 是两条独立路径。服务进程的现场栈若停在
`alloc_extend_kernel -> linalg_to_bin_enable_npu_compile_A2_A3`，应使用本节的
独立用例，而不是用固定 KDA config 推断它是否恢复。

用例在四节点各启动 16 个 rank，依次记录：

```text
before_compile -> after_compile -> after_launch -> after_sync
               -> before_dp_all_gather -> after_dp_all_gather
```

`exact-dynamic` 原样调用同一 SGLang checkout 的生产 kernel；`static-bound` 只把
Part 2 的运行时动态循环恢复成 PR #19898 之前的 constexpr 上界。K3 现场参数是
`page_size=128`、单请求 `bs_upper=1`。`pre_lens/seq_lens` 的值不是编译 key，默认
70 token 足以触发相同编译；另跑 4096/16384 token 用来覆盖动态循环的设备执行。

每台机器分别运行，`NODE_RANK` 使用 0/1/2/3：

```bash
NODE_RANK=<0|1|2|3> \
CASE_NAME=ta322-dynamic-p128-bs1-cold \
PHASE=cold \
VARIANT=exact-dynamic \
SGLANG_SOURCE=/home/wzy/sglang-ta-ab \
CACHE_ROOT=/tmp/alloc-extend-network/ta322-dynamic-p128-bs1-cold \
RESULT_ROOT=/home/wzy/alloc-extend-network-results \
MASTER_ADDR=192.168.25.209 \
MASTER_PORT=30231 \
bash /home/wzy/kernel-regression-repros/k3_ta_autotune_dp/run_alloc_extend_network.sh
```

冷跑不会删除 cache，而是在 `${CACHE_ROOT}/nodeN` 已存在时直接退出码 3，结果目录
已存在也拒绝覆盖。这样日志中的 `node_cache_preexisting=false` 和生成后的
`cache_manifest.tsv` 可以证明没有复用前一轮二进制。每个 TA 版本、variant、
cache layout 都必须使用全新的 `CASE_NAME`、`CACHE_ROOT` 和 `MASTER_PORT`。

建议矩阵：

| CANN/torch-npu | TA | variant | cache layout | 输入 |
|---|---|---|---|---|
| 完全相同 | 3.2.1 | exact-dynamic | per-node | bs1/page128/70 |
| 完全相同 | 3.2.2 | exact-dynamic | per-node | bs1/page128/70 |
| 完全相同 | 3.2.2 | exact-dynamic | per-rank | bs1/page128/70 |
| 完全相同 | 3.2.2 | static-bound | per-node | bs1/page128/70 |
| 完全相同 | 3.2.2 | exact-dynamic | per-node | bs1/page128/16384 |

判定：

- 只随 TA 3.2.1/3.2.2 变化，且停在 `before_compile`：TA 编译回归；
- 3.2.2 的 per-node 挂而 per-rank 通过：同节点 16 进程共享 Triton cache 的
  lock/launcher build 竞争；
- 3.2.2 只有 `exact-dynamic` 挂而 `static-bound` 通过：SGLang allocator 应在
  NPU 避免运行时动态 loop，最小修改点为 `alloc_extend_kernel`；
- `after_compile` 有而 `after_sync` 没有：不是编译，而是 kernel 执行/设备同步；
- 两个单算子 variant 均通过：`alloc_extend` 的一次 py-spy 采样只是当时正在编译，
  继续在框架首请求的 rank 调度顺序、cache 共享和 DP batch metadata 对齐处定位。

把四个 `nodeN` 目录收集到同一个 case/phase 目录后，可直接汇总多轮 A/B：

```bash
python3 k3_ta_autotune_dp/summarize_alloc_extend_network.py \
  /collected/ta321-dynamic/cold \
  /collected/ta322-dynamic/cold \
  /collected/ta322-static/cold
```

汇总器会拒绝比较 allocator SHA256 或输入 shape 不一致的结果，并列出每轮 64 个
rank 的最后事件、未完成 rank、编译耗时 min/median/max 和数值正确性。

### 单机 16-rank 收缩用例

定位 TA 编译、动态 loop lowering 和节点内共享 Triton cache 竞争时，不需要先跑
四机。因为编译器进程和 `/tmp/TRITON_CACHE_DIR` 都是节点本地资源，单机同时启动
16 个 rank 已能保留完整服务在每个节点上的主要并发条件：

```bash
NODE_RANK=0 \
NNODES=1 \
NPROC_PER_NODE=16 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=30231 \
CASE_NAME=ta322-dynamic-single-node-cold \
PHASE=cold \
VARIANT=exact-dynamic \
SGLANG_SOURCE=/home/wzy/sglang-ta-ab \
CACHE_ROOT=/tmp/alloc-extend-network/ta322-dynamic-single-node-cold \
RESULT_ROOT=/home/wzy/alloc-extend-network-results \
CACHE_LAYOUT=per-node \
BATCH_SIZE=1 \
PAGE_SIZE=128 \
PREFIX_TOKENS=0 \
EXTEND_TOKENS=70 \
bash /home/wzy/kernel-regression-repros/k3_ta_autotune_dp/run_alloc_extend_network.sh
```

单节点没有真实的跨节点 DP4 group。为保留“一个 rank 编译落后、其他 rank 在
collective 等待”的诊断模式，用例在 `NNODES=1` 时让全部 16 rank 进入默认 world
all-gather；四节点时仍使用 `[0,16,32,48]` 等真实 DP4 group。

单机矩阵先跑以下四轮，每轮使用从未存在过的 case/cache 目录：

1. 同一 CANN/torch-npu + TA 3.2.1，`exact-dynamic/per-node`；
2. 同一 CANN/torch-npu + TA 3.2.2，`exact-dynamic/per-node`；
3. TA 3.2.2，`exact-dynamic/per-rank`；
4. TA 3.2.2，`static-bound/per-node`。

单机可以证明 TA 版本回归、dynamic/static 差异、compile/launch/sync 阶段以及
节点内 cache 并发问题；不能证明跨节点 HCCL、真实 DP4 group 或完整 SGLang
scheduler 的问题。只有单机全部通过而完整服务仍挂时，才需要扩大到四节点。

### 真实 NPU allocator 的 64-rank 验证

`exact-dynamic` 导入 SGLang 通用 allocator，并不等同于 NPU monkeypatch 的最终
调用路径。要验证 K3 在 NPU 上实际使用的算子，使用 `npu-production`：

```bash
NODE_RANK=<0|1|2|3> \
NNODES=4 \
NPROC_PER_NODE=16 \
MASTER_ADDR=192.168.25.209 \
MASTER_PORT=30241 \
CASE_NAME=ta322-npu-production-64r-cold \
PHASE=cold \
VARIANT=npu-production \
SGLANG_SOURCE=/home/wzy/sglang-ta-ab \
CACHE_ROOT=/tmp/alloc-extend-network/ta322-npu-production-64r-cold \
RESULT_ROOT=/home/wzy/alloc-extend-network-results \
CACHE_LAYOUT=per-node \
BATCH_SIZE=1 \
PAGE_SIZE=128 \
PREFIX_TOKENS=0 \
EXTEND_TOKENS=70 \
bash /home/wzy/kernel-regression-repros/k3_ta_autotune_dp/run_alloc_extend_network.sh
```

这个变体直接导入
`sgl_kernel_npu.mem_cache.allocator.alloc_extend_kernel`，并按 NPU allocator
生产调用传入 `next_power_of_2(extend_num_tokens)` 作为 constexpr。结果中的
`allocator_source`、`allocator_sha256` 和 `sgl-kernel-npu` 版本用于拒绝把通用
SGLang kernel 的结果误当成 K3 NPU 生产路径。
