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
