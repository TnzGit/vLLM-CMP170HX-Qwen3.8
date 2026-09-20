# 在 NVIDIA CMP 170HX（SM80，64 GiB）上运行 Qwen3.8-27B

[English](README.md) | **简体中文**

这是一个基于 vLLM 0.27.1 的研究与生产分支，目标是在**单张 NVIDIA CMP 170HX** 上运行 **Qwen3.8-27B + DFlash2 推测解码**。

当前主线性能路径是 **M7 mixed-FP8 verifier**：target KV 使用 FP8，draft KV 使用 BF16，配合面向 SM80 的自定义 split-KV speculative verifier、target/draft 异构缓存页几何，以及 FULL CUDA Graph 执行。

> **项目状态（2026-09-20）：** M7 路径已经完成重建、历史冻结结果复现、与修复后的 INT8 替代路径的同条件 A/B，并已确立为这套硬件/runtime 上推荐的长上下文生产路径。完整研究账本保留在 PR #1；只有通过验证的生产切片合并进了 `main`。

本 README 专门描述 **CMP 170HX 路径**。仓库中仍保留了从上游继承的 RTX 3090 / batch / KVarN / Docker 资产，它们仍可用于其他实验，但**不是**下文性能结果对应的配置。

---

## TL;DR

对这套确定的硬件与软件栈：

| profile | 上下文 | DFlash `k` | `MAX_SEQS` | 实际 attention block | 适用场景 |
|---|---:|---:|---:|---:|---|
| **short** | ≤ 32K | **5** | **4** | 816 | acceptance 较高的短/中上下文交互请求 |
| **long** | ≥ 48K | **3** | **1** | 800 | 长上下文 / 126K / 250K 服务 |

分界点大约在 **48K**。**不要**在同一个 engine 里实现动态 `k` 切换：在这个 fork 中，`k` 会改变派生出来的 cache/page 几何（k=3/5/7 分别对应 800/816/832），因此它属于 **engine 级 service profile**，而不是 per-request knob。

在 126K 和 250K，`MAX_SEQS=1` 不是单纯的“低延迟偏好”。实测 service A/B 中，它在**所有测量维度上都优于 2**，包括总 makespan。

---

## 硬件与软件契约

仓库中正式 qualification 的数字对应以下环境：

| 组件 | 值 |
|---|---|
| GPU | **NVIDIA CMP 170HX**，SM80，64 GiB |
| 正式 benchmark 频率 | **1350 MHz 锁频** |
| 功耗上限 | **180 W** |
| vLLM | **0.27.1** |
| PyTorch | **2.13** |
| CUDA | **13.x** |
| Triton | **3.7.1** |
| 最终 qualification 使用的 target model | `Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4` |
| drafter | Qwen3.8 DFlash2 W4A16 |
| target attention/KV | FlashInfer + **FP8 E4M3 KV** |
| draft attention/KV | FlashAttention + **BF16 KV** |
| speculative verifier | 自定义 SM80 split-KV，q≤8 / GQA6 / D256 |
| verifier segmentation | **NSEG=35** → 140 CTA / 140 SM |
| graph mode | **FULL** |
| 已 qualification 的最大 model length | **262144** |

checkpoint 必须视为实验契约的一部分。不同 target checkpoint 之间的 performance / acceptance 不应被直接当作“纯路径性能差异”来比较。

---

## 为什么需要这个 fork

vLLM 0.27.1 的常规路径并不足以高效覆盖这张卡和这个 workload：

- Qwen3.8 包含 **16 层 full attention + 48 层 GDN**。
- target 使用 24 个 query heads、4 个 KV heads、GQA=6、head dimension=256。
- 长上下文 speculative verification 会反复扫描非常大的 paged KV cache。
- 在 64 GiB 显存上，KV allocation 可以轻易超过 2 GiB 地址算术边界，即便 block table 本身仍是 int32。
- target 和 draft 的最佳 KV 格式不同，因此 allocator 必须支持异构 attention page geometry。
- verifier 必须针对 q≤8 保持 CUDA Graph 稳定并足够快，而不能只依赖通用 dense-attention 路径。

M7 的本质，是把这些硬件/runtime 约束显式编码成一条 SM80 专用执行路径，而不是把 CMP 170HX 当作“普通 CUDA GPU”。

---

## 当前生产架构

```text
Qwen3.8-27B W4A16 target
        |
        |  FlashInfer target attention
        |  FP8 E4M3 target KV cache
        v
custom split-KV speculative verifier (SM80)
        |
        |  q <= 8
        |  Hq=24 / Hkv=4 / GQA=6 / D=256
        |  NSEG=35
        |  FP8 LUT decode
        |  page-local block-table carry
        |  stride 乘法前将 physical block id 扩展为 int64
        v
DFlash2 W4A16 drafter
        |
        |  FlashAttention
        |  BF16 draft KV
        v
FULL CUDA Graph replay
```

目录名 `experimental/cmp170hx-mixed-fp8/` 为了保留历史没有改名。其内部的 M0→M7 系列现在已经是**这个 fork 中经过 qualification 的生产性能路径**。

---

## 安装

### 1. 创建 vLLM 0.27.1 环境

本仓库 patch 固定针对 vLLM 0.27.1。

可以直接使用仓库 Docker build 对应的依赖集合：

```bash
git clone https://github.com/TnzGit/vLLM-CMP170HX-Qwen3.8
cd vLLM-CMP170HX-Qwen3.8

python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r docker/requirements.txt
```

确认版本：

```bash
venv/bin/python - <<'PY'
import vllm, torch, triton
print("vllm", vllm.__version__)
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

预期是 vLLM 0.27.1，并且 GPU capability 为 SM80。

### 2. 应用基础 patch stack

```bash
SP=$(venv/bin/python - <<'PY'
import inspect, pathlib, vllm
print(pathlib.Path(inspect.getfile(vllm)).resolve().parent)
PY
)

for p in patches/*.patch; do
    echo "== $p"
    patch -p1 -d "$SP" < "$p"
done
```

基础 stack 现在包含最终 qualification 阶段确认的重要 correctness hardening：

- split-KV verifier 中的 int64 physical block addressing；
- Qwen GDN gate/token alignment backport（#51812）；
- draft 独立 RNG stream / reproducibility hardening（#54282 invariant）。

### 3. 应用 CMP 170HX mixed-FP8 M0→M7 系列

先 dry-run：

```bash
bash experimental/cmp170hx-mixed-fp8/install.sh --dry-run
```

然后正式应用：

```bash
bash experimental/cmp170hx-mixed-fp8/install.sh --apply
```

检查安装后的 vLLM：

```bash
bash scripts/audit-mixed-fp8-prereqs.sh
bash experimental/cmp170hx-mixed-fp8/install.sh --check
```

完整系列顺序明确记录在：

```text
experimental/cmp170hx-mixed-fp8/series
```

### 4. 提供 target 和 draft checkpoint

设置主机上的实际路径：

```bash
export MODEL=/path/to/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4
export DRAFT=/path/to/Qwen3.8-27B-DFlash2-W4A16
```

本 README 中的最终数字使用的是上述 checkpoint family。如果替换成其他 W4A16 target，应把它视为一个新的 qualification 点：即使 kernel cost 不变，acceptance 也可能变化。

---

## 启动已 qualification 的 profile

mixed-FP8 公共 runtime contract：

```bash
export SPEC=dflash2
export CTX=cmp-mixed-fp8

export CMP_TARGET_ATTN_BACKEND=FLASHINFER
export CMP_TARGET_KV_DTYPE=fp8
export DFLASH_ATTN_BACKEND=FLASH_ATTN
export DFLASH_KV_CACHE_DTYPE=bfloat16

export SPEC_ATTN=1
export VLLM_SPEC_DECODE_ATTN_SEGMENTS=35
export VLLM_FP8_SPEC_VERIFY=1
export VLLM_FP8_SPEC_FULL_CG=1
export VLLM_ALIGN_HETEROGENEOUS_ATTN_PAGES=1
export CUDAGRAPH_MODE=FULL

export LOOKUP=0
export PREFIX_CACHE=1
```

在 CMP 主机上复现正式 benchmark 条件：

```bash
sudo nvidia-smi -pl 180
sudo nvidia-smi -lgc 1350,1350
```

### Short profile — ≤32K

```bash
export MAX_LEN=32768
export DFLASH_MAX_LEN=$MAX_LEN
export DFLASH_TOKENS=5
export MAX_SEQS=4
export PORT=8000

bash single-user/start_qwen.sh
```

### Long profile — ≥48K，包括 126K/250K

```bash
export MAX_LEN=262144
export DFLASH_MAX_LEN=$MAX_LEN
export DFLASH_TOKENS=3
export MAX_SEQS=1
export PORT=8000

bash single-user/start_qwen.sh
```

研究/qualification 阶段统一使用 **8002**，目的就是避免触碰 production 8000。

### 单 profile 保守方案

如果运维简单性比榨取短上下文最后几个百分点更重要：

```text
k=3
MAX_SEQS=1
```

是保守的单 profile 选择。它会放弃短上下文中由 workload acceptance 带来的收益（实测 corpus 上 4K 约损失 8%），但从结构上看，k=3 在所有测量上下文上每次 speculative iteration 的成本都是最低的。

---

## 实测性能

### M7 历史结果复现

按重建出的历史协议（180 W、不锁固定频率、FULL graph、NSEG=35）重放 M7：

| 上下文 | frozen M7 ms/spec-iteration | reconstructed | 差异 |
|---:|---:|---:|---:|
| 4K | 22.315 | **22.254** | **-0.27%** |
| 126K | 35.230 | **35.717** | **+1.38%** |
| 250K | 46.941 | **48.329** | **+2.96%** |

这个结果足以把原始 M7 视为已复现。长上下文剩余的 1–3% 漂移，不被解释为历史数字有误。

### M7 对比 repaired INT8 path

同 checkpoint、同 prompt IDs、FULL graph、1350 MHz / 180 W：

| 上下文 | M7 mixed-FP8 ms/iter | repaired INT8 ms/iter | INT8 penalty |
|---:|---:|---:|---:|
| 4K | **22.306** | 24.065 | +7.9% |
| 65K | **29.258** | 46.438 | +58.7% |
| 126K | **35.685** | 68.827 | +92.9% |
| 250K | **48.292** | 115.408 | +139.0% |

除 250K 外，acceptance 基本接近，因此差距随 context 增长而扩大的主要原因是 **target verifier iteration cost**，不是 drafter efficiency。

修复后的 INT8 path 仍保留为 correctness-complete alternate/fallback，但不再是长上下文性能默认路径。

### Production k crossover

C1、每个 cell fresh engine、exact-token unique corpus、1350 MHz / 180 W：

| 上下文 | k=3 tok/s | k=5 tok/s | k=7 tok/s | production 选择 |
|---:|---:|---:|---:|---|
| 4K | 123.49 | **134.68** | 129.67 | **k=5** |
| 32K | 134.64 | **147.02** | 137.62 | **k=5** |
| 48K | **91.68** | 91.57 | 88.58 | **k=3** |
| 65K | **100.05** | 90.64 | 94.56 | **k=3** |
| 126K | 89.55 | 88.18 | **90.21** | 运维上选 k=3；k=7 只领先 0.7%，在 spread 内 |
| 250K | **64.90** | 58.68 | 59.69 | **k=3** |

需要特别强调：output tok/s 会受 acceptance 影响。去掉 acceptance 影响后的结构性指标 `ms/spec-iteration` 在**所有上下文都由 k=3 胜出**。k=5 只在短上下文赢，是因为更高 acceptance 足以抵消更贵的 verify pass。

完整数据和 caveat：[docs/production-service-profile.md](docs/production-service-profile.md)。

---

## 长上下文并发策略

qualification 中一个非常重要的结论是：在这张单卡上，**允许更多长上下文请求同时 resident 反而更差**。

Service-policy A/B：

| 上下文 | policy | makespan | useful aggregate | first completion | 最慢请求 |
|---:|---|---:|---:|---:|---:|
| 126K | **MAX_SEQS=1** | **215.8 s** | **4.75 tok/s** | **107.6 s** | **84.2 tok/s** |
| 126K | MAX_SEQS=2 | 219.8 s | 4.66 tok/s | 218.4 s | 4.4 tok/s |
| 250K | **MAX_SEQS=1** | **561.9 s** | **1.82 tok/s** | **278.1 s** | **68.7 tok/s** |
| 250K | MAX_SEQS=2 | 589.8 s | 1.74 tok/s | 589.8 s | 1.7 tok/s |

在 126K 和 250K，`MAX_SEQS=2` 连总 makespan 都更差。一个长请求已经足以把 GPU 吃满；第二个 ultra-long prefill 主要带来 GPU interference，并消耗 KV headroom。

prefill token budget 也**不是**解决这个问题的 knob：4× budget sweep 对 steady-state ITL 的影响只有约 0.4%。

在实测配置下，250K 从物理 capacity 上最多能同时容纳 **2 个 resident sequence**，但 production 仍应使用 **1**，因为 2 的实际表现更差。

详见 [docs/maxseqs-service-policy-ab.md](docs/maxseqs-service-policy-ab.md)。

---

## M0 → M7 优化历史

成功的性能链保留了原始 commit 历史：

| milestone | 主要变化 | 4K ms/pass | 126K ms/pass | 250K ms/pass |
|---|---|---:|---:|---:|
| M0 | mixed-FP8 qualified baseline，16 segments | 23.100 | 62.800 | 103.300 |
| M1 | verifier 16 → 32 segments | 23.000 | 50.000 | 77.500 |
| M2 | E4M3FN → BF16 LUT decode | 22.500 | 41.700 | 61.200 |
| M3 | 减少 spill / 重复 block-ID work | ~22.6 | ~38.4 | 54.200 |
| M4 | q≤8 / GQA6 专用化 | 22.500 | ~37.2 | ~51.5 |
| M5 | page-local block-table carry | 22.500 | 37.200 | 51.600 |
| M6 | read-only-cache LUT | 22.500 | 36.300 | 50.000 |
| **M7** | **NSEG=35 / 140-CTA resident-wave alignment** | **22.315** | **35.230** | **46.941** |

foundation 与 M1–M7 分别通过 PR **#2**、**#3** 合并，并且没有 squash 原始研究 commits。

---

## 重要 correctness 修复

### 1. Large-KV int32 地址溢出

长上下文 illegal memory access 的根因是：

```python
blk = tl.load(...)              # int32 block id
k_ptr = k_ptr + blk * stride_kb # 可能在参与 pointer arithmetic 前先发生 int32 overflow
```

当 KV pool 约为 5.44 GiB、block stride 约为 898 KiB 时，一个完全合法的 physical block id 就可能让 `blk * stride_kb` 超过 signed int32 上限并发生负向 wrap。

修复：

```python
blk = tl.load(...).to(tl.int64)
```

地址算术与 Compute Sanitizer 实际观察到的 fault displacement 能精确闭合。

修复后的 requalification 在以下测试中均保持 Xid 31 delta=0：

- DFlash2 16K：120/120
- DFlash2 65K：12/12
- DFlash2 126K：8/8
- MTP 16K：24/24
- C4 16K：32/32
- FULL graph 16K：24/24

4K paired ABBA 的真实性能代价只有 **+0.64% C1 / +0.27% C2**，低于 material-regression gate。

### 2. Qwen GDN gate/token 对齐（#51812）

vLLM 0.27.1 在 mixed speculative batch 中已经对 Q/K/V rows 做了 gather，但两个 recurrent-update call site 仍可能把原始、未 permutation 的 GDN gate rows `a/b` 传给 kernel。

这类 bug 不一定会 crash；它可能静默更新错误的 recurrent state。

backport 后，`a/b` 会使用与 Q/K/V 相同的 `spec_token_indx` / `non_spec_token_indx` mapping。该修复已经作为 P0 correctness hardening 合并进 `main`。

### 3. Draft RNG 隔离（#54282 invariant）

v0.27.1 的 draft proposal path 使用 process-global default RNG 生成 exponential noise。这意味着 unrelated/concurrent global RNG traffic 会影响 proposal draws。

backport 后，draft proposal 使用独立 generator。

**作用域修正：** 在我们这条 v0.27.1 路径里，target/rejection/residual sampling 原本已经使用 per-request seeded generator；200k-trial test 也没有发现 shared-noise distribution bias。因此这个 patch 应描述为 **reproducibility / RNG isolation hardening**，而不是“旧路径 target distribution 有偏”的证据。

---

## Greedy equivalence caveat

不同 k 之间 byte-identical，只证明**这条 speculative path 内部**的一致性；它不能证明 block-shaped speculative verification forward 与普通 target-only q_len=1 greedy forward 在 token 层面严格一致。

upstream vLLM issue [#54928](https://github.com/vllm-project/vllm/issues/54928) 报告过 Qwen3.8 的案例：target-only 和 DFlash2 各自都完全 deterministic，但在接近 logit tie 时，多位置 verifier forward 与单 token target forward 的 argmax 不同，因此输出发生分叉。

本仓库**尚未证明**最终 W4A16 + M7 CMP170HX stack 存在相同现象。所以未来如果要宣称 target-only greedy equivalence，必须在**完全相同 checkpoint/runtime**上同时记录：

- target-only emitted token；
- verifier target-logit argmax；
- speculative emitted token；
- 首次 divergence 附近 target-only top-1/top-2 margin。

在没有这些证据前，不要把这种 divergence 自动归因于 RNG 或 cache corruption。

---

## Profiling 状态，以及为什么当前优化周期已经结束

当前 head 的新 profile：

| 组件 | 126K | 250K |
|---|---:|---:|
| target Marlin GEMMs | ~48.8% | ~36.4% |
| verifier | ~35.9% | **~52.2%** |
| 其余 kernels | balance | balance |

250K 下 verifier 仍然是最大单项，但当前低风险结构性方案已经基本耗尽：

- tensor-core softmax denominator：**慢 11–14%**；
- two-level FP32 PV accumulation：在写完整 kernel 前就被 register gate 杀掉；
- production verifier：**252 registers/thread，0 spill**，正好保持 2 resident CTA/SM；
- projected FP32 long-term accumulator：约 300 regs/thread，超过 SM80 limit；
- register-release audit 没找到不引入新 memory/dataflow 代价、且可信释放 ≥16 regs 的方案；
- fused DFlash2 grouped convolution 在 whole-step 占比太小，不值得做；
- 之前 E21–E42 已覆盖大量其他 verifier 设计。

即使 verifier 再快 3%，在 250K 的 whole-step 上也只有约 **1.5%**。

因此当前建议是：**交付已经 qualification 的 M7，而不是继续做 verifier 微优化。**

---

## Benchmark 规则

最终 qualification 中，同一种测量 defect 连续出现了三次：ratio 的 numerator 与 denominator 覆盖的时间区间不一致。

永久规则记录在 [docs/benchmark-rules.md](docs/benchmark-rules.md)。最重要的规则：

1. 每一个 ratio 都必须明确 numerator 和 denominator 对应的时间区间。
2. 区间不一致时，只能报告 `null` 或显式 bound，不能报告“看起来合理”的估算值。
3. concurrent decode 必须使用真正 steady-state window：
   ```text
   steady_start = max(first_output_i)
   steady_end   = min(finish_i)
   ```
4. global server counters 不能除以某一个 request 的时间窗口。
5. runtime 事实来自**parsed/running engine**，不能来自 launcher intent。
6. 每个正式 context/config cell 使用 fresh engine。
7. 正式长上下文 benchmark 使用 exact-token、unique prompts。
8. 报告 Xid **delta**，而不是 dmesg 的绝对累计数。
9. engine crash 后视为 poisoned，下一 cell 前必须重启。
10. standalone microbench、kernel attribution、C1 decode throughput 和 service-policy measurement 必须保持分类清晰，不能混为同一种指标。

---

## 仓库结构

| 路径 | 用途 |
|---|---|
| `patches/` | vLLM 0.27.1 基础 patch stack，包括最终 correctness hardening |
| `experimental/cmp170hx-mixed-fp8/` | 已 qualification 的 CMP mixed-FP8 M0→M7 系列；目录名仅为历史保留 |
| `single-user/start_qwen.sh` | CMP profile 使用的 launcher |
| `scripts/audit-mixed-fp8-prereqs.sh` | base/install contract 专项检查 |
| `deploy/` | qualification 阶段使用的 systemd/env 示例 |
| `bench/` | kernel、path、concurrency、k-sweep 和 regression harness |
| `docs/production-service-profile.md` | production profile 的权威测量结论 |
| `docs/benchmark-rules.md` | 永久 benchmark 方法学 |
| `docs/cmp170hx-mixed-fp8-engineering.md` | mixed-FP8 工程记录 |
| `docs/gotchas.md` | implementation/runtime gotchas |
| `handover.md` | 完整 investigation chronology |
| `batch/`、`kvarn/`、Docker assets | 从上游继承的替代路径；不是 CMP M7 主 production 推荐 |

`deploy/cmp170hx-mixed-fp8-*.service.example` 是 qualification snapshot，其中部分文件有意保留了历史测试时的 k / max-seq 值。当前 production policy 以 **docs/production-service-profile.md** 和本 README 为准。

---

## GitHub 历史与项目收口

研究历史与生产历史被刻意分开：

- **PR #1** — 已关闭，保留完整 research ledger；**不要 merge**。
- **PR #2** — 已合并 mixed-FP8 foundation / M0。
- **PR #3** — 已合并 M1–M7 成功优化链，并保留原始 commit history。
- **PR #4** — 已关闭为 superseded；其 int64 widening 已包含于 #2。
- **PR #5** — 已合并 Qwen GDN gate/token correctness backport。
- **PR #6** — 已合并 draft RNG isolation / reproducibility hardening。
- **PR #7** — 已合并 production profiles 和 benchmark rules。

M7 之后的 negative research 没有被删除，而是继续保留在研究账本中，避免未来重复走同样的失败路线。

---

## 已知边界

- 这条 production path 是**硬件相关**的：CMP 170HX 的具体 SM80 行为是实验契约的一部分。
- 最终 profile 数字同时依赖 **checkpoint 和 workload**，尤其是 DFlash acceptance。
- 短上下文下 k=5 的优势由 acceptance 驱动；k=3 在所有上下文的 verify iteration 结构成本都更低。
- 250K C4 在当前 KV pool 下不是一个有效的四路 steady-decode benchmark。
- 长上下文 `MAX_SEQS=1` 是这套**单 GPU stack**上的实测 service policy，不是通用 vLLM 建议。
- 仓库 Docker image 会应用从上游继承的基础 patch stack 和 KVarN 路径，但**不会自动应用 CMP mixed-FP8 M0→M7 系列**。要运行已 qualification 的 CMP path，请使用上文 host/venv 安装方式，或自行扩展 image。
- 不要因为 launcher/env 文件里设置了某个变量，就假设 runtime 实际走了那个路径。需要检查最终 engine config，必要时读取 `/proc/<engine-pid>/environ`。

---

## 延伸阅读

- [Production service profile](docs/production-service-profile.md)
- [Benchmark rules](docs/benchmark-rules.md)
- [Long-context MAX_SEQS A/B](docs/maxseqs-service-policy-ab.md)
- [CMP mixed-FP8 engineering notes](docs/cmp170hx-mixed-fp8-engineering.md)
- [Gotchas](docs/gotchas.md)
- [完整 handover / research chronology](handover.md)

---

## License 与上游血缘

本仓库保留原项目的代码、脚本和 license，同时加入本文所描述的 CMP 170HX / Qwen3.8 专用化工作。

这个 fork 的重点，是在固定的 vLLM 0.27.1 stack 上实现可复现的工程结果。它**不代表**这些硬件专用 patch 应该取代 current upstream vLLM 在其他 GPU 上的默认行为。
