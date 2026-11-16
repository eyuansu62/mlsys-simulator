# Step 2 预注册：9070XT 低并发＋长输入盲留出（course9070sla20261116）

测前冻结设计。**本设计先于任何 GPU 测量**；预测与 SHA 承诺在本文件与配套 bundle 生成后即固定，
正式测量不得改网格、改种子、改阈值、回拟合或挑选有利重复。`hardware_validated=false`。

## 0. 为什么要做

11-11 研究（`course9070sla20261111`）的负结果结构（见 [../course-sla-9070xt-20261111/README.md](../course-sla-9070xt-20261111/README.md)）：
TPOT 0/39 达精度目标、预测安全并发 16 而实测最大全通过点 C4、C1→C8 批处理斜率实测 +39% 而模型仅 +3.6%。
Step 1（[README.md](README.md)）证明：11-16 runtime-cost 校准**域外**于本场景（执行模式 graph vs eager、上下文
1024 vs ≈256），不能合法应用；且短上下文下名义宏被批处理不变的权重搬运主导、斜率结构性偏平。

因此需要一个**新**的、eager 域、低并发、短上下文的留出，用来：
1. 在低并发（C1/2/4）长输入上做**盲**精度检验（TPOT 是否收敛）；
2. 重测低并发批处理斜率（C1→C4）是否收敛；
3. 提供 eager 域、短上下文下可识别每步固定截距与 KV/发射/混批随批处理增长斜率的独立观测。

原 39-cohort 留出（网格 8/16/32）已消耗，见
[../course-sla-9070xt-20261111/HOLDOUT-CONSUMED.json](../course-sla-9070xt-20261111/HOLDOUT-CONSUMED.json)，**不得复用**。

## 1. 硬件 / 运行时 / 模型（与 11-11 完全一致，保证可比）

- 单 RX9070XT / gfx1201 / 16 GiB（`memory_bytes=17095983104`，`memory_bandwidth_gbps=640`，`compute_tflops=195`）。
- Qwen3-4B BF16，TP1×PP1，`enforce_eager=true`，`async_scheduling=false`。
- 实际 vLLM 构建与 11-11 相同（镜像 tag 0.29.0 不当作实际版本，以运行时自报为准）。
- `max_num_seqs=16`、`max_num_batched_tokens=2048`、`max_model_len=2048`、`kv_cache_memory_bytes=3221225472`、
  `block_size=16`、Prefix Cache 关、Chunked Prefill 开（每请求 chunk 上限 512）、混批开。
- 实验对象直接复用 11-11 冻结的
  [experiment.json](../../examples/course/9070-sla/frozen-commitment/experiment.json)（`profile_path=null`，纯 CPU 预测）。

## 2. 新预注册并发网格（不复用 8/16/32）

**并发网格 C ∈ {1, 2, 4}**（低并发）。刻意不包含 8/16/32，避免复用已消耗留出点。
C1→C8 全斜率无法在本研究盲测（C8 已消耗），本研究提供盲 C1→C4 斜率并以 11-11 C1→C8 为参照。

| cohort | 输入长度（等权类别） | 并发 | 单元数 | 盲/重测 |
| --- | --- | --- | ---: | --- |
| baseline | 256 | 1,2,4 | 3 | 重测（11-11 基线 C1/2/4，独立流；斜率臂 C1→C4） |
| short | 64,128 | 1,2,4 | 3 | **盲**（11-11 短留出仅测 C8/16/32） |
| long | 512,1024 | 1,2,4 | 3 | **盲**（11-11 长留出仅测 C8/16/32） |

合计 **9 单元 × 3 fresh Runtime = 27 Runtime，27,000 请求**。

## 3. 负载与采样（与 11-11 同协议）

- 协议 `finite_cohort_closed_loop_v1`，观测边界 `server_token_ready_v1`（服务端 Scheduler）。
- 每单元 3 次 fresh Runtime，每次 1000 请求；`think_ms=0`。
- 输出固定 16 tokens，`temperature=0`（greedy）、`ignore_eos=true`、`prompt_token_id=100`（重复填充，不推断 chat 模板长度）。
- **新 seed 20261116**（独立请求流；同一 replication 的 ID/长度序列不随并发改变，实际到达由前序完成反馈决定）。
- 每点先做声明形状 warmup、排空并核对 KV 空闲块恢复；Runner 在正式计时前关闭，仅保留 Scheduler 逐 token 事件。

## 4. 判定标准（预注册，采样前固定）

- **SLA**：P99 E2E ≤ 500 ms、P99 TTFT ≤ 500 ms、P99 TPOT ≤ 50 ms、聚合 elapsed 输出吞吐 ≥ 20 tokens/s；每次重复均须满足。
- **精度**：P99 TTFT / P99 TPOT / P99 E2E 相对误差 ≤ 25%，elapsed 输出吞吐相对误差 ≤ 15%。
- 相对误差统一 `abs(predicted−measured)/measured`。

## 5. 失败与诚信策略

- 保留全部请求的单元/重复/失败/超时/部分/未知结果；**不重调、不挑有利种子、不扩网格、不重采到通过为止**。
- 预测本地留档；远端在正式采集前收到 raw/canonical SHA 承诺；正式采集只按冻结 policy 与请求流执行。
- 全部 GPU 单元结束后，仅用冻结代码与输入在 CPU 重算预测，两种 SHA 均须与测前承诺一致，再比较实测。
- **不得以实测 arrival 重跑预测**；不得对本研究或 11-11 已观察点回拟合。

## 6. 决策规则（测前声明）

- 若盲 short/long 在 C1/2/4 出现“三次全部通过”的点，记录为该网格下最大已观察全通过点，不向未测并发外推。
- 若 TPOT 精度在低并发仍系统性偏低（方向稳定、超出重复范围），则确认失准为模型侧（β 高估 decode 带宽效率 / 斜率偏平），
  应立项 eager 域独立成本 Profile，而非继续用 graph 域校准外推。
- 若 TPOT 精度与斜率在修正后收敛，则记录收敛的并发/输入区间与残差，作为后续 eager 域校准的独立留出。
- 任何 SLA FAIL 由有意过载或真实失败引起时，按 11-11 口径区分“SLA 可服务性”与“模型精度”，不混同。

## 7. 边界与免责

- 有限 cohort（N=1000 为停止规则后 drain），不是稳态到达窗吞吐或稳定性定理。
- 服务端边界，不是 HTTP 客户端 SLA。
- 未采集逐 kernel/功耗轨迹；不能唯一分离 HBM 效率、CPU 调度、kernel 融合与降频。
- 本设计与预测不构成对未测策略/配置/硬件的覆盖主张；`hardware_validated=false`。

## 8. 执行状态

### 8.1 CPU 冻结（已完成，2026-11-16）

由 [scripts/freeze_course_sla_20261116_local.py](../../scripts/freeze_course_sla_20261116_local.py) 生成，
复现命令即该脚本。产物在 `examples/course/9070-sla-20261116/frozen-commitment/`：
`experiment.json`（复用 11-11，`89ec0d93…`）、3 个 cohort spec（网格 `[1,2,4]`、seed `20261116`）、
`jobs.json`（27 Runtime 的确定性执行序）、`predictions-summary.json`（逐 run 摘要）、
`selection.json`（预注册盲网格声明，**不做容量搜索、不据预测挑选**）、
`validation-contract.template.json`（与 11-11 同，`8f0bb4ee…`）、`manifest.json`。
完整预测 payload 已本地留档（scratch `step2-20261116-payloads/`），其字节与 raw 承诺一致。

**承诺 SHA（预测 payload，raw / canonical）：**

| cohort | raw | canonical |
| --- | --- | --- |
| baseline | `7f3d0f77dc10498f519ca2d8089a01a1a176256854c912b41a500607cbb1791c` | `d9d704bf5e7303023d7be38919517ca2b70f85052ab5cabb6e0acdd3c255e9aa` |
| short | `e0e1c9639dafcfbaffa78bd641141ad15bc67020c7e0023c2455900a1c358a7a` | `b4ce2005b09b127ae84a42ba34c28747d18629946c8c89cad673a444979ebb2c` |
| long | `678de9e5682815b0bde2fbb64b508149f6166609fd8ede847d6d9a7ba91be28f` | `a4392775774cc0368d6cbec975ec350b8b36e387ede8113db2f55602be775784` |

identity `cc6665e8…`（与 11-11 同源，因 experiment 未改）。**复现性已验证**：用冻结
`experiment.json` + 各 `spec.json` 重跑 `predict_closed_loop`，raw 与 canonical 两种 SHA 均与上表一致
（揭示公式见 [scripts/analyze_course_sla_study.py](../../scripts/analyze_course_sla_study.py) 第 70–75 行）。
`files_sha256` / `code_files_sha256` 均已对本地 bundle 与 `mlsys_sim/*.py` 核对一致。

**冻结的盲预测（P99 ms，3 次 fresh 重复的中位）**——即测后要核对的数值：

| cohort | C | TTFT | TPOT | E2E | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline(256) | 1 / 2 / 4 | 18.3 / 34.9 / 68.1 | 14.3 / 14.4 / 14.5 | 233 / 251 / 286 | 68.6 / 127.5 / 223.5 |
| short(64,128) | 1 / 2 / 4 | 15.1 / 18.3 / 34.9 | 14.3 / 14.3 / 14.4 | 229 / 233 / 251 | 69.8 / 138.6 / 264.2 |
| long(512,1024) | 1 / 2 / 4 | 71.0 / 138.7 / 173.3 | 14.5 / 17.6 / 25.2 | 289 / 361 / 465 | 59.4 / 104.8 / 166.0 |

预期：TPOT 预测偏低（11-11 实测约 21–29 ms），本盲留出的作用正是检验其是否收敛、以及 C1→C4 斜率。

### 8.2 远端冻结 + 测量（GPU 主机，待 16 GiB 卡空闲）

本地 manifest 的 host 侧字段（`checkpoint_manifest`、`runtime_execution_config`、
`runtime_config_sha256`、`scheduler_config_sha256`、`design_contract_sha256`）现为 `REMOTE_PENDING`，
**不在本地伪造**。在 9070XT 上按序执行：

1. 同步本仓库（含 `examples/course/9070-sla-20261116/` 与 `mlsys_sim/`）到测量主机，使
   `files_sha256` / `code_files_sha256` 与本地一致。
2. 由主机侧 producer 用**真实**测量主机数据补全 manifest：模型 checkpoint 清单与其 SHA、
   运行时自报的 vLLM 执行配置及其 SHA、scheduler 配置 SHA、主机侧 design-contract SHA；重算
   `local_manifest_sha256`，写出最终 `manifest.json` 并记下其 SHA256。
3. 在补全后的 bundle 上运行
   `python scripts/freeze_course_sla_commitment.py --bundle <bundle> --code-root <mlsys_sim> --expected-manifest-sha256 <步骤2的SHA>`，
   签发 `freeze-receipt.json` + 逐 cohort `validation-contract.json`（`measurement_started=false`）。
4. 确认 16 GiB 卡（`memory_bytes=17095983104`，11-11 为 GPU 1）无其他在跑任务，且固定到正确的物理卡
   （rocm-smi 的 product-name/显存标签有歧义，须以 `memory_bytes` 为准核对），避免污染本盲留出与在跑的 11-16 校准采样。
5. 在 `ch4-mlsys`（rocm/vllm）容器内按 `jobs.json` 的 27 个 fresh Runtime 采集（每点声明形状 warmup、
   排空、核对 KV 空闲块恢复；正式计时前关 Runner，仅保留 Scheduler 逐 token 事件）。
6. 全部单元结束后运行 `scripts/analyze_course_sla_study.py`（CPU-only 揭示）：重算预测并核对 raw/canonical
   两 SHA，再与实测比较，产出逐 cohort 验证与 SLA/精度判定。

`hardware_validated=false` 直至第 6 步完成且回执链完整。
