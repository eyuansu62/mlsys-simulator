# Step 2 历史设计：9070XT 低并发＋长输入（course9070sla20261116）

本页整理 PR #10 原 HEAD `a075347` 的2026-11-16设计和本地冻结记录，状态说明更新于2026-11-19。
原九文件 snapshot 已冻结，但该 **27 × 1000 请求 GPU 合同没有执行**，host 字段仍为 `REMOTE_PENDING`，
`hardware_validated=false`。本次文字整理不是新的测前登记，原日期与承诺应由历史提交和原冻结字节核对。

后续 main PR #13 已另行完成 **54 × 512 请求**的 BF16/eager/MRv1、CPU10、显式 Core 后端验证：
53/54格达标，唯一 short-C2 TTFT P99误差30.59%，整体仍为FAIL。
见[独立报告](../qwen4b-online-20261117/README.md)。它不补签或完成原27格合同，也不修改旧默认 β。

## 0. 为什么要做

11-11 研究（`course9070sla20261111`）的负结果结构（见 [../course-sla-9070xt-20261111/README.md](../course-sla-9070xt-20261111/README.md)）：
TPOT 0/39 达精度目标、预测安全并发 16 而实测最大全通过点 C4、闭环TPOT的C1→C8相对增长实测 +39% 而模型仅 +3.6%。
Step 1（[README.md](README.md)）检查：11-16 runtime-cost 校准**域外**于本场景（Graph配置 vs eager、上下文
1024 vs ≈256），不能合法应用；在原参数化下短上下文的名义 `linear` 项占比高，批大小响应较平。
分量占比和由闭环 TPOT 推算的带宽是模型内解释／代理，不是独立 kernel 带宽测量或唯一根因证明。

当时据此提出新的 eager 域低并发计划，用来：

1. 在低并发（C1/2/4）长输入上做当时尚未观测配置的精度检验；
2. 重测闭环TPOT随低并发（C1→C4）变化的形态；
3. 为后续 eager 域独立校准提供观测；本留出本身不用于拟合，闭环指标也不能单独识别每步固定截距或纯 kernel 斜率。

原39-cohort数据已经观察：baseline含C1/2/4/8/16/32/64，short/long分布留出才是C8/16/32。见
[../course-sla-9070xt-20261111/HOLDOUT-CONSUMED.json](../course-sla-9070xt-20261111/HOLDOUT-CONSUMED.json)，**不得复用**。
该JSON及其中历史引用保持原字节；本说明不重写它。PR #13现已覆盖本计划的输入与并发配置，不能再称当前未见盲点。

## 1. 原计划的硬件 / 运行时 / 模型（host 绑定未完成）

- 单 RX9070XT / gfx1201 / 16 GiB（`memory_bytes=17095983104`，`memory_bandwidth_gbps=640`，`compute_tflops=195`）。
- Qwen3-4B BF16，TP1×PP1，`enforce_eager=true`，`async_scheduling=false`。
- 原计划要求 vLLM 构建与 11-11 相同，但尚未完成host实际绑定（镜像tag 0.29.0不当作实际版本）。
- `max_num_seqs=16`、`max_num_batched_tokens=2048`、`max_model_len=2048`、`kv_cache_memory_bytes=3221225472`、
  `block_size=16`、Prefix Cache 关、Chunked Prefill 与混批开。原模拟器每请求 chunk 上限512，
  不是已经核验的原生vLLM每请求限制；后续PR #13另行按实际剩余token预算对齐。旧snapshot选择不因此改写。
- 实验对象直接复用 11-11 冻结的
  [experiment.json](../../examples/course/9070-sla/frozen-commitment/experiment.json)（`profile_path=null`，纯 CPU 预测）。

## 2. 原预注册并发网格与新旧配置关系

**并发网格 C ∈ {1, 2, 4}**（低并发）。它与旧 short/long 的 C8/16/32 不相交，
但 baseline C1/2/4 已实测。C1→C4的闭环增长对照不能冒称固定 batch Decode kernel 斜率。

| cohort | 输入长度（等权类别） | 并发 | 单元数 | 盲/重测 |
| --- | --- | --- | ---: | --- |
| baseline | 256 | 1,2,4 | 3 | 重测（11-11 基线 C1/2/4，独立流；斜率臂 C1→C4） |
| short | 64,128 | 1,2,4 | 3 | 相对11-11分布留出新增；当前PR #13已观测 |
| long | 512,1024 | 1,2,4 | 3 | 相对11-11分布留出新增；当前PR #13已观测 |

原合同合计 **9 单元 × 3 fresh Runtime = 27 Runtime，27,000 请求**；这是计划量，不是完成量。

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

## 5. 失败保留与独立验证规则

- 保留全部请求的单元/重复/失败/超时/部分/未知结果；**不重调、不挑有利种子、不扩网格、不重采到通过为止**。
- 预测本地留档；远端在正式采集前收到 raw/canonical SHA 承诺；正式采集只按冻结 policy 与请求流执行。
- 全部 GPU 单元结束后，仅用冻结代码与输入在 CPU 重算预测，两种 SHA 均须与测前承诺一致，再比较实测。
- **不得以实测 arrival 重跑预测**；不得对本研究或 11-11 已观察点回拟合。

## 6. 原决策规则与归因限制

- 原计划若 short/long 在 C1/2/4 出现“三次全部通过”的点，记录为该网格下最大已观察全通过点，不向未测并发外推。
- 低并发 TPOT 若持续偏低，只能说明该预测与闭环观测不符；不能单凭它确认 β、HBM带宽或某个GPU因素为唯一根因。
  eager 域独立采样与校准应另立协议，不能继续借用Graph配置域参数。
- 若以后建立新实验，应记录其并发/输入区间及残差，并使用独立新验证；不能在当前已观察点上拟合后仍称盲留出。
- 任何 SLA FAIL 由有意过载或真实失败引起时，按 11-11 口径区分“SLA 可服务性”与“模型精度”，不混同。

## 7. 边界与免责

- 有限 cohort（N=1000 为停止规则后 drain），不是稳态到达窗吞吐或稳定性定理。
- 服务端边界，不是 HTTP 客户端 SLA。
- 未采集逐 kernel/功耗轨迹；不能唯一分离 HBM 效率、CPU 调度、kernel 融合与降频。
- 本设计与预测不构成对未测策略/配置/硬件的覆盖主张；`hardware_validated=false`。

## 8. 执行状态

### 8.1 CPU 冻结（已完成，2026-11-16）

由原 HEAD `a075347` 的 [scripts/freeze_course_sla_20261116_local.py](../../scripts/freeze_course_sla_20261116_local.py)
生成。以下目录及其九个文件是受保护的历史产物，不是当前脚本的默认输出：
`examples/course/9070-sla-20261116/frozen-commitment/`：
`experiment.json`（复用 11-11，`89ec0d93…`）、3 个 cohort spec（网格 `[1,2,4]`、seed `20261116`）、
`jobs.json`（27 Runtime 的确定性执行序）、`predictions-summary.json`（逐 run 摘要）、
`selection.json`（预注册盲网格声明，**不做容量搜索、不据预测挑选**）、
`validation-contract.template.json`（与 11-11 同，`8f0bb4ee…`）、`manifest.json`。
旧脚本将完整预测payload写至作者私有scratch，未随PR发布；仓库九文件只含输入、摘要与双SHA承诺。
这里保留当时已记录的字节承诺与数值，不保证当前main能重现原完整payload或raw SHA，
也不能用当前代码生成的文件替代或补写其历史承诺。

**承诺 SHA（预测 payload，raw / canonical）：**

| cohort | raw | canonical |
| --- | --- | --- |
| baseline | `7f3d0f77dc10498f519ca2d8089a01a1a176256854c912b41a500607cbb1791c` | `d9d704bf5e7303023d7be38919517ca2b70f85052ab5cabb6e0acdd3c255e9aa` |
| short | `e0e1c9639dafcfbaffa78bd641141ad15bc67020c7e0023c2455900a1c358a7a` | `b4ce2005b09b127ae84a42ba34c28747d18629946c8c89cad673a444979ebb2c` |
| long | `678de9e5682815b0bde2fbb64b508149f6166609fd8ede847d6d9a7ba91be28f` | `a4392775774cc0368d6cbec975ec350b8b36e387ede8113db2f55602be775784` |

identity `cc6665e8…`（与11-11同源，因experiment未改）。原HEAD的复现记录称当时重跑的raw/canonical SHA均与上表一致，
且核对了 `files_sha256` / `code_files_sha256`。这些是原提交的历史记录，不保证当前代码产生同样完整payload字节；
当前代码、报告字段和hash演进不得反写原manifest或更新上表承诺。

**原冻结预测（P99 ms，3次计划重复的中位）**——保留历史数值，尚无原27格的GPU核验：

| cohort | C | TTFT | TPOT | E2E | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline(256) | 1 / 2 / 4 | 18.3 / 34.9 / 68.1 | 14.3 / 14.4 / 14.5 | 233 / 251 / 286 | 68.6 / 127.5 / 223.5 |
| short(64,128) | 1 / 2 / 4 | 15.1 / 18.3 / 34.9 | 14.3 / 14.3 / 14.4 | 229 / 233 / 251 | 69.8 / 138.6 / 264.2 |
| long(512,1024) | 1 / 2 / 4 | 71.0 / 138.7 / 173.3 | 14.5 / 17.6 / 25.2 | 289 / 361 / 465 | 59.4 / 104.8 / 166.0 |

原文预期TPOT偏低（11-11闭环观测约21–29ms）；这一参照不是新的测量，也不能据它改原预测或拟合参数。

### 8.2 原远端合同未执行，不补写历史 snapshot

原manifest的host侧字段（`checkpoint_manifest`、`runtime_execution_config`、
`runtime_config_sha256`、`scheduler_config_sha256`、`design_contract_sha256`）仍为 `REMOTE_PENDING`。
原计划须先完成实际checkpoint/Runtime绑定、远端SHA接收回执，再运行27次独立采集并揭示；这条链没有完成。
原稿记载当时16 GiB卡被其它工作占用，故未并行启动；这不是当前GPU占用状态。

如果以后需要新的实验，应使用新study ID、新目录、新政策与独立数据，保留原九文件和
[HOLDOUT-CONSUMED.json](../course-sla-9070xt-20261111/HOLDOUT-CONSUMED.json) 原字节。
不得就地补全旧manifest、覆盖承诺、借当前代码声称重放了旧完整payload，或将PR #13的54 × 512结果填回本合同。

### 8.3 当前代码的 CPU 本地 draft（不采样、不替代历史冻结）

当前可移植脚本要求显式 `--out` 和 `--study-id`，拒绝已有输出及受保护历史目录。可先运行小规模CPU烟测：

```bash
python scripts/analyze_decode_domain_gap.py
python scripts/freeze_course_sla_20261116_local.py \
  --study-id course9070sla-local-draft-20261119 \
  --out runs/course9070sla-local-draft \
  --requests-per-run 2 --replications 1
```

输出标注 `current-code local draft`、`hardware_validated=false` 和远端冻结待办；完整payload保存为
`OUT/predictions/<cohort>.prediction-payload.json`。小烟测不具备GPU验证意义，也不按旧实验的1000请求合同解释。
脚本保持显式课程参数，不拟合、不默认修改β；需要真实GPU验证时必须另行设计、绑定和冻结。
