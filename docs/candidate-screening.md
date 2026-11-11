# 联合约束下的候选配置筛选

`screen` 在**同一冻结工作负载**上执行显式给定的完整候选配置，先检查显存、P99、吞吐和任务墙钟，再对合格结果按声明的目标排序。它复用原模拟引擎，不修改请求内容，也不预测未给出的全局最优配置。

```bash
mlsys-sim screen --plan examples/course/screen-simulation.plan.json --out /tmp/screen-simulation
mlsys-sim screen --plan examples/course/screen-sweep.plan.json --out /tmp/screen-sweep
mlsys-sim screen --plan examples/course/screen-dag.plan.json --out /tmp/screen-dag
```

打开各运行目录 `artifacts/report.html` 查看候选表、逐约束实际值和最紧资源约束。同源数据还保存在 `report.json`、`candidates.csv`、`checks.csv` 和数字编号的候选 JSON 文件中。全部结果是 CPU 模型预测；`PASS` 不表示硬件实测、SLA 认证或全局最优证明。

## 计划与冻结契约

```json
{
  "schema_version": 1,
  "kind": "simulation",
  "source": "synthetic_teaching_trace",
  "description": "说明请求、设备和计时来源",
  "constraints": {
    "schema_version": 1,
    "min_completed_samples": 100,
    "memory": {"max_budget_fraction": 0.95},
    "latency": {"p99_e2e_ms": 70},
    "throughput": {"basis": "elapsed", "min_requests_per_s": 1000}
  },
  "objective": {"metric": "elapsed_ms", "direction": "min"},
  "candidates": [
    {"candidate_id": "configuration-a", "experiment": {}}
  ]
}
```

上面的空实验仅展示结构。运行文件中的每个候选独立声明一个完整 `Experiment`，采用现有严格 schema 和默认值，没有公共基配置加 patch 的隐式合并。

一个计划只接受一种 `kind`：

| kind | 每个候选必须提供 | 冻结比较对象 |
|---|---|---|
| `simulation` | `candidate_id`、内联 `experiment` | 全部 RequestSpec 字段、execution seed、完整 evaluation 契约 |
| `sweep` | `candidate_id`、内联 `experiment`、`load_plan` | 完整负载生成计划、seed、replications、扫描轴、长度分布、服务时间分布、drain，以及实验评价契约 |
| `dag` | `candidate_id`、内联单配置 `dag_plan` | DAG source、所有节点内容/目标模型/依赖/release，以及全局 execution |

普通请求的 ID、arrival、input/output tokens、priority、timeout、token 身份和 cache namespace 都参与冻结身份。不能通过缩短输出、更换 seed、删掉难点或改变测量窗口提高排名。候选模型、精度、TP/PP、副本、KV 预算和独立 Profile 可以不同，原配置完整保留在报告中。

`sweep` 实验的 `workload` 必须为空，由 `load_plan` 唯一生成负载。DAG 候选必须提供一个内联 `pools` 放置，不能再嵌套候选枚举。`pool_constraints` 仅对 DAG 可用，按 `model_id` 声明额外分池约束；被指定的模型池必须存在于每个候选中。

计划文件内的计算、P2P、Workspace Profile 及各副本 Profile 路径相对计划文件目录解析。Python API 已经捕获 dict 时可以传 `base_dir=...`，避免为了处理相对路径再次读取可变计划文件。

## 判定与最紧约束

每条约束保留 `actual`、`operator`、`limit`、`unit`、`sample_count`、`verdict`、`reason` 和归一化 headroom。

| 判定 | 处理 |
|---|---|
| `PASS` | 所有执行、样本及声明约束均通过，并且目标可观测，才能进入排名 |
| `FAIL` | 任意失败、超时、未完成、未排空或超限会否决候选；完整失败原因仍保留 |
| `UNKNOWN` | 低样本、缺失指标/分母或未声明目标，不能推定通过，也不当成零 |

显存使用原 MemoryPlanner 的逐卡 `budget_peak_bytes`，每卡限制取物理容量、`max_budget_fraction` 和可选 `max_budget_bytes_per_rank` 的最小值。账本已经包含 KV 预留、workspace、安全余量等预算，不能再加一次逻辑 KV 使用量。

P99 保留原请求级线性插值口径。默认最低完成样本为 100，也可明确设置教学样本下限。样本门槛本身保留在 checks 中；最紧资源列表只包含 memory、latency、throughput、task，不能让“刚好达到 100 个样本”冒充显存或延迟瓶颈。

上限约束 headroom 为 `1 - actual/limit`，下限约束为 `actual/limit - 1`。负值表示超限，越小越紧；不同量纲按这个无量纲余量排序。阈值为零时不构造比值，不伪造无限余量。

扫描要求**每个声明点、每次 replication**都通过联合约束，且满足原 `load_plan` 的 SLO 与有限观测分类门槛。一个低负载点通过，不能掩盖高负载点失败；多个重复中也不能只挑一个有利样本。原 legacy SLO 和新的联合约束保留各自字段。

DAG 同时检查整体与显式分池约束，且要求所有节点完成、反馈/DMA/PP 引用排空。最后输出 token 的 `makespan_ms` 与资源排空的 `elapsed_ms` 分开报告：即便任务输出已完成，排空被截断的候选仍不排名。

## 支持的排序目标

`objective.direction` 显式声明 `min` 或 `max`；省略 objective 或使用 null 时，约束评估仍保存，但选择判定为 UNKNOWN，不排名。相同原始数值保留并列名次，采用竞争排名，例如 1、1、3；不按 candidate ID 破除平局。

| kind | metric | 指标口径 |
|---|---|---|
| simulation | `elapsed_ms` | 原模拟结束时间，包含到达间隙及需要的 drain |
| simulation | `p99_e2e_ms` | 完成请求的原 E2E P99；样本门槛仍适用 |
| simulation | `throughput_rps` | constraints.throughput.basis 选定的实际完成请求率；未声明 basis 为 UNKNOWN |
| simulation | `goodput_rps` | 原引擎按 evaluation SLA 计算的 goodput，保持原到达 cohort/窗口口径 |
| simulation | `elapsed_output_tokens_per_s` | 原 token 时间戳在 elapsed 区间的输出 token 数 / elapsed |
| simulation | `window_output_tokens_per_s` | 原 token 时间戳在声明测量窗口内的输出 token 数 / 窗口长度 |
| sweep | `worst_p99_e2e_ms` | 所有点、所有重复中最大的单 run E2E P99，不是合并后挑低值 |
| sweep | `min_throughput_rps` | 全部点、重复中按声明 basis 计算的最小请求率 |
| sweep | `min_window_throughput_rps` | 全部点、重复中最小的实际窗口内完成请求率 |
| sweep | `min_window_output_tokens_per_s` | 全部点、重复中最小的实际窗口内输出 token 率 |
| dag | `makespan_ms` / `elapsed_ms` | 任务最后输出 / 资源观测结束，两种口径分别使用 |

窗口到达 cohort 为 `[0,T)`；完成和 token 事件在右端点 `T` 提交时计入窗口，即 token 计数区间为 `[0,T]`。drain 后才产生的 token 不会因为请求最终完成而回填到窗口吞吐。缺少明确窗口或原时间戳时指标保持缺失。

扫描的 minimum/worst 目标是在**同一个冻结 grid**上对全部观测取保守值。它不混合不同负载的均值、不插值容量、不声称数学稳定性。改变 grid、seed、replication 或测量契约需要另一份计划，不能与原候选混排。

## 不可行候选与无效输入

合法配置的静态显存不足保留为 `NOT_STARTED_STATIC_INFEASIBLE`，继续执行其它候选。因为同一 base 实验的启动预算不依赖负载点，该候选的所有声明点/重复都标明未启动，不伪造显存数值或延迟样本。运行中的 KV、Workspace、swap 或超时失败直接保留原请求结果并否决候选。

无效配置、损坏或身份不匹配的 Profile、Profile 校准域错误不会改名为 OOM 后继续排名。所有候选选中的 Profile schema/identity 在第一个候选运行前检查；输入变更导致整次运行 `INPUT_CHANGED`，而不是生成混合来源的比较。

## 字节快照与离线复现

运行前捕获完整候选集的 timing、native PP 通信、Workspace Profile，以及所有 replica 角色的原始字节、SHA256 与来源。实际引擎只读取这些字节的只读快照，原源文件的临时 A→B→A 替换不会改变计算使用的字节。持续变更仍会否决整份报告。

报告递归精确恢复原路径，不包含随机临时目录。`configuration_sha256` 用 Profile 字节 SHA 替代路径：同一路径内容改变会产生新身份，复制到新的重放目录不会改变语义配置身份。

顶层 `profile_inputs` 是全部实际使用字节的扁平列表，带 `candidate_id`；各候选也有自己的 `profile_inputs`，DAG 项额外带 `model_id` 与 `role`。写入器保存固定数字目录下的原始 Profile、`inputs/manifest.json` 和 `replay.plan.json`。删除源 Profile 后仍可执行重放计划，得到相同工作负载/配置 SHA 与排名。

单计划最多 32 个候选；启动前还检查整个集合不超过声明的 250,000 次请求和 5,000,000 次事件预算，warmup 与各扫描重复包含在预算内。Profile 原始字节总量限制为 64 MiB。超限需要缩小明确边界，不能静默截短候选或样本。

## 教学示例的预期结果

- `screen-simulation.plan.json`：100 个冻结 FIFO 请求；一个联合通过，另三个分别仅触发静态显存、P99、吞吐失败。synthetic 计时与 toy 模型不是真机数据。
- `screen-sweep.plan.json`：同一 50/100/200 requests/s 周期负载、每点两次重复；较慢配置的高负载点失败，不能由低负载点抵消。
- `screen-dag.plan.json`：原 52 节点人工教学图、四张 48 GiB 声明设备、32B/7B 三套部署。25,000 ms 墙钟约束下 BF16 TP2 与 INT4 单卡候选通过，FP8 单卡候选不通过。这些排名只对声明的 dense 几何和解析性能假设成立，不是实际 Claude 轨迹或 W7900D 集群测量。

Python API：

```python
from mlsys_sim.screening import run_candidate_screening, write_screening_report

report = run_candidate_screening("examples/course/screen-simulation.plan.json")
write_screening_report(report, "/tmp/screening-artifacts")
```
