# 第 7 章：负载、并发与 SLA 扫描

`generate_workload()` 把有界、带 seed 的开放负载展开为现有 `RequestSpec`，
`run_load_sweep()` 在每个扫描点、每次独立重复创建新的模拟器，保留实际请求、逐请求结果及 provenance。
固定 `Experiment.workload` 的原有模拟接口不变；扫描会显式用生成的工作负载替换它。

```bash
mlsys-sim generate-workload --plan examples/course/load-workload.spec.json --out runs/course-workload
mlsys-sim sweep --experiment examples/course/load-queue.experiment.json \
  --plan examples/course/load-clients.plan.json --out runs/course-clients
```

```python
import json
from mlsys_sim.config import load_experiment
from mlsys_sim.load_sweep import run_load_sweep, write_load_report

experiment = load_experiment("examples/course/load-queue.experiment.json")
plan = json.loads(open("examples/course/load-clients.plan.json").read())
report = run_load_sweep(experiment, plan)
write_load_report(report, "runs/course-clients")  # 输出目录须尚不存在
```

`report.html` 将同一份报告中的汇总表、四张 SVG 与观察边界放在一起；
`points.csv` 保存逐重复原指标，`report.json` 和 `runs/` 保存完整展开请求、种子、哈希及原结果。
报告标题明确标记 **CPU 模型预测**，不能把这些曲线称为新增的真机验证。

## 两种扫描轴

| 计划字段 | 到达过程 | 扫描的含义 |
|---|---|---|
| `rate_grid_rps` | `poisson` 或 `periodic` | 开放系统：到达不等待服务完成；改变 offered requests/s。 |
| `concurrency_grid` | `closed_loop` | 闭环系统：每个 client 最多一条 outstanding 请求；收到完成、失败或超时反馈后，等待 `think_ms` 再发下一条。 |

两种 grid 互斥、严格递增，不能把 `max_batch_size` 当成在线 client 数。
周期流第一条在 0 ms；Poisson 流第一条在第一次指数间隔后。
所有新到达均限制在半开窗口 `[0,duration_ms)`，窗口结束后最多排空 `drain_ms`。
完成与超时在同一时间戳优先于新到达，且遵循现有引擎优先级。
TP、PP、多个副本、Prefix Cache、CPU swap 和混合批处理由原引擎执行，闭环不会另起简化排队服务。

开放负载的严格 schema：

```json
{
  "schema_version": 1,
  "seed": 42,
  "arrival": {"process": "poisson", "rate_rps": 50, "duration_ms": 10000, "max_requests": 2000},
  "tokens": {"input": {"min": 128, "max": 1024}, "output": 32}
}
```

输入和输出长度可为正整数或含端点的均匀整数范围 `{min,max}`；两者与到达各有独立 RNG 子流。
可选字段为 `request_prefix`、`priority`、`timeout_ms`、`cache_namespace` 和 `prompt_token_ids`。
显式 prompt token 身份需要固定且匹配的 input 长度，才能用于 Prefix Cache 实验。
闭环计划的 arrival 改为
`{"process":"closed_loop","duration_ms":5000,"max_requests":2000,"think_ms":0}`，
不带无意义的开放 `rate_rps`。闭环展开结果依赖模型服务及资源，不能预先伪造固定到达。

## 服务时间分布与模型口径

默认 `service_time={"kind":"deterministic"}`，保留解析模型或实测 Profile 的原始时长。
教学用 synthetic backend 可显式声明
`{"kind":"empirical_step_scale","samples":[0.7,0.8,1.0,1.2,1.3]}`：
每个 phase batch / PP stage 独立、有放回地抽取一个正乘子，乘以已有 synthetic 时长。
种子按 replication、replica、stage 派生；同一计划完整重放一致。

该分布的统计单位是**阶段批次**，并非单请求服务时间；batch 大于 1 时同批请求共享服务步骤。
它只用于显式 synthetic 课程实验，不扰动解析或 Profile 结果，也不声称复原 GPU 尾延迟与相关噪声。
示例 `load-queue.experiment.json` 使用 FIFO、batch=1 和一次输出，10 ms 服务均值的意义因此直观。
`load-poisson.plan.json` 扫开放到达率，`load-clients.plan.json` 扫真实闭环 client 数。

## 分位数、失败分母和观察边界

单次运行直接保留原报告的 TTFT、TPOT、E2E P50/P95/P99，不另造定义。
汇总只从所有声明重复的原逐请求样本重算经验 P50/P90/P95/P99/P999，使用 `(n−1)q` 线性插值。
ITL 仍保留在原运行结果；单 token 输出没有 TPOT 样本。不能将缺失 TPOT 写成 0。

失败、超时和未完成请求保留在 completion/SLO 分母中；它们会否决整点通过。
每次重复都必须满足最小完成样本数及所有阈值，才能声明整个点通过；不选择最好的 seed。
缺少声明指标样本或低于 `min_completed_samples` 返回未知；默认每次重复至少 100 条完成请求，
这只是一个透明的教学门槛，并非尾部分位数置信保证。
同一时刻已经释放但因事件预算尚未 admission 的 client 也保留为 `UNFINISHED`；
尚未出现完成反馈的未来 client 请求不会被虚构出来。
`NOT_ADMITTED` 表示测量窗外，始终不作为过载证据。

吞吐保留两个不同分母契约，避免排空阶段制造误解：

| 字段 | 计数方法 |
|---|---|
| 原 `throughput.requests_per_s` / `mean_requests_per_s` | 到达窗口内的请求，包含 drain 后完成数，除以到达窗口时长。 |
| `window_completed_requests_per_s` / 图中吞吐 | 仅完成时间不晚于窗口结束的请求，除以窗口时长。 |

例如 100 ms 内到达 20 条请求，服务需 10 ms/条，200 ms 时全部排空：
原 cohort 吞吐为 200 requests/s，窗口内完成吞吐为 100 requests/s。
两者都不是长期随机系统稳定性的数学证明。

开放扫描的 `observed_stable` 操作定义为：足够样本、全部成功排空，
且窗口末 backlog 减去四分之一窗口 backlog 不超过声明容忍度。
默认容忍度 `max(2,ceil(0.05*请求数))`，可用 `backlog_growth_tolerance` 显式修改。
超过容忍度返回 `observed_overloaded`；任何失败/预算中断返回 `observed_failure`。
闭环 client 数天然有界，使用 `observed_completed`，不把它称为开放队列稳定性。

报告保留相邻 observed stable→overloaded 与 SLO pass→fail bracket 和网格分辨率。
`sla_max_observed` 只从所有重复均通过、且没有观察到开放过载的实际点中选择；不插值、不外推。
随机 P99 可下降，SLO/稳定状态可非单调，报告完整保留这些事实。

## 可复现性与资源边界

- seed 是带符号 64 位整数，bool 不算整数；replication 也是显式整数。
- 子流以 SHA-256 派生，永不使用随机化的 Python `hash()`。
- provenance 保存 generator 版本、原 spec SHA-256、展开请求 SHA-256、种子及重复号。
- 扫描开始冻结全部 timing / native PP communication / Workspace Profile 的角色、原始字节、SHA-256 和来源类型；整个扫描只从这些只读快照计算，不会在不同点重新加载可变的源文件。
- 每次运行后仍复读原文件对照，持续变化则拒绝整个混源曲线并报 `INPUT_CHANGED`；即使源文件短暂变化又恢复，实际计算仍读取已冻结的同一批字节。
- 原配置和原文件不改变，报告将临时快照路径还原为原始路径，保证同 seed 报告可重现；报告内嵌原内容，`inputs/` 保留原字节副本，角色与文件映射在 `profile_inputs` 中，可供离线审计和复现。
- 每次生成最多 100000 条请求；每个计划最多 256 个运行，所有声明请求上限之和最多 250000。
- 服务分布表最多 4096 个有限正数，整个扫描累计处理事件最多 1000000（关闭详细日志也检查）；超限失败而不是剪掉请求、seed 或图点。
- 超出 `max_requests`、非有限参数、时钟溢出、浮点下溢或零时长不前进都会显式报错。
- 输出目录独占创建，不能覆盖已冻结的预测或留出报告。

第 9.2 章可冻结模型、Profile 和 SLO，另存短 prompt 与长 prompt 负载计划，
在相同声明 client 网格、种子及重复数下各自扫描。比较物理 KV 容量和尾延迟允许的最大 client 水位；
较长 prompt 改变解析/Profile 服务与 KV 用量时，应使用相应有效的校准域。
这一步的预测依旧需要课程真机选点对照；本模块不自动把未来实测数据当成拟合材料。
