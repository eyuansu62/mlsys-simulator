# 冻结 DAG 与异构模型池

第 9.3 章要求在同一冻结任务图上比较部署策略。新协调器在父请求实际完成时释放子请求，32B 与 7B 池共享一个因果时钟，各池复用已有 TP/PP、多副本、Prefix Cache、CPU swap、Chunked Prefill 与混合批处理引擎。

```bash
mlsys-sim dag --plan examples/course/dag-fork-join.plan.json --out /tmp/dag-fork-join
mlsys-sim dag --plan examples/course/dag-candidates.plan.json --out /tmp/dag-candidates
```

输出保存在 CLI 运行目录的 `artifacts/` 下。`report.html` 可直接查看摘要、时间线与表格；`report.json`、`nodes.csv`、`devices.csv` 与 `gantt.svg` 分别提供完整数据、逐节点时间、每卡利用率/KV 水位与甘特图。候选枚举另有 `candidates.csv`，总 HTML 按数字编号链接各候选的 HTML、SVG 和 CSV。写入器使用新目录，不覆盖既有预测。写入的 JSON 带 UTC `stored_at` 与 `prediction_id`。

每个被选择的计算、通信、Workspace Profile 都保留原始 JSON 字节、SHA256、角色与来源。预测前后、候选之间检查文件不变；同一路径的字节改变会改变 `configuration_sha256`。写入器复制这些字节到 `inputs/`，生成 `inputs/manifest.json` 和引用副本的 `replay.plan.json`。删除或移动原 Profile 后仍可从保存的 replay 计划离线重放；原路径不参与 Profile 内容身份，不会因复制到新的输出目录而改变配置 SHA。

## 输入与冻结边界

```json
{
  "schema_version": 1,
  "trace_source": "synthetic_teaching_trace",
  "trace_description": "明确说明轨迹与硬件假设",
  "nodes": [
    {"node_id": "planner", "model_id": "32B", "input_tokens": 512, "output_tokens": 24},
    {"node_id": "search", "model_id": "7B", "input_tokens": 768, "output_tokens": 16,
     "dependencies": ["planner"], "release_ms": 0}
  ],
  "pools": [
    {"model_id": "32B", "role": "planning", "experiment": {}},
    {"model_id": "7B", "role": "retrieval", "experiment": {}}
  ]
}
```

这里的空 `experiment` 只是结构示意，不能构成两个独占设备池。可运行文件声明完整实验；每池使用现有 `Experiment` 的严格 schema，`workload` 必须为空，节点是唯一被测负载来源。设备 ID 必须在所有池之间独占，设备顺序仍为 replica → PP stage → TP rank。各池可以有不同模型、TP/PP、副本数、KV 预算与独立 Profile。

节点还接受 `priority`、`timeout_ms`、`prompt_token_ids`、`cache_namespace`。Prefix Cache 命中需要显式 token 身份，单有相同长度不会推断命中。依赖 ID 必须存在、不可重复，不允许自环或有向环。未知字段、布尔型 token 数、无穷时间等均在运行前拒绝。

首版接受内联的完整 `Experiment`，不接受把 `experiment` 或其 section 写成外部文件名。传入计划文件路径时，池内计算、通信、Workspace Profile 路径及每副本路径都相对该计划文件解析；Python API 传入 dict/dataclass 时，可通过 `load_dag_plan(mapping, base_dir=...)` 声明相对路径根目录，省略时保留调用者提供的路径。Profile 的模型、dtype、TP/PP、分片与硬件身份检查继续由原性能模型执行，跨模型不共享 Profile。

`trace_sha256` 覆盖轨迹来源和所有节点的规范化内容。候选枚举仅替换 `pools`，不会修改 prompt 长度、输出目标、依赖或节点顺序。更换模型精度与副本数属于 infra 策略；改变节点的请求内容或目标模型则是另一条轨迹。

## 因果顺序与时间口径

节点的实际入池时间为：

\[
t_{ready}(v)=\max\left(t_{release}(v),\max_{u\in parents(v)}t_{complete}(u)\right)
\]

所有父节点成功完成后才有可用的 `ready_ms`。`release_ms` 是相对于测量起点的“不早于”约束，不是来自旧部署的实测到达时间。每个时刻先提交所有计算/通信完成与超时，再按冻结节点顺序接收入池请求，最后启动所有池的可运行工作。新请求产生的同刻零超时也先于下一次计算启动处理。池之间的同优先级事件按计划中的池顺序确定性排序。

调度与路由只看到已接收的请求状态、在途引用和本地缓存。输出长度仅作为原引擎的执行 oracle，在提交生成 token 时判断结束，不用于预测排队或选择最短任务。

| 字段 | 含义 |
|---|---|
| `dependency_wait_ms` | `ready_ms - release_ms`，因依赖尚未满足而等待的时间 |
| `arrival_ms` | 实际接收入模型池的时间；正常成功接收时等于 `ready_ms` |
| `first_compute_start_ms` | 首次真正开始计算的时间，PP 取 stage 0 真正启动时刻 |
| `queue_wait_ms` | 首次计算开始减 `ready_ms`；未开始计算时为 null |
| `completion_ms` | 该请求最后一个输出 token 提交时刻 |
| `makespan_ms` | 所有节点完成时的最大 `completion_ms`；存在未完成节点时为 null |
| `elapsed_ms` | 协调器观测时钟，包含完成后的必要资源排空 |
| `resources_drained` | 没有在途 PP wave、反馈、DMA 或未完成请求 |

例如 PP 最后 token 已经提交，反馈引用仍可能保持到稍后时刻。此时任务 makespan 不会被反馈尾部加长；协调器仍排空资源。若在排空期间触发预算，保留任务 makespan，整体状态为 `UNRESOLVED`，候选不进入排名。

`critical_path` 给出从最晚完成节点逆向选择最晚完成父节点得到的**观测依赖链**。`node_elapsed_ms` 包含资源排队、抢占及恢复，`release_wait_ms` 显式补充 release 间隙，两者之和等于链的 `elapsed_ms`。它不是隔离执行服务时间的理论下界，不能据此把资源竞争当成零开销。

## 失败、预算与缓存生命周期

失败、超时或被截断的父节点使其尚未接收的后代变成 `BLOCKED_DEPENDENCY`，这些节点不会进入任何模型引擎。健康子图继续执行。在途超时仍遵守原引擎的 KV 引用、PP feedback 与 DMA 生命周期，不会提前释放内存或生成未提交 token。

`execution.max_events` 与 `max_simulated_ms` 是整个 DAG 的预算；各池实验仍保留自己的上限。事件数只计接收请求和原引擎事件，不把 DAG 依赖 bookkeeping 重复计数。池的局部时间上限也是相对于测量起点的绝对时钟，不会在空闲后重置。已健康完成且空闲的池到达其局部上限时正常关闭；之后才释放到该池的节点将报告 `MODEL_POOL_UNAVAILABLE`。

空闲池不会因为暂时没有请求而立即 finalize，Prefix Cache 可供后续 DAG 节点复用。每池 `execution.warmup` 独立运行并检查成功，测量阶段使用全新空 KV/Prefix 状态，与原 `simulate` 的 warmup 语义一致。全局 `execution.warmup` 不带模型标注，因此拒绝；请在相应池中声明。有限 DAG 实验不使用 `measurement_window_ms` 到达窗口，避免将依赖解锁和窗口外排除混为一谈。

每卡 `utilization` 是其同步 TP 组或 PP stage 的已执行 busy 时间除以整个协调器 `elapsed_ms`，不是实测 GPU occupancy。KV 水位直接来自原 MemoryPlanner 的逐卡高水位。各模型池拥有独立链路、host pool 与 DMA lane；跨池共享 PCIe/CPU/host 内存竞争、检索、DAG 控制开销及跨模型数据传输延迟尚未建模，报告会标明这些边界。

## 第 9.3 章三配置示例

`dag-candidates.plan.json` 是 **52 节点的人工教学轨迹**，来源明确为 `synthetic_teaching_trace`，不是课程未提供的真实 claudecode 采样。8 条检索、分析、推理、验证与报告分支经过 join、critic、final 收束，同一份冻结节点供三套部署执行。

| 候选 | 32B 池 | 7B 池 | 设备总数 |
|---|---|---|---|
| `bf16-tp2-small-r2` | BF16、TP2 | FP8、单卡 × 2 副本 | 4 |
| `fp8-single-small-r3` | FP8、单卡 | FP8、单卡 × 3 副本 | 4 |
| `int4-single-small-r3` | INT4、单卡 | FP8、单卡 × 3 副本 | 4 |

示例仅声明 4 张 48 GiB 设备，100 TFLOP/s、800 GB/s 与链路参数是教学假设，不能当作 W7900D 的测量或已核实规格。`course32B`/`course7B` 是显式给定的 dense GQA 宏观几何，不冒充某个官方 Qwen3-7B revision。INT4 有效宽度 0.515625 byte/weight 包含声明的 GPTQ-128 类 scale 元数据近似，FP8/INT4 使用 fused dequant 解析假设；它们不是量化内核性能或质量验证。

枚举结果只对全部节点完成、资源排空且没有任何整体错误的候选按 makespan 排名。换成课程真实冻结轨迹与独立校准 Profile 后，先保存预测，再实测选定配置；留出配置应更改 7B 副本数或 KV 预算等 infra 维度，保持同一 `trace_sha256`。本功能补齐课程模拟器实验，不等于完成 ≥4 卡 W7900D 端到端真机闭环。

## Python API

```python
from mlsys_sim.dag import load_dag_plan, run_dag_candidates, simulate_dag, write_dag_report

plan = load_dag_plan("examples/course/dag-candidates.plan.json")
result = run_dag_candidates(plan)
write_dag_report(result, "/tmp/frozen-dag-prediction")
```

`EventDriver` 是给新工作流使用的内部引擎适配器。它复用现有多副本路由与调度，普通 `simulate` 不经过此层。修改原引擎 private 接口时，需要同时运行 `test_event_driver` 与 `test_dag` 的差分、手算及生命周期回归。
