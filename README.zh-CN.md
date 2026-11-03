# LLM 推理系统模拟器

[English](README.md) · [Python API](docs/api.md) · [贡献说明](CONTRIBUTING.md)

当前正在准备初始库版本，尚未发布到 PyPI。许可证待选择，当前未授予开源许可；正式公开发布前需完成[发布清单](docs/releasing.md)。

这是开发计划 v0.3 的首个可执行版本。模拟器在 CPU 上运行，输出请求时间线、服务指标、显存预算和调度事件。它支持单卡与单个 TP group 的在线模拟，以及 PP 的解析分析。

所有内置设备参数和性能系数都是教学假设。`synthetic` 为手算测试使用的固定耗时；`analytical` 为未校准解析估计。当前版本没有 AMD 实机校准、模型推理或真实 CUDA/HIP 分配。

## 运行

需要 Python 3.11 或更新版本。JSON、TOML 输入和全部核心功能只使用标准库，无必需第三方运行依赖。在源码目录安装：

```bash
python -m pip install .
```

安装后可在源码目录运行示例：

```bash
mlsys-sim simulate --experiment examples/fifo.json --out runs/fifo
mlsys-sim simulate --experiment examples/memory_pressure.json --out runs/memory
mlsys-sim simulate --experiment examples/tp2.json --out runs/tp2
mlsys-sim predict --experiment examples/pp2_analytical.json --out runs/pp2
```

原有源码入口继续可用：

```bash
python3 simulate.py --experiment examples/fifo.json --out runs/fifo
python3 simulate.py --experiment examples/memory_pressure.json --out runs/memory
python3 simulate.py --experiment examples/tp2.json --out runs/tp2
python3 predict.py --experiment examples/pp2_analytical.json --out runs/pp2
python3 -m unittest discover -s tests -v
```

也可以使用 `python3 -m mlsys_sim simulate --experiment ... --out ...`。
输出目录必须不存在，重复使用会返回 `OUTPUT_EXISTS`，不会覆盖实验。
CLI 最后一行是 JSON，含 `status`、`result_path`、`error_code`。
退出码 0 表示模拟执行完成，2 表示配置或执行错误，3 表示预算耗尽。运行完成与满足 SLA 是两件事。

| 示例 | 用途 |
| --- | --- |
| `fifo.json` | 三请求手算：首 Token 在 10、24、34ms，完成于 14、24、36ms |
| `memory_pressure.json` | 仅三个 KV blocks 的池，两个长输出请求触发抢占与重算 |
| `tp2.json` | 两 rank 分片、通信和在线 Batch 的解析估计 |
| `pp2_analytical.json` | 不均匀层切分的 PP 显存与服务时间；在线模拟会明确拒绝 |

YAML 输入在系统已安装 PyYAML 时可用；缺少它时给出明确错误，并可直接使用 JSON/TOML。项目没有把 PyYAML 声明为必需依赖。

## Python 库接口

```python
from mlsys_sim import CalibrationSpec, Experiment, RequestSpec, simulate

experiment = Experiment(
    workload=(RequestSpec("r1", arrival_ms=0, input_tokens=16, output_tokens=4),),
    calibration=CalibrationSpec(backend="synthetic"),
)
run = simulate(experiment)
print(run.result["counts"])
```

`load_experiment(path)` 加载配置，`experiment_from_dict(mapping)` 校验内联配置，`predict_experiment(experiment)` 进行解析分析。Python API 返回内存结果，CLI 负责创建实验目录；详细语义见 [API 文档](docs/api.md)。

## 配置

顶层为 `schema_version` 和六类输入：`workload`、`infra`、`strategy`、`calibration`、`evaluation`、`execution`。未提供字段采用 `schema.py` 的默认值，未知字段、非有限数值和非法分片会被拒绝。配置可以内联或引用相对于当前配置文件的独立文件。

`workload` 为请求列表，包含 request_id、arrival_ms、input_tokens、output_tokens，以及可选 priority、timeout_ms。`output_tokens` 只用于执行回放的终止条件；调度和 victim 选择不使用未来的实际输出长度。较大的 priority 表示更重要，timeout_ms 相对于到达时间。

`strategy.model` 定义 Dense GQA 模型形状和精度。当前分片采用无 Sequence Parallel 的 Megatron 风格方案，Embedding 与 LM head 不共享权重。设备按 stage、TP rank 的顺序映射；`stage_layers` 给出各 stage 的层数。只支持一个模型副本。

权重、KV 和激活只接受每元素 2、4、8 bytes 的浮点存储假设。8-bit 和 packed 量化格式明确拒绝，尚未建模 scales/zeros 等元数据。

`calibration.backend` 支持 `analytical` 和 `synthetic`。前者对 Linear、Attention、Misc、输出头与通信分别估计；后者使用每个阶段固定的 Batch 耗时，只用于逻辑测试。真实测量 Profile、拟合/插值和自动校准将在后续增量实现。

## 调度与恢复

`fifo` 独占执行一个请求直至结束。`continuous_batching` 在每个安全 Step 边界重选 Batch，优先处理可准入的 Prefill，然后执行 Decode；每个 Batch 只有一个阶段，不支持 Chunked Prefill 或 Prefill/Decode 混合 Batch。这是明确的简化策略，并非完整 vLLM 调度器。

首次 Prefill 为输入构建 KV 并产生第一个输出 Token。最新采样 Token 的 KV 在下一 Decode 才计算，因而已输出数量与已计算 KV 位置分别记录。

每个 Step 前分别检查物理峰值预算和逻辑 KV block 增量，所有 rank 都通过后才提交预留。KV 不足时缩小 Batch、限制新准入；无法推进时，在安全边界优先抢占低优先级且最近到达的请求，并保留输出历史。恢复时重新构建已知前缀 KV，不重复产生历史输出。压力期间通过排空存活工作保证进展。

固定 KV pool 的逻辑 block 回收不会降低已预留的物理显存。Workspace 形状超预算时缩小执行形状，单请求仍不可行则明确失败；不会靠无限 eviction 重试掩盖不可行配置。CPU swap、Prefix Cache、任意 Kernel 中途抢占和真实故障恢复尚未实现。

## 指标与结果

每次运行生成：

- `manifest.json`：展开配置、内容哈希、代码哈希、seed、版本与来源。
- `result.json`：完成状态、TTFT/TPOT/ITL 分位数、吞吐、Goodput、失败统计、重算和显存账本。
- `requests.jsonl`：请求到达、首 Token、结束、生成数量、计算位置和抢占记录。
- `events.jsonl`：详细时间线；关闭 detailed_events 仅关闭记录，不改变模拟。
- `logs.txt`：诊断；配置失败也保留结果和日志。

`TTFT = first_token - arrival`。正常完成且输出至少两个 Token 时，`TPOT = (last_token-first_token)/(output_tokens-1)`；单 Token 的 TPOT 为 null。ITL 包括抢占与重算引入的间隔，首 Token 时刻不会在恢复时重置。失败和未完成请求不从完成率分母中剔除。

可选 measurement_window_ms 定义从零开始的到达窗口，窗口内到达请求作为测量 cohort，其完成允许 drain；报告显式窗口和 cohort 统计。未指定时使用从零到模拟结束的时长。Goodput 只计完成且满足已声明 SLA 的请求；若要求 TPOT SLA，TPOT 不适用的请求不计为达标。分位数采用排序后线性插值，位置为 `(样本数-1) × q`；ITL 汇总保留已产生输出的失败请求间隔，TTFT/TPOT 分位数只使用正常完成请求。

同刻事件先处理完成/释放，再处理超时，再处理到达，最后调度。已经在执行中的超时请求在 Step 完成前仍占资源；预算截止会保留明确未完成状态和在途占用。

## 范围与下一步

本版已建立可执行的配置—解析模型—内存规划—事件模拟—结果链路。PP 仅提供解析服务时间和逐 rank 容量，在线 SLA 指标为空；在线 PP 和多个副本明确拒绝。

下一步是目标 AMD/Runtime 的低干扰服务时间、Workspace 与 RCCL 数据采集；随后加入实测 Profile 覆盖检查、留出比较及 PP stage/microbatch 事件。当前测试验证机制和公式，不证明真实硬件上的绝对延迟准确率。

计划依据：[开发计划 v0.3](https://gist.github.com/eyuansu62/f91df8b7597801ca4fd91bb3bb8c98cc)。当前支持范围见[能力矩阵](docs/coverage-matrix.md)。
