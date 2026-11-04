# LLM 推理系统模拟器

[English](README.md) · [Python API](docs/api.md) · [校准](docs/calibration.md) ·
[验证规范](docs/validation-protocol.md) · [系统规范](docs/specification.md) · [PP 与 CPU swap](docs/pipeline-and-swap.md) · [Prefix Cache 与分块预填充](docs/prefix-cache-and-chunking.md) · [混合批处理](docs/mixed-batching.md) · [多副本路由](docs/multi-replica.md) · [ROCm 采集](docs/rocm-collection.md) · [Qwen3 与 vLLM 内部 hook](docs/qwen3-and-vllm-hooks.md) · [原生 PP 采样与 Profile](docs/native-pp-capture.md) · [原生 PP 通信微基准与校准](docs/native-pp-communication.md) · [Workspace 与容量验证](docs/workspace-and-capacity.md)

1.10 版增加原生 PP 小消息 lookup 校准、训练拐点保护和分尺寸独立留出报告；保留逐 rank Workspace Profile、完整 Runtime scratch 归因契约与全新 Runtime 容量／OOM 区间验证。已有能力包括原生 PP 通信微基准与 Profile、Qwen3／vLLM 内部采集、多副本调度、TP/PP、缓存、分块、混合批处理、CPU swap、校准和验证。用户提供的 AMD/NVIDIA 参考包用于格式适配与审计；原始包及 prompt/output 全文不随库分发，现有覆盖不代表通用硬件准确率。仓库为[私有仓库](https://github.com/eyuansu62/mlsys-simulator)，许可证待选择，尚未发布 PyPI。

## 安装和运行

需要 Python 3.11 或更新版本。核心没有必需第三方运行依赖，JSON/TOML 使用标准库；已安装 PyYAML 时可读 YAML。
实际 stage 采集另需 AMD 机器上已安装的 ROCm PyTorch 和 Runtime Adapter；CPU 模拟与 fixture 验证不会安装 GPU 软件或退回 CPU 冒充 GPU 计时。

```bash
python -m pip install .
mlsys-sim simulate --experiment examples/fifo.json --out runs/fifo
mlsys-sim simulate --experiment examples/memory_pressure.json --out runs/memory
mlsys-sim simulate --experiment examples/tp2.json --out runs/tp2
mlsys-sim predict --experiment examples/pp2_analytical.json --out runs/pp2
```

也可使用 `python -m mlsys_sim`，或保留的 `simulate.py`／`predict.py` 源码入口。每次运行必须使用新的输出目录，已有结果不会覆盖。

| 示例 | 用途 |
| --- | --- |
| `fifo.json` | 三请求手算时间线 |
| `memory_pressure.json` | 小 KV pool 触发请求级抢占和重算 |
| `tp2.json` | 双 rank 分片、通信和在线调度 |
| `pp2_analytical.json` | 不等长 PP stage 的容量和孤立前向时间分析 |

## PP 与 CPU swap 示例

```bash
mlsys-sim simulate --experiment examples/v1_1/pp_golden.json --out runs/pp-golden
mlsys-sim simulate --experiment examples/v1_1/cpu_swap.json --out runs/cpu-swap
mlsys-sim simulate --experiment examples/v1_1/pp_cpu_swap.json --out runs/pp-cpu-swap
```

PP 以 stage／microbatch 独立推进，同一请求必须等上一迭代的采样反馈后再进入下一 Decode。CPU swap 模拟有限主机 KV pool 和 D2H／H2D 搬运，恢复时不重算历史 KV。PP 手算例显式关闭 P2P，用于理想流水线对照。这些机制都在 CPU 上计时和记账，不执行真实设备传输，详见 [PP 与 CPU swap](docs/pipeline-and-swap.md)。

## Prefix Cache 与 Chunked Prefill 示例

```bash
mlsys-sim simulate --experiment examples/v1_2/prefix_cache.json --out runs/prefix-cache
mlsys-sim simulate --experiment examples/v1_2/chunked_prefill.json --out runs/chunked-prefill
mlsys-sim simulate --experiment examples/v1_2/pp_prefix_chunk.json --out runs/pp-prefix-chunk
```

Prefix Cache 使用显式 `prompt_token_ids`、namespace 和完整 token 前缀匹配不可变 KV block；缺少 token ID 时绕过缓存，不按输入长度猜命中。缓存与请求共用固定显存池，无人引用的 block 按 LRU 淘汰，正在使用的 block 不提前释放。Chunked Prefill 保留前缀上下文、分次计算输入，只在最后一块完成后输出首 Token；连续批处理在启用分块时优先调度 Decode，未开启混合策略时各批保持同一 phase。完整配置、PP／swap 组合和计时边界见[专门说明](docs/prefix-cache-and-chunking.md)。

## Prefill/Decode 混合批处理

```bash
mlsys-sim simulate --experiment examples/v1_3/mixed_batching.json --out runs/mixed-batching
mlsys-sim simulate --experiment examples/v1_3/pp_mixed_cache_swap.json --out runs/pp-mixed-cache-swap
```

`strategy.mixed_batching=true` 将就绪 Decode 优先放入批次，再用剩余 Token 预算加入 Prefill，两类工作共用一次 stage 执行。分块是独立开关；未启用分块时，完整剩余 prompt 能放入才会加入。FIFO 仍独占一个请求。PP 保留逐成员进度与反馈依赖，Prefix Cache／CPU swap 的安全引用规则不变。

Synthetic 混合批次使用独立的 `calibration.mixed_ms`。Profile 必须提供包含 `request_phases` 的精确混合形状，不能通过相加／取最大纯阶段耗时补齐缺失数据。配置、形状和计时约定见[混合批处理说明](docs/mixed-batching.md)。

## 多副本调度与负载均衡

```bash
mlsys-sim simulate --experiment examples/v1_4/round_robin.json --out runs/round-robin
mlsys-sim simulate --experiment examples/v1_4/prefix_aware.json --out runs/prefix-aware
mlsys-sim simulate --experiment examples/v1_4/replicas_pp_mixed_cache_swap.json --out runs/replicas-pp-mixed-cache-swap
```

`strategy.replicas` 配置独立模型副本，设备按副本／PP stage／TP rank 排列。`replica_routing` 支持轮询、最小未完成负载和最小负载内的缓存亲和选择。路由只使用到达时可见状态，不读取未来输出长度；请求选定副本后保持绑定。各副本独立计 KV、缓存、Host swap 容量和 DMA，时钟、测量窗口和执行预算则是全局一份。Profile 后端通过 `replica_profile_paths` 为每个副本提供严格匹配的 Profile，见[多副本说明](docs/multi-replica.md)。

## 校准与验证闭环

下面的数据是**手算测试 fixture，不是实测**。显式允许 fixture 后，所有派生 Profile 和报告仍保留 fixture 来源，不会转成实测证据。

```bash
mlsys-sim calibrate --dataset examples/v1/step_measurements.fixture.json \
  --profile-version fixture-v1 --out runs/calibration --allow-test-fixture
mlsys-sim simulate --experiment examples/v1/profile_serving.json \
  --out runs/profile-serving
mlsys-sim compare --predicted runs/profile-serving \
  --observed examples/v1/online_observation.fixture.json \
  --out runs/compare --allow-test-fixture
mlsys-sim validate --dataset examples/v1/step_measurements.fixture.json \
  --profile-version holdout-v1 --holdout-group prefill-16 \
  --out runs/holdout --allow-test-fixture
```

`profile_serving.json` 引用首个命令生成的 `profile.json`，并在配置中允许 fixture。校准按精确形状汇总重复观测；留出按配置／分组连通分量切分；在线比较使用独立的观测运行契约。实测接入方法见[校准文档](docs/calibration.md)和[验证规范](docs/validation-protocol.md)。

## 真机采集与证据验收

```bash
mlsys-sim collect --kind probe --out runs/rocm-probe
mlsys-sim validate-hardware --plan examples/v1_5/fixture-validation.plan.json \
  --out runs/acceptance-fixture --allow-test-fixture
```

Probe 会报告硬件可用性及版本证据。上面的验收示例全部为手算 fixture，必须保留 `hardware_validated=false`。实际 stage 采集计量同步后的 Runtime 调用；RCCL AllReduce 检查计时与正确性；在线采集保留服务端 Token-ready 与客户端 SSE 到达两种边界。验收报告分别给出 `PASS`、`FAIL`、`INSUFFICIENT_EVIDENCE` 或 `TEST_FIXTURE_ONLY`，命令执行完成本身不等于真机验收通过。完整计划、来源约束和参考数据缺口见[ROCm 采集说明](docs/rocm-collection.md)。

## Qwen3 模型导入与组合示例

```bash
mlsys-sim import-model --kind qwen3 --config examples/v1_6/qwen3-4b.hf.json \
  --name Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --out runs/qwen3-model
mlsys-sim predict --experiment examples/v1_6/qwen3-4b-predict.json --out runs/qwen3-predict
mlsys-sim simulate --experiment examples/v1_6/qwen3-r2-tp2-pp2.json --out runs/qwen3-combined
```

Qwen3-4B 的 residual hidden width=2560，query projection width=32×128=4096，保持真实维度。权重和性能模型补充 tied embeddings、Q/K Norm 与最终 RMSNorm；同一物理 rank 的词表权重去重，PP 两端位于不同设备时各有副本。组合示例使用 synthetic 时间，实际触发 R2×TP2×PP2、缓存、混合批处理和 CPU swap，不加载模型权重或证明真机吞吐。配置指纹与内部采集边界见[专门说明](docs/qwen3-and-vllm-hooks.md)。

## vLLM 原生 PP 延迟接收与切片传输

```bash
mlsys-sim simulate --experiment examples/v1_7/qwen3-native-r2-tp2-pp2.json \
  --out runs/qwen3-native-combined
```

`strategy.pipeline_transport="vllm_native"` 显式使用 Qwen3 的 `hidden_states`／`residual` 双张量、按张量判断的 TP 切片 P2P 与接收端 AllGather。固定 MRv1 版本的 hook 只观察实际自然触发的接收，不提前读取 lazy tensor 或消费输出；阶段 Profile 扣除接收等待及其 AllGather，仍计输入拷贝与计算。Profile 保留实际 padding／张量描述并只允许精确查找；边界传输单独计费一次，可显式选择通信 Profile，未选择时沿用解析模型。接收端完整驻留 Buffer 不随 wire slice 除以 TP。示例仍是 CPU 上的 synthetic stage 模拟，实机采集、校准方法与支持边界见[原生 PP 文档](docs/native-pp-capture.md)。

## 独立校准原生 PP 通信

通信采集使用单向 src send／dst receive 和接收端 TP AllGather，预分配缓冲区，按物理位置、dtype 与实际软件栈分别校准。它需要已安装 HIP PyTorch／RCCL 的真机，不增加 CPU 核心的必需 GPU 依赖。显式选中的通信 Profile 替代原生边界解析估计，不改阶段 Profile identity，也不重复计局部模型 collective；缺少 primitive、位置或字节覆盖会明确失败。未选择通信 Profile 时保留原解析路径。采集、直接仿射拟合、独立留出验证和仅用于软件验证的 fixture 示例见[通信校准文档](docs/native-pp-communication.md)。

```bash
mlsys-sim simulate --experiment examples/v1_8/native-pp-serving.fixture.json \
  --out runs/native-comm-fixture-serving
```

此示例使用手算通信 Profile 和 synthetic 阶段耗时，保留 fixture 来源，不代表真机测量。

显式使用 `calibrate --native-pp-communication --native-pp-model size-lookup` 可按消息尺寸拟合全部重复观测的均值，保留非单调拐点，拒绝跨跳变／稀疏区间及外推；旧 affine 仍为默认。留出同时报告逐次误差、逐尺寸均值误差和拒绝覆盖率，详见[小消息校准](docs/native-pp-communication.md)。

## Workspace 归因与容量／OOM 验证

Workspace 校准只接收完整的 Runtime-owned allocation ledger：按实际底层 storage 去重 alias，等释放完成才扣除占用，逐 rank 区分真实零和未知。独立 Workspace Profile 只替换动态 Workspace 预算，不改变阶段耗时或重复计固定储备；显式选择后按精确形状查询，不插值、不解析回退。默认 vLLM WorkspaceManager 观察器只提供部分 managed scratch 信息，不能自动代表所有 temporary Workspace。

容量扫描为每个候选点和重复实验建立全新 Runtime／进程，区分 typed device OOM、逻辑 KV 准入拒绝、超时及其他错误，并与整个有效配置的模拟结果比较。报告只给有证据的相邻成功→OOM 区间和分辨率；不稳定、非单调或删失结果不能推成一个精确通用阈值。适配器集成契约、fixture 示例和证据边界见[专门说明](docs/workspace-and-capacity.md)。

```bash
mlsys-sim predict --experiment examples/v1_9/native-pp-workspace.fixture.json \
  --out runs/workspace-fixture-predict
```

示例联合使用手算 Workspace／通信 Profile 和 synthetic 阶段耗时，不代表真机测量。

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

`load_experiment(path)` 加载配置，`experiment_from_dict(mapping)` 校验内联配置；`predict_experiment(experiment)` 分析形状、显存和通信，不模拟排队。校准／留出／比较同样提供 Python API，见 [API 文档](docs/api.md)。

## 机制与模型边界

每个副本的在线引擎支持单卡、同步 TP group 和多个 TP group 组成的 PP 流水线；多个副本在同一全局事件时间线上独立推进。单组 FIFO 独占一个请求直至结束；Continuous Batching 在安全边界重选工作。PP 默认每个 microbatch 只有一个 phase；开启混合策略后，可在同一 microbatch 中放入 Prefill 和 Decode。不同 stage 可以推进不同 microbatch，这仍是声明了简化条件的调度器。

首次 Prefill 构建输入 KV，最终输入块完成后生成首 Token；最新采样 Token 的 KV 在下一 Decode 才计算。已输出数量和已计算 KV 位置分别记录。每步先检查物理预算和逻辑 KV，再跨 rank 原子预留；PP 在进入 stage 0 前为整个流水线预留 KV，并另计常驻激活 Buffer。压力下限制准入／缩 Batch，必要时在安全边界按配置重算恢复或经 CPU swap 保存／恢复 KV；历史输出不重复生成，停顿计入延迟。PP 回收先排空在途 wave，DMA 期间不启动新的 GPU 工作。

固定 KV pool 释放逻辑 block 不会缩小 GPU 预分配 tensor。显式 Workspace Profile 逐 rank 精确替换动态预算，缺形状不回退；未选择时仍使用阶段 Profile 中的受支持观测或明确标记的解析假设。张量分配器碎片和 Kernel 周期不做细粒度模拟。

性能后端分为 `synthetic`、`analytical` 和 `profile`。Profile 严格核对模型、设备、软件版本、并行布局与计时边界；ragged 形状只精确匹配，同质 Batch 仅在所需角点齐全时插值。混合形状还按有序成员 phase 精确匹配；原生 PP Profile 也只允许精确形状，并保留实际 padding／传输描述，二者均不做插值或外推。其它形状越界默认拒绝，显式外推会记录来源。Profile 的整 stage 时间已包含局部 collective，不重复加通信。

Warmup 使用独立时间线，不进入测量指标，也不把 KV 留给测量。当前仅支持 `empty_cache` 初始状态。测量到达窗口为半开区间，窗外请求为 `NOT_ADMITTED`；是否 drain 及最大 drain 时间显式配置。

PP 在线模拟已支持 stage 队列、P2P 和反馈依赖；`predict` 仍是未分块、无缓存的形状分析，不执行在线混合调度或输出在线 SLA。NVMe swap 和量化存储格式仍未实现。混合 stage 使用共享权重与通信的解析假设或精确 Profile，不执行真实 GPU Kernel。软件功能不等于真实硬件验证完成，具体见[能力矩阵](docs/coverage-matrix.md)。

## 指标和输出

输出包括 `manifest.json`、`result.json`、`requests.jsonl`、可选 `events.jsonl` 和日志。Prefix／分块运行另报缓存占用、命中和实际输入计算量。PP 结果补充各 stage 忙时／空泡和在途进度，swap 结果补充主机占用、DMA 状态及逐请求搬运次数／字节／耗时。多副本结果补充逐请求路由和各副本资源诊断，并在统一窗口重算全局指标。Manifest 保存展开配置、代码／输入哈希、各副本 Profile 快照／来源及比较契约；校准和验证报告保存数据来源、训练／留出成员及覆盖域。

`TTFT = first_token - arrival`。正常完成且输出至少两个 Token 时，`TPOT = (last_token-first_token)/(outputs-1)`；单 Token 的 TPOT 为 null。ITL 包括抢占与重算停顿，恢复不重置首 Token 时刻。完成率保留已准入请求中的失败、超时和未完成项，排除窗外未准入请求。

Goodput 只计完成且满足全部已声明 SLA 的请求。要求 TPOT SLA 时，TPOT 不适用的请求不算达标。分位数采用排序后 `(n-1) × q` 位置的线性插值。命令执行完成不代表请求全部成功、覆盖率完整或 SLA 达标。

## 开发

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check .
mypy --check-untyped-defs --ignore-missing-imports mlsys_sim
python -m build
twine check dist/*
```

实际检查结果以验证报告和 CI 运行记录为准。正式发布前仍需选择许可证；PyPI 发布是独立步骤，见[发布清单](docs/releasing.md)。
