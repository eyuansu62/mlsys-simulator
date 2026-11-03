# LLM 推理系统模拟器

[English](README.md) · [Python API](docs/api.md) · [校准](docs/calibration.md) ·
[验证规范](docs/validation-protocol.md) · [系统规范](docs/specification.md)

V1 软件实现覆盖 CPU 在线模拟、解析模型、版本化 Profile、校准、分组留出验证和运行／候选方案比较。当前没有 AMD 实测数据，也未建立真实硬件预测准确率。仓库为[私有仓库](https://github.com/eyuansu62/mlsys-simulator)，许可证待选择，尚未发布 PyPI。

## 安装和运行

需要 Python 3.11 或更新版本。核心没有必需第三方运行依赖，JSON/TOML 使用标准库；已安装 PyYAML 时可读 YAML。

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

在线引擎支持单卡或单个同步 TP group。FIFO 独占一个请求直至结束；Continuous Batching 在安全 Step 边界优先准入 Prefill，再执行 Decode。每个 Batch 只有一个阶段，不等同于完整 vLLM 调度器。

首次 Prefill 构建输入 KV 并生成首 Token；最新采样 Token 的 KV 在下一 Decode 才计算。已输出数量和已计算 KV 位置分别记录。每步先检查物理预算和逻辑 KV，再跨 rank 原子预留。压力下限制准入／缩 Batch，必要时暂停低优先级请求、释放 KV，之后重算恢复；历史输出不重复生成，停顿计入延迟。

固定 KV pool 释放逻辑 block 不会缩小 GPU 预分配 tensor。Workspace 可使用 Profile 中的观测；缺少数据时明确回退解析假设，不声称实测。张量分配器碎片和 Kernel 周期不做细粒度模拟。

性能后端分为 `synthetic`、`analytical` 和 `profile`。Profile 严格核对模型、设备、软件版本、并行布局与计时边界；ragged 形状只精确匹配，同质 Batch 仅在所需角点齐全时插值。越界默认拒绝，显式外推会记录来源。Profile 的整 stage 时间已包含局部 collective，不重复加通信。

Warmup 使用独立时间线，不进入测量指标，也不把 KV 留给测量。当前仅支持 `empty_cache` 初始状态。测量到达窗口为半开区间，窗外请求为 `NOT_ADMITTED`；是否 drain 及最大 drain 时间显式配置。

PP 仅提供解析分析。在线 PP、多副本、CPU swap、Prefix Cache、完整 Chunked Prefill 和量化存储格式仍未实现，并明确拒绝。校准数据接入与留出工具已经实现，不等于真实硬件验证完成，具体见[能力矩阵](docs/coverage-matrix.md)。

## 指标和输出

输出包括 `manifest.json`、`result.json`、`requests.jsonl`、可选 `events.jsonl` 和日志。Manifest 保存展开配置、代码／输入哈希、Profile 来源及比较契约；校准和验证报告保存数据来源、训练／留出成员及覆盖域。

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
