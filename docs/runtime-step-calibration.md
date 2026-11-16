# 整步骤校准与跨进程稳定性

整步仿射拟合是显式选择的成本族，适合只采集了完整步骤时间的场景。
原分量拟合、Profile、CLI、Python、Web 和 Agent 默认行为不变。

## 模型和门禁

`时间 = a × (nominal 非 Attention 成本 + nominal Attention 成本) + c`

`a` 和 `c` 采用非负仿射最小二乘。每个独立采集进程、阶段、形状的中位数
具有相同权重；给某个进程重复追加步骤不会增加它的拟合权重。
导出到既有 `AnalyticalCostSpec` 时，两项 scale 都为 `a`，Attention 独立固定项为零，
Runtime 固定项为 `c`。这是整步骤经验模型，不能据此归因独立的 GPU Attention
效率或测得的 CPU 时间。预测 breakdown 中的公式族名称也不代表拟合过程的分量归因。

`freeze_step_fit_spec` 在拟合前固定算法、聚合方式和稳定性门槛，并绑定旧 split 的 SHA。
每个训练域和每个训练形状至少有两个进程；至少两个形状的 nominal 成本可辨识。
同一形状的训练进程中位数 `max/min` 默认不能超过 1.25，边界包含在内。
不满足条件时返回 `step_cost_fit_rejected_v1`，保留全部诊断，不生成可执行校准参数。
这个训练门禁不能保证未来进程仍然稳定，仍需要独立的测前冻结预测验证。

原有模型、设备、Runtime 配置、执行模式、Attention 后端、调度策略和形状范围的
匹配条件继续生效。固定 KV 池、线程或 ROCm 环境开关属于实际 Runtime 控制，采集方
必须把它们纳入配置指纹；不能仅修改预测标签，也不能使用目标延迟反推快/慢标签。

## CLI

先执行 `mlsys-sim-cost freeze --dataset dataset.json --membership membership.json
--step-affine --max-training-process-ratio 1.25 --out runs/step-split`。
再执行 `mlsys-sim-cost fit --dataset dataset.json --split runs/step-split/split.json
--fit-spec runs/step-split/fit-spec.json --calibration-id controlled-step-v1 --out runs/step-fit`。
验证使用 `mlsys-sim-cost validate --dataset dataset.json --fit runs/step-fit/fit.json
--out runs/step-validation`。

`freeze` 不加 `--step-affine` 时不生成新 fit policy；`fit` 不加 `--fit-spec` 时仍运行
原分量算法，不会因为分量记录缺失就自动改用整步骤模型。
拒绝拟合时，CLI 保存 `fit-rejected.json` 和输入字节 manifest，返回 3；输入或完整性
错误返回 2。成功产物使用原 `fit.json`、实验和校准导出格式，原 `predict` 可以加载。
所有输出目录必须是新的，原数据不被覆盖；fixture 仍需要显式许可。

## Python

```python
from mlsys_sim import freeze_step_fit_spec, fit_step_costs, validate_analytical_costs

policy = freeze_step_fit_spec(split, max_training_process_ratio=1.25)
fit = fit_step_costs(dataset, split, calibration_id="controlled-step-v1", fit_spec=policy)
if fit["kind"] == "analytical_cost_fit_v1":
    report = validate_analytical_costs(dataset, fit)
else:
    diagnostics = fit["diagnostics"]
```

`split` 仍由 `freeze_cost_split` 对实际已有观测生成，并绑定完整数据集；它是回顾性划分。
不得用虚构的未来观测占位再改写同一个数据集。前瞻实验需要另存测前预测表、输入哈希、
采样计划和主机回执，再采集新的进程；新数据只用于验证，不回流到同一次拟合。
