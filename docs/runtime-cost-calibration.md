# 固定开销与 Attention 成本校准

本页记录1.21的原分量族和历史回顾性结果。1.22的新增能力见
[整步骤拟合与稳定性门禁](runtime-step-calibration.md)，新真机数据见
[0.6B队列对照和独立前瞻报告](runtime-step-20261117/README.md)。原冻结参数与原始失败结果均保留。

1.21增加显式的运行域成本校准，不覆盖旧默认参数，不改变既有整阶段Profile ABI。
默认CLI、Python、Web和Agent仍使用原先的模型；只有显式加载新的不可变成本参数时才校正。
本轮没有新增运行依赖、远程执行或GPU采集。

## 校正模型与运行域

每个模型／阶段／模式／后端分开选择参数，公式为：

`校正时间 = sN×原nominal非Attention成本 + sA×原nominalAttention成本 + 层数×Attention固定项 + 阶段固定项`

- nominal基线使用效率1、launch/framework开销0。旧0.5效率和旧启动开销不会再重复叠加。
- 非Attention项包含Linear、Embedding/LM-head和Misc；用独立profiler进程的这些GPU分类时间拟合比例。
- Attention比例与按层固定项用非负仿射拟合；至少需要两个可辨识的分量形状。
- 阶段固定项取训练墙钟残差中位数的非负值，保留原始负残差诊断。
  **这是经验上的未建模工作，不是测得的CPU、launch或每层kernel启动时间。**

参数保存在不可变 `AnalyticalCostSpec`，通过 `CalibrationSpec.analytical_costs` 内联加载；
不在运行中重新读取、拟合或替换可变Profile文件。
匹配条件包括模型／设备／Runtime版本与配置摘要、独立墙钟边界、Prefill/Decode、
配置的执行模式、Attention backend、核心调度策略及保留形状范围。
eager与Graph不能互借参数；模型、后端、策略或形状改变后不匹配就报 `CALIBRATION_MISMATCH`。

第一参数族只覆盖BF16、TP1/PP1/单副本、独立Prefill/Decode。混批、recompute、
TP/PP、多副本等旧工作流继续支持原模型，不把本次窄域成本自动推广过去。
Profile和synthetic后端不叠加此校正；fixture参数需要显式opt-in。
对已有Web和Agent接口不新增未协商字段，也不静默切换目录或校准版本。

## 本次来源与冻结划分

原始 `summary_package(1).zip` 本地原件及解压trace保留未修改，SHA256为
`32567daea53b3a0700d07be9a0f332577fddb2c87070f6515d43df15cb49bb85`。
导入器核对完整ZIP和所有消费文件的原始字节，公开去除本地路径后的观测与来源SHA，
不公开整包环境日志。所有6174条归一化观测均无损保存，包括未使用记录。

| 分区 | 成员 | 观测数 |
| --- | --- | ---: |
| 训练 | 正式进程1/2：S128/S2048、B1/B16；对应profiler端点分类 | 2748 |
| 主留出 | 正式进程3：未参与拟合的S512、B4 | 684 |
| 保留形状重复 | 正式进程3：端点形状 | 1368 |
| 未使用、仍保留 | 其余中间形状记录 | 1374 |

主留出为6个阶段／形状单元，来自3个独立进程。逐步样本嵌套于进程，
不能把684步当作684次独立硬件试验。主留出与重复分区共享同一组进程，分别报告。
训练/验证进程互斥，主留出形状互斥。成员、数据SHA与30%单元中位数误差门槛在拟合前冻结，
之后没有按留出结果更换成员、阈值或参数。

全部18点汇总此前已经被查看，因此本轮是**事后分组留出，不是新的盲测采集**。
当前记录只证明MRv2/async/FULL_AND_PIECEWISE配置和 `RocmAttentionImpl`，没有逐步实际
Graph/eager分派证据；不建立eager或其他后端精度结论。
实际步骤的Decode上下文由已验证的调度前缀重建多重集，并保留来源方法和上下界；
profiler聚合未给所选16步索引，使用自身调度记录的保守上下文上界代表值，明确标近似。
GPU分量与墙钟来自不同采集渠道。4B进程1/2的KV分配状态差异保留为诊断，
配置摘要不把这些分配结果当控制参数，也不据此证明唯一误差成因。

## 执行与重放

安装本库后可以重放公开的去敏数据，不需要访问原机器或下载模型：

```bash
mlsys-sim-cost freeze --dataset examples/runtime_cost_20261116/dataset.json.gz --recommended --out runs/cost/split
mlsys-sim-cost fit --dataset examples/runtime_cost_20261116/dataset.json.gz --split runs/cost/split/split.json --calibration-id local-cost-v1 --out runs/cost/fit
mlsys-sim-cost validate --dataset examples/runtime_cost_20261116/dataset.json.gz --fit runs/cost/fit/fit.json --out runs/cost/validation
mlsys-sim-cost predict --experiment examples/runtime_cost_20261116/experiments/model-2.json --phase prefill --query-tokens 512 --context-lengths 0 --out runs/cost/prefill
```

源码中也可用 `python -m mlsys_sim.cost_cli`。每个输出目录必须是新的，freeze/fit JSON设为只读；
哈希和只读属性用于追溯／防误改，不是对恶意重写者的数字签名或硬件证明。
validate成功返回0，完整验证但精度判定失败返回3；输入、成员或身份不一致返回2，失败报告仍保留。
输出包含全部逐步误差、分组CSV、SVG和HTML。发布的大数据文件使用标准gzip无损压缩。

`predict` 是单阶段估计，要求内联JSON（可gzip），输入读取一次并按同一字节快照求SHA；
它不运行显存可行性、排队或SLA验证。原 `predict` 会同时计算Prefill和Decode，
不能把只覆盖C≈1024的成本参数用于其它Decode context，也不静默回退。
原CLI继续支持其文件引用方式，新阶段工具只接受内联实验。

原包可另行导入：`PYTHONPATH=. python scripts/import_cost_trace.py --package-root <解压目录> --archive <原ZIP> --out <新目录>`。
不提供并核对原ZIP时，导入结果保留为fixture/reference，不能提升成已核实来源。

Python接口为 `freeze_cost_split`、`fit_analytical_costs`、`validate_analytical_costs`，
以及 `AnalyticalModel.estimate_step`。导出的 `calibrations/model-N.json` 与完整实验均可作为普通
配置加载，原数据和冻结参数不会因验证而重新拟合。

## 结果

详见[完整图文报告](runtime-cost-20261116/README.md)。主留出单元MAPE从37.82%降至9.72%，
6/6满足拟合前冻结的30%门槛；逐步MAPE从39.99%降至11.47%。
保留形状重复单元MAPE从34.77%降至8.64%，12/12通过。
少数本来接近的单元略有退步，全部记录保留，没有逐点择优切换模型。
这些是本数据集和声明运行域内的结果，不推广为全域硬件、端到端服务P99或显存准确性。
