# 受控0.6B整步成本（2026-11-17）

这是RX9070XT、BF16、TP1/PP1单副本的显式校准包。实际采样环境为GPU_MAX_HW_QUEUES=1、固定2GiB KV池、同一ROCm/vLLM镜像与源SHA。仅适用于已声明的离线内部步骤边界，不自动用于vLLM在线SLA或其它队列/硬件/TP/PP/混批/缓存策略。

- dataset.json.gz：1824条无损归一化步骤，含原始报告字段、辅助nohook观测及每个消费文件的SHA。912条训练，456条内部形状留出，456条内部重复。
- fit.json.gz、fit-spec.json、split.json：不可变拟合与预冻结稳定性策略；两项scale共享，Attention独立固定项为零。
- models.json、experiments/、calibrations/：既有CLI/Python可加载的内联格式；预测范围Prefill256–1024，Decode batch2–8/context1024–1039，独立阶段。
- frozen/：另行前瞻的原始冻结输入/预测表（gzip无损压缩）及commitment；raw SHA按解压后的JSON原始字节核对。
- receipts/：AMD主机测前回执、完整三进程状态与原始输出文件SHA。
- forward-report.json.gz：独立三进程24格/2736个报告步骤的全部结果，不回流拟合。

从仓库可重放内部划分与拟合：

```sh
mlsys-sim-cost freeze --dataset examples/runtime_step_20261117/dataset.json.gz --recommended --step-affine --out runs/controlled/split
mlsys-sim-cost fit --dataset examples/runtime_step_20261117/dataset.json.gz --split runs/controlled/split/split.json --fit-spec runs/controlled/split/fit-spec.json --calibration-id rx9070xt-qwen3-06b-queues1-step-v1 --out runs/controlled/fit
mlsys-sim-cost validate --dataset examples/runtime_step_20261117/dataset.json.gz --fit runs/controlled/fit/fit.json --out runs/controlled/internal
mlsys-sim-cost predict --experiment examples/runtime_step_20261117/experiments/model-0.json --phase decode --query-tokens 1 1 1 1 --context-lengths 1024 1024 1024 1024 --out runs/controlled/b4
```

freeze/validate针对真实已存在的内部观测，selection_kind仍为retrospective_group_holdout；它不是新的盲测。真正的新采集使用frozen/与receipts/，参数和预测均在采样前冻结。前瞻结果可由tests/test_published_step_study.py逐格及逐步重算，并让实际AnalyticalModel重放全部308条冻结预测。所有输出目录必须为新的。

原始自动/多队列数据与旧四参数包保留，没有用新参数改写它们；相同GPU与软件名称不意味着可以跨Runtime配置借用此包。排队、显存可行性、CPU型号变化、在线交付和其它GPU仍需各自证据。hardware_validated=false不否认这次实测，而是避免将窄域结果升级为整个Profile/硬件范围背书；本轮事实看scoped_hardware_measurement_performed和verdict。

[完整图文报告](../../docs/runtime-step-20261117/README.md) · [新拟合族说明](../../docs/runtime-step-calibration.md)
