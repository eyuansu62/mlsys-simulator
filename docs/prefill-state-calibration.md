# Prefill 执行状态与稳态整步 Profile

新能力显式使用 `reported_l2_step_hook_wall_v1` 时间边界，仅用于经过声明的
TP1/PP1、单副本 Profile。原 Profile 默认边界、参数、序列化和算法保持原行为。
不能把同一观测重命名成旧边界，也不把新数据自动套入默认 Web/Agent 配置。

## 先固定测量条件，再拟合

一次进程的中位数稳定，并不说明这段采样过程平稳。前50次慢、后50次快的数据
可能仍与另一个进程有相近中位数。新政策在排序、聚合和生成 Profile 之前检查源顺序：

- Prefill 每个正式形状100条记录，按连续20条分块；Decode128条，按32条分块。
- 每个训练组至少两个独立会话；同组的工作负载和源顺序必须一致。
- 同形状的块中位数、跨会话中位数比值不得超过预冻结门槛，默认1.10。
- 所有正式记录恰好覆盖一次，不允许缺样、丢弃尾块或按record ID排序冒充时序。
- 单一 Prefill series 不能混入其它形状来掩盖不稳定；Decode 的实际 lookup 形状同样检查。

拒绝时保存完整原始数据、政策和诊断，返回 `step_profile_rejected_v1`，没有可执行samples。
它是数据一致性证据，不是数字签名，也不能仅靠JSON证明某次物理测量确已发生。

新 Profile 保存原始训练数据和政策 proof，加载时校验一次；模型查询复用只读快照。
新墙钟域还绑定调度、缓存、chunk、混批、抢占和完整MemoryConfig，改KV池或策略不能
沿用同一个opaque Runtime SHA借用参数。旧Profile域不新增这些要求。

## API 与 CLI

```python
from mlsys_sim import freeze_stationarity_policy, calibrate_step_profile

policy = freeze_stationarity_policy()
profile = calibrate_step_profile(
    dataset, policy, profile_version="prefill-steady-v1"
)
```

`dataset.notes`需要`wall_domain`和`ordered_series`；后者逐项声明session_id、group_id
及按源时间顺序排列的record_ids。`step_wall_domain(experiment)`生成前者。
策略文件必须在拟合前冻结，真实采集中的窗口、暖机、线程条件还应在采样前预登记。

已有CLI增加显式选择：

```sh
mlsys-sim calibrate --dataset dataset.json --stationarity-policy policy.json --profile-version prefill-steady-v1 --out runs/profile
```

不稳定时输出`dataset.json`与`profile-rejected.json`，返回3。没有政策的旧calibrate
路径继续原行为；新边界缺失proof会拒绝。Fixture仍须显式允许。

`mlsys-sim-cost predict`可对一个独立阶段加载Profile；相对profile_path按实验或独立校准
文件的目录解析，manifest记录实际读取的Profile原始字节SHA。普通CLI/Python预测、
Compare和已有Profile路径都保留原接口，不会静默跨越时间边界。

## 统计口径和适用范围

新边界的duration采用各会话exact-shape中位数的中位数，workspace仍取观测MAX；
原边界保留原mean/MAX。所有outlier保留。Profile内部网格可分段线性插值，默认禁止
网格外推；这不能代替独立新进程、新形状的留出验证。

若采用固定phase-specific warmup，它定义的是准备后的成本；冷/非平稳阶段另行记录，
不能把慢请求事后标成warmup。应在新留出前冻结预测、输入字节、Profile、政策和主机回执，
之后只评分，不用留出结果调整门槛、形状或参数。工作空间、在线SLA和不同CPU/队列状态
仍需自己的证据，不由一个窄域步时间Profile自动认证。
