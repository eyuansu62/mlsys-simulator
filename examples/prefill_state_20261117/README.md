# Prefill 稳态 Profile（2026-11-17）

本包只覆盖报告中声明的RX9070XT/Qwen3-0.6B BF16、queues1、2GiB KV、CPU10主线程与固定逐形状warmup。旧默认模型、旧边界和所有原始失败数据不覆盖、不替换。

- dataset.json：两个全新进程1968条正式观测，保留源顺序、原始字段、辅助观测、采集文件哈希。
- profile.json：54个实测网格点；新边界使用各会话exact-shape中位数的中位数，保留全部raw proof与1.10政策。
- experiment.json：相对Profile路径，既有CLI/Python可加载；仅在测量网格内插值，默认禁止外推。
- frozen/：原始输入、71点预测与Profile的gzip无损副本、commitment；SHA按解压后原JSON字节核对。
- receipts/：测前主机回执和三个全新进程状态、原始文件SHA。
- forward-report.json.gz、forward-stationarity*：全部33格/3636正式步骤的误差与固定稳定性检查。
- train-a/b-warmup.json.gz：1536条训练前置请求，固定规则排除于参数拟合，原值保留。

```sh
mlsys-sim calibrate --dataset examples/prefill_state_20261117/dataset.json --stationarity-policy examples/prefill_state_20261117/stationarity-policy.json --profile-version q06-steady-prefill-v1 --out runs/prefill/profile
mlsys-sim-cost predict --experiment examples/prefill_state_20261117/experiment.json --phase prefill --query-tokens 768 --context-lengths 0 --out runs/prefill/q768
```

新数据不是旧留出的再拟合。训练后固定Profile/政策，再作三次新进程新形状验证；完整warmup单独保存，不能事后改标角色。Profile的hardware_validated=false避免将窄域结果升级成全硬件范围认证，实测事实看scoped_hardware_measurement_performed、scope与verdict。

[图文报告](../../docs/prefill-state-20261117/README.md) · [API与方法](../../docs/prefill-state-calibration.md)
