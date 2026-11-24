# Qwen3-4B 短输入 Prefill

本研究覆盖一张 RX 9070 XT，Qwen3-4B BF16、TP1/PP1、同步 eager、ROCM_ATTN，输入为 64 或 128 token，输出固定 16 token。[测前方案](plan.json) 列出校准与留出的请求流、顺序和门槛。

每个新进程运行 512 个闭环请求。对每个零上下文纯 Prefill 的有序形状，先取进程内步耗时中位数，再对两个校准进程的中位数取中位数。并发 1、2、4 共覆盖 22 种形状。每个进程中每种形状至少需要四步，校准和验收均逐进程检查 5% 的最大相对误差门槛。完整请求的端到端 P99 使用首次 Scheduler 入队到最后一个输出 token 就绪的时长，验收门槛为 25%。

## 在模拟器中使用

从仓库根目录运行以下命令，生成与基础模型、证据和硬件配置绑定的运行时 Profile。生成的 `experiment.json` 可用于原有 CLI 和 Python API。为常规模拟填入 workload，或者调用 `predict_closed_loop` 生成闭环负载。

```sh
study=examples/qwen4b_short_prefill_20261124
python scripts/short_prefill_study.py materialize \
  --plan "$study/plan.json" --evidence "$study/calibration.json" --out runs/short-prefill
mlsys-sim-cost predict --experiment runs/short-prefill/experiment.json \
  --phase prefill --query-tokens 64 128 --context-lengths 0 0 --out runs/short-prefill-step
```

旧的 v1 Profile 仍可直接加载。选择新 Profile 时，读入器会重算有序形状中位数并核验来源和基础模型。其他查询继续使用基础模型自身的运行域与成本规则。

## 复算验收

```sh
study=examples/qwen4b_short_prefill_20261124
python scripts/short_prefill_study.py score --plan "$study/plan.json" \
  --evidence "$study/holdout.json" --predictions "$study/predictions.json" \
  --out runs/short-prefill-verdict.json
```
