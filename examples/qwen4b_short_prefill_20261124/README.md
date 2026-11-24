# Qwen3-4B 短输入 Prefill

本研究覆盖一张 RX 9070 XT，Qwen3-4B BF16、TP1/PP1、同步 eager、ROCM_ATTN，输入为 64 或 128 token，输出固定 16 token。[测前方案](plan.json) 列出校准与留出的请求流、顺序和门槛。

每个新进程运行 512 个闭环请求。对每个零上下文纯 Prefill 的有序形状，先取进程内步耗时中位数，再对两个校准进程的中位数取中位数。并发 1、2、4 共覆盖 22 种形状。每个进程中每种形状至少需要四步，校准和验收均逐进程检查 5% 的最大相对误差门槛。完整请求的端到端 P99 使用首次 Scheduler 入队到最后一个输出 token 就绪的时长，验收门槛为 25%。

## 9070 XT 独立实测

2026 年 11 月 24 日的校准和留出各使用六个独立 vLLM 进程，每阶段 3,072 个请求。模型覆盖 22 种有序形状，含此次补采的并发 2 和 4 的全部 16 种非等长排列。留出逐进程评分，共 44 项中位数和 6 项端到端 P99。原始 Scheduler、API 和 EngineCore 文件保存在测量主机，各进程的 [校准样本](calibration.json) 与 [留出样本](holdout.json) 含逐步 Prefill 墙钟和原始文件的 SHA-256。

| 并发 | 重复 | 基础模型最大形状误差 | 新模型最大形状误差 | 新模型端到端 P99（ms） | 真机端到端 P99（ms） | 端到端 P99 误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 14.31% | 0.52% | 311.56 | 318.50 | 2.18% |
| 2 | 1 | 14.09% | 0.25% | 295.58 | 318.39 | 7.16% |
| 1 | 1 | 5.53% | 0.31% | 275.84 | 295.55 | 6.67% |
| 1 | 2 | 6.19% | 0.62% | 275.84 | 296.98 | 7.12% |
| 2 | 2 | 14.26% | 0.15% | 295.58 | 311.77 | 5.19% |
| 4 | 2 | 14.06% | 1.52% | 311.56 | 325.49 | 4.28% |

形状验收通过 **44/44** 项，端到端 P99 通过 **6/6** 项。同一批请求上，基础模型的两项通过数分别为 10/44 和 6/6。每项形状误差分别比较该进程的一种有序形状，以真机中位数为分母。表中的最大值保留该进程最差的一项。

[模型与预测承诺](predictions.json)、[冻结的基础模型对照](baseline.json) 和 [逐形状评分](score.json) 可按下文命令复核。覆盖范围为单卡同步 eager 的热身后短输入闭环请求。更长输入、混批、TP/PP，以及罕见单步尖峰需要各自的精度证据。

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
