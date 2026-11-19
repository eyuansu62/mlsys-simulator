# LLM 推理系统模拟器使用教程

本教程用两个 CPU 实验带你跑通安装、配置、运行和结果分析。需要 Python 3.11 或更新版本，无需 GPU 或模型权重。

## 1. 安装并运行第一个例子

```bash
git clone https://github.com/eyuansu62/mlsys-simulator.git
cd mlsys-simulator
python -m pip install .
python -m mlsys_sim simulate --experiment examples/fifo.json --out runs/fifo
```

`fifo.json` 使用预设的合成耗时，模拟三个先后到达、按 FIFO 顺序处理的请求。打开 `runs/fifo/result.json`，应看到 **3 个请求全部完成、TTFT 中位数 19 ms、输出吞吐约 166.7 tokens/s**。重跑时换一个新的 `--out` 目录，避免覆盖已有结果。

## 2. 看懂配置

| 配置 | 决定什么 |
| --- | --- |
| `workload` | 请求何时到达，输入和输出各有多少 Token |
| `infra` | 设备容量、计算与带宽参数，以及设备间链路 |
| `strategy` | 模型结构、调度策略、批大小、KV 缓存池和 TP／PP 配置 |
| `calibration` | 如何估算耗时。`synthetic` 使用预设耗时，`analytical` 根据模型和硬件参数估算 |
| `evaluation`、`execution` | 延迟目标、随机种子、事件记录等设置 |

## 3. 做一次 KV 缓存池对照实验

在仓库根目录复制示例，然后用编辑器打开副本：

```bash
cp examples/memory_pressure.json memory_large.json
```

只把 `memory_large.json` 中的 `strategy.memory.kv_pool_bytes` 从 `24576` 改为 `49152`。请求、调度策略和合成耗时保持相同，这样才能把两次运行的差异归因到 KV 池配置。

```bash
python -m mlsys_sim simulate --experiment examples/memory_pressure.json --out runs/memory-small
python -m mlsys_sim simulate --experiment memory_large.json --out runs/memory-large
```

这两组在当前示例下的结果如下：

| KV 池 | 完成请求 | 抢占次数 | 重算 Token | 端到端延迟中位数 |
| --- | ---: | ---: | ---: | ---: |
| 24 KiB | 2/2 | 1 | 32 | 50 ms |
| 48 KiB | 2/2 | 0 | 0 | 28 ms |

两组请求都完成了。小池需要抢占请求并重算历史 KV，因此端到端延迟更高。这个对照用于理解模拟机制，数值来自示例中的合成耗时。

## 4. 读结果并排查异常

1. 在 `result.json` 里先看 `status` 和 `counts`，确认有多少请求完成、失败、超时或未完成。
2. 查看 `latency_ms` 中的 `ttft`（到达至首 Token）、`tpot`（首 Token 之后的平均每 Token 时间）和 `end_to_end`（到达至完成）。只有一个输出 Token 的请求没有 TPOT 样本。
3. 用 `throughput.output_tokens_per_s` 比较整个模拟时间内的输出吞吐。比较延迟时也要核对完成数，以免把完成子集的低延迟误读成整体改善。
4. 用 `requests.jsonl` 定位具体请求。要跟踪事件，可以在运行前设置 `execution.detailed_events=true`，再看 `events.jsonl`。分享结果时一起保留实验配置、输出目录和 `manifest.json`。

## 5. 换用网页或进阶入口

本地网页适合快速配置单卡解析模拟：

```bash
python -m mlsys_sim.web
```

打开 `http://127.0.0.1:8765`，选择 Qwen3 预设，填写请求数量和输入、输出长度，点击「运行模拟」。试着选择固定间隔到达，只提高到达速率，对照两组的 TTFT P99、吞吐和完成数。点击「导出配置」可得到能传给 `simulate --experiment` 的 JSON。按 Ctrl+C 停止服务。[Web 实验台](web-interface.md)

| 实验问题 | 入口 |
| --- | --- |
| TP、PP、连续批处理或缓存策略对延迟有什么影响？ | 用 `simulate` 运行相应示例，例如 `examples/tp2.json` |
| 单独估算前向耗时和容量 | 用 `predict`，例如 `examples/pp2_analytical.json` |
| 到达率或客户端并发升高时，在哪些采样点违反 SLO？ | 用 `sweep` 扫描预先声明的负载点，结合完成数和延迟分位数读报告 |
| 有依赖的任务怎样在多个模型池上执行？ | 用 `dag` 运行课程计划 |

需要与 AMD 真机对齐时，先在匹配的硬件、运行时和计时边界下采集、校准，再用独立数据验证。分析结果时记录使用的是合成耗时、解析估算还是实测校准配置。[真机采集](rocm-collection.md) · [校准](calibration.md) · [验证规范](validation-protocol.md)

[完整中文使用说明](../README.zh-CN.md) · [课程实验命令与示例](course-workflows.md)
