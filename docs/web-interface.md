# 本地 Web 实验台

Web 界面将表单转换为现有 `Experiment`，使用原有 CPU 解析引擎执行，然后展示原始结果。
不加载模型权重，不调用 vLLM，也不会用浏览器中的计算替换模拟器。
使用本地 HTML/CSS/JavaScript 和 Python 标准库，无新增运行依赖或外部 CDN。

## 启动

在安装了本库的 Python 3.11+ 环境中运行：

```bash
mlsys-sim-web
# 或
python -m mlsys_sim.web
```

浏览器打开终端显示的 `http://127.0.0.1:8765`。用 `--port 8766` 可修改端口；
按 Ctrl+C 停止服务。源码 checkout 也可直接执行模块入口。
服务始终绑定本机 `127.0.0.1`，不提供远程监听开关。

## 一个实验的交互

1. 选择 Qwen3-0.6B、1.7B、4B，或填入自定义 Dense/GQA 结构。预设来自现有课程几何，
   包括独立 Query 投影宽度、共享词表权重与 Q/K Norm。自定义结构采用原 `ModelSpec` 默认值；
   本表单自定义项不声明 Q/K Norm 或 final Norm。当前表单使用 BF16、单 GPU。
2. 选择或修改显存（GiB）、计算峰值（TFLOP/s）和显存带宽（GB/s）。
   RX9070XT 算力/带宽是课程声明的名义输入，24/48 GiB 项是人工教学配置，均非硬件自动探测。
3. 填写固定请求总数、输入/输出 token 长度、到达方式、速率和种子。
   可选固定间隔、Poisson 或全部同时到达；每次运行保留全部声明请求并尝试排空。
   这不是固定时长稳态扫描，客户端并发/到达率扫描仍使用现有 `sweep` 命令。
4. 按需展开高级配置，修改 FIFO/连续批处理、batch/token 预算、Chunked Prefill、混批和 KV 池。
5. 点击「运行模拟」。原引擎完成后，页面显示延迟、吞吐、显存图表和全部请求明细。
   配置修改后会标记旧结果；新运行失败会清除旧图并显示原始错误码。

「导出配置」下载完整的可重放 `Experiment` JSON，可以交给原 `simulate --experiment` 命令。
「导出结果」下载同一份原始汇总、逐请求结果、图表数据与来源元数据。
表单的 KV 池及运行时/安全余量等实际预算都会出现在导出的配置中。

## 图表口径

| 显示项 | 数据及分母 |
| --- | --- |
| TTFT P50、E2E P99、延迟分位数 | 原引擎 `result.latency_ms` 的完成请求统计；一个输出 Token 没有 TPOT 样本，显示未知而非 0 |
| 总体输出吞吐 | 原引擎 `result.throughput.output_tokens_per_s`；从模拟 0 时刻到实际结束，包含到达空闲及排空 |
| 吞吐时间线 | 原逐请求 `token_times_ms` 按实际时间分箱，tokens 除以每箱真实时长；失败前已经输出的 Token 也保留 |
| 显存预算峰值 | 原逐设备账本的 `budget_peak_bytes`，包括安全余量和固定 KV 池；属于累计峰值，不是瞬时 GPU 实测 |
| 逻辑 KV 使用量 | 原事件内的 `kv_used_bytes`；已包含在固定预留池中，不再加到物理预算上 |

显存时间线保留原内存事件与最终快照；超过展示上限时按桶保留 KV/预算峰值并记录
`meta.memory_thinned=true`。这不是对每个分配/释放事件的完整瞬时重建。
失败或未完成请求会显示状态，完成子集的延迟不能代表所有请求通过。
静态显存不可行直接显示 `STATIC_MEMORY_INFEASIBLE`，不绘制虚假的零占用。

## 支持范围与执行限制

本界面是单卡解析模型的简单入口；TP/PP、多副本、Prefix Cache、CPU swap、DAG、
实测 Profile 与校准/验证仍通过现有 CLI/Python API 使用。
所有 Web 运行保持 `source=analytical_uncalibrated`、`hardware_validated=false`。
现有 9070XT 实测精度限制不会因增加界面而改变。

每次最多 500 条请求、8192 输入和 512 输出 tokens；事件数上限 100000，
模拟时间上限 600000 ms，独立进程墙钟超时 20 秒。所有请求共享一个运行槽，
并发提交返回忙碌提示。服务不持久化实验或读取用户指定的文件路径，不执行 shell 命令。
若达到预算，应减少负载后重新运行；失败/未完成样本不会被删除以制造通过结果。

## GitHub Pages 与在线部署

GitHub Pages 可以发布静态说明、图表和 HTML 报告，但不支持运行服务端 Python，
见 [GitHub 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)。
当前界面的 `/api/presets` 和 `/api/simulate` 依赖本地 Python 服务，直接把静态文件
上传到 Pages 不会得到可运行的模拟器。

若需要打开公网网址即可配置并运行，可以另行部署支持 Python 的服务，或将核心
适配为浏览器内运行的 Python/WebAssembly。前者需要处理公开服务的访问与资源限制；
后者需要处理运行时加载、浏览器工作线程和执行限制，均尚未包含在本版本。
当前服务器保留 loopback 与同源约束，不提供将本地服务直接公开的开关。
