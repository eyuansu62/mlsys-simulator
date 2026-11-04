# 1.11 P2P 小消息协议策略与独立留出

2026-11-04。在授权的 RX7900XTX → RX9070XT、BF16、TP1×PP2 物理 pair 上，
显式关闭 P2P LL/LL128 阈值路径后，两次独立留出的逐次 MAPE 为 **6.96% / 8.09%**，
覆盖均为 100%。这解决了本测量域中原约 1ms 的双峰主峰，保留所有慢尾样本。
它是显式协议策略变更；旧默认 LL 条件下的双峰仍能复现。

## 定位与对照

实际加载 RCCL 为 `2.30.7-HEAD:7da4302`，库 SHA256 为
`77d5f37cc04837d93094a1c93487ec1c2ddbb294f8c94e3392a8e469af08b0cb`。
同镜像实际 Python3.14.8、PyTorch2.13.0+rocm10.1.0、HIP build7.16.26385。
两卡无直接 peer 访问，设置 `NCCL_P2P_DISABLE=1`；实际 INFO 连接为
`SHM/direct/direct`。固定 `NCCL_PROTO=Simple`、`TORCH_NCCL_BLOCKING_WAIT=0`。

七组完成等待对照未消除旧双峰：原设备同步、事件同步、事件查询、流查询、
显式 HIP Spin、关闭 HSA interrupt，以及重复默认基线。设备原本读回的 schedule
flag 已为 Spin。SHM CE 对照也没有消除双峰。因此没有以 CPU 唤醒或漏同步定因，
没有扣掉同步阶段、改用 GPU 事件时间或丢弃慢样本。

固定的 [RCCL 源码](https://github.com/ROCm/rocm-systems/blob/7da4302/projects/rccl/src/enqueue.cc)
显示 P2P 有独立的 LL/LL128 阈值选择，不受 collective 的 `NCCL_PROTO` override
控制。本构建默认 P2P LL 阈值为每 active channel 4096B；LL128 有独立分支。
独立 COLL/INIT/ENV 日志直接观察到双方实际协议：

| 设置 | 256B / 4096B | 4098B / 8192B | 32768B |
| --- | --- | --- | --- |
| 默认阈值，`NCCL_PROTO=Simple` | LL | Simple | Simple |
| LL 阈值 16384，LL128 阈值0 | LL | LL | Simple |
| 两个 P2P 阈值均0 | Simple | Simple | Simple |

扩大阈值同时扩大约 1ms 慢区间；两个独立零阈值诊断会话均去掉原来的慢主峰。
这组源码、选路日志和参数对照将可控因素定位至该 P2P 协议路径；没有声称已证明
LL 内部某条 GPU 指令或底层硬件的最终根因。观察协议的日志探针与正式训练/留出
分开运行，未把高日志开销的诊断样本混入拟合。

## 显式策略与兼容

新增 opt-in `native_pp_primitives_v3`，沿用 v2 lazy 初始化和完整设备同步计时。
只在子 worker 的环境中设置 `NCCL_P2P_LL_THRESHOLD=0` 及
`NCCL_P2P_LL128_THRESHOLD=0`，显式非零冲突会在 probe/启动前拒绝。
worker 在导入 Torch 前和 PG 初始化后核验实际值；所有 rank、benchmark、stack
和 Profile 均绑定策略。既有 v1/v2、affine/mean lookup 和预测消费接口不变。

vLLM Hook 对显式设置的两个阈值及 SHM copy policy 记录 Runtime 指纹；未设置时
保持旧指纹。真实 Runtime 要在启动前配置同样策略并采集匹配的阶段 Profile；
不能把旧默认域的 Profile 悄悄用于新策略。跨协议/阈值的父级校准、留出和原始
导入均拒绝混用。计时函数、拟合规则、均值目标、间隔 guards 均未改变。

CPU 回归覆盖 TP/PP 两 tensor 与 receiver AllGather 只计一次、缺域在 KV 提交前
失败，以及三 Profile 与多副本、Prefix Cache、chunking、swap、混合批处理。
未新增运行依赖；旧默认 v1/v2 的完整 CPU capture JSON 与上一版逐字一致。

## 冻结与独立验证

先采一组新的 v2 默认路径对照，1,920 条、32 个训练网格尺寸；它不进入 v3 拟合。
四个全新 v3 训练会话共 **19,200 条、32 个尺寸**，覆盖 256B–16MiB：
前两会话每尺寸60次，后两会话每尺寸240次，各尺寸另有20次真实预热。
全部重复观测参与均值拟合，无阻断间隔；规则在最终留出前固定。

Lookup 与同域 affine Profile 于 **11:45:55 UTC** 冻结。留出 A 于 **11:47:04 UTC**、
B 于 **11:47:20 UTC** 分别启动，使用新进程/communicator/session；每组31个尺寸
各300次，共 **18,600 条独立留出**。每组30个尺寸未参加训练，另有4096B的独立
重复；两个未见网格也不同。记录/组/session 无训练泄漏，启动前检查输出不存在、
计划及源文件 SHA，之后未调参或重拟合。

冻结与数据的 SHA256：

- Lookup：`2d8d084d2ed9348e724ccf9a1b4537dd7abee1dec4add95d1031b46da3b7506e`
- Affine：`89536951d81dde53f9c0a491449f7d9540e5426194636791341b05947bea4927`
- 留出 A：`786025b147eae353344f19839dba4cafde722b659627c5afe73f593ee032fd98`
- 留出 B：`fca2bcd4e3b0f684563fa243be997c1c32df4a182738786efd6d57d514614b0a`

| 冻结 lookup 的指标 | 留出 A | 留出 B |
| --- | ---: | ---: |
| 覆盖 | 9,300/9,300，100% | 9,300/9,300，100% |
| 逐次 MAPE | **6.9636%** | **8.0895%** |
| 逐次 MAE | 0.015185ms | 0.017983ms |
| 逐尺寸均值 MAPE | **2.3954%** | **3.1421%** |
| ≤64KiB 逐次 MAPE（各6,900条） | **7.2396%** | **8.5345%** |
| ≤64KiB 逐尺寸均值 MAPE | 2.6330% | 2.8106% |
| 最大单次绝对误差 | 1.052438ms | 1.332299ms |

预先设定每组覆盖≥95%、均值 MAPE≤10%、逐次及小消息 MAPE≤15%，两组均满足。
正确性错误为0，无拒绝、删失或裁剪；少量大误差仍在结果中。
同一批 v3 数据的 affine 逐次 MAPE 为7.3648%/8.3909%，均值 MAPE 为
2.9495%/2.6488%。说明主要改善来自可控协议策略，不应归功于增加模型复杂度。

## 双峰与残留慢尾

| 4096B 观测 | 新默认路径对照（60次） | v3 留出 A（300次） | v3 留出 B（300次） |
| --- | ---: | ---: | ---: |
| 均值 | 0.710107ms | 0.143139ms | 0.139633ms |
| p10 | 0.123264ms | 0.128981ms | 0.126949ms |
| p50 | 1.066092ms | 0.134479ms | 0.133536ms |
| p90 | 1.129483ms | 0.152311ms | 0.146373ms |
| 标准差 | 0.476250ms | 0.044192ms | 0.041576ms |

预先定义的探索统计 `duration>0.5ms`：默认对照≤4096B网格为446/600；
两个不同留出网格分别8/3000与12/3000。该阈值仅统计尾部比例，未用于删除观测、
切换预测或授予 PASS。网格与重复数不同，不能把这个比例当作精确总体因果效应。
新的分布报告保留全部 min/max、p10/p50/p90/p99 与 stddev。

## 真实模型兼容性烟测

独立 P2P 留出结束后，安装完整 wheel 并核对全部源文件与冻结快照一致，再在
相同两卡运行 Qwen3-1.7B BF16 的 TP1×PP2、TP2×PP1。两个阈值均在 Runtime
启动前设0。每种布局三组 greedy 请求（8/16/32 个输入 token，各输出4个），
输出 token 与此前默认策略烟测完全一致；各 rank 的12条 Hook 记录全部完成，
无失败或不支持，Scheduler 记录完整。这是有限正确性/采集兼容检查，不是新的
整模型性能准确率证明。首次仅复制源码的测试安装缺少插件入口，已保留失败证据；
完整 wheel 重试解决安装问题，未修改或重拟合冻结源码/模型。
真实 Hook 导出另经公开 CLI 规范化与校准：PP2 接受24组、TP2 接受12组，均无
隔离/缺失 channel，`canonical_stage_ready=true`，阶段 Profile 生成成功；实际
Runtime snapshot 中两个阈值均为字符串0。

## 范围限制

结论限于这对有向 GPU、BF16、当前软件栈及显式策略下的无竞争独立 primitive。
`hardware_validated=false` 保留：本轮满足的是 P2P 点误差目标，不是整模型/在线
SLA/吞吐/策略排序、Workspace 或真实 OOM 的整体验收。
未新增反向 pair、其他精度、多机/共享 fabric、receiver AllGather 精度或四卡
TP2×PP2 实测。历史 v1.10 的60.25%来自不同协议域，不能当作本轮同数据模型对照。

完整 rank 原始证据、协议日志、失败假设、源码/冻结 SHA 与逐次结果保存在独立
验证目录。仓库只分发手算 fixture 和脱敏报告，不含连接材料或原始真机日志。
