# 首个可执行版本的能力矩阵

本表描述当前代码，而非六周计划的最终承诺。

| 能力 | 当前状态 | 范围 |
| --- | --- | --- |
| 六类输入与严格配置 | 已实现 | JSON/TOML、JSONL workload、相对文件引用；YAML 可选 |
| C01 宏观性能模型 | 已实现，未校准 | 逐序列形状、Linear/Attention/Misc、Embedding/输出头 |
| C02 显存规划 | 已实现，未校准 | 逐 rank 物理预算、固定 KV pool、逻辑 blocks、形状相关 Workspace |
| C03 在线事件 | 已实现 | FIFO、prefill-first Continuous Batching、Token、超时、预算 |
| KV 抢占与恢复 | 已实现 | 安全 Step 边界、整请求 KV 释放、重算前缀、输出历史保留 |
| TP | 已实现，未校准 | 单个同步组，TP 1/2/4、GQA 分片/复制、原子预留 |
| C04 通信 | 已实现，未校准 | Ring AllReduce、logits AllGather、PP P2P、无争用/重叠 |
| PP 解析 | 已实现，未校准 | 各 stage 层数、显存、孤立 Batch 服务时间与边界传输 |
| PP 在线流水线 | 未实现并明确拒绝 | 后续 stage/microbatch/反馈事件 |
| 多副本、Chunked Prefill、Prefix Cache | 未实现并明确拒绝 | 不隐式降级或忽略配置 |
| CPU/NVMe swap | 未实现并明确拒绝 | 当前恢复方式为 recompute |
| 量化格式 | 未实现并明确拒绝 | 仅 2/4/8 bytes 浮点存储假设，不省略量化元数据 |
| 实测查表、校准和留出比较 | 待实现 | 当前不接收 measured backend |
| AMD 真机有效性 | 未验证 | 需要固定 GPU、Runtime 和测量数据 |

实际分配器碎片、张量存活期和 Kernel 执行不做细粒度模拟。Workspace 模型是
`base + query_tokens × bytes_per_token`，需要后续实测替换。

首次版本用单个实现明确三层边界；配置、分片、内存与事件各自独立，避免重复维护两套 DES。当前没有运行时第三方依赖。
