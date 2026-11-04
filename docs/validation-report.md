# V1.6 软件验证记录

日期：2026-11-04  
环境：Python 3.14.3，macOS，CPU 执行  
版本：1.6.0

## 自动检查

- `python3 -m unittest discover -s tests -v`：660 项 CPU 测试通过，覆盖 Qwen3 几何/权重、新 vLLM 内部 Hook/聚合、ROCm/RCCL/Runtime 采集适配、显式硬件验收、参考包导入、多副本路由、混合 Prefill/Decode、Prefix Cache、Chunked Prefill、PP、CPU swap、资源引用、Profile、校准、Compare、留出与 CLI 闭环；含固定 seed 的有界生成负载。
- Ruff 0.17.3：检查通过。
- mypy 2.4.1：`mypy mlsys_sim`，28 个源文件检查通过。
- `python3 -m compileall -q mlsys_sim tests scripts`：通过。

Ruff/mypy 为临时开发检查工具，没有加入项目运行依赖。测试覆盖配置拒绝、
公式与单位、GQA 复制、跨 rank 原子预留、物理与逻辑显存分离、确定性时间线、
超时在途引用、重算不重复输出、失败后的资源释放和重新准入、指标及输出不可覆盖。
预算在 Step 中途截止时计入已执行片段，并保留未提交状态；超大整数和浮点上溢/下溢明确拒绝。
生成负载还检查了关闭详细日志不会改变结果。
新增检查覆盖完整设备/模型/Runtime/精度身份、FP16/BF16 区分、重复观测计数、缺角点拒绝、显式外推、训练泄漏、零值与缺失指标、候选排序、原始 Profile 快照和配置数字规范化。
PP 新增 stage/link 不重叠、气泡手算、反馈因果、微批容量、TP×PP、部分执行与排队等待的截止统计。Swap 检查完整／部分拷贝、双侧预留、取消、Host 不足、显式回退及无重复输出。共享状态与指标抽取先通过原 175 项回归，再实现新调度路径。
Prefix Cache 检查 namespace、完整 token ancestry 的 SHA256 链、保留最终 prompt token、共享引用与同池计账、LRU/pin、失败准入回滚、长 prompt 线性身份存储和 warmup 隔离。Chunked Prefill 检查逐块 context、decode 优先、无中间输出、Profile Workspace 一致、TP×PP 反馈边界及部分 prompt 的 swap／重算；新增 CLI 示例执行与重放字节一致性、旧 Compare 指纹兼容测试。缓存命中后准入失败曾可能阻塞已有 decoder，已用单组和 PP 的 preemption=none 复现锁定修复。
混合批处理检查 decode 优先装箱与共享 Token/序列预算、同一次 stage 估计/预留、按成员提交进度、非最终 Prefill 不采样、联合 Workspace 的跨 rank 原子失败，以及 PP feedback/cache/swap 生命周期。混合 Profile 测试保留有序成员 phase、单 Token Prefill 与 Decode 的区分、精确形状域、校准重复统计与训练泄漏检查。在线 Profile 手算只提供首次 Prefill 与实际混合形状而不提供虚构的单成员 Decode 样本；单组和 TP2×PP2 均通过。混合开关默认关闭，旧 Compare 契约补默认值；生成负载覆盖日志关闭、截止、Host 不足及显式重算。
多副本检查共享事件时序、轮询／最少未完成请求／受负载约束的缓存亲和路由、到达时可见状态、固定归属、独立Host/DMA与纯读缓存探测。两条本地引擎仅抽取可逐事件驱动和幂等finalization接口，驱动结果与原run路径的请求、事件和资源报告逐项相等。全局预算/窗口、PP反馈和swap截止保留引用；局部停止不阻断健康副本，全部副本不可用时未来请求保持未路由。全局分位数从合并请求重算，三请求手算全局TTFT中位数为2ms，而局部中位数平均为2.5ms。Profile与CLI检查每副本身份、原始字节快照、有序hash、版本/来源、完整集群Compare，以及warmup缓存和路由游标重置。

## CLI 示例证据

Qwen3 新增手算验证保留真实 residual/query 两种宽度，计入 Q/K Norm、最终 Norm、同 rank tied 词表去重及 PP 端点物理复制。数学参数量为 1.7B 模型1,720,574,976、4B 模型4,022,468,096；4B residual=2560、query=4096，KV=147456bytes/token。Runtime 词表 padding 和 allocator 额外占用仍需实测。旧默认 Profile 身份/数值保持兼容，新架构与实际执行配置各自按 SHA256 区分。

内部 Hook 按固定 vLLM `98dff2a81` 源码接口使用 CPU source-shaped Runner/Scheduler 夹具测试，未安装或运行真实 vLLM/GPU。实际 Hook 导出直接进入聚合器，覆盖完整采样/已发布 Token 生命周期、cache salt 与 prompt SHA、参数/配置/PP层范围、rank 完整性、原调用返回/异常、record 上限、Future 释放和 close 竞态。异步消费、SP、恢复历史、lazy incoming、额外 residual 及 TP-sliced/receiver-AllGather 与模型传输不一致的区间保留 raw，不能作为纯 stage Profile。收集插件默认 dormant，只有显式配置才安装；支持域见[Qwen3 与 Hook 说明](qwen3-and-vllm-hooks.md)。

采集适配测试使用 CPU fake backend、合成 RCCL 日志及本机 HTTP/SSE 服务，未运行 AMD GPU kernel。检查 HIP 与 CUDA 区分、编译版本与实际 RCCL 库版本分离、真实 TP 的 Runtime 库版本/hash绑定、同步主机计时、allocator allocated/reserved 原始值与 Workspace 归因分离、精度/单位/rank/正确性、超时与部分失败证据。流式采集检查 delta token IDs/最终 usage、heartbeat 绝对期限、全局排队期限、重定向不转发凭据，以及客户端合并 chunk 不伪造服务端 Token 时刻。规范化 server trace 核对到达/输出数/逐 Token 索引及 execution/window/drain/request timeout 边界。验收测试覆盖组/形状泄漏、完整请求配对、重复内容去重、显式阈值与缺失证据；fixture 永不成为硬件 PASS，`--require-pass` 对非实测 PASS 返回3。

基础示例与新增的校准闭环均通过公开 CLI 执行。生成的 `runs/` 和 `sample-runs/` 不纳入仓库或分发包；可按 README 重新生成。

| 示例 | 观察结果 |
| --- | --- |
| FIFO 手算 | 3 个请求完成；首 Token 为 10/24/34ms，结束为 14/24/36ms |
| 内存压力 | 2 个请求各输出24 Token；1 次抢占、32 Token 重算、重算5ms，总模拟56ms，最终 KV 全部释放 |
| TP 2 | 3 个请求完成；输出各 rank 账本及通信分量 |
| PP 2 解析 | 2 个不等长 stage；predict 仅作形状分析，在线指标应使用 simulate |
| PP 在线手算 | 两 stage 各1ms、三个 microbatch、显式零 P2P/feedback；请求于2/3/4ms完成，makespan 4ms |
| CPU swap | 两请求各24 Token；1次D2H及1次H2D，85.768ms完成，零重算，Host/GPU KV最终释放 |
| PP＋CPU swap | 两请求各24 Token；1次D2H及1次H2D，约88.228781ms完成，零重算，Host/GPU KV最终释放 |
| Prefix Cache | 两请求完成；后者复用8 Token；总计计算16 prompt Token、完成2次 Prefill；请求引用全部释放，3个缓存block保留 |
| Chunked Prefill | 长prompt 20 Token超过batch预算8；两请求均完成，总计6 chunks／24 prompt Token，只在各自最终chunk输出首Token |
| PP＋Prefix Cache＋Chunked Prefill | 两请求完成，复用8 Token；4 chunks／16 prompt Token，feedback安全边界发布，最终无请求引用 |
| 混合 Prefill/Decode | 两请求完成，2次真实混合批次，混合计算6ms，总模拟9ms；非最终prompt chunk不输出 |
| PP＋混合＋Cache＋CPU swap | 5请求完成，2混合微批，复用12 Token，D2H/H2D各1次，总模拟53.6584ms；零重算，最终请求/Host引用全部释放 |
| 多副本轮询 | 4请求路由0/1/0/1，R2 FIFO总模拟6ms；同工作负载单副本FIFO为12ms |
| 多副本缓存亲和 | 3请求路由0/1/1，repeat在副本1实际复用4 Token，总模拟7ms |
| R2＋PP2＋混合＋Cache＋CPU swap | 10请求完成，4混合微批、24缓存Token，D2H/H2D各2次，总模拟53.6584ms，零重算；请求/Host引用全部释放 |
| Step 校准 | 6 条手算观测，5 个精确形状；重复观测取均值，版本及输入哈希保留 |
| Profile 模拟 | 1 个请求，3 个输出 Token；完整 Step 不再叠加解析 TP 通信 |
| Online Compare | 19 项指标均可比较，覆盖率 100%；请求按 ID 配对 |
| 分组留出 | 2 条同配置观测整体留出；域内覆盖率 100%，MAE 0.1ms，MAPE 约 1.61% |
| 通信校准 | 独立 Ring/P2P 手算数据，拟合约 5µs 启动延迟与 20GB/s 带宽；输出显式 Infra patch |
| ROCm Probe | 本机 Darwin/arm64、无 PyTorch/ROCm；诊断流程完成，hardware_available=false，不回退为 CPU 测量 |
| 显式硬件验收 fixture | 校准→分组留出→模拟→Compare 数值条件通过；verdict=TEST_FIXTURE_ONLY，hardware_validated=false |
| Runtime trace fixture | server-token-ready 事件转换为观察，TTFT6.1ms、TPOT1.1ms、E2E8.3ms；来源仍为fixture |
| RX9070XT 参考包 | 195 runs/65 cells，9750正式＋1950预热请求；已读取已知数据文件校验匹配，规范化验收仍INSUFFICIENT_EVIDENCE |
| 旧 NVIDIA 参考包 | 48 runs/2400正式请求，README哈希不一致被保留；不重标AMD或stage Profile |
| Qwen3-4B 导入/预测 | hidden2560与query4096保持独立；BF16数学权重8,044,936,192 bytes；原始/规范化架构哈希保留 |
| Qwen3 R2×TP2×PP2 组合 | 10请求完成、4混批、24缓存Token，D2H/H2D各2次，零重算，总模拟52.413216ms，请求/Host引用归零 |
| 内部 Hook fixture | 完整同步rank样本可生成stage dataset并校准；无Scheduler生命周期时online不可比较，hardware_validated=false |

内存压力示例使用 synthetic 固定阶段时间；仓库中分发的校准样例全部标记 test_fixture。本次另读取用户提供的真实外部采样包，仅作边界、覆盖与导入审计，没有重新运行 GPU 采集。完整性检查限定在导入器读取的已知文件，未重放嵌套原始归档。原始 prompt/response、归档代码与数据包不纳入仓库或分发。详见[参考包审计](reference-data-audit.md)。这些检查不构成真实 GPU 时延准确率证明。

## 当前限制

本地执行环境为 Python 3.14.3；Python 3.11 至 3.14 的远程结果以该提交对应的 Actions 记录为准。
Profile、PP 在线流水线、CPU swap、Prefix Cache、Chunked Prefill、混合 Prefill/Decode 与多副本调度均已实现。用户现已提供 RX9070XT 请求级数据，但缺少匹配的整 stage 样本、服务端逐 Token 时间和 Workspace 归因；硬件留出误差、真实容量拐点、在线 SLA 和策略排序的预测有效性仍未建立。模型采用全 stage 预留、持久激活缓冲上界、排空 wave 后回收，以及 DMA 不与 GPU 计算重叠的保守策略；没有真实 CUDA/HIP DMA、NVMe 或分配器细粒度模拟。混合解析模型假设共享权重和宏观通信，混合 Profile 只接受精确成员形状，不推断真实 Kernel 效率；每chunk保守计入输出／sampling服务边界，缓存CPU开销未测量，swap恢复为完整私有KV。缓存占用在请求结束后保留是预期行为。组合手算示例显式使用单inflight微批和零P2P，另有R2×TP2×PP2启用P2P与正反馈延迟的组合检查。副本资源和DMA独立；未模拟共享fabric争用、跨副本请求迁移或实测路由开销。

## 库化与分发验证

- Setuptools 84.2.1 通过标准 PEP 517 隔离构建生成 sdist，再从该 sdist 构建 wheel。
- Twine 7.0.0 对两种分发文件的元数据检查通过。
- `scripts/verify_distribution.py` 在源码目录以外创建新虚拟环境，离线安装 wheel；公共 API、PP 在线、单组／PP swap、单组／PP Prefix Cache＋Chunked Prefill、混合批处理和PP/缓存/swap组合、多副本三种路由/组合与逐副本Profile模拟和预测、ROCm probe/服务端trace/客户端token-ID fixture/显式验收命令、Qwen模型导入/预测/组合与dormant vLLM插件和Hook数据集校准、PP 解析、校准→Profile→Compare→留出及通信校准 CLI 均通过。
- Wheel 内含 `py.typed`，安装元数据无必需第三方运行依赖；sdist 包含测试、示例和公共文档。
- CI 覆盖 Python 3.11 至 3.14 的回归测试、lint、类型和分发检查，使用固定 Actions SHA。

生成日志和内部协作笔记未纳入分发内容。仓库为私有，许可证仍待选择，没有发布 PyPI。
