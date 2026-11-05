# V1.13.0 软件与限定真机验证记录

日期：2026-11-05  
环境：Python 3.14.3，macOS，CPU 执行  
版本：1.13.0

## 1.13 Decode 与多卡 Workspace

新增 Worker 级采集保留真实 Prefill/Decode、逐 rank Torch storage generation、
Native 通信分配、PP 跨迭代引用与自然 shutdown 证据。完整装配器重放这些证据后，
才允许生成 canonical Workspace 数据集；采集进程成功或组件审核通过均不能代替完整归因。

单卡 Qwen3-1.7B BF16、同步 eager MRv1、单序列、512-token 上限、256MiB KV
域完成了训练与冻结后的独立 Runtime 重复。训练和验证各 24 条正式观测、8 种形状，
覆盖 Prefill16/32 和 Decode contexts16/17/18/32/33/34。验证记录、rank 和保留形状
覆盖率均为100%，最大绝对字节误差0、低估0；冻结 Profile 原始字节保持不变。
六种 Decode 的 requested-storage 峰值均为920,844 bytes。普通公开 simulate CLI
使用同一 Profile 完成非空请求，Workspace 来源为显式 Profile；此消费检查不构成新时延精度证据。

TP2 训练与独立重复也已完整归因并通过公开验证：每个 cohort 各24条 canonical记录、
48条rank观测和8种精确形状，记录/rank/保留形状覆盖率均100%，最大绝对字节误差0、
低估0。Profile于2026-11-05 05:49:37.167722 UTC冻结，SHA256为
`f2e6a8a3024586e086e1e9d8c1327b93a391d81c920c283e1c09ecbeac6c546c`，
新Runtime重复前后原始字节未变。两rank的20个通信backing分别独立证明，各17,113,392 bytes。

PP2 已完成实际采集与逐 rank 组件回放，但完整校准仍以全部 Native 和同步
资源门禁闭合为前提。PP 自然退出后仍有未证明归属/销毁的 formal HSA signal generation；
进程退出诊断的 SDK stop 实际返回 CONTEXT_NOT_FOUND，不能改写为成功。
这些 partial 结果不进入 Workspace 拟合或独立验证，也不能据此推断GPU泄漏或OOM。

真实 PP2 大形状压力观测还发现 sender 跨迭代保留峰值8,372,224 bytes；原单 inflight
预算4,194,304 bytes不足，修正为 `max(inflight,2)` 对输出缓冲后的8,388,608 bytes可覆盖。
该压力诊断不进入训练，不推广为任意异步或多序列上界。

新增独立 `validate --workspace-repeats` 核对训练 SHA、重算的拟合来源、独立采集身份、
逐 rank/shape 覆盖与字节低估。现有 TP/PP、缓存、swap、混批组合仍受精确 Profile
身份和形状域约束。实采域未启用这些在线策略，CPU 组合回归不授予它们新增硬件精度结论。
所有报告保留 `hardware_validated=false`。完整范围与冻结 SHA 见
[Worker Workspace 验证记录](workspace-stage-validation.md)。

本轮本地1115项CPU回归检查通过；沙箱跳过的两项loopback HTTP测试另行补跑通过。
Mac缺C++标准头而跳过的可选原生路径formatter测试，在实际Linux容器中使用同一
测试类与实际probe源码编译执行通过，零跳过。Ruff通过，mypy对52个源文件检查通过，
源码编译检查通过。最终提交的跨Python结果以该提交对应的GitHub Actions记录为准。

## 1.12 新增证据与验收范围

本轮沿用获授权的 RX9070XT 与 RX7900XTX，按 Workspace、物理容量、
独立多卡 Stage、在线策略行为的顺序采集。真实执行使用现有 ROCm/Torch/vLLM
镜像，版本与实际二进制/源码证据分别绑定；没有把镜像标签当作实际版本。
本轮本地检查：987项CPU回归通过（沙箱跳过的两项loopback HTTP检查另行补跑通过），
Ruff检查通过；mypy对核心、分发smoke脚本和公开审计工具共46个文件检查通过。
远程CI结果以推送后该提交对应的Actions记录为准。

| 项目 | 本轮结果 | 有效范围与保留问题 |
| --- | --- | --- |
| 完整 Workspace 归因 | 单卡 prefill16/32 的动态峰值分别为982,284和1,703,940 bytes；冻结后独立Runtime各3次重复，覆盖100%，无低估 | owned requested-storage口径；固定Runtime组件210,783,373 bytes计入独立1GiB预算；不覆盖decode、多卡或其他策略 |
| RX9070XT 物理KV容量 | 12,977,176,576 bytes成功、13,153,337,344 bytes原生KV allocator OOM，各独立确认2次，分辨率168MiB | `BRACKETED`仅表示观察端点；默认分析预算仍将成功点判为不可行，未授予容量预测PASS |
| RX7900XTX 物理KV容量 | 21,491,613,696 bytes成功、21,843,935,232 bytes原生KV allocator OOM，各独立确认2次，分辨率336MiB | 中间21,726,494,720 bytes两次worker异常退出，保留ERROR/CENSORED；不推断区间单调或精确阈值 |
| TP2独立未见shape验证 | 24条正式观测，覆盖100%，MAPE2.9520%，MAE0.723308ms | 新输入长度12/24；原形状泄漏门禁保留，无外推，单active sequence、eager限定域 |
| PP2独立exact重复 | 72条正式观测，覆盖100%，MAPE6.1565%，MAE0.865454ms | 冻结Profile后新Runtime采样，保留原生payload；只证明训练保留shape内重复，不称为未见shape验证 |
| Prefix Cache / Chunked Prefill / 混批行为 | 64-token请求实际复用48-token前缀；分块出现3个不采样partial prefill；3次执行同时含prefill/decode，输出完整 | 正确性smoke，与后续条件延迟验证分开 |
| Prefix Cache / Chunked Prefill条件延迟 | 两个策略各3次fresh Runtime，冻结Profile后45条正式Stage exact覆盖100%，请求9/9配对；每次TTFT/TPOT/E2E p50误差均≤15% | 短固定cohort、真实条件arrival，保留所有seed；不证明open-loop吞吐、尾延迟、SLA或策略排序 |
| 混批精度覆盖缺口 | 真机训练首批为`[8,8]`，模拟器按真实微秒arrival立即启动solo`[8]`，exact Profile缺该solo形状而报告域外 | 保留真实arrival和缺域，不把功能smoke称为精度通过，不用验证数据补拟合 |
| 两个物理副本 | 两张卡各一个独立TP1×PP1 Runtime；4请求按0/1/0/1轮询，输出全部完成，原生ID归属与逐实例公开规范化检查通过 | 固定路由/输出smoke；不验证least-loaded、prefix-aware、路由开销、争用或性能收益 |

完整Workspace依赖已绑定的Runtime对象、原生Torch allocator事件、九个SDK域、
成功原生分配返回、scratch整池预算与CPU返回值证明共同闭合。单个组件的
`CLOSED_CANDIDATE` 或 `NATIVE_BACKING_CHECKS_PASSED` 不会单独成为完整归因。
公开可选工具的CPU fixtures检验审计逻辑，不升级为硬件观测。
最终公开driver/merger另读取实际软件、GPU、Runner并行与执行配置，拒绝只靠输入JSON
贴身份；收紧门禁后重新训练、冻结并在新Runtime重复，六点均完整归因且误差为零。

新增 `validate_profile_repeats` 和 `validate --exact-repeats` 核对冻结训练数据的
canonical SHA、重算的拟合来源、不同session/capture/record/group。该路径只接受
保留exact shape；普通留出与硬件验收的形状泄漏检查没有放宽。typed容量OOM仍需
设备allocator的`num_ooms`增长及真实最终KV分配调用；其他worker失败不重标OOM。

详见[Workspace实测](workspace-live-validation.md)、[容量实测](capacity-live-validation.md)、
[多卡Stage独立验证](multigpu-stage-validation.md)和
[策略/副本行为实测](online-policy-live-validation.md)。原始观测、失败尝试、冻结/进程回执
及源码/输入哈希保存在独立证据目录；密钥、地址、模型权重与原始机器日志不进入仓库。
报告保留 `hardware_validated=false`，不表示整体硬件验收通过。

上述条件延迟Profile于2026-11-04 17:10:15 UTC冻结，6个独立Runtime在
17:10:48至17:13:14 UTC执行，没有重试、重拟合或改动Profile bytes。预设条件为
覆盖≥95%且每项p50相对误差≤15%。Prefix Cache三次TTFT p50误差分别为
1.1345%/8.9372%/0.8793%，Chunked Prefill为4.2252%/2.6750%/3.1030%；
完整TPOT/E2E与冻结SHA见[条件在线验证记录](online-policy-live-validation.md#independent-conditional-prefix-cache-and-chunking-repeats)。
Prefix Cache后一个arrival受实际seed完成影响，属于条件重放，不能解释为开放环负载预测。

## 1.11 自动检查基线

- `python3 -m unittest discover -s tests -v`：912 项 CPU 回归通过（沙箱跳过的两项 loopback HTTP 检查另行补跑通过），新增 29 项 v3策略、实际阈值／Runtime指纹、分布和组合专项；保留 36 项 lookup/v2/TP/PP/三 Profile 专项；保留真实 checkpoint tying、默认 RoPE 别名、跨 session 留出及加载后证明回归；保留完整 Workspace ledger/独立逐rank校准、三 Profile 联用、fresh Runtime 容量/OOM 区间与来源验证；保留独立 P2P/AllGather 微基准、同次运行校验、按位置校准/留出及验收来源绑定；保留原生 PP 自然接收、准备流、持久池容量、双张量/切片及 exact Profile 闭环；覆盖 Qwen3 几何/权重、vLLM 内部 Hook/聚合、ROCm/RCCL/Runtime 采集适配、显式硬件验收、参考包导入、多副本路由、混合 Prefill/Decode、Prefix Cache、Chunked Prefill、PP、CPU swap、资源引用、Profile、校准、Compare、留出与 CLI 闭环；含固定 seed 的有界生成负载。
- Ruff 0.17.3：检查通过。
- mypy 2.4.1：`mypy mlsys_sim scripts/verify_distribution.py`，37 个文件检查通过。
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

Workspace 新增完整 Runtime-owned storage ledger、alias 去重、异步 release-completed 与逐 rank 完成证明。完整动态归因才进入独立 exact Profile，重复 shape 逐 rank 取最大；零值合法，未知/partial/超限不可伪成零。显式 Profile 替换动态预算，并保留所有固定池、PP backing、通信与 Runtime 预算。每 rank 不同峰值、一字节容量边界、缺形状原子失败与三类 Profile 的 R2×TP2×PP2/cache/chunk/swap/mixed 重放已测试；硬件验收纳入 Workspace 原始字节 SHA 和来源。

容量扫描每个 axis point/repeat 使用独立 Runtime 进程，绑定完整有效配置、模型/设备、所选 Profile 字节及实际应用的轴。typed GPU OOM 还需实际设备 allocator `num_ooms` 增量；只有可验证的 prepare/run 证据可形成区间，startup/load/cleanup、timeout、锁错误、KV 逻辑拒绝与缺校准域各自保留状态。KV 轴成功需实际 Tensor storage 证明，分配失败需明确目标请求与设备 OOM 证明。验证不假定单调，混合重复和非单调结果明确标记；只记录相邻成功/OOM 的观察区间和分辨率，精确阈值为 null。CPU fixture 的 `[4,8]` 区间不构成真机容量结论。详见[Workspace 与容量验证](workspace-and-capacity.md)。

原生 PP 独立微基准使用单向 P2P 与 out-of-place AllGather；以完整参与 rank 的本地同步主机时长取最大值。AllGather 拟合字节数是完整输出，P2P 则是实际 wire payload。CPU 测试锁定预热、分配、CPU Gloo 控制 barrier 和正确性检查均在计时外，`Work.wait()` 后还需设备同步；同一 cohort 的完整 rank 输出才可归并，超时和进程组失败保留原始证据。没有运行真实 HIP 通信。

新通信 Profile 按 primitive、接收 stage、配对 TP rank 与有序参与设备分别拟合非负直接 affine，不将有效参数冒充每跳 Ring 系数。默认拒绝缺组和域外查询，显式外推保留标记；独立留出拒绝 record/group 泄漏并报告覆盖和误差。运行只加载一次已选 Profile，CLI 保存原始字节快照及 SHA；同一张量先取各配对 P2P 最大值，再加一次 receiver AllGather，阶段 Profile 的本地通信不再叠加。原生 padding 与 R2/TP2/PP2/cache/chunk/swap/mixed 两种 Profile 的组合回归通过。硬件验收将实际通信 Profile 来源与输入 SHA 纳入范围，fixture 参数不能得到硬件 PASS。详见[独立通信微基准与校准](native-pp-communication.md)。

Qwen3 新增手算验证保留真实 residual/query 两种宽度，计入 Q/K Norm、最终 Norm、同 rank tied 词表去重及 PP 端点物理复制。数学参数量为 1.7B 模型1,720,574,976、4B 模型4,022,468,096；4B residual=2560、query=4096，KV=147456bytes/token。Runtime 词表 padding 和 allocator 额外占用仍需实测。旧默认 Profile 身份/数值保持兼容，新架构与实际执行配置各自按 SHA256 区分。

内部 Hook 的早期开发按固定 vLLM `98dff2a81` 源码接口使用 CPU source-shaped Runner/Scheduler 夹具测试；1.9.1 另完成下述限定真机试跑。实际 Hook 导出直接进入聚合器，覆盖完整采样/已发布 Token 生命周期、cache salt 与 prompt SHA、参数/配置/PP层范围、rank 完整性、原调用返回/异常、record 上限、Future 释放和 close 竞态。异步消费、SP、恢复历史及未知传输区间仍保留 raw。默认 full-hidden 契约不接受原生双张量切片传输；显式 native 契约通过独立边界校验后可校准。收集插件默认 dormant，只有显式配置才安装；支持域见[Qwen3 与 Hook 说明](qwen3-and-vllm-hooks.md)。

原生 PP 新增实际双张量、padding、TP 切片/全量分支、完整相邻 stage/rank 证据和严格接收区间算术检查。自然 wait 入口先完成当前 compute stream 的本轮输入准备，再观察原始 wait/postprocess AllGather 并同步其完成；阶段 Profile 扣除后者区间，保留输入准备、持久 buffer copy、计算与采样。已由外部消费的输入另先建立完成同步。缺少边界、构建或标准 NCCL/RCCL 路径绑定的实测声明拒绝。等待上游的耗时不是独立网络样本；网络成本由实际每张量 wire slice 与 receiver AllGather 分析估计并单独计费一次。原生 Profile 保存物理 descriptor，仅精确查找，拒绝同形状不同 padding 的聚合。显存预算包含接收原始 full backing、AllGather 输出及持久 copy 目标，不按 TP 缩小驻留预算。详见[原生 PP 采样说明](native-pp-capture.md)。

真机原生接收阶段另需绑定已加载 Runner 的 `max_num_tokens` 与完整持久输入 buffer 的实际容量、精度及形状；超过配置上界的池不因本次采样批次较小而被接受。

CPU 采集适配回归使用 fake backend、合成 RCCL 日志及本机 HTTP/SSE 服务，不包含 GPU kernel；实际远端测量独立列于后节。检查 HIP 与 CUDA 区分、编译版本与实际 RCCL 库版本分离、真实 TP 的 Runtime 库版本/hash绑定、同步主机计时、allocator allocated/reserved 原始值与 Workspace 归因分离、精度/单位/rank/正确性、超时与部分失败证据。流式采集检查 delta token IDs/最终 usage、heartbeat 绝对期限、全局排队期限、重定向不转发凭据，以及客户端合并 chunk 不伪造服务端 Token 时刻。规范化 server trace 核对到达/输出数/逐 Token 索引及 execution/window/drain/request timeout 边界。验收测试覆盖组/形状泄漏、完整请求配对、重复内容去重、显式阈值与缺失证据；fixture 永不成为硬件 PASS，`--require-pass` 对非实测 PASS 返回3。

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
| 原生 PP Hook → Profile | source-shaped Hook 导出经过TP/相邻stage归并、校准后完成精确Profile预测/在线模拟及CLI导入；接收等待保留raw且不充当网络时延 |
| 原生 Qwen3 R2×TP2×PP2 组合 | synthetic与fixture Profile路径均完成10请求，4混批、24缓存Token，D2H/H2D各2次，最终资源引用归零；网络成本仍为分析估计 |
| PP primitive fixture 闭环 | collect→独立 affine 校准→留出→predict/simulate；5请求完成、D2H/H2D各1次，来源仍为fixture |
| 双 Profile 组合 | 每副本同时选择阶段与通信 Profile，10请求完成、24缓存Token、D2H/H2D各2次，结束引用归零 |
| 三 Profile 组合 | 逐副本阶段/通信/Workspace同时选择，10请求完成、24缓存Token、D2H/H2D各2次；Workspace按rank计量，结束引用归零 |
| Workspace 独立校准 | 完整归因夹具按精确shape/rank重复取max，校准→预测→模拟通过；manager与allocator诊断保持partial/raw |
| 容量验证 fixture | 4个fresh trial，2成功/2typedOOM；观察区间4至8、分辨率4、exact_threshold=null；fixture身份保持 |

内存压力示例使用 synthetic 固定阶段时间；仓库中分发的校准样例全部标记 test_fixture。本次另读取用户提供的真实外部采样包，仅作边界、覆盖与导入审计，没有重新运行 GPU 采集。完整性检查限定在导入器读取的已知文件，未重放嵌套原始归档。原始 prompt/response、归档代码与数据包不纳入仓库或分发。详见[参考包审计](reference-data-audit.md)。这些检查不构成真实 GPU 时延准确率证明。

## 1.9.1 初轮 AMD 真机（限定范围）

在调用方授权的两张不同物理 GPU 上执行：RX7900XTX/gfx1100 与 RX9070XT/gfx1201；PyTorch2.13.0+rocm10.1.0、HIP build7.16.26385、RCCL2.30.7、Python3.14.8、实际vLLM0.29.1.dev0+g98dff2a81.d20261002。镜像标签0.29.0没有被当作实际版本。两方向直接peer访问均不可用，多卡明确禁用peer路径/custom AllReduce；固定协议复测单独记录Simple。

- Qwen3-1.7B BF16在两卡分别完成真实中文HTTP流式生成；9070上的正式客户端采集CLI完成，客户端交付与服务端token-ready边界保持区分。
- 修复后9070单卡同步MRv1训练76条、独立Runtime52条原始Stage记录均完整，无隔离；实际checkpoint临时untie而loaded Parameter/storage重新共享的行为得到证明。
- 明确排除独立留出的4条warmup后，48条未见shape覆盖100%，Stage MAPE0.7995%、MAE0.1539ms。范围是短输入、单active request、eager、无prefix/chunking和固定配置。
- 7请求条件在线重放完整配对；TTFT/TPOT/E2E p50相对误差约5.05%/5.09%/4.99%。样本少且包含初始warmup请求，不证明通用尾部、吞吐或SLA。
- TP2×PP1与TP1×PP2均完成真实模型生成，3组greedy输出token一致；各rank记录完整，分别归并12/24组且拟合成功。没有多卡独立精度留出，也没有4卡TP2×PP2覆盖。PP有效端点配置明确保留与原始逻辑tying的区别，没有假跨设备alias。
- 独立P2P正确性错误为0。首个窄网格线性拟合因负斜率被拒；固定Simple宽网格训练/留出各300条，覆盖100%，但留出MAPE101.52%、MAE0.2104ms，精度不能验收。独立AllGather2诊断90组正确性错误为0，不伪装为4卡原生PP场景。
- 真实LLM fresh-runtime安全容量点16/64/128输入token全部成功，无删失/无OOM，报告UNBRACKETED。没有找到真实OOM阈值。实际manager容量0保持partial，不晋升完整Workspace。

完整原始日志、软件/文件SHA和失败对照仅保存在独立验证目录，未纳入仓库/分发；共享模型保持只读，测试容器和自有进程已清理。密钥、主机地址与账号不进入源代码或报告分发。硬件预测的一般有效性仍未建立。

## 1.10 小消息 P2P 真机复测

同一对真实 GPU 完成四会话 19,080 条训练，模型冻结后另起会话采集 9,300 条独立留出，30 个未见尺寸加 4KB 独立重复，正确性错误为0。显式v2 lazy协议、`NCCL_PROTO=Simple`请求和blocking wait=0的同域对照中，逐尺寸均值MAPE由affine的100.3340%降至lookup的4.8772%，覆盖100%；逐次MAPE由120.0551%降至60.2526%，仍未达到预设15%目标。所有慢样本保留。4KB以下双峰波动仍在，结果只支持本pair/精度/环境的均值预测改善，不授予整体硬件PASS。训练拐点加密在留出前完成，完整时间与SHA证明保存在独立证据目录。详见[小消息独立留出记录](p2p-small-message-validation.md)。

## 1.11 显式 P2P Simple-only 复测

固定RCCL实际构建7da4302的源码与COLL日志确认：P2P LL阈值独立于collective的`NCCL_PROTO`。默认4KB及以下选择LL并复现双峰；扩大阈值扩大慢区间，两个阈值设0后双方选择Simple。新增v3仅在子worker设置阈值，保留原完整设备同步计时与全部观测；v1/v2默认兼容。四训练会话19,200条后冻结，两次独立留出各9,300条、各30个未见尺寸，lookup逐次MAPE为6.9636%/8.0895%，小消息为7.2396%/8.5345%，覆盖均100%，两组均达到预设目标。未将策略变化伪装为旧LL域模型准确性改善，也未声称消除所有慢尾。新策略下真实Qwen3 TP2/PP2三组greedy输出与此前一致，逐rank12条Hook记录完整；公开CLI规范化接受12/24组且校准成功。详见[协议策略与独立留出](p2p-protocol-validation.md)。

## 当前限制

本地执行环境为 Python 3.14.3；Python 3.11 至 3.14 的远程结果以该提交对应的 Actions 记录为准。
Profile、PP 在线流水线、CPU swap、Prefix Cache、Chunked Prefill、混合 Prefill/Decode 与多副本调度均已实现。旧 RX9070XT 请求级参考包仍缺匹配的整stage/服务端逐Token/Workspace归因；新增的小域Stage、条件重放与Workspace实测分别按上述范围成立。广域Workspace、精确OOM阈值、通用容量预测、在线SLA与策略排序的有效性仍未建立。只有两张物理GPU，不能覆盖四卡TP2×PP2及其接收端AllGather；当前真机vLLM版本缺少原生CPU KV swap路径，尚无端到端evict/restore硬件对照。模型采用全 stage 预留、持久激活缓冲上界、排空 wave 后回收，以及 DMA 不与 GPU 计算重叠的保守策略；没有真实 CUDA/HIP DMA、NVMe 或分配器细粒度模拟。混合解析模型假设共享权重和宏观通信，混合 Profile 只接受精确成员形状，不推断真实 Kernel 效率；每chunk保守计入输出／sampling服务边界，缓存CPU开销未测量，swap恢复为完整私有KV。缓存占用在请求结束后保留是预期行为。组合手算示例显式使用单inflight微批和零P2P，另有R2×TP2×PP2启用P2P与正反馈延迟的组合检查。副本资源和DMA独立；未模拟共享fabric争用、跨副本请求迁移或实测路由开销。

## 库化与分发验证

- Setuptools 84.2.1 通过标准 PEP 517 隔离构建生成 sdist，再从该 sdist 构建 wheel。
- Twine 7.0.0 对两种分发文件的元数据检查通过。
- `scripts/verify_distribution.py` 在源码目录以外创建新虚拟环境，离线安装 wheel；公共 API、PP 在线、单组／PP swap、单组／PP Prefix Cache＋Chunked Prefill、混合批处理和PP/缓存/swap组合、多副本三种路由/组合与逐副本Profile模拟和预测、ROCm probe/服务端trace/客户端token-ID fixture/显式验收命令、Qwen模型导入/预测/组合与dormant vLLM插件和Hook数据集校准、PP 解析、校准→Profile→Compare→留出及通信校准 CLI 均通过。
- Wheel 内含 `py.typed`，安装元数据无必需第三方运行依赖；sdist 包含测试、示例和公共文档。
- CI 覆盖 Python 3.11 至 3.14 的回归测试、lint、类型和分发检查，使用固定 Actions SHA。

生成日志和内部协作笔记未纳入分发内容。仓库为私有，许可证仍待选择，没有发布 PyPI。
