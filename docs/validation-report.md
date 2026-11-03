# V1.1 软件验证记录

日期：2026-11-03  
环境：Python 3.14.3，macOS，CPU 执行  
版本：1.1.0

## 自动检查

- `python3 -m unittest discover -s tests -v`：242 项测试通过，覆盖 PP、CPU swap、资源引用、Profile、校准、Compare、留出与 CLI 闭环；含固定 seed 的有界生成负载。
- Ruff 0.17.3：检查通过。
- mypy 2.4.1：`mypy mlsys_sim`，17 个源文件检查通过。
- `python3 -m compileall -q mlsys_sim tests scripts`：通过。

Ruff/mypy 为临时开发检查工具，没有加入项目运行依赖。测试覆盖配置拒绝、
公式与单位、GQA 复制、跨 rank 原子预留、物理与逻辑显存分离、确定性时间线、
超时在途引用、重算不重复输出、失败后的资源释放和重新准入、指标及输出不可覆盖。
预算在 Step 中途截止时计入已执行片段，并保留未提交状态；超大整数和浮点上溢/下溢明确拒绝。
生成负载还检查了关闭详细日志不会改变结果。
新增检查覆盖完整设备/模型/Runtime/精度身份、FP16/BF16 区分、重复观测计数、缺角点拒绝、显式外推、训练泄漏、零值与缺失指标、候选排序、原始 Profile 快照和配置数字规范化。
PP 新增 stage/link 不重叠、气泡手算、反馈因果、微批容量、TP×PP、部分执行与排队等待的截止统计。Swap 检查完整／部分拷贝、双侧预留、取消、Host 不足、显式回退及无重复输出。共享状态与指标抽取先通过原 175 项回归，再实现新调度路径。

## CLI 示例证据

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
| Step 校准 | 6 条手算观测，5 个精确形状；重复观测取均值，版本及输入哈希保留 |
| Profile 模拟 | 1 个请求，3 个输出 Token；完整 Step 不再叠加解析 TP 通信 |
| Online Compare | 19 项指标均可比较，覆盖率 100%；请求按 ID 配对 |
| 分组留出 | 2 条同配置观测整体留出；域内覆盖率 100%，MAE 0.1ms，MAPE 约 1.61% |
| 通信校准 | 独立 Ring/P2P 手算数据，拟合约 5µs 启动延迟与 20GB/s 带宽；输出显式 Infra patch |

内存压力示例使用 synthetic 固定阶段时间；新增校准数据全部标记 test_fixture，未收集 GPU 测量。
这些检查验证程序机制与假设内部一致性，不构成真实 GPU 时延准确率证明。

## 当前限制

本地执行环境为 Python 3.14.3；Python 3.11 至 3.14 的远程结果以该提交对应的 Actions 记录为准。
Profile、PP 在线流水线与 CPU swap 均已实现。当前没有 AMD 原始测量数据，硬件留出误差、真实容量拐点、在线 SLA 和策略排序的预测有效性仍未建立。模型采用全 stage 预留、持久激活缓冲上界、排空 wave 后回收，以及 DMA 不与 GPU 计算重叠的保守策略；没有真实 CUDA/HIP DMA、NVMe 或分配器细粒度模拟。

## 库化与分发验证

- Setuptools 84.2.1 通过标准 PEP 517 隔离构建生成 sdist，再从该 sdist 构建 wheel。
- Twine 7.0.0 对两种分发文件的元数据检查通过。
- `scripts/verify_distribution.py` 在源码目录以外创建新虚拟环境，离线安装 wheel；公共 API、PP 在线、单组／PP swap、PP 解析、校准→Profile→Compare→留出及通信校准 CLI 均通过。
- Wheel 内含 `py.typed`，安装元数据无必需第三方运行依赖；sdist 包含测试、示例和公共文档。
- CI 覆盖 Python 3.11 至 3.14 的回归测试、lint、类型和分发检查，使用固定 Actions SHA。

生成日志和内部协作笔记未纳入分发内容。仓库为私有，许可证仍待选择，没有发布 PyPI。
