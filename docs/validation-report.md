# 首个可执行版本验证记录

日期：2026-11-03  
环境：Python 3.14.3，macOS，CPU 执行  
版本：0.1.0

## 自动检查

- `python3 -m unittest discover -s tests -v`：69 项测试通过，其中包含 130 组固定 seed 的生成负载，以及公共 Python API 调用。
- Ruff 0.17.3：检查通过。
- mypy 2.4.1：`--check-untyped-defs --ignore-missing-imports mlsys_sim`，9 个源文件检查通过。
- `python3 -m compileall -q mlsys_sim tests simulate.py predict.py`：通过。

Ruff/mypy 为临时开发检查工具，没有加入项目运行依赖。测试覆盖配置拒绝、
公式与单位、GQA 复制、跨 rank 原子预留、物理与逻辑显存分离、确定性时间线、
超时在途引用、重算不重复输出、失败后的资源释放和重新准入、指标及输出不可覆盖。
预算在 Step 中途截止时计入已执行片段，并保留未提交状态；超大整数和浮点上溢/下溢明确拒绝。
生成负载还检查了关闭详细日志不会改变结果。

## CLI 示例证据

四个示例均通过公开 CLI 执行。生成的 `sample-runs/` 保留在开发工作区，不纳入仓库或分发包；可按 README 中的命令重新生成。

| 示例 | 观察结果 |
| --- | --- |
| FIFO 手算 | 3 个请求完成；首 Token 为 10/24/34ms，结束为 14/24/36ms |
| 内存压力 | 2 个请求各输出24 Token；1 次抢占、32 Token 重算、重算5ms，总模拟56ms，最终 KV 全部释放 |
| TP 2 | 3 个请求完成；输出各 rank 账本及通信分量 |
| PP 2 解析 | 2 个不等长 stage；在线指标为 null，原因 PP_ONLINE_DES_NOT_IMPLEMENTED |

内存压力示例中的耗时来自 synthetic 固定阶段 Profile；TP/PP 的数值来自未校准解析模型。
这些检查验证程序机制与假设内部一致性，不构成真实 GPU 时延准确率证明。

## 当前限制

当前只验证了 Python 3.14.3 上的执行；代码声明最低 Python 3.11，尚未建立跨版本 CI。
没有进行 AMD 实机测量、Runtime Trace 对齐或留出误差验证；真实 Profile、CPU swap 和
PP 在线流水线列为后续工作。能力边界见 `coverage-matrix.md`。

## 库化与分发验证

- Setuptools 84.2.1 通过标准 PEP 517 隔离构建生成 sdist，再从该 sdist 构建 wheel。
- Twine 7.0.0 对两种分发文件的元数据检查通过。
- `scripts/verify_distribution.py` 在源码目录以外创建新虚拟环境，离线安装 wheel；公共 API、内存压力模拟、PP 解析和安装后的 `mlsys-sim` 命令通过。
- Wheel 内含 `py.typed`，安装元数据无必需第三方运行依赖；sdist 包含测试、示例和公共文档。
- 本地 editable 安装后的公共 API 和控制台入口通过。
- CI 已配置 Python 3.11 至 3.14 的回归测试、lint、类型和分发检查，使用固定 Actions SHA；远程 CI 尚未运行。

生成日志和内部协作笔记未纳入分发内容。许可证按项目所有者的决定暂不选择，当前只准备本地仓库和草稿分发文件，未公开发布或上传 PyPI。
