# Agent 协议联调样例

来源与固定commit见 [sources.json](sources.json)，接口与支持范围见
[Agent 接入说明](../../docs/agent-integration.md)。

- `task-9.x.unconfirmed.request.json`：逐字节保留课程团队的原始请求。
  其中待确认目录、模型和硬件必须拒绝，不能当作已能执行的课程基线。
- `task-9.x.protocol.request.json`：明确命名的独立协议测试变体，使用本服务已公布的
  Qwen3/BF16架构与RX9070XT课程名义配置。ID、长度、到达和依赖间隔保持原样；
  部署与目标模型ID、硬件、精度、目录版本及测试标签显式改变。
- `request.schema.json`、`response.schema.json`：对方固定版本的结构契约，
  不表示本服务支持所有Schema允许的模型/精度/策略组合；还需检查服务目录。

这些变体验证9.2原始列表、9.1终态顺序和9.3多模型因果依赖的接口行为，
**不是原3B/32B/7B、FP8或W7900D基线的性能结果或双方验收**。
原始JSONL的 `source_sha256` 保留为来源声明，传输body另由服务计算 `input_sha256`。
服务与课程平台完成目录协商、平台接入并使用正式配置联调后，才可更新基线验收状态。
