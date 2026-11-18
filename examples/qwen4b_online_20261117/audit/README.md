# 测前冻结的独立CPU评估器

`evaluate_forward.py`按主机回执和人工审阅确认的原始SHA `34cde56cbb9388796ee1fe65e519015f419a96171e8295a587560fa50af88b69` 逐字节发布。它是独立辅助脚本，不声称包含在模拟器库的源码提交中；其单独来源见[source.json](source.json)。

```sh
python examples/qwen4b_online_20261117/audit/evaluate_forward.py \
  --study /PATH/TO/COMPLETE_PRIVATE_RAW_STUDY \
  --expected-receipt-sha256 2b7bb2c2a3b49613deeb04d97cfd1b22c8d98749b0280ad9bcec40995e471b82 \
  --code-root /PATH/TO/FROZEN_LIBRARY_CHECKOUT/mlsys_sim \
  --out runs/q4b-independent-raw-audit
```

库源码应checkout/copy冻结提交`4169cbafc7b614bf5c3540647f81ab7ae45944db`，原始模块字节必须与回执SHA一致。评估器只核对这些文件，不导入或执行模拟器，也不需要Torch/vLLM/GPU。

重新评分需要**完整原始study**：测前冻结目录、全部54次尝试、原始Scheduler/API/core记录及SHA。公开summary、cells.csv或仅有模型包不能替代这些原始证据。完整raw由所有者保留，私有资料不会因此自动公开。

公开model/spec可以独立重放模拟预测；该步骤和对私有原始真机证据重新评分是两种不同操作。`freeze_online.py`和`forward_controller.py`负责准备/硬件控制，未在本包作为CPU重放代码发布；其原始SHA仍由冻结manifest及保留归档索引记录。
