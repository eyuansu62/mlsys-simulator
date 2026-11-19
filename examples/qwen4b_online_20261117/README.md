# Qwen3-4B在线成本：冻结预测与重放包

本包是推理性能校准与验证，不训练Qwen权重。原始大JSON使用标准gzip无损保存；SHA按解压后的原始字节核对。

```sh
python - <<'PY'
import gzip, hashlib, json
from pathlib import Path
root = Path('examples/qwen4b_online_20261117')
index = json.loads((root / 'replay-inventory.json').read_text())
destination = Path('runs/q4b-replay')
for item in index['artifacts']:
    stored = (root / item['published_file']).read_bytes()
    raw = gzip.decompress(stored) if item['storage'] == 'gzip' else stored
    assert hashlib.sha256(raw).hexdigest() == item['source_raw_sha256']
    if item['replay_target'] is None:
        continue
    target = destination / Path(item['replay_target']).relative_to('replay')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
PY
mlsys-sim simulate --experiment runs/q4b-replay/experiment.json --out runs/q4b-runtime
```

解压后的`runs/q4b-replay/experiment.json`是独立可移植副本，只将artifact路径改成相对路径；原冻结experiment字节仍保存在frozen包。默认workload若为空，普通simulate只作配置/加载检查；要重放闭环研究，调用`predict_closed_loop`并使用相应spec。

```python
from pathlib import Path
import json
from mlsys_sim.config import load_experiment
from mlsys_sim import predict_closed_loop
root = Path('runs/q4b-replay')
experiment = load_experiment(root / 'experiment.json')
spec = json.loads((root / 'baseline.spec.json').read_text())
prediction = predict_closed_loop(experiment, spec)
```

重放只使用冻结的策略和请求流，不用真机实际到达时间。可移植副本的路径与原环境不同，原raw/canonical预测字节的验证应针对frozen存档；数值重放与源SHA核对分开。跨Python版本的浮点末位和nominal-span边界诊断标志可能不同，不能据此声称跨版本序列化或诊断标志逐位等价；数值成本/指标应按明确容差复核，原冻字节SHA保持原值。

- `frozen/`：接收方冻结的JSON/gzip、主机回执与原始字节/canonical SHA映射。
- `calibration/`：接受的模型、归一化正式校准数据和source lineage；未知Workspace保持null。
- `rejections/`：接受模型之前的拒绝校准，保持REJECTED，不混入fit。
- `replay-inventory.json`：每个文件的原始、canonical及存储SHA；`replay_target`明确哪些文件解压加载。
- `source-code-sha256.json`：测前源码提交及各模块SHA；不随最终文档提交改写。

- `audit/`：最终冻版独立CPU评估器原字节、单独来源SHA与调用说明。其源不是冒称来自库commit。完整raw评分需要私有原始study；公开本包足够重放冻结模拟预测。

[完整图文报告](../../docs/qwen4b-online-20261117/README.md) · [API与方法](../../docs/runtime-core-costs.md)

[cpu-regression-cases.json](cpu-regression-cases.json)保存独立标注的CPU数值回归参考。它来自所列接受模型的CPU API重放，不是新真机采样或P99/SLA验收；原字节复制并纳入本目录sha256清单，冻结inventory格式不变。
