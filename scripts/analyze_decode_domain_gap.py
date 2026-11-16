"""Step 1 (CPU-only): quantify what the 2026-11-16 runtime-cost calibration
explains for the 2026-11-11 eager 9070XT SLA study.

Every out-of-domain number printed below is an ANALYSIS EXTRAPOLATION, not a
prediction: the retained decode specs cover context 1024-1043 only, and the
11-11 study ran eager MRv1 while the fit is bound to
graph_full_and_piecewise_configured / RocmAttentionImpl. The tooling itself
refuses to apply the fit at the 11-11 shape (see the SCOPE line).
"""
import gzip
import json
import statistics
import sys
from dataclasses import replace

sys.path.insert(0, '/mnt/luyuzhou/hello-mlsys/mlsys-simulator')
from mlsys_sim import load_experiment
from mlsys_sim.analytical_cost import (
    nominal_cost_features,
    corrected_analytical_step,
)
from mlsys_sim.performance import AnalyticalModel
from mlsys_sim.schema import SimulationError

BASE = '/mnt/luyuzhou/hello-mlsys/mlsys-simulator/examples/runtime_cost_20261116'
exp_cal = load_experiment(f'{BASE}/experiments/model-2.json')  # Qwen3-4B + fitted costs

# 1) 11-11 default spec: course eta/beta transfer, no analytical costs.
cal = replace(exp_cal.calibration,
              compute_efficiency=0.6243, bandwidth_efficiency=0.9054,
              launch_overhead_us=2.0, framework_overhead_us=5.0,
              analytical_costs=())
exp1111 = replace(exp_cal, calibration=cal)
am = AnalyticalModel(exp1111)


def default_decode(b: int, ctxs) -> float:
    return am.estimate_step('decode', [1] * b, [c for c in ctxs]).duration_ms


# 11-11 baseline: input 256, output 16. TPOT spans decode steps 2..16 (15
# intervals); context before step k+1 is 256 + k, i.e. 257..271.
ctxs = list(range(257, 272))
print('DEFAULT (11-11 course eta/beta, no analytical costs) decode median ms:')
for B in (1, 2, 4, 8):
    per = [default_decode(B, [c] * B) for c in ctxs]
    print(f'  B={B:2d}: median {statistics.median(per):8.3f}  '
          f'(ctx257 {per[0]:.3f} -> ctx271 {per[-1]:.3f})')

# 2) Scope enforcement: the fitted model must refuse the 11-11 shape.
try:
    corrected_analytical_step(exp_cal, 'decode', [1], [257])
    print('SCOPE: unexpectedly accepted ctx=257')
except SimulationError as e:
    print(f'SCOPE ENFORCED: {e}')

# 3) Out-of-domain extrapolation with the fitted 4B decode spec (analysis only).
fit = json.load(gzip.open(f'{BASE}/fit.json.gz', 'rt'))
spec = [c for c in fit['cost_specs'] if c['phase'] == 'decode'
        and c['identity_sha256'].startswith('11efc636')][0]
layers = exp_cal.strategy.model.num_layers
print(f'\nFitted 4B decode spec: nonattention x{spec["nonattention_scale"]:.4f}, '
      f'attention x{spec["attention_scale"]:.4f}, '
      f'attn_fixed {spec["attention_fixed_us_per_layer"]:.3f}us/layer x{layers}, '
      f'runtime_fixed {spec["runtime_fixed_us"]:.1f}us. Retained ctx '
      f'{spec["min_context_tokens"]}-{spec["max_context_tokens"]}, '
      f'B {spec["min_batch_size"]}-{spec["max_batch_size"]}.')


def extrap(b: int, ctxs) -> float:
    per = []
    for c in ctxs:
        f = nominal_cost_features(exp1111, 'decode', [1] * b, [c] * b, 0)
        ms = (spec['nonattention_scale'] * f['nonattention_ms']
              + spec['attention_scale'] * f['attention_ms']
              + (spec['attention_fixed_us_per_layer'] * layers
                 + spec['runtime_fixed_us']) / 1000.0)
        per.append(ms)
    return statistics.median(per)


print('\nEXTRAPOLATED calibrated decode (fitted 4B spec, OUT OF DOMAIN):')
for B in (1, 2, 4, 8):
    print(f'  B={B:2d}: median {extrap(B, ctxs):8.3f} ms')

# 4) Compare with 11-11 measured decode (median of 3 fresh runtimes).
measured = {1: 21.204, 2: 22.584, 4: 22.684, 8: 29.461}
print('\nB   measured  default  def_err   extrap_cal  extrap_err')
for B in sorted(measured):
    m = measured[B]
    d = statistics.median([default_decode(B, [c] * B) for c in ctxs])
    e = extrap(B, ctxs)
    print(f'{B:2d}  {m:8.3f}  {d:7.3f}  {(d / m - 1) * 100:+7.1f}%  '
          f'{e:11.3f}  {(e / m - 1) * 100:+7.1f}%')

print('\nC1->C8 slope: measured %+.1f%%  default %+.1f%%  extrap %+.1f%%' % (
    (measured[8] / measured[1] - 1) * 100,
    (statistics.median([default_decode(8, [c] * 8) for c in ctxs])
     / statistics.median([default_decode(1, [c] * 1) for c in ctxs]) - 1) * 100,
    (extrap(8, ctxs) / extrap(1, ctxs) - 1) * 100))

# 5) Component breakdown at a single representative context (C=265).
print('\nComponent breakdown (median ctx=265), nominal 11-11 spec, ms:')
for B in (1, 4, 8):
    est = am.estimate_step('decode', [1] * B, [265] * B)
    parts = est.breakdown
    row = {k: parts[k]['duration_ms'] for k in
           ('linear', 'embedding_and_output', 'misc', 'framework',
            'attention', 'communication')}
    tot = est.duration_ms
    print(f'  B={B:2d} total {tot:7.3f} | ' +
          '  '.join(f'{k} {v:.3f}' for k, v in row.items()))
