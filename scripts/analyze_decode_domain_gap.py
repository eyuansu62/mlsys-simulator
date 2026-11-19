"""CPU historical Decode diagnostic, without fitting or hardware attribution.

The retained cost API rejects short-context/eager transfer. Out-of-domain
coefficient arithmetic below is a diagnostic proxy, never a hardware prediction.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mlsys_sim import load_experiment  # noqa: E402
from mlsys_sim.analytical_cost import (  # noqa: E402
    analytical_cost_identity, corrected_analytical_step, nominal_cost_features,
)
from mlsys_sim.performance import AnalyticalModel  # noqa: E402
from mlsys_sim.schema import Experiment, SimulationError  # noqa: E402

BASE = REPO / "examples/runtime_cost_20261116"
HISTORICAL_TPOT_MS = {1: 21.204, 2: 22.584, 4: 22.684, 8: 29.461}


def _scope_rejection(experiment: Experiment, context: int) -> dict[str, str]:
    try:
        corrected_analytical_step(experiment, "decode", [1], [context])
    except SimulationError as exc:
        if exc.code != "CALIBRATION_MISMATCH":
            raise
        return {"error_code": exc.code, "message": str(exc)}
    raise RuntimeError("Retained cost unexpectedly accepted an out-of-domain diagnostic query")


def analyze(base: Path = BASE) -> dict[str, Any]:
    """Return structured CPU arithmetic; import performs no analysis or file reads."""
    calibrated = load_experiment(base / "experiments/model-2.json")
    calibration = calibrated.calibration
    identity = analytical_cost_identity(calibrated)
    candidates = [entry for entry in calibration.analytical_costs
        if (entry.identity_sha256, entry.phase, entry.execution_mode, entry.attention_backend)
        == (identity, "decode", calibration.decode_execution_mode, calibration.decode_attention_backend)]
    if len(candidates) != 1:
        raise ValueError("Expected one retained 4B Decode cost in its exact declared domain")
    cost = candidates[0]
    # Preserve the contributor's historical eta/beta transfer unchanged.
    historical = replace(calibrated, calibration=replace(calibration,
        compute_efficiency=0.6243, bandwidth_efficiency=0.9054,
        launch_overhead_us=2.0, framework_overhead_us=5.0, analytical_costs=()))
    model = AnalyticalModel(historical)
    contexts = list(range(257, 272))
    rows, components = [], []
    for batch, reference in HISTORICAL_TPOT_MS.items():
        defaults, proxies = [], []
        for context in contexts:
            defaults.append(model.estimate_step("decode", [1] * batch, [context] * batch).duration_ms)
            features = nominal_cost_features(historical, "decode", [1] * batch, [context] * batch)
            # Diagnostic only: never widen bounds or install extrapolated costs.
            proxies.append(cost.nonattention_scale * features["nonattention_ms"]
                + cost.attention_scale * features["attention_ms"]
                + (cost.attention_fixed_us_per_layer * features["layers"] + cost.runtime_fixed_us) / 1000)
        default, proxy = statistics.median(defaults), statistics.median(proxies)
        estimate = model.estimate_step("decode", [1] * batch, [265] * batch)
        linear = estimate.breakdown["linear"]
        rows.append({"batch_size_proxy": batch, "closed_loop_concurrency_reference": batch,
            "historical_closed_loop_tpot_reference_ms": reference, "default_decode_median_ms": default,
            "default_context_endpoints_ms": [defaults[0], defaults[-1]], "out_of_domain_cost_proxy_ms": proxy,
            "default_vs_reference_signed_pct": (default / reference - 1) * 100,
            "proxy_vs_reference_signed_pct": (proxy / reference - 1) * 100})
        components.append({"batch_size": batch, "context_tokens": 265, "model_total_ms": estimate.duration_ms,
            "model_component_ms": {name: estimate.breakdown[name]["duration_ms"] for name in
                ("linear", "embedding_and_output", "misc", "framework", "attention", "communication")},
            "linear_fraction_model_internal": linear["duration_ms"] / estimate.duration_ms,
            "linear_hbm_bytes_model_internal": linear["hbm_bytes_per_rank"],
            "bandwidth_proxy_gbps": linear["hbm_bytes_per_rank"] / (reference * 1e6)})
    eager_probe = replace(calibrated, calibration=replace(calibration, decode_execution_mode="eager"))
    return {"kind": "historical_decode_cpu_diagnostic_v1", "hardware_validated": False, "fitted": False,
        "scope": {"short_context_probe": _scope_rejection(calibrated, 257),
                  "eager_selector_probe": _scope_rejection(eager_probe, cost.min_context_tokens)},
        "retained_decode_cost": asdict(cost), "context_proxy_tokens": contexts, "rows": rows,
        "model_internal_components": components,
        "c1_to_c8_reference_change_pct": (rows[-1]["historical_closed_loop_tpot_reference_ms"]
            / rows[0]["historical_closed_loop_tpot_reference_ms"] - 1) * 100,
        "b1_to_b8_default_model_change_pct": (rows[-1]["default_decode_median_ms"]
            / rows[0]["default_decode_median_ms"] - 1) * 100,
        "b1_to_b8_out_of_domain_proxy_change_pct": (rows[-1]["out_of_domain_cost_proxy_ms"]
            / rows[0]["out_of_domain_cost_proxy_ms"] - 1) * 100,
        "interpretation": [
            "The 21.2ms-class references are historical closed-loop TPOT, not isolated batch/kernel timings.",
            "Homogeneous B1/B2/B4/B8 CPU queries are concurrency-reference proxies, not captured runtime batches.",
            "Configured Graph/RocmAttentionImpl costs grant no eager or short-context transfer.",
            "Linear shares (including approximately 88%) are analytical-model attribution, not profiler evidence.",
            "Bandwidth proxies divide modeled Linear bytes by closed-loop TPOT; traffic and bandwidth were not measured.",
            "These numbers establish no kernel/CPU root cause, new calibration or holdout accuracy result.",
        ]}


def print_analysis(report: dict[str, Any]) -> None:
    print("CPU HISTORICAL DIAGNOSTIC: hardware_validated=false; fitted=false")
    print("References are closed-loop TPOT; B denotes a homogeneous batch proxy.")
    print("B/C  TPOT reference  default model  out-of-domain proxy (ms)")
    for row in report["rows"]:
        print(f"{row['batch_size_proxy']:3d}  {row['historical_closed_loop_tpot_reference_ms']:14.3f}"
              f"  {row['default_decode_median_ms']:13.3f}  {row['out_of_domain_cost_proxy_ms']:19.3f}")
    for name, rejection in report["scope"].items():
        print(f"SCOPE ENFORCED ({name}): {rejection['error_code']}: {rejection['message']}")
    print("Model-internal context-265 components; bandwidth is a TPOT-denominator proxy:")
    for row in report["model_internal_components"]:
        print(f"  B={row['batch_size']:2d} total={row['model_total_ms']:.3f}ms"
              f" Linear share={row['linear_fraction_model_internal']:.1%}"
              f" bandwidth_proxy={row['bandwidth_proxy_gbps']:.3f}GB/s")
    for message in report["interpretation"]:
        print("- " + message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE, help="Retained cost example directory")
    parser.add_argument("--json", action="store_true", help="Print structured CPU diagnostic")
    args = parser.parse_args(argv)
    report = analyze(args.base)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print_analysis(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
