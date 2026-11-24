"""Fit the phase increments from a preregistered two-process C2 CPU crossover."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlsys_sim import RuntimeCostModel  # noqa: E402
from mlsys_sim.config import load_experiment  # noqa: E402


def fit(base, plan_bytes, measured_bytes):
    plan, observed = json.loads(plan_bytes), json.loads(measured_bytes)
    digest = hashlib.sha256(plan_bytes).hexdigest()
    measured = observed["runs"]
    jobs = plan["roles"]["calibration"]["jobs"]
    if (
        observed["role"] != "calibration"
        or observed["plan_sha256"] != digest
        or len(measured) != 2
        or len(jobs) != 2
        or len({r["session"] for r in measured}) != 2
    ):
        raise ValueError("Require the two preregistered independent calibration sessions")
    cache = {}
    summaries = []
    increments = defaultdict(list)
    passed = True
    for run, job in zip(measured, jobs):
        if run["label"] != job["label"] or run["stream"] != job["stream"] or run["role"] != "calibration":
            raise ValueError("Wrong calibration stream or acquisition role")
        sample, residual, delivery = defaultdict(list), defaultdict(list), defaultdict(list)
        clean = [
            {k: v[k] for k in ("start", "stop", "center", "condition", "query_tokens")}
            for v in run["windows"]
        ]
        declared = json.dumps(dict(role="calibration", windows=clean)).encode()
        if hashlib.sha256(declared).hexdigest() != job["schedule_sha256"]:
            raise ValueError("Wrong calibration pulse assignment")
        for window in run["windows"]:
            condition = window["condition"]
            delivery[condition].append(window["cpu_fraction"])
            for i in range(window["start"], window["stop"]):
                phase, q, c, wall = run["steps"][i]
                key = phase, tuple(q), tuple(c)
                if key not in cache:
                    cache[key] = base.estimate_step(phase, q, c).duration_ms
                sample[phase, condition].append(wall)
                residual[phase, condition].append(wall - cache[key])
        phases = {}
        for phase in ("prefill", "decode"):
            med = {k: median(sample[phase, k]) for k in ("sibling", "separate", "sham")}
            delta = median(residual[phase, "sibling"]) - median(residual[phase, "sham"])
            ok = (
                med["sibling"] / med["sham"] >= 1.10
                and abs(med["separate"] - med["sham"]) < med["sibling"] - med["sham"]
                and delta > 0
            )
            passed &= ok
            phases[phase] = {"observed_medians_ms": med, "increment_ms": delta, "passed": ok}
            increments[phase].append(delta)
        summaries.append(
            dict(
                label=run["label"],
                session=run["session"],
                phases=phases,
                child_cpu_fraction={k: median(v) for k, v in delivery.items()},
            )
        )
    return dict(
        schema_version=1,
        kind="c2_smt_sibling_exposure_v1",
        base_model_sha256=base.provenance["model_sha256"],
        plan_sha256=digest,
        calibration_sha256=hashlib.sha256(measured_bytes).hexdigest(),
        calibration_sessions=[r["session"] for r in measured],
        summary=summaries,
        gate=passed,
        increment_ms={k: median(v) for k, v in increments.items()},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("experiment", "plan", "calibration", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    measured = a.calibration.read_bytes()
    if a.calibration.suffix == ".gz":
        measured = gzip.decompress(measured)
    result = fit(RuntimeCostModel(load_experiment(a.experiment)), a.plan.read_bytes(), measured)
    with a.out.open("x") as handle:
        json.dump(result, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return 0 if result["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
