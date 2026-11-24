"""Score frozen forecasts against independent C2 native-worker captures."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def measures(steps, windows):
    """Use unexposed exact-geometry steps as the reference for neighboring Decode."""
    labels = {}
    by_shape = defaultdict(list)
    by_condition = defaultdict(list)
    window_counts = Counter()
    end = 0
    for window in windows:
        start, stop, cond = window["start"], window["stop"], window["condition"]
        require(
            cond in {"sibling", "sham", "separate"} and end <= start < stop <= len(steps), "Invalid exposure"
        )
        require(
            start <= window["center"] < stop and steps[window["center"]][0] == "prefill",
            "Misaligned exposure",
        )
        end = stop
        window_counts[cond] += 1
        labels.update(dict.fromkeys(range(start, stop), cond))
    shapes = []
    for i, (phase, q, c, y) in enumerate(steps):
        require(math.isfinite(y) and y > 0, "Invalid duration")
        shape = (phase, tuple(q), tuple(c))
        shapes.append(shape)
        if i in labels:
            by_condition[phase, labels[i]].append(y)
        else:
            by_shape[shape].append(y)
    normal = {k: median(v) for k, v in by_shape.items()}
    hits = Counter()
    for i, (phase, _, _, y) in enumerate(steps):
        if i == 0 or phase != "prefill" or y < 38:
            continue
        if any(
            steps[j][0] == "decode" and steps[j][3] >= 1.3 * normal[shapes[j]]
            for j in (i - 1, i + 1)
            if j < len(steps)
        ):
            hits[labels.get(i, "uncontrolled")] += 1
    require(
        set(by_condition) == {(p, c) for p in ("prefill", "decode") for c in ("sibling", "separate", "sham")},
        "Missing condition",
    )
    return {f"{p}-{c}": median(v) for (p, c), v in by_condition.items()}, hits, window_counts


def metric(estimate, observation):
    require(all(math.isfinite(v) and v > 0 for v in (estimate, observation)), "Invalid metric")
    return {"predicted": estimate, "observed": observation, "relative_error": abs(estimate / observation - 1)}


def score(plan_bytes, forecast, evidence, receipt_bytes):
    freeze = json.loads(receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    pred_bytes = (json.dumps(forecast, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    expected = [v for k, v in freeze["files_sha256"].items() if k.endswith("-predictions.json")]
    require(expected == [hashlib.sha256(pred_bytes).hexdigest()], "Frozen prediction bytes differ")
    plan = json.loads(plan_bytes)
    require(
        evidence["role"] == "holdout"
        and evidence["plan_sha256"] == forecast["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(),
        "Unbound holdout or plan",
    )
    require(evidence["receipt_sha256"] == receipt_sha, "Wrong hardware freeze")
    jobs = plan["roles"]["holdout"]["jobs"]
    require(len(evidence["runs"]) == len(forecast["runs"]) == len(jobs) == 3, "Incomplete forward cohort")
    seen = set(forecast["calibration_sessions"])
    rows = []
    for j, p, r in zip(jobs, forecast["runs"], evidence["runs"]):
        require(
            r["session"] not in seen
            and datetime.fromisoformat(r["started"]) > datetime.fromisoformat(freeze["frozen_at_utc"]),
            "Reused session or measurement before freeze",
        )
        seen.add(r["session"])
        require(
            j["label"] == p["label"] == r["label"] and j["stream"] == p["stream"] == r["stream"],
            "Wrong request stream",
        )
        require(
            [x[:3] for x in r["steps"]] == [x[:3] for x in p["steps"]],
            "Different native and simulated geometries",
        )
        clean = [
            {k: v[k] for k in ("start", "stop", "center", "condition", "query_tokens")} for v in r["windows"]
        ]
        schedule = json.dumps(dict(role="holdout", windows=clean)).encode()
        require(
            hashlib.sha256(schedule).hexdigest() == j["schedule_sha256"] == p["schedule_sha256"],
            "Different pulse schedule",
        )
        a, ah, n = measures(r["steps"], r["windows"])
        f, fh, _ = measures(p["steps"], r["windows"])
        med = {k: metric(f[k], a[k]) for k in a}
        p99 = {k: metric(p[k], r[k]) for k in ("ttft", "e2e")}
        positive = ah["sibling"] > 0 and fh["sibling"] > 0
        tail_ok = positive and abs(fh["sibling"] - ah["sibling"]) / n["sibling"] <= 0.20
        rows.append(
            dict(
                label=r["label"],
                condition_medians=med,
                latency_p99=p99,
                observed_hits=dict(ah),
                predicted_hits=dict(fh),
                sibling_windows=n["sibling"],
                typical_passed=all(v["relative_error"] <= 0.10 for v in med.values()),
                p99_passed=all(v["relative_error"] <= 0.25 for v in p99.values()),
                historical_paired_gate="PASS"
                if tail_ok
                else "NO_OBSERVED_SIBLING_POSITIVE"
                if not ah["sibling"]
                else "MISSED"
                if not fh["sibling"]
                else "RATE_ERROR",
            )
        )
    return dict(
        role="independent_controlled",
        rows=rows,
        typical=all(v["typical_passed"] for v in rows),
        p99=all(v["p99_passed"] for v in rows),
        strict_paired_positive=all(v["historical_paired_gate"] == "PASS" for v in rows),
        freeze_sha256=evidence["receipt_sha256"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("plan", "forecast", "evidence", "freeze", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    result = score(a.plan.read_bytes(), load(a.forecast), load(a.evidence), a.freeze.read_bytes())
    with a.out.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return 0 if result["typical"] and result["p99"] and result["strict_paired_positive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
