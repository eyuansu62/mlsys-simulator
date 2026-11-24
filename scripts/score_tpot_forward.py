"""Score frozen complete-window TPOT forecasts against a separate native cohort."""

import argparse
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def raw(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def digest(data):
    return hashlib.sha256(data).hexdigest()


def p99(values):
    values = sorted(values)
    pos = 0.99 * (len(values) - 1)
    return values[math.floor(pos)] + (pos % 1) * (values[math.ceil(pos)] - values[math.floor(pos)])


def summarize(rows, key):
    errors = [r[key] for r in rows]
    return {
        "cells": len(rows),
        "under": sum(e < 0 for e in errors),
        "over": sum(e > 0 for e in errors),
        "mean_signed": mean(errors),
        "median_signed": median(errors),
        "mape": mean(abs(e) for e in errors),
        "max_absolute": max(map(abs, errors)),
    }


def score(plan_bytes, forecast_bytes, evidence, receipt_bytes):
    plan = json.loads(plan_bytes)
    pred = json.loads(forecast_bytes)
    receipt = json.loads(receipt_bytes)
    require(
        evidence["role"] == "holdout"
        and digest(plan_bytes) == pred["plan_sha256"] == evidence["plan_sha256"],
        "Wrong role/plan",
    )
    require(
        receipt["files_sha256"]["prediction.json"] == digest(forecast_bytes)
        and evidence["receipt_sha256"] == digest(receipt_bytes),
        "Prediction receipt mismatch",
    )
    require(evidence["base_file_sha256"] == pred["base_file_sha256"], "Wrong runtime model")
    measured = {r["label"]: r for r in evidence["sessions"]}
    estimates = {r["label"]: r for r in pred["runs"]}
    jobs = plan["jobs"]["holdout"]
    require(
        len(measured) == len(evidence["sessions"]) == len(estimates) == len(pred["runs"]) == len(jobs)
        and set(measured) == set(estimates) == {j["label"] for j in jobs},
        "Incomplete or duplicated forward cells",
    )
    sessions = set(pred["calibration_sessions"])
    hashes = set(pred["calibration_captures"])
    past = set(pred["calibration_streams"])
    rows = []
    for j in jobs:
        m = measured[j["label"]]
        f = estimates[j["label"]]
        spec = plan["specs"][j["spec_file"]]
        require(
            all(m[k] == v for k, v in j.items()) and all(f[k] == v for k, v in j.items()),
            "Cell binding changed",
        )
        require(
            m["spec"] == spec
            and m["session_id"] not in sessions
            and m["capture_sha256"] not in hashes
            and m["stream_sha256"] not in past
            and datetime.fromisoformat(m["started"]) > datetime.fromisoformat(receipt["frozen_at_utc"]),
            "Holdout isolation failed",
        )
        sessions.add(m["session_id"])
        hashes.add(m["capture_sha256"])
        windows = m["windows"]
        require(
            len(windows) == spec["requests_per_run"] and {v[0] for v in windows} == set(range(len(windows))),
            "Incomplete server cohort",
        )
        require(all(math.isfinite(v[2]) and v[2] > 0 for v in windows), "Invalid observed TPOT")
        observed = p99([v[2] for v in windows])
        require(math.isclose(observed, m["tpot_p99_ms"], abs_tol=1e-7), "Server TPOT P99 mismatch")
        base = f["window_forecast"]["base_p99_ms"]
        candidate = f["window_forecast"]["p99_ms"]
        require(
            f["window_forecast"]["profile_sha256"] == pred["profile_sha256"]
            and all(math.isfinite(x) and x > 0 for x in (base, candidate)),
            "Invalid forecast",
        )
        rows.append(
            dict(
                label=j["label"],
                cohort=j["cohort"],
                concurrency=j["concurrency"],
                replication=j["replication"],
                base_ms=base,
                predicted_ms=candidate,
                observed_ms=observed,
                base_signed=(base - observed) / observed,
                candidate_signed=(candidate - observed) / observed,
            )
        )
    primary = [r for r in rows if r["replication"] == 0]
    require(
        len(primary) == 18
        and len(rows) == 20
        and len({(r["cohort"], r["concurrency"]) for r in primary}) == 18,
        "Missing primary workload point",
    )
    base = summarize(primary, "base_signed")
    candidate = summarize(primary, "candidate_signed")
    policy = plan["acceptance"]
    gates = {
        "individual": max(abs(r["candidate_signed"]) for r in rows)
        <= policy["individual_relative_error_max"],
        "mape_improved": candidate["mape"] < base["mape"],
        "bias_improved": abs(candidate["mean_signed"]) < abs(base["mean_signed"]),
        "bias_limit": abs(candidate["mean_signed"])
        <= policy["primary_absolute_mean_signed_relative_error_max"],
        "direction_balance": max(candidate["under"], candidate["over"])
        <= policy["primary_under_and_over_max_each"],
    }
    return dict(
        passed=all(gates.values()),
        gates=gates,
        primary_baseline=base,
        primary_candidate=candidate,
        all_baseline=summarize(rows, "base_signed"),
        all_candidate=summarize(rows, "candidate_signed"),
        rows=rows,
        receipt_sha256=digest(receipt_bytes),
        profile_sha256=pred["profile_sha256"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "forecast", "evidence", "receipt", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    result = score(raw(a.plan), raw(a.forecast), json.loads(raw(a.evidence)), raw(a.receipt))
    with a.out.open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
