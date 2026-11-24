"""Freeze or independently score the RX 9070 XT short-Prefill study."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlsys_sim import RuntimeCostModel, fit_short_prefill, predict_closed_loop  # noqa: E402
from mlsys_sim.config import load_experiment  # noqa: E402
from mlsys_sim.short_prefill import BATCHES, LIMIT, MIN_STEPS  # noqa: E402

BASE = Path(__file__).resolve().parents[1] / "examples/qwen4b_online_20261117"
PROFILE_ID = "q4b-short-prefill-shapes-v2"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw_json(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validated_runs(plan_path, evidence, role):
    plan = json.loads(plan_path.read_bytes())
    check(
        plan["shape_error_limit"] == LIMIT
        and plan["min_shape_steps_per_session"] == MIN_STEPS
        and plan["end_to_end_p99_error_limit"] == 0.25,
        "Unknown study thresholds",
    )
    check(
        evidence["plan_sha256"] == sha(plan_path.read_bytes()) and evidence["role"] == role,
        "Wrong evidence role or plan",
    )
    stage = plan["phases"][role]
    rows = evidence["rows"]
    check(len(rows) == len(stage["jobs"]) and len(rows) == len(BATCHES) * 2, "Incomplete experiment")
    check(
        {(r["concurrency"], r["replication"]) for r in rows} == set(product(BATCHES, (0, 1))),
        "Missing required batch or repetition",
    )
    check(
        len({r["session_id"] for r in rows}) == len(rows)
        and len({r["source_files"]["runtime-cost-capture.json"] for r in rows}) == len(rows),
        "Acquisition reused in study",
    )
    for job, row in zip(stage["jobs"], rows):
        check(all(row[k] == v for k, v in job.items()), "Wrong job")
        check(
            row["stream_sha256"] == stage["streams"][str(job["replication"])]
            and row["runtime_sha256"] == plan["expected_runtime_sha256"]
            and row["checkpoint_sha256"] == plan["expected_checkpoint_sha256"],
            "Acquisition identity mismatch",
        )
        batch = job["concurrency"]
        keys = set(product((64, 128), repeat=batch))
        check(
            len(row["shapes"]) == len(keys) and {tuple(x["query_tokens"]) for x in row["shapes"]} == keys,
            "Missing or duplicated ordered Prefill shape",
        )
        for item in row["shapes"]:
            costs = item["durations_ms"]
            check(
                len(costs) >= MIN_STEPS
                and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in costs),
                "Invalid or insufficient shape observations",
            )
        check(
            sum(len(x["durations_ms"]) for x in row["shapes"]) * batch == stage["spec"]["requests_per_run"],
            "Incomplete Prefill accounting",
        )
    return rows


def materialize(plan_path, evidence, out):
    rows = validated_runs(plan_path, evidence, "calibration")
    base = json.loads(gzip.decompress((BASE / "calibration/runtime-cost-model.json.gz").read_bytes()))
    domain = base["domain"]["domain_sha256"]
    check(all(r["domain_sha256"] == domain for r in rows), "Base runtime domain mismatch")
    sessions = [
        dict(
            session_id=r["session_id"],
            batch_size=r["concurrency"],
            capture_sha256=r["source_files"]["runtime-cost-capture.json"],
            shapes=r["shapes"],
        )
        for r in rows
    ]
    fit = fit_short_prefill(
        base,
        dict(
            domain_sha256=domain, plan_sha256=evidence["plan_sha256"], role="calibration", sessions=sessions
        ),
        calibration_id=PROFILE_ID,
    )
    out.mkdir(parents=True, exist_ok=False)
    model = out / "runtime-cost-model.json"
    model.write_bytes(raw_json(fit))
    exp = json.loads((BASE / "portable-experiment.json").read_bytes())
    exp["calibration"]["profile_version"] = PROFILE_ID
    config = out / "experiment.json"
    config.write_bytes(raw_json(exp))
    return fit, rows, config


def freeze(plan_path, evidence, out):
    plan = json.loads(plan_path.read_bytes())
    fit, rows, config = materialize(plan_path, evidence, out)
    stream = predict_closed_loop(load_experiment(config), plan["phases"]["holdout"]["spec"])
    check(stream["status"] == "COMPLETED", "Could not freeze complete predictions")
    runs = []
    for run in stream["runs"]:
        check(run["resources_drained"] and run["status"] == "COMPLETED", "Incomplete prediction")
        times = run["result"]["latency_ms"]
        runs.append(
            dict(
                concurrency=run["concurrency"],
                replication=run["replication"],
                stream_sha256=run["request_stream_sha256"],
                e2e_p99_ms=times["end_to_end"]["p99"],
                ttft_p99_ms=times["ttft"]["p99"],
                tpot_p99_ms=times["tpot"]["p99"],
                requests_sha256=sha(raw_json(run["requests"])),
            )
        )
    # Preserve request predictions separately. The large embedded Profile already exists in this directory.
    (out / "predicted-requests.json.gz").write_bytes(
        gzip.compress(raw_json([r["requests"] for r in stream["runs"]]), mtime=0)
    )
    backend = RuntimeCostModel(load_experiment(config))
    shape_predictions = {}
    for key in fit["short_prefill"]["table"]:
        phase, stage, queries, contexts = json.loads(key)
        shape_predictions[key] = backend.estimate_step(phase, queries, contexts, stage).duration_ms
    evidence_files = {
        name: sha((out / name).read_bytes())
        for name in ("experiment.json", "runtime-cost-model.json", "predicted-requests.json.gz")
    }
    code = Path(__file__).resolve().parents[1] / "mlsys_sim"
    prediction = dict(
        plan_sha256=evidence["plan_sha256"],
        model_sha256=fit["model_sha256"],
        domain_sha256=fit["domain"]["domain_sha256"],
        scoring_program_sha256=sha(Path(__file__).read_bytes()),
        files_sha256=evidence_files,
        code_sha256={p.name: sha(p.read_bytes()) for p in sorted(code.glob("*.py"))},
        calibration_sessions=[r["session_id"] for r in rows],
        calibration_captures=[r["source_files"]["runtime-cost-capture.json"] for r in rows],
        calibration_streams=sorted({r["stream_sha256"] for r in rows}),
        shape_durations_ms=shape_predictions,
        runs=runs,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    (out / "predictions.json").write_bytes(raw_json(prediction))


def score(plan_path, evidence, prediction):
    rows = validated_runs(plan_path, evidence, "holdout")
    receipt = evidence["freeze"]
    check(
        receipt["prediction_sha256"] == sha(raw_json(prediction))
        and receipt["plan_sha256"] == prediction["plan_sha256"] == evidence["plan_sha256"]
        and receipt["model_sha256"] == prediction["model_sha256"],
        "Unbound holdout or prediction",
    )
    remote_freeze = datetime.fromisoformat(receipt["frozen_at_utc"])
    runs = {(r["concurrency"], r["replication"]): r for r in prediction["runs"]}
    check(len(runs) == len(rows) == len(prediction["runs"]), "Incomplete frozen predictions")
    measurements = []
    for row in rows:
        pred = runs[row["concurrency"], row["replication"]]
        check(
            datetime.fromisoformat(row["started_at_utc"]) > remote_freeze
            and row["session_id"] not in prediction["calibration_sessions"]
            and row["source_files"]["runtime-cost-capture.json"] not in prediction["calibration_captures"]
            and row["stream_sha256"] not in prediction["calibration_streams"]
            and row["domain_sha256"] == prediction["domain_sha256"]
            and pred["stream_sha256"] == row["stream_sha256"],
            "Holdout independence or runtime identity mismatch",
        )
        shape_rows = []
        for shape in row["shapes"]:
            q = shape["query_tokens"]
            key = json.dumps(["prefill", 0, q, [0] * len(q)], separators=(",", ":"))
            shape_rows.append(
                dict(
                    query_tokens=q,
                    count=len(shape["durations_ms"]),
                    **compare(prediction["shape_durations_ms"][key], median(shape["durations_ms"]), LIMIT),
                )
            )
        e2e = compare(pred["e2e_p99_ms"], row["e2e_p99_ms"], 0.25)
        measurements.append(
            dict(
                label=row["label"],
                shapes=shape_rows,
                e2e_p99_ms=e2e,
                ttft_p99_ms={"predicted": pred["ttft_p99_ms"], "observed": row["ttft_p99_ms"]},
                passed=all(s["passed"] for s in shape_rows) and e2e["passed"],
            )
        )
    return dict(
        passed=all(r["passed"] for r in measurements),
        measurements=measurements,
        plan_sha256=prediction["plan_sha256"],
        model_sha256=prediction["model_sha256"],
    )


def compare(pred, actual, limit):
    check(
        type(pred) in (float, int)
        and math.isfinite(pred)
        and pred > 0
        and type(actual) in (float, int)
        and math.isfinite(actual)
        and actual > 0,
        "Invalid metric",
    )
    error = abs(pred - actual) / actual
    return dict(predicted=pred, observed=actual, relative_error=error, passed=error <= limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["materialize", "freeze", "score"])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_bytes())
    if args.action == "materialize":
        materialize(args.plan, evidence, args.out)
        return 0
    if args.action == "freeze":
        freeze(args.plan, evidence, args.out)
        return 0
    if args.predictions is None:
        parser.error("score requires --predictions")
    result = score(args.plan, evidence, json.loads(args.predictions.read_bytes()))
    with args.out.open("xb") as f:
        f.write(raw_json(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
