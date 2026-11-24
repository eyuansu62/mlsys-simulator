"""Reconstruct server TPOT windows and replay their native schedule on a frozen cost model."""

from bisect import bisect_right
from collections import defaultdict
import math
import random
from statistics import median


def check(condition, message):
    if not condition:
        raise ValueError(message)


def geometry(row):
    members = row["members"]
    roles = tuple(m["request_phase"] for m in members)
    phase = "idle" if not roles else roles[0] if len(set(roles)) == 1 else "mixed"
    return (
        phase,
        tuple(m["query_tokens"] for m in members),
        tuple(m["context_length"] for m in members),
        roles if phase == "mixed" else None,
    )


def _edges(steps, ends, start, end):
    i = bisect_right(ends, start)
    while i < len(steps) and steps[i][0] < end:
        a, b, _, wall = steps[i]
        clipped = (min(b, end) - max(a, start)) / 1e6
        yield i, clipped, clipped / wall
        i += 1


def replay_windows(requests, capture, estimate):
    """Clip overlapping steps at both TOKEN_READY boundaries, including other requests' work.

    Durations of partial calls scale by the observed overlap fraction. An idle call
    retains its native wall in both counterfactuals. No measured arrival is used
    when the simulator later predicts a different cohort.
    """
    check(capture["closed"] and not capture.get("capture_errors"), "Incomplete Core capture")
    steps: list[tuple] = []
    durations = defaultdict(list)
    for row in capture["iterations"]:
        if row["capture_phase"] == "warmup":
            check(not steps, "Warmup after measured steps")
            continue
        start, end = row["started_ns"], row["finished_ns"]
        check(
            row["capture_phase"] == "formal"
            and row["status"] == "COMPLETED"
            and end > start
            and (not steps or start >= steps[-1][1]),
            "Invalid Core chronology",
        )
        wall = (end - start) / 1e6
        check(math.isclose(wall, row["duration_ms"], abs_tol=1e-7), "Native duration mismatch")
        key = geometry(row)
        durations[key].append(wall)
        steps.append((start, end, key, wall))
    check(bool(steps), "Missing measured steps")
    ends = [s[1] for s in steps]
    typical = {k: median(v) for k, v in durations.items()}
    nominal = {k: estimate(k) for k in typical if k[0] != "idle"}
    windows = []
    for req in requests:
        times = req["token_ready_ns"]
        count = len(times) - 1
        check(
            count == 15 and all(a <= b for a, b in zip(times, times[1:])), "Expected all 15 ordered intervals"
        )
        start, end = times[0], times[-1]
        inside = denoised = replay = 0.0
        for i, clipped, fraction in _edges(steps, ends, start, end):
            _, _, key, wall = steps[i]
            inside += clipped
            denoised += fraction * (typical[key] if key[0] != "idle" else wall)
            replay += fraction * nominal.get(key, wall)
        actual = (end - start) / 1e6 / count
        check(
            math.isclose(actual, req["tpot_ms"], abs_tol=1e-7) and actual + 1e-6 >= inside / count,
            "Window disagrees with server tokens or Core accounting",
        )
        windows.append((req["stream_index"], actual, inside / count, denoised / count, replay / count))
    return windows


def chronology_control(requests, capture):
    """Retrospective permutation control on already audited Core and token clocks."""
    from mlsys_sim.reporting import _percentiles

    steps = [
        (r["started_ns"], r["finished_ns"], geometry(r), r["duration_ms"])
        for r in capture["iterations"]
        if r["capture_phase"] == "formal"
    ]
    ends = [row[1] for row in steps]
    groups = defaultdict(list)
    for i, row in enumerate(steps):
        groups[row[2]].append(i)
    spans = []
    for req in requests:
        times = req["token_ready_ns"]
        edges = list(_edges(steps, ends, times[0], times[-1]))
        outside = (times[-1] - times[0]) / 1e6 - sum(steps[i][3] * fraction for i, _, fraction in edges)
        spans.append((outside, edges, len(times) - 1))

    def quantile(costs):
        values = [
            (gap + sum(costs[i] * fraction for i, _, fraction in edges)) / count
            for gap, edges, count in spans
        ]
        return _percentiles(values)["p99"]

    wall = [row[3] for row in steps]
    shuffled = []
    for seed in range(11242026, 11242026 + 64):
        rng = random.Random(seed)
        costs = wall.copy()
        for ids in groups.values():
            values = [wall[i] for i in ids]
            rng.shuffle(values)
            for i, v in zip(ids, values):
                costs[i] = v
        shuffled.append(quantile(costs))
    return {"seed_start": 11242026, "original_ms": quantile(wall), "permutation_p99s_ms": shuffled}


def main():
    import argparse
    from functools import lru_cache
    import hashlib
    import json
    from pathlib import Path
    import runpy
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlsys_sim import RuntimeCostModel
    from mlsys_sim.config import load_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker-dir", "experiment", "reference-receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--chronology-control", action="store_true", help="Include the fixed 64 within-shape permutations"
    )
    args = parser.parse_args()
    path = args.worker_dir
    evaluator = (
        Path(__file__).resolve().parents[1] / "examples/qwen4b_online_20261117/audit/evaluate_forward.py"
    )
    check(
        hashlib.sha256(evaluator.read_bytes()).hexdigest()
        == "34cde56cbb9388796ee1fe65e519015f419a96171e8295a587560fa50af88b69",
        "Expected the frozen native evaluator",
    )
    ev = runpy.run_path(str(evaluator))
    read = ev["read_json"]
    summary, feedback = read(path / "summary.json"), read(path / "api-feedback.json")
    check(summary["status"] == "COMPLETED" and summary["shutdown"] == "completed", "Incomplete worker")
    schedule = read(path / "captures" / Path(summary["scheduler_capture_file"]).name)
    capture, dataset = read(path / "runtime-cost-capture.json"), read(path / "runtime-cost-dataset.json")
    model = RuntimeCostModel(load_experiment(args.experiment))
    ev["validate_runtime_capture"](
        capture,
        dataset,
        summary,
        read(args.reference_receipt),
        model.profile,
        schedule,
        feedback["formal_start_ns"],
        feedback,
        read(path / "runner-binding.json"),
    )
    raw = ev["recompute_server_run"](
        schedule, feedback, summary["spec"], summary["concurrency"], summary["replication"]
    )
    ev["crosscheck_producer"](raw, read(path / "observed-run.json"), read(path / "server-trace.json"))

    @lru_cache(maxsize=100_000)
    def estimate(key):
        phase, q, c, roles = key
        return model.estimate_step(phase, q, c, request_phases=roles).duration_ms

    rows = replay_windows(raw["requests"], capture, estimate)
    doc = {
        "session_id": summary["session_id"],
        "fields": ["stream_index", "observed_ms", "inside_core_ms", "matched_median_ms", "replay_ms"],
        "windows": rows,
        "base_file_sha256": model.provenance["sha256"],
        "source_capture_sha256": hashlib.sha256(
            (path / "runtime-cost-capture.json").read_bytes()
        ).hexdigest(),
    }
    if args.chronology_control:
        doc["chronology_control"] = chronology_control(raw["requests"], capture)
    with args.out.open("x") as f:
        json.dump(doc, f, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
