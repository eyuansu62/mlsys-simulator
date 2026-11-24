"""Compare C2 Prefill/Decode episodes with matching within-session native steps."""

from __future__ import annotations
import argparse
import bisect
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return tuple((m["request_phase"], m["query_tokens"], m["context_length"]) for m in row["members"])


def phase(shape):
    phases = {p for p, _, _ in shape}
    return next(iter(phases)) if len(phases) == 1 else "other"


def ref(rows, current):
    """Leave the event under inspection out of its reference population."""
    return (
        median([v for order, v in rows if order != current]) if any(o != current for o, _ in rows) else None
    )


def stage_index(timeline, core):
    if timeline is None:
        return {}
    checked(
        timeline["kind"] == "c2_stage_timeline_v1" and timeline["session_id"] == core["session_id"],
        "Stage timeline does not match raw EngineCore acquisition",
    )
    bounds = {
        r["source_order"]: (r["started_ns"], r["finished_ns"])
        for r in core["iterations"]
        if r["capture_phase"] == "formal"
    }
    result: dict[int, dict] = defaultdict(dict)
    for order, name, *stamps in timeline["rows"]:
        checked(order in bounds and len(stamps) == 12, "Unbound stage record")
        a, b = stamps[:6], stamps[6:]
        checked(
            all(type(v) is int for v in stamps) and all(y >= x for x, y in zip(a, b)), "Invalid stage counter"
        )
        start, end = bounds[order]
        checked(
            (a[0] <= start <= end <= b[0]) if name == "core" else (start <= a[0] <= b[0] <= end),
            "Mismatched stage clock or source order",
        )
        slot = result[order].setdefault(
            name,
            {
                "wall_ms": 0.0,
                "thread_ms": 0.0,
                "voluntary": 0,
                "involuntary": 0,
                "minor_faults": 0,
                "major_faults": 0,
                "calls": 0,
            },
        )
        slot["calls"] += 1
        for i, metric in enumerate(
            ("wall_ms", "thread_ms", "voluntary", "involuntary", "minor_faults", "major_faults")
        ):
            slot[metric] += (b[i] - a[i]) / (1e6 if i < 2 else 1)
    checked(
        set(result) == set(bounds) and all("core" in stages for stages in result.values()),
        "Incomplete full-session timeline",
    )
    return result


def load_sensors(path):
    if path is None:
        return None
    with path.open() as f:
        header = json.loads(next(f))
        rows = [json.loads(line) for line in f]
    checked(all(len(r) == len(header["keys"]) + 2 and r[0] <= r[1] for r in rows), "Invalid sensor series")
    times = [r[0] for r in rows]
    checked(times == sorted(times), "Sensor chronology reversed")
    return header, rows, times


def analysis(core, timeline=None, sensors=None):
    checked(core["closed"] and not core.get("capture_errors"), "Incomplete native acquisition")
    steps = [r for r in core["iterations"] if r["capture_phase"] == "formal"]
    checked(
        steps
        and all(
            r["status"] == "COMPLETED"
            and r["started_ns"] <= r["finished_ns"]
            and (
                i == 0
                or steps[i - 1]["source_order"] + 1 == r["source_order"]
                and steps[i - 1]["finished_ns"] <= r["started_ns"]
            )
            and math.isclose((r["finished_ns"] - r["started_ns"]) / 1e6, r["duration_ms"], abs_tol=1e-6)
            for i, r in enumerate(steps)
        ),
        "Invalid native chronology",
    )
    stages = stage_index(timeline, core)
    groups = defaultdict(list)
    stage_groups = defaultdict(list)
    for row in steps:
        k = key(row)
        order = row["source_order"]
        groups[k].append((order, row["duration_ms"]))
        for name, values in stages.get(order, {}).items():
            stage_groups[k, name].append((order, values["wall_ms"]))

    def describe(i):
        row = steps[i]
        order = row["source_order"]
        k = key(row)
        center = ref(groups[k], order)
        times = {}
        for name, vals in stages.get(order, {}).items():
            times[name] = {**vals, "same_shape_median_ms": ref(stage_groups[k, name], order)}
        return dict(
            source_order=order,
            phase=phase(k),
            shape=k,
            wall_ms=row["duration_ms"],
            same_shape_median_ms=center,
            thread_ms=row["thread_cpu_ns"] / 1e6,
            since_start_s=(row["started_ns"] - steps[0]["started_ns"]) / 1e9,
            stages=times,
        )

    first_prefill = next((i for i, r in enumerate(steps) if phase(key(r)) == "prefill"), None)
    events = []
    for i, row in enumerate(steps):
        if phase(key(row)) != "prefill" or row["duration_ms"] < 38.0:
            continue
        nrange = range(max(0, i - 2), min(len(steps), i + 3))
        before = steps[nrange.start]["started_ns"]
        after = steps[nrange.stop - 1]["finished_ns"]
        around = [describe(j) for j in nrange]
        paired = any(
            abs(r["source_order"] - row["source_order"]) == 1
            and r["phase"] == "decode"
            and r["same_shape_median_ms"]
            and r["wall_ms"] >= 1.30 * r["same_shape_median_ms"]
            for r in around
        )
        extra = {}
        if sensors:
            header, records, clock = sensors
            lo = bisect.bisect_left(clock, before)
            hi = bisect.bisect_right(clock, after)
            extra["sensors"] = {
                "keys": header["keys"],
                "labels": header["hwmon_labels"],
                "window_rows": [
                    [(r[0] - row["started_ns"]) / 1e6, (r[1] - row["started_ns"]) / 1e6, *r[2:]]
                    for r in records[lo:hi]
                    if r[1] <= after
                ],
            }
        if timeline:
            extra["overlapping_gc"] = [v for v in timeline["gc"] if v[0] <= after and v[3] >= before]
        events.append(
            dict(
                source_order=row["source_order"],
                first_online=i == first_prefill,
                paired=bool(i != first_prefill and paired),
                neighbors=around,
                **extra,
            )
        )
    summary = {
        p: {"count": len(vals), "p50_ms": median(vals), "max_ms": max(vals)}
        for p in ("prefill", "decode")
        if (vals := [r["duration_ms"] for r in steps if phase(key(r)) == p])
    }
    sensor_summary = {}
    if sensors:
        header, records, clock = sensors
        select = [r for r in records if steps[0]["started_ns"] <= r[0] <= r[1] <= steps[-1]["finished_ns"]]
        for col, name in enumerate(header["keys"], 2):
            v = [r[col] for r in select if r[col] is not None]
            sensor_summary[name] = {
                "count": len(v),
                "median": median(v) if v else None,
                "min": min(v) if v else None,
                "max": max(v) if v else None,
            }
    return dict(
        session=core["session_id"],
        phases=summary,
        prefill_shapes=[
            {"shape": shape, "count": len(values), "median_ms": median(duration for _, duration in values)}
            for shape, values in sorted(groups.items()) if phase(shape) == "prefill"
        ],
        prefill_ge38=len(events),
        paired=sum(x["paired"] for x in events),
        sensor_summary=sensor_summary,
        missing_stages=timeline["missing"] if timeline else [],
        events=events,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker_dir", type=Path)
    p.add_argument("--sensor", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.worker_dir
    core = root / "runtime-cost-capture.json"
    timeline = root / "stage-timeline.json"
    sources = [core] + ([timeline] if timeline.is_file() else []) + ([args.sensor] if args.sensor else [])
    report = analysis(
        json.loads(core.read_bytes()),
        json.loads(timeline.read_bytes()) if timeline.is_file() else None,
        load_sensors(args.sensor),
    )
    report["source_sha256"] = {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    with args.out.open("x") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
    print(json.dumps({"prefill_ge38": report["prefill_ge38"], "paired": report["paired"]}))


if __name__ == "__main__":
    main()
