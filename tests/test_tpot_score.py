from copy import deepcopy
import hashlib
import json
import unittest

from scripts.score_tpot_forward import score


def encode(x):
    return json.dumps(x, sort_keys=True, allow_nan=False).encode()


def fixture():
    jobs = [
        dict(label=f"{c}-{b}-r0", cohort=c, concurrency=b, replication=0, spec_file=c)
        for c in ("short", "baseline", "long")
        for b in (1, 2, 4, 8, 16, 32)
    ]
    jobs.extend(
        dict(label=f"short-2-r{k}", cohort="short", concurrency=2, replication=k, spec_file="short")
        for k in (1, 2)
    )
    for j in jobs:
        j["stream_sha256"] = j["label"]
    specs = {k: {"requests_per_run": 2} for k in ("short", "baseline", "long")}
    plan = encode(
        {
            "specs": specs,
            "jobs": {"holdout": jobs},
            "acceptance": {
                "individual_relative_error_max": 0.25,
                "primary_absolute_mean_signed_relative_error_max": 0.02,
                "primary_under_and_over_max_each": 12,
            },
        }
    )
    plan_sha = hashlib.sha256(plan).hexdigest()
    preds = [
        dict(
            **j, window_forecast={"base_p99_ms": 9, "p99_ms": (9.9 if i % 2 else 10.1), "profile_sha256": "p"}
        )
        for i, j in enumerate(jobs)
    ]
    prediction = encode(
        dict(
            plan_sha256=plan_sha,
            profile_sha256="p",
            base_file_sha256="m",
            calibration_sessions=["training"],
            calibration_captures=["cal"],
            calibration_streams=["old"],
            runs=preds,
        )
    )
    receipt = encode(
        dict(
            files_sha256={"prediction.json": hashlib.sha256(prediction).hexdigest()},
            frozen_at_utc="2026-11-24T00:00:00+00:00",
        )
    )
    rows = [
        dict(
            **j,
            started="2026-11-24T01:00:00+00:00",
            spec=specs[j["cohort"]],
            session_id=j["label"],
            capture_sha256=j["label"],
            windows=[[0, 9, 10], [1, 9, 10]],
            tpot_p99_ms=10,
        )
        for j in jobs
    ]
    return (
        plan,
        prediction,
        dict(
            role="holdout",
            plan_sha256=plan_sha,
            base_file_sha256="m",
            sessions=rows,
            receipt_sha256=hashlib.sha256(receipt).hexdigest(),
        ),
        receipt,
    )


class ScoreTests(unittest.TestCase):
    def test_primary_and_hard_cell_evidence(self):
        p, f, e, r = fixture()
        result = score(p, f, e, r)
        self.assertTrue(result["passed"])
        self.assertEqual(result["primary_candidate"]["under"], 9)
        self.assertEqual(result["all_candidate"]["cells"], 20)
        # The secondary repetitions can veto the individual limit.
        e["sessions"][-1].update(windows=[[0, 9, 20], [1, 9, 20]], tpot_p99_ms=20)
        self.assertFalse(score(p, f, e, r)["gates"]["individual"])

    def test_reject_leakage_partial_or_tampered_study(self):
        p, f, e, r = fixture()
        for case in ["missing", "training", "stream", "early", "window", "receipt"]:
            q = deepcopy(e)
            s = q["sessions"][0]
            if case == "missing":
                q["sessions"].pop()
            elif case == "training":
                s["session_id"] = "training"
            elif case == "stream":
                s["stream_sha256"] = "old"
            elif case == "early":
                s["started"] = "2026-11-23T01:00:00+00:00"
            elif case == "window":
                s["windows"].pop()
            else:
                q["receipt_sha256"] = "wrong"
            with self.subTest(case=case), self.assertRaises(ValueError):
                score(p, f, q, r)
