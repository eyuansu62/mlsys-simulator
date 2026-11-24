"""Score every declared shape and E2E P99 against the premeasurement freeze."""

from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from scripts.short_prefill_study import raw_json, score, sha


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.plan = Path(self.scratch.name) / "plan.json"
        jobs = [
            {"concurrency": b, "replication": r, "label": f"b{b}r{r}"} for b in (1, 2, 4) for r in range(2)
        ]
        self.plan.write_bytes(
            raw_json(
                {
                    "shape_error_limit": 0.05,
                    "end_to_end_p99_error_limit": 0.25,
                    "min_shape_steps_per_session": 4,
                    "expected_checkpoint_sha256": "ck",
                    "expected_runtime_sha256": "rt",
                    "phases": {
                        "holdout": {
                            "jobs": jobs,
                            "streams": {"0": "s0", "1": "s1"},
                            "spec": {"requests_per_run": 256},
                        }
                    },
                }
            )
        )
        self.pred = {
            "model_sha256": "model",
            "domain_sha256": "domain",
            "plan_sha256": sha(self.plan.read_bytes()),
            "calibration_sessions": ["training"],
            "calibration_captures": ["old"],
            "calibration_streams": ["trainstream"],
            "runs": [],
            "shape_durations_ms": {},
        }
        rows = []
        for job in jobs:
            b = job["concurrency"]
            shapes = []
            for shape in product((64, 128), repeat=b):
                key = json.dumps(["prefill", 0, shape, [0] * b], separators=(",", ":"))
                self.pred["shape_durations_ms"][key] = 20.0
                shapes.append({"query_tokens": list(shape), "durations_ms": [20.0] * (256 // b // 2**b)})
            run = {**job, "stream_sha256": f"s{job['replication']}", "e2e_p99_ms": 250.0, "ttft_p99_ms": 21.0}
            self.pred["runs"].append(run)
            rows.append(
                {
                    **run,
                    "shapes": shapes,
                    "checkpoint_sha256": "ck",
                    "runtime_sha256": "rt",
                    "source_files": {"runtime-cost-capture.json": job["label"]},
                    "session_id": job["label"],
                    "domain_sha256": "domain",
                    "started_at_utc": "2026-11-24T02:00:00+00:00",
                }
            )
        self.obs = {
            "plan_sha256": self.pred["plan_sha256"],
            "role": "holdout",
            "rows": rows,
            "freeze": {
                "prediction_sha256": sha(raw_json(self.pred)),
                "model_sha256": "model",
                "plan_sha256": self.pred["plan_sha256"],
                "frozen_at_utc": "2026-11-24T01:00:00+00:00",
            },
        }

    def test_one_bad_shape_or_e2e_fails_the_whole_run(self):
        self.assertTrue(score(self.plan, self.obs, self.pred)["passed"])
        cost = self.obs["rows"][0]["shapes"][0]["durations_ms"]
        cost[:] = [21.5] * len(cost)
        self.assertFalse(score(self.plan, self.obs, self.pred)["passed"])
        cost[:] = [20.0] * len(cost)
        self.obs["rows"][-1]["e2e_p99_ms"] = 400.0
        self.assertFalse(score(self.plan, self.obs, self.pred)["passed"])

    def test_reused_session_stale_freeze_and_incomplete_grid(self):
        row = self.obs["rows"][0]
        for key, value in [
            ("session_id", "training"),
            ("started_at_utc", "2026-11-24T00:00:00+00:00"),
            ("shapes", row["shapes"][:-1]),
        ]:
            with self.subTest(key=key):
                original = row[key]
                row[key] = value
                with self.assertRaises(ValueError):
                    score(self.plan, self.obs, self.pred)
                row[key] = original
        self.pred["runs"][0]["e2e_p99_ms"] = 42.0
        with self.assertRaises(ValueError):
            score(self.plan, self.obs, self.pred)
