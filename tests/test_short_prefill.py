"""The short Prefill overlay must guard coverage and leave the frozen base replayable."""

import json
import tempfile
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path

from mlsys_sim.runtime_cost import RuntimeCostModel, freeze_runtime_cost_policy
from mlsys_sim.schema import SimulationError
from mlsys_sim.short_prefill import fit_short_prefill

try:
    from .test_runtime_cost import fixture, fit, digest
except ImportError:
    from test_runtime_cost import fixture, fit, digest


def base_and_sessions():
    experiment, data, policy = fixture(
        selected=(("prefill", 1), ("prefill", 2), ("prefill", 4), ("decode", 1))
    )
    for cell in policy["cells"]:
        cell.update(
            max_query_tokens=128 if cell["phase"] == "prefill" else 1,
            max_total_query_tokens=512,
            max_sequence_tokens=256,
            max_total_tokens=1024,
        )
    base = fit(data, freeze_runtime_cost_policy(policy["cells"]))
    sessions = []
    for batch in (1, 2, 4):
        for rep in range(2):
            session = f"fresh-{batch}-{rep}"
            shapes = [
                {
                    "query_tokens": list(q),
                    "durations_ms": [sum(q) / 10 + rep / 10 + (0.4 if q[0] == 64 else 0)] * 4,
                }
                for q in product((64, 128), repeat=batch)
            ]
            sessions.append(
                dict(session_id=session, batch_size=batch, capture_sha256=digest(session), shapes=shapes)
            )
    return (
        experiment,
        base,
        {
            "role": "calibration",
            "domain_sha256": base["domain"]["domain_sha256"],
            "plan_sha256": "a" * 64,
            "sessions": sessions,
        },
    )


class ShortPrefillTests(unittest.TestCase):
    def test_dispatch_and_legacy(self):
        exp, base, evidence = base_and_sessions()
        candidate = fit_short_prefill(base, evidence, calibration_id="shape-v2", allow_test_fixture=True)
        self.assertNotIn("short_prefill", base)
        with tempfile.TemporaryDirectory() as tmp:

            def model(payload):
                path = Path(tmp) / "model.json"
                path.write_text(json.dumps(payload))
                return RuntimeCostModel(
                    replace(exp, calibration=replace(exp.calibration, profile_path=str(path)))
                )

            before, after = model(base), model(candidate)
            value = after.estimate_step("prefill", (128, 64), (0, 0))
            self.assertAlmostEqual(value.duration_ms, 19.25)
            self.assertAlmostEqual(after.estimate_step("prefill", [64, 128], [0, 0]).duration_ms, 19.65)
            self.assertEqual(value.breakdown["runtime_cost"]["method"], "ordered_shape_median")
            for phase, q, c in [("prefill", [64, 128], [1, 0]), ("decode", [1], [8]), ("prefill", [32], [0])]:
                self.assertEqual(
                    after.estimate_step(phase, q, c).duration_ms,
                    before.estimate_step(phase, q, c).duration_ms,
                )

    def test_gate_on_missing_duplicate_shifted_and_holdout_sources(self):
        for case in ("missing", "duplicate", "shifted", "holdout", "wrong-domain"):
            with self.subTest(case=case):
                _, base, e = base_and_sessions()
                if case == "missing":
                    e["sessions"][0]["shapes"].pop()
                elif case == "duplicate":
                    e["sessions"][1] = e["sessions"][0]
                elif case == "shifted":
                    e["sessions"][0]["shapes"][0]["durations_ms"] = [100] * 4
                elif case == "holdout":
                    e["role"] = "holdout"
                else:
                    e["domain_sha256"] = "b" * 64
                with self.assertRaises(SimulationError):
                    fit_short_prefill(base, e, calibration_id="shape-v2", allow_test_fixture=True)
