import unittest
from copy import deepcopy
from mlsys_sim.schema import SimulationError
from mlsys_sim.tpot_windows import TpotWindowProfile


def spec():
    return dict(
        schema_version=1,
        study_id="fixture",
        cohort="short",
        concurrency_grid=[2],
        requests_per_run=4,
        replications=2,
        seed=12,
        think_ms=0,
        input_lengths=[64, 128],
        output_tokens=16,
        timeout_ms=1000,
        max_run_ms=10000,
        prompt_token_id=100,
    )


def evidence():
    rows = []
    for i in range(2):
        rows.append(
            dict(
                session_id=f"worker-{i}",
                capture_sha256=str(i + 1) * 64,
                stream_sha256=str(i + 3) * 64,
                concurrency=2,
                spec=spec(),
                windows=[[0, 20, 21 + i], [1, 10, 11 + i], [2, 30, 35 + i], [3, 10, 13 + i]],
            )
        )
    return dict(
        kind="tpot_window_calibration_v1",
        role="calibration",
        source="test_fixture",
        plan_sha256="a" * 64,
        base_file_sha256="b" * 64,
        sessions=rows,
    )


class TPOTTests(unittest.TestCase):
    def test_rank_transfers_whole_windows_and_equal_weights_sessions(self):
        p = TpotWindowProfile(evidence(), allow_test_fixture=True)
        # Model TPOT in new cohort differs from the native replay by +2 at every rank.
        v = p.forecast([32, 12, 22, 12], spec(), 2, "b" * 64, "f" * 64)
        self.assertAlmostEqual(v["base_p99_ms"], 31.7)
        self.assertAlmostEqual(v["p99_ms"], 37.08)
        self.assertEqual(v["sessions"], 2)

    def test_reject_leakage_scope_mismatch_and_partial_windows(self):
        raw = evidence()
        with self.assertRaises(SimulationError):
            TpotWindowProfile(raw)
        for failure in ("duplicate", "incomplete", "nonfinite", "role", "stream", "session-count"):
            data = deepcopy(raw)
            if failure == "duplicate":
                data["sessions"][1]["session_id"] = "worker-0"
            if failure == "incomplete":
                data["sessions"][0]["windows"].pop()
            if failure == "nonfinite":
                data["sessions"][0]["windows"][0][-1] = float("nan")
            if failure == "role":
                data["role"] = "holdout"
            if failure == "stream":
                data["sessions"][1]["stream_sha256"] = data["sessions"][0]["stream_sha256"]
            if failure == "session-count":
                data["sessions"].pop()
            with self.subTest(failure=failure), self.assertRaises(SimulationError):
                TpotWindowProfile(data, allow_test_fixture=True)
        p = TpotWindowProfile(raw, allow_test_fixture=True)
        for s, c, base, stream in [
            (dict(spec(), output_tokens=32), 2, "b" * 64, "f" * 64),
            (spec(), 2, "c" * 64, "f" * 64),
            (spec(), 2, "b" * 64, "3" * 64),
            (spec(), 1, "b" * 64, "f" * 64),
        ]:
            with self.assertRaises(SimulationError):
                p.forecast([1] * 4, s, c, base, stream)


class TPOTIntegrationTests(unittest.TestCase):
    def test_opt_in_forecast_and_validation_preserve_token_ledger(self):
        import hashlib
        import json
        from dataclasses import replace
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from mlsys_sim import predict_closed_loop, validate_closed_loop
        from mlsys_sim.workload import canonical_sha256

        try:
            from .test_runtime_cost_integration import fixture, study
            from .test_load_validation import observations, validation_contract
        except ImportError:
            from test_runtime_cost_integration import fixture, study
            from test_load_validation import observations, validation_contract
        exp, model = fixture()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            data = json.dumps(model).encode()
            path.write_bytes(data)
            exp = replace(exp, calibration=replace(exp.calibration, profile_path=str(path)))
            s = study()
            ordinary = predict_closed_loop(exp, s)
            raw = evidence()
            raw["base_file_sha256"] = hashlib.sha256(data).hexdigest()
            for proc in raw["sessions"]:
                proc.update(spec=s, concurrency=1, windows=[[i, 1, 2] for i in range(3)])
            profile = TpotWindowProfile(raw, allow_test_fixture=True)
            predicted = predict_closed_loop(exp, s, tpot_window_profile=profile)
            r = predicted["runs"][0]
            self.assertEqual(r["requests"], ordinary["runs"][0]["requests"])
            self.assertEqual(r["result"], ordinary["runs"][0]["result"])
            self.assertAlmostEqual(r["tpot_forecast"]["p99_ms"], r["result"]["latency_ms"]["tpot"]["p99"] + 1)
            self.assertNotIn("tpot_forecast", ordinary["runs"][0])
            contract = validation_contract(
                min_completed_samples=3,
                max_relative_error={"window_tpot_p99_ms": 1},
                prediction_payload_sha256=canonical_sha256(predicted),
            )
            contract["constraints"]["min_completed_samples"] = 3
            audit = validate_closed_loop(predicted, observations(ordinary), contract)
            row = audit["runs"][0]["metrics"]["window_tpot_p99_ms"]
            self.assertEqual(row["predicted"], r["tpot_forecast"]["p99_ms"])
            self.assertEqual(row["measured"], r["result"]["latency_ms"]["tpot"]["p99"])
            contract.pop("prediction_payload_sha256")
            with self.assertRaises(SimulationError):
                validate_closed_loop(predicted, observations(ordinary), contract)
