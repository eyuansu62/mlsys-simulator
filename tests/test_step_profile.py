"""Stationarity, raw proof and policy-scoped reported-wall lookup checks."""

from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim.calibration import calibrate_dataset
from mlsys_sim.config import validate_experiment
from mlsys_sim.profiles import (ProfileModel, estimate_profile, profile_identity,
                                validate_identity, validate_profile)
from mlsys_sim.schema import CalibrationSpec, Experiment, MemoryConfig, SimulationError
from mlsys_sim.step_profile import (REPORTED_STEP_BOUNDARY, STEP_PROFILE_REJECTION_KIND,
    STEP_PROFILE_DURATION_STATISTIC, calibrate_step_profile, check_step_stationarity,
    freeze_stationarity_policy, step_wall_domain)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fixture(*, decode=False, sessions=("session-a", "session-b")):
    experiment = Experiment(calibration=CalibrationSpec(backend="profile", profile_path="placeholder.json",
        timing_boundary=REPORTED_STEP_BOUNDARY, runtime_name="fixture-runtime", runtime_version="fixture-v1",
        runtime_config_sha256="a" * 64, allow_test_fixture=True))
    experiment = replace(experiment, strategy=replace(experiment.strategy, preemption="none"))
    shapes = [("decode", [1], [context], 6 + context) for context in (4, 8)] if decode else [
        ("prefill", [query], [0], 1 + 2 * query) for query in (4, 8)]
    records, series = [], []
    for session in sessions:
        for phase, queries, contexts, duration in shapes:
            group = f"{phase}-{queries}-{contexts}"
            count = 128 if phase == "decode" else 100
            names = []
            for index in range(count):
                # IDs intentionally run opposite to source chronology.
                name = f"{session}:{group}:{count-index:03d}"
                names.append(name)
                records.append({"record_id": name, "group_id": group, "phase": phase,
                    "stage_index": 0, "query_tokens": queries[:], "context_lengths": contexts[:],
                    "duration_ms": duration, "workspace_peak_bytes": index})
            series.append({"session_id": session, "group_id": group, "record_ids": names})
    dataset = {"schema_version": 1, "dataset_id": "reported-wall-fixture", "identity": profile_identity(experiment),
        "source": {"kind": "test_fixture", "reference": "Synthetic whole-step chronology"},
        "records": records, "notes": {"wall_domain": step_wall_domain(experiment), "ordered_series": series}}
    return experiment, dataset


def fit(data):
    return calibrate_step_profile(data, freeze_stationarity_policy(), profile_version="wall-v1", allow_test_fixture=True)


def rehash_proof(profile):
    proof = profile["notes"]["reported_step_profile"]
    proof["proof_sha256"] = digest({key: value for key, value in proof.items() if key != "proof_sha256"})


def decode_sweep():
    experiment, data = fixture(decode=True)
    data["records"], data["notes"]["ordered_series"] = [], []
    for session in ("session-a", "session-b"):
        names = []
        for index in range(128):
            name = f"{session}:{128-index:03d}"
            names.append(name)
            data["records"].append({"record_id": name, "group_id": "decode-sweep", "phase": "decode",
                "stage_index": 0, "query_tokens": [1], "context_lengths": [4 + index % 16],
                "duration_ms": 10, "workspace_peak_bytes": None})
        data["notes"]["ordered_series"].append({"session_id": session, "group_id": "decode-sweep", "record_ids": names})
    return experiment, data


class StepProfileTests(unittest.TestCase):
    def test_prefill_mixed_shapes_cannot_hide_a_minority_workload(self):
        _, data = fixture()
        names = {name for item in data["notes"]["ordered_series"] if item["group_id"].startswith("prefill-[4]")
                 for name in item["record_ids"][::5]}
        for row in data["records"]:
            if row["record_id"] in names:
                row["query_tokens"] = [8]
                row["duration_ms"] = 40 if row["record_id"].startswith("session-b:") else 10
            else:
                row["duration_ms"] = 20
        with self.assertRaisesRegex(SimulationError, "one exact"):
            fit(data)

    def test_decode_exact_context_cross_session_and_chronology_gates(self):
        _, data = decode_sweep()
        profile = fit(data)
        self.assertEqual(len(profile["samples"]), 16)
        for change in ("session", "halves"):
            _, data = decode_sweep()
            chosen = [row for row in data["records"]
                      if row["record_id"].startswith("session-b:") and row["context_lengths"] == [4]]
            for row in chosen if change == "session" else chosen[:4]:
                row["duration_ms"] = 40
            rejected = fit(data)
            self.assertEqual(rejected["kind"], STEP_PROFILE_REJECTION_KIND)
            report = rejected["stationarity"]
            self.assertTrue(all(row["passed"] for row in report["series"] + report["cross_session_groups"]))
            self.assertTrue(any(not row["passed"] for row in report["lookup_shapes"]))
            failing = next(row for row in report["lookup_shapes"] if not row["passed"])
            if change == "halves":
                session = next(row for row in failing["sessions"] if row["session_id"] == "session-b")
                self.assertEqual(session["source_ordered_half_medians_ms"], [40, 10])

    def test_decode_lookup_context_requires_repeats_and_fixed_unit_query(self):
        for change in ("repeat_count", "query", "batch"):
            _, data = decode_sweep()
            row = data["records"][0]
            if change == "repeat_count":
                row["context_lengths"] = [5]
            elif change == "query":
                row["query_tokens"] = [2]
            else:
                row["query_tokens"], row["context_lengths"] = [1, 1], [4, 4]
            with self.subTest(change=change), self.assertRaises(SimulationError):
                fit(data)

    def test_reported_duration_target_resists_outliers_without_dropping_them(self):
        _, data = fixture()
        data["records"][0]["duration_ms"] = 1_000_000
        data["records"][0]["workspace_peak_bytes"] = 9999
        profile = fit(data)
        sample = next(row for row in profile["samples"] if row["query_tokens"] == [4])
        self.assertEqual(sample["duration_ms"], 9)
        self.assertEqual(sample["sample_count"], 200)
        self.assertEqual(sample["workspace_peak_bytes"], 9999)
        proof = profile["notes"]["reported_step_profile"]
        self.assertEqual(proof["raw_dataset"]["records"][0]["duration_ms"], 1_000_000)
        self.assertEqual(proof["duration_statistic"], STEP_PROFILE_DURATION_STATISTIC)
        self.assertEqual(profile["notes"]["duration_statistic"], STEP_PROFILE_DURATION_STATISTIC)
        sample["duration_ms"] = (1_000_000 + 199 * 9) / 200
        with self.assertRaises(SimulationError):
            validate_profile(profile, allow_test_fixture=True)

    def test_each_lookup_shape_gives_sessions_equal_weight(self):
        _, data = fixture()
        data["records"], data["notes"]["ordered_series"] = [], []
        for session, group, duration in (("a", "g1", 10), ("b", "g1", 11), ("a", "g2", 10), ("c", "g2", 11)):
            names = []
            for index in range(100):
                name = f"{session}:{group}:{index}"
                names.append(name)
                data["records"].append({"record_id": name, "group_id": group, "phase": "prefill", "stage_index": 0,
                    "query_tokens": [4], "context_lengths": [0], "duration_ms": duration, "workspace_peak_bytes": None})
            data["notes"]["ordered_series"].append({"session_id": session, "group_id": group, "record_ids": names})
        profile = fit(data)
        # Session a contributes 200 rows; b and c each contribute 100. Every session has one vote.
        self.assertEqual(profile["samples"][0]["duration_ms"], 11)
        self.assertEqual(profile["samples"][0]["sample_count"], 400)
        profile["samples"][0]["duration_ms"] = 10.5
        with self.assertRaises(SimulationError):
            validate_profile(profile, allow_test_fixture=True)

    def test_whole_step_lookup_and_piecewise_grid_reuse(self):
        _, data = fixture()
        before = deepcopy(data)
        profile = fit(data)
        self.assertEqual(profile["method"], "lookup")
        self.assertEqual(profile["identity"]["runtime"]["timing_boundary"], REPORTED_STEP_BOUNDARY)
        middle = estimate_profile(profile, "prefill", [6], [0], allow_test_fixture=True)
        self.assertEqual(middle.duration_ms, 13)
        self.assertEqual(middle.breakdown["profile"]["lookup"], "multilinear")
        self.assertEqual(middle.breakdown["profile"]["workspace_peak_bytes"], 99)
        self.assertEqual(profile["samples"][0]["sample_count"], 200)
        self.assertEqual(profile["notes"]["reported_step_profile"]["stationarity"]["verdict"], "PASS")
        self.assertEqual(check_step_stationarity(data, freeze_stationarity_policy(), allow_test_fixture=True),
                         profile["notes"]["reported_step_profile"]["stationarity"])
        self.assertEqual(data, before)
        _, data = fixture(decode=True)
        middle = estimate_profile(fit(data), "decode", [1], [6], allow_test_fixture=True)
        self.assertEqual(middle.duration_ms, 12)

    def test_prefill_contiguous_blocks_detect_drift_before_id_sorting(self):
        _, data = fixture()
        chosen = data["notes"]["ordered_series"][0]["record_ids"]
        by_id = {row["record_id"]: row for row in data["records"]}
        for index, name in enumerate(chosen):
            by_id[name]["duration_ms"] = 10 if index < 40 else 20
        rejected = fit(data)
        self.assertEqual(rejected["kind"], STEP_PROFILE_REJECTION_KIND)
        self.assertNotIn("samples", rejected)
        self.assertEqual(rejected["stationarity"]["series"][0]["block_medians_ms"], [10, 10, 20, 20, 20])
        self.assertEqual(len(rejected["raw_dataset"]["records"]), len(data["records"]))
        self.assertEqual(rejected["stationarity"]["series"][0]["record_ids"], chosen)
        wrong_order = deepcopy(data)
        wrong_order["notes"]["ordered_series"][0]["record_ids"] = sorted(chosen)
        with self.assertRaisesRegex(SimulationError, "ID sorting"):
            fit(wrong_order)

    def test_decode_uses_four_complete_32_sample_blocks(self):
        _, data = fixture(decode=True)
        names = set(data["notes"]["ordered_series"][0]["record_ids"][:32])
        for row in data["records"]:
            if row["record_id"] in names:
                row["duration_ms"] *= 2
        rejected = fit(data)
        first = rejected["stationarity"]["series"][0]
        self.assertEqual(len(first["block_medians_ms"]), 4)
        self.assertEqual(first["sample_count"], 128)
        self.assertFalse(first["passed"])

    def test_cross_session_instability_or_one_session_rejects_point_profile(self):
        for factor, sessions in ((1.2, ("session-a", "session-b")), (1, ("session-a",))):
            _, data = fixture(sessions=sessions)
            for row in data["records"]:
                if row["record_id"].startswith("session-b:"):
                    row["duration_ms"] *= factor
            rejected = fit(data)
            self.assertEqual(rejected["kind"], STEP_PROFILE_REJECTION_KIND)
            self.assertNotIn("samples", rejected)
            self.assertTrue(any(not row["passed"] for row in rejected["stationarity"]["cross_session_groups"]))

    def test_policy_thresholds_counts_and_digest_cannot_change_implicitly(self):
        _, data = fixture()
        for field, value in (("max_within_series_median_ratio", 1.5), ("minimum_training_sessions_per_group", 1)):
            policy = freeze_stationarity_policy()
            policy[field] = value
            with self.subTest(field=field), self.assertRaises(SimulationError):
                calibrate_step_profile(data, policy, profile_version="bad", allow_test_fixture=True)
        policy = freeze_stationarity_policy()
        policy["phases"]["prefill"]["block_size"] = 25
        policy["policy_sha256"] = digest({key: value for key, value in policy.items() if key != "policy_sha256"})
        with self.assertRaises(SimulationError):
            calibrate_step_profile(data, policy, profile_version="bad", allow_test_fixture=True)

    def test_policy_rejects_bool_and_float_integer_aliases_even_when_rehashed(self):
        _, data = fixture()
        for change in ("schema", "block_size", "sample_count", "minimum_sessions"):
            for rehashed in (False, True):
                policy = freeze_stationarity_policy()
                if change == "schema":
                    policy["schema_version"] = True
                elif change == "minimum_sessions":
                    policy["minimum_training_sessions_per_group"] = 2.0
                else:
                    policy["phases"]["prefill"][change] = float(policy["phases"]["prefill"][change])
                if rehashed:
                    policy["policy_sha256"] = digest({key: value for key, value in policy.items() if key != "policy_sha256"})
                with self.subTest(change=change, rehashed=rehashed), self.assertRaises(SimulationError):
                    check_step_stationarity(data, policy, allow_test_fixture=True)

    def test_extreme_finite_durations_keep_finite_diagnostics(self):
        _, data = fixture()
        for row in data["records"]:
            row["duration_ms"] = 1e308
        report = check_step_stationarity(data, freeze_stationarity_policy(), allow_test_fixture=True)
        self.assertEqual(report["verdict"], "PASS")
        self.assertTrue(all(row["median_ms"] == 1e308 for row in report["series"]))
        chosen = set(data["notes"]["ordered_series"][0]["record_ids"][:20])
        for row in data["records"]:
            if row["record_id"] in chosen:
                row["duration_ms"] = 1e-300
        rejected = fit(data)
        self.assertEqual(rejected["kind"], STEP_PROFILE_REJECTION_KIND)
        self.assertIsNone(rejected["stationarity"]["series"][0]["within_series_ratio"])
        json.dumps(rejected, allow_nan=False)

    def test_malformed_wall_policy_raises_simulation_error(self):
        _, original = fixture()
        for field in ("scheduler", "preemption"):
            data = deepcopy(original)
            data["notes"]["wall_domain"]["core_policy"][field] = []
            with self.subTest(field=field), self.assertRaises(SimulationError):
                fit(data)

    def test_recursive_new_boundary_payload_raises_simulation_error(self):
        _, data = fixture()
        data["notes"]["recursive"] = data
        with self.assertRaises(SimulationError):
            fit(data)
        _, data = fixture()
        profile = fit(data)
        profile["notes"]["recursive"] = profile
        with self.assertRaises(SimulationError):
            estimate_profile(profile, "prefill", [6], [0], allow_test_fixture=True)

    def test_missing_tail_duplicate_extra_and_unknown_ids_are_not_trimmed(self):
        _, original = fixture()
        for change in ("missing", "duplicate", "unknown", "uncovered"):
            data = deepcopy(original)
            names = data["notes"]["ordered_series"][0]["record_ids"]
            if change == "missing":
                removed = names.pop()
                data["records"] = [row for row in data["records"] if row["record_id"] != removed]
            elif change == "duplicate":
                names[-1] = names[0]
            elif change == "unknown":
                names[-1] = "not-a-record"
            else:
                data["notes"]["ordered_series"].pop()
            with self.subTest(change=change), self.assertRaises(SimulationError):
                fit(data)

    def test_group_schedule_cannot_mix_workloads_across_sessions(self):
        _, data = fixture()
        chosen = set(data["notes"]["ordered_series"][2]["record_ids"])
        for row in data["records"]:
            if row["record_id"] in chosen:
                row["query_tokens"] = [5]
        with self.assertRaisesRegex(SimulationError, "shape schedule"):
            fit(data)

    def test_raw_and_sample_duration_changes_cannot_bypass_consistency_checks(self):
        _, data = fixture()
        for change in ("sample", "raw", "gate"):
            profile = fit(data)
            if change == "sample":
                profile["samples"][0]["duration_ms"] += 1
            elif change == "raw":
                profile["notes"]["reported_step_profile"]["raw_dataset"]["records"][0]["duration_ms"] += 1
            else:
                profile["notes"]["reported_step_profile"]["stationarity"]["series"][0]["within_series_ratio"] = 0
                rehash_proof(profile)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                validate_profile(profile, allow_test_fixture=True)

    def test_wall_domain_is_more_than_an_opaque_runtime_digest(self):
        experiment, data = fixture()
        profile = fit(data)
        with tempfile.TemporaryDirectory(prefix="step-profile-") as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile))
            configured = replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path)))
            for field, value in (("scheduler", "fifo"), ("max_batch_size", 4), ("max_batch_tokens", 2048),
                ("max_active_requests", 2), ("chunked_prefill", True), ("prefill_chunk_tokens", 128),
                ("mixed_batching", True), ("prefix_cache", True), ("preemption", "recompute")):
                changed = replace(configured, strategy=replace(configured.strategy, **{field: value}))
                self.assertEqual(profile_identity(changed), profile["identity"])
                with self.subTest(field=field), self.assertRaises(SimulationError) as caught:
                    ProfileModel(changed)
                self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_complete_memory_policy_is_bound_at_load_and_each_query(self):
        experiment, data = fixture()
        wall = data["notes"]["wall_domain"]
        self.assertEqual(set(wall["memory_policy"]), {field.name for field in fields(MemoryConfig)})
        self.assertEqual(wall["memory_policy"], asdict(experiment.strategy.memory))
        profile = fit(data)
        with tempfile.TemporaryDirectory(prefix="step-profile-") as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile))
            configured = replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path)))
            model = ProfileModel(configured)
            changed = replace(configured, strategy=replace(configured.strategy,
                memory=replace(configured.strategy.memory, kv_pool_bytes=2 * configured.strategy.memory.kv_pool_bytes)))
            self.assertEqual(profile_identity(changed), profile["identity"])
            with self.assertRaises(SimulationError) as caught:
                ProfileModel(changed)
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")
            model.experiment = changed
            for call in (lambda: model.estimate_step("prefill", [6], [0]),
                         lambda: model.workspace_bytes([6], "prefill", context_lengths=[0])):
                with self.assertRaises(SimulationError) as caught:
                    call()
                self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_cached_model_guards_each_estimate_and_workspace_query(self):
        experiment, data = fixture()
        profile = fit(data)
        with tempfile.TemporaryDirectory(prefix="step-profile-") as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile))
            configured = replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path)))
            model = ProfileModel(configured)
            model.experiment = replace(configured, strategy=replace(configured.strategy, max_batch_tokens=2048))
            for call in (lambda: model.estimate_step("prefill", [6], [0]),
                         lambda: model.workspace_bytes([6], "prefill", context_lengths=[0])):
                with self.assertRaises(SimulationError) as caught:
                    call()
                self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_snapshot_verifies_raw_proof_once_without_public_validation_bypass(self):
        experiment, data = fixture()
        profile = fit(data)
        with tempfile.TemporaryDirectory(prefix="step-profile-") as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile))
            configured = replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path)))
            with patch("mlsys_sim.profiles.validate_profile", wraps=validate_profile) as validator:
                model = ProfileModel(configured)
                self.assertEqual(validator.call_count, 1)
                path.write_text("changed after snapshot")
                for _ in range(3):
                    self.assertEqual(model.estimate_step("prefill", [6], [0]).duration_ms, 13)
                    self.assertEqual(model.workspace_bytes([6], "prefill", context_lengths=[0]), 99)
                self.assertEqual(validator.call_count, 1)
                leaked = model.profile
                leaked["samples"][0]["duration_ms"] = 1000
                self.assertEqual(model.estimate_step("prefill", [6], [0]).duration_ms, 13)
            bad = deepcopy(profile)
            bad["samples"][0]["duration_ms"] = 1000
            with self.assertRaises(SimulationError):
                estimate_profile(bad, "prefill", [6], [0], allow_test_fixture=True)

    def test_snapshot_fixture_opt_in_remains_required_for_workspace_early_return(self):
        experiment, data = fixture()
        with tempfile.TemporaryDirectory(prefix="step-profile-") as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(fit(data)))
            configured = replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path)))
            model = ProfileModel(configured)
            model.experiment = replace(configured, calibration=replace(configured.calibration, allow_test_fixture=False))
            for call in (lambda: model.estimate_step("prefill", [6], [0]),
                         lambda: model.workspace_bytes([6], "prefill", context_lengths=[0]),
                         lambda: model.workspace_bytes([1], "decode")):
                with self.assertRaises(SimulationError) as caught:
                    call()
                self.assertEqual(caught.exception.code, "TEST_FIXTURE_NOT_ALLOWED")

    def test_new_boundary_requires_profile_backend_and_wrapper(self):
        experiment, data = fixture()
        self.assertEqual(validate_identity(profile_identity(experiment)), profile_identity(experiment))
        for backend in ("analytical", "synthetic"):
            changed = replace(experiment, calibration=replace(experiment.calibration,
                backend=backend, profile_path=None, allow_test_fixture=False))
            with self.subTest(backend=backend), self.assertRaises(SimulationError):
                validate_experiment(changed)
        with self.assertRaisesRegex(SimulationError, "calibrate_step_profile"):
            calibrate_dataset(data, profile_version="not-gated", allow_test_fixture=True)
        profile = fit(data)
        profile["notes"].pop("reported_step_profile")
        with self.assertRaises(SimulationError):
            validate_profile(profile, allow_test_fixture=True)

    def test_reported_runtime_digest_is_mandatory_without_changing_legacy(self):
        experiment, _ = fixture()
        for runtime_hash in ("", "a" * 63, "g" * 64):
            changed = replace(experiment, calibration=replace(experiment.calibration, runtime_config_sha256=runtime_hash))
            with self.subTest(runtime_hash=runtime_hash), self.assertRaises(SimulationError):
                step_wall_domain(changed)
            identity = profile_identity(changed)
            with self.assertRaises(SimulationError):
                validate_identity(identity)
        validate_experiment(Experiment())

    def test_reported_proof_cannot_be_downgraded_to_legacy_boundary(self):
        _, data = fixture()
        profile = fit(data)
        profile["identity"]["runtime"]["timing_boundary"] = "step_includes_output_and_sampling_v1"
        for call in (lambda: validate_profile(profile, allow_test_fixture=True),
                     lambda: estimate_profile(profile, "prefill", [6], [0], allow_test_fixture=True)):
            with self.assertRaisesRegex(SimulationError, "downgraded"):
                call()

    def test_fixture_permission_and_simulated_source_are_not_promoted(self):
        _, data = fixture()
        with self.assertRaises(SimulationError) as caught:
            calibrate_step_profile(data, freeze_stationarity_policy(), profile_version="fixture")
        self.assertEqual(caught.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        data["source"]["kind"] = "simulated"
        with self.assertRaises(SimulationError):
            fit(data)

    def test_legacy_boundary_profile_remains_unchanged(self):
        identity = profile_identity(Experiment())
        data = {"schema_version": 1, "dataset_id": "legacy", "identity": identity,
            "source": {"kind": "test_fixture", "reference": "Legacy fixture"},
            "records": [{"record_id": name, "group_id": name, "phase": "prefill", "stage_index": 0,
                         "query_tokens": [q], "context_lengths": [0], "duration_ms": duration}
                        for name, q, duration in (("a", 4, 9), ("b", 8, 17))]}
        before = deepcopy(data)
        profile = calibrate_dataset(data, profile_version="legacy-v1", allow_test_fixture=True)
        self.assertEqual(profile, validate_profile(profile, allow_test_fixture=True))
        self.assertNotIn("reported_step_profile", profile["notes"])
        self.assertEqual(estimate_profile(profile, "prefill", [6], [0], allow_test_fixture=True).duration_ms, 13)
        with self.assertRaises(SimulationError):
            calibrate_dataset(data, profile_version="legacy-v1", allow_test_fixture=True, _reported_step_proof={})
        self.assertEqual(data, before)

    def test_reported_wall_domain_rejects_unmeasured_parallel_layouts(self):
        experiment, _ = fixture()
        for field in ("tensor_parallel_size", "pipeline_parallel_size", "replicas"):
            changed = replace(experiment, strategy=replace(experiment.strategy, **{field: 2}))
            with self.subTest(field=field), self.assertRaises(SimulationError):
                step_wall_domain(changed)


if __name__ == "__main__":
    unittest.main()
