"""Synthetic EngineCore costs: mixed shapes, phase/batch scope and raw lineage."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import mlsys_sim.runtime_cost as runtime_cost
from mlsys_sim.performance import AnalyticalModel
from mlsys_sim.runtime_cost import (RUNTIME_COST_BOUNDARY, RUNTIME_COST_KIND, RUNTIME_COST_REJECTION_KIND,
    RuntimeCostModel, _nnls, fit_runtime_costs, freeze_runtime_cost_policy, load_runtime_cost,
    runtime_cost_domain, validate_runtime_cost_model)
from mlsys_sim.schema import Experiment, SimulationError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rehash(model):
    model["model_sha256"] = digest({key: value for key, value in model.items() if key != "model_sha256"})


def fixture(*, selected=(("prefill", 1), ("decode", 1), ("mixed", 2)), sessions=("session-a", "session-b")):
    """A pure reusable CPU fixture; it never reads repository experiment artifacts."""
    experiment = Experiment()
    experiment = replace(experiment, strategy=replace(experiment.strategy, preemption="none", mixed_batching=True),
        calibration=replace(experiment.calibration, backend="runtime", profile_path="future-runtime.json",
            timing_boundary=RUNTIME_COST_BOUNDARY, runtime_name="fixture-runtime", runtime_version="fixture-v1",
            runtime_config_sha256="a" * 64, allow_test_fixture=True))
    nominal_exp = replace(experiment, calibration=replace(experiment.calibration, backend="analytical",
        timing_boundary="step_includes_output_and_sampling_v1", profile_path=None, allow_test_fixture=False))
    nominal = AnalyticalModel(nominal_exp)
    bounds = [{"phase": phase, "batch_size": batch, "min_query_tokens": 1, "max_query_tokens": 32,
        "min_context_tokens": 0, "max_context_tokens": 128, "min_total_query_tokens": 1,
        "max_total_query_tokens": batch * 32, "min_total_context_tokens": 0,
        "max_total_context_tokens": batch * 128, "max_total_tokens": batch * 160,
        "max_sequence_tokens": 160} for phase, batch in selected]
    for row in bounds:
        if row["phase"] == "decode":
            row["max_query_tokens"] = 1
    policy = freeze_runtime_cost_policy(bounds)
    records = []
    for session in sessions:
        order = 0
        for phase, batch in selected:
            for point in (1, 2, 3):
                if phase == "prefill":
                    queries, contexts, roles = [2 * point] * batch, [point - 1] * batch, None
                elif phase == "decode":
                    queries, contexts, roles = [1] * batch, [8 * point] * batch, None
                else:
                    queries, contexts = [1] + [2 * point] * (batch - 1), [8 * point] + [point - 1] * (batch - 1)
                    roles = ["decode"] + ["prefill"] * (batch - 1)
                scale, intercept = (5, 7) if batch == 5 else (2, 3)
                duration = scale * nominal.estimate_step(phase, queries, contexts, request_phases=roles).duration_ms + intercept
                for _ in range(8):
                    row = {"record_id": f"{session}:{10000-order}", "session_id": session,
                        "group_id": f"{phase}-{batch}-{point}", "source_order": order, "phase": phase,
                        "stage_index": 0, "query_tokens": queries[:], "context_lengths": contexts[:],
                        "duration_ms": duration, "workspace_peak_bytes": 100 + point}
                    if roles is not None:
                        row["request_phases"] = roles[:]
                    records.append(row)
                    order += 1
    data = {"schema_version": 1, "kind": "runtime_cost_dataset_v1", "dataset_id": "runtime-fixture",
        "domain": runtime_cost_domain(experiment), "source": {"kind": "test_fixture", "reference": "Synthetic CPU EngineCore costs"},
        "records": records, "notes": {"real_hardware": False}}
    return experiment, data, policy


def fit(data, policy):
    return fit_runtime_costs(data, policy, calibration_id="runtime-fixture-v1", allow_test_fixture=True)


def naive_sum(values, start=0):
    result = start
    for value in values:
        result += value
    return result


def precise_sum(values, start=0):
    values = list(values)
    if type(start) is int and all(type(value) is int for value in values):
        return naive_sum(values, start)
    return math.fsum([start, *values])


def rich_fixture():
    experiment, data, policy = fixture(selected=(("prefill", 1),))
    nominal = AnalyticalModel(replace(experiment, calibration=replace(experiment.calibration,
        backend="analytical", profile_path=None, allow_test_fixture=False,
        timing_boundary="step_includes_output_and_sampling_v1")))
    rng, anchors = random.Random(0), []
    for point in range(47):
        queries, contexts = [1 + point % 32], [(point * 11) % 127]
        coordinate = nominal.estimate_step("prefill", queries, contexts).duration_ms
        anchors.append((queries, contexts, 2.3456789 * coordinate + .456789 + rng.uniform(-.003, .003)))
    data["records"] = []
    for session in ("session-a", "session-b"):
        order = 0
        for point, (queries, contexts, duration) in enumerate(anchors):
            for _ in range(8):
                data["records"].append({"record_id": f"{session}:{order}", "session_id": session,
                    "group_id": f"anchor-{point}", "source_order": order, "phase": "prefill", "stage_index": 0,
                    "query_tokens": queries, "context_lengths": contexts, "duration_ms": duration,
                    "workspace_peak_bytes": None})
                order += 1
    return experiment, data, policy


class RuntimeCostTests(unittest.TestCase):
    def test_naive_solver_sum_replays_frozen_coefficients_without_replacing_them(self):
        _, data, policy = rich_fixture()
        with patch("mlsys_sim.runtime_cost.sum", precise_sum, create=True):
            artifact = fit(data, policy)
        before = deepcopy(artifact)
        with patch("mlsys_sim.runtime_cost.sum", naive_sum, create=True):
            replayed_fit = fit(data, policy)
            validated = validate_runtime_cost_model(artifact, allow_test_fixture=True)
        self.assertNotEqual(artifact["parameters"], replayed_fit["parameters"])
        self.assertEqual(validated, before)
        self.assertEqual(artifact, before)
        self.assertEqual(validated["model_sha256"], before["model_sha256"])

    def test_naive_nominal_and_solver_sums_allow_near_zero_sse_replay(self):
        _, data, policy = fixture()
        with (patch("mlsys_sim.runtime_cost.sum", precise_sum, create=True),
              patch("mlsys_sim.performance.sum", precise_sum, create=True)):
            artifact = fit(data, policy)
        self.assertTrue(all(row["fit_diagnostics"]["normalized_sum_squared_error"] < 1e-24
                            for row in artifact["parameters"]))
        with (patch("mlsys_sim.runtime_cost.sum", naive_sum, create=True),
              patch("mlsys_sim.performance.sum", naive_sum, create=True)):
            replayed_fit = fit(data, policy)
            validated = validate_runtime_cost_model(artifact, allow_test_fixture=True)
        self.assertNotEqual(artifact["diagnostics"]["fitting_cells"], replayed_fit["diagnostics"]["fitting_cells"])
        self.assertEqual(validated, artifact)

    def test_near_zero_recomputed_constraint_label_can_change(self):
        _, data, policy = fixture(selected=(("prefill", 1),))
        for row in data["records"]:
            row["duration_ms"] -= 3
        artifact = fit(data, policy)
        original_solver = runtime_cost._nnls
        self.assertTrue(math.isclose(artifact["parameters"][0]["intercept_ms"], 0, abs_tol=1e-12))
        def boundary_roundoff(xs, ys):
            scale, intercept, diagnostics = original_solver(xs, ys)
            diagnostics["constraint_active"] = not diagnostics["constraint_active"]
            diagnostics["unconstrained_intercept_ms"] = (
                -math.ulp(1.0) if diagnostics["constraint_active"] else math.ulp(1.0))
            return scale, intercept, diagnostics
        with patch("mlsys_sim.runtime_cost._nnls", boundary_roundoff):
            self.assertEqual(validate_runtime_cost_model(artifact, allow_test_fixture=True), artifact)

    def test_nonboundary_and_material_constraint_label_tampering_still_fails(self):
        for constrained in (False, True):
            _, data, policy = fixture(selected=(("prefill", 1),))
            if constrained:
                for row in data["records"]:
                    row["duration_ms"] = 10 - row["duration_ms"]
            artifact = fit(data, policy)
            diagnostics = artifact["parameters"][0]["fit_diagnostics"]
            self.assertEqual(diagnostics["constraint_active"], constrained)
            diagnostics["constraint_active"] = not diagnostics["constraint_active"]
            rehash(artifact)
            with self.assertRaises(SimulationError):
                validate_runtime_cost_model(artifact, allow_test_fixture=True)

    def test_replay_tolerance_never_weakens_checksum_or_structural_guards(self):
        _, data, policy = fixture()
        original = fit(data, policy)
        changed = deepcopy(original)
        changed["raw_dataset"]["records"][0]["duration_ms"] = math.nextafter(
            changed["raw_dataset"]["records"][0]["duration_ms"], math.inf)
        with self.assertRaisesRegex(SimulationError, "checksum"):
            validate_runtime_cost_model(changed, allow_test_fixture=True)
        for change in ("coefficient", "nominal", "residual", "geometry", "domain"):
            artifact = deepcopy(original)
            if change == "coefficient":
                row = artifact["parameters"][0]
                row["scale"] += 1e-6
                row["fit_diagnostics"]["scale"] += 1e-6
                row["fit_diagnostics"]["unconstrained_scale"] += 1e-6
            elif change == "nominal":
                artifact["diagnostics"]["fitting_cells"][0]["nominal_ms"] += 1e-6
            elif change == "residual":
                artifact["diagnostics"]["training_residuals"][0]["signed_error_ms"] += 1e-6
            elif change == "geometry":
                artifact["diagnostics"]["fitting_cells"][0]["sample_count"] = 8.0
            else:
                artifact["domain"]["strategy"]["max_batch_tokens"] = float(artifact["domain"]["strategy"]["max_batch_tokens"])
            rehash(artifact)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                validate_runtime_cost_model(artifact, allow_test_fixture=True)

    def test_recovers_known_affine_coefficients_and_retains_all_raw_points(self):
        experiment, data, policy = fixture()
        before = deepcopy(data)
        model = fit(data, policy)
        self.assertEqual(model["kind"], RUNTIME_COST_KIND)
        self.assertEqual(model["domain"]["timing_boundary"], RUNTIME_COST_BOUNDARY)
        self.assertEqual(model["training_record_ids"], [row["record_id"] for row in data["records"]])
        self.assertEqual(len(model["diagnostics"]["training_residuals"]), len(data["records"]))
        for parameter in model["parameters"]:
            self.assertAlmostEqual(parameter["scale"], 2, places=7)
            self.assertAlmostEqual(parameter["intercept_ms"], 3, places=7)
        self.assertEqual(validate_runtime_cost_model(model, experiment=experiment, allow_test_fixture=True), model)
        self.assertEqual(data, before)

    def test_nonnegative_solver_boundary_has_hand_computed_answers(self):
        scale, intercept, _ = _nnls([1, 2, 3], [5, 7, 9])
        self.assertAlmostEqual(scale, 2)
        self.assertAlmostEqual(intercept, 3)
        scale, intercept, _ = _nnls([1, 2, 3], [3, 2, 1])
        self.assertEqual(scale, 0)
        self.assertAlmostEqual(intercept, 2)
        scale, intercept, _ = _nnls([1, 2, 3], [1, 3, 5])
        self.assertEqual(intercept, 0)
        self.assertAlmostEqual(scale, 22 / 14)

    def test_four_repeat_policy_is_explicit_and_all_repeats_have_equal_cell_weight(self):
        _, data, policy = fixture(selected=(("prefill", 1),))
        policy = freeze_runtime_cost_policy(policy["cells"], min_samples_per_shape_session=4)
        data["records"] = [row for row in data["records"] if row["source_order"] % 8 < 4]
        self.assertEqual(fit(data, policy)["kind"], RUNTIME_COST_KIND)
        _, data, policy = fixture(selected=(("prefill", 1),))
        for row in data["records"]:
            if row["session_id"] == "session-a":
                row["duration_ms"] *= 1.02
        original = fit(data, policy)
        first_session = [row for row in data["records"] if row["session_id"] == "session-a"]
        repeated = []
        for index, row in enumerate(first_session):
            extra = deepcopy(row)
            extra.update(record_id=row["record_id"] + ":extra", source_order=24 + index)
            repeated.append(extra)
        data["records"] = first_session + repeated + [row for row in data["records"] if row["session_id"] == "session-b"]
        duplicated = fit(data, policy)
        self.assertEqual(original["parameters"], duplicated["parameters"])
        self.assertAlmostEqual(duplicated["parameters"][0]["scale"], 2.02, places=7)
        self.assertAlmostEqual(duplicated["parameters"][0]["intercept_ms"], 3.03, places=7)

    def test_single_extreme_outlier_is_preserved_without_changing_median_target(self):
        _, data, policy = fixture(selected=(("prefill", 1),))
        expected = fit(data, policy)
        data["records"][0]["duration_ms"] = 1_000_000
        artifact = fit(data, policy)
        self.assertEqual(artifact["parameters"], expected["parameters"])
        self.assertEqual(artifact["raw_dataset"]["records"][0]["duration_ms"], 1_000_000)
        self.assertGreater(abs(artifact["diagnostics"]["training_residuals"][0]["signed_error_ms"]), 900_000)

    def test_batch_four_and_five_are_independent_without_batch_interpolation(self):
        experiment, data, policy = fixture(selected=(("prefill", 4), ("prefill", 5)))
        artifact = fit(data, policy)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(artifact))
            model = RuntimeCostModel(replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path))))
            four = model.estimate_step("prefill", [4] * 4, [1] * 4)
            five = model.estimate_step("prefill", [4] * 5, [1] * 5)
            self.assertAlmostEqual(four.breakdown["runtime_cost"]["scale"], 2)
            self.assertAlmostEqual(five.breakdown["runtime_cost"]["scale"], 5)
            with self.assertRaises(SimulationError) as caught:
                model.estimate_step("prefill", [4] * 3, [1] * 3)
            self.assertEqual(caught.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_ragged_mixed_uses_joint_nominal_and_explicit_roles(self):
        experiment, data, policy = fixture()
        artifact = fit(data, policy)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(artifact))
            model = RuntimeCostModel(replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path))))
            result = model.estimate_step("mixed", [1, 3], [11, 1], request_phases=["decode", "prefill"])
            info = result.breakdown["runtime_cost"]
            self.assertAlmostEqual(result.duration_ms, 2 * info["nominal_duration_ms"] + 3)
            self.assertFalse(info["shape_observed_in_training"])
            self.assertFalse(info["hardware_validated"])
            for roles in (None, ["prefill", "prefill"], ["prefill", "decode"]):
                with self.subTest(roles=roles), self.assertRaises(SimulationError):
                    model.estimate_step("mixed", [1, 3], [11, 1], request_phases=roles)

    def test_shape_series_and_cross_process_instability_reject_without_parameters(self):
        for change in ("within", "cross", "missing_session", "missing_repeats", "unidentified"):
            _, data, policy = fixture(selected=(("prefill", 1),))
            if change == "within":
                for row in data["records"][:4]:
                    row["duration_ms"] *= 2
            elif change == "cross":
                for row in data["records"]:
                    if row["session_id"] == "session-b":
                        row["duration_ms"] *= 2
            elif change == "missing_session":
                data["records"] = [row for row in data["records"] if row["session_id"] == "session-a"]
            elif change == "missing_repeats":
                data["records"].pop(0)
            else:
                for row in data["records"]:
                    row["query_tokens"], row["context_lengths"] = [2], [0]
            rejected = fit(data, policy)
            self.assertEqual(rejected["kind"], RUNTIME_COST_REJECTION_KIND)
            self.assertNotIn("parameters", rejected)
            self.assertEqual(len(rejected["raw_dataset"]["records"]), len(data["records"]))
            self.assertFalse(rejected["fit_attempted"])

    def test_frozen_policy_and_source_order_cannot_be_changed_or_trimmed(self):
        _, original, policy = fixture(selected=(("prefill", 1),))
        for change in ("order", "duplicate_id", "bounds", "role_metadata"):
            data = deepcopy(original)
            if change == "order":
                data["records"] = sorted(data["records"], key=lambda row: row["record_id"])
            elif change == "duplicate_id":
                data["records"][1]["record_id"] = data["records"][0]["record_id"]
            elif change == "bounds":
                data["records"][0]["context_lengths"] = [129]
            else:
                data["records"][0]["request_phases"] = ["prefill"]
            with self.subTest(change=change), self.assertRaises(SimulationError):
                fit(data, policy)
        for rehashed in (False, True):
            changed = deepcopy(policy)
            changed["min_samples_per_shape_session"] = 8.0
            if rehashed:
                changed["policy_sha256"] = digest({key: value for key, value in changed.items() if key != "policy_sha256"})
            with self.assertRaises(SimulationError):
                fit(original, changed)

    def test_full_runtime_memory_model_and_nominal_basis_are_bound(self):
        experiment, data, policy = fixture(selected=(("prefill", 1),))
        artifact = fit(data, policy)
        changes = [replace(experiment, strategy=replace(experiment.strategy, max_batch_tokens=2048)),
            replace(experiment, strategy=replace(experiment.strategy, memory=replace(experiment.strategy.memory, kv_pool_bytes=8 * 1024 * 1024))),
            replace(experiment, calibration=replace(experiment.calibration, launch_overhead_us=3)),
            replace(experiment, calibration=replace(experiment.calibration, runtime_config_sha256="b" * 64)),
            replace(experiment, calibration=replace(experiment.calibration, prefill_execution_mode="graph")),
            replace(experiment, calibration=replace(experiment.calibration, decode_attention_backend="other-backend")),
            replace(experiment, infra=replace(experiment.infra, devices=(replace(experiment.infra.devices[0], name="other-device"),)))]
        for changed in changes:
            with self.assertRaises(SimulationError) as caught:
                validate_runtime_cost_model(artifact, experiment=changed, allow_test_fixture=True)
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_snapshot_query_and_workspace_guard_permission_and_domain_each_time(self):
        experiment, data, policy = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(fit(data, policy)))
            configured = replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path)))
            with patch("mlsys_sim.runtime_cost.validate_runtime_cost_model", wraps=validate_runtime_cost_model) as validator:
                model = RuntimeCostModel(configured)
                path.write_text("changed after snapshot")
                for _ in range(3):
                    model.estimate_step("prefill", [3], [0])
                self.assertEqual(validator.call_count, 1)
            exposed = model.profile
            exposed["parameters"][0]["scale"] = 0
            self.assertNotEqual(exposed, model.profile)
            model.experiment = replace(configured, calibration=replace(configured.calibration, allow_test_fixture=False))
            for call in (lambda: model.estimate_step("prefill", [3], [0]), lambda: model.workspace_bytes([1], "decode")):
                with self.assertRaises(SimulationError) as caught:
                    call()
                self.assertEqual(caught.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
            model.experiment = replace(configured, strategy=replace(configured.strategy, max_batch_size=4))
            with self.assertRaises(SimulationError) as caught:
                model.estimate_step("prefill", [3], [0])
            self.assertEqual(caught.exception.code, "CALIBRATION_MISMATCH")

    def test_envelope_and_missing_phase_are_hard_failures(self):
        experiment, data, policy = fixture(selected=(("prefill", 1),))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(fit(data, policy)))
            model = RuntimeCostModel(replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path))))
            for phase, queries, contexts in (("decode", [1], [8]), ("prefill", [33], [0]),
                                            ("prefill", [1], [129]), ("prefill", [1, 1], [0, 0])):
                with self.subTest(phase=phase, queries=queries), self.assertRaises(SimulationError) as caught:
                    model.estimate_step(phase, queries, contexts)
                self.assertEqual(caught.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_total_query_context_and_combined_budgets_are_checked_separately(self):
        experiment, data, policy = fixture(selected=(("mixed", 2),))
        for changed, queries, contexts in (({"max_total_query_tokens": 32}, [1, 32], [8, 0]),
            ({"max_total_context_tokens": 30}, [1, 2], [20, 20]),
            ({"max_total_tokens": 50}, [1, 30], [20, 20])):
            rule = {**policy["cells"][0], **changed}
            artifact = fit(data, freeze_runtime_cost_policy([rule]))
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "runtime.json"
                path.write_text(json.dumps(artifact))
                model = RuntimeCostModel(replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path))))
                with self.subTest(changed=changed), self.assertRaises(SimulationError) as caught:
                    model.estimate_step("mixed", queries, contexts, request_phases=["decode", "prefill"])
                self.assertEqual(caught.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_missing_declared_cell_cannot_emit_a_partial_executable_model(self):
        _, data, policy = fixture()
        data["records"] = [row for row in data["records"] if row["phase"] == "prefill"]
        rejected = fit(data, policy)
        self.assertEqual(rejected["kind"], RUNTIME_COST_REJECTION_KIND)
        self.assertNotIn("parameters", rejected)

    def test_one_overlong_sequence_is_rejected_even_when_batch_sum_fits(self):
        experiment, data, policy = fixture(selected=(("mixed", 2),))
        rule = {**policy["cells"][0], "max_sequence_tokens": 32}
        policy = freeze_runtime_cost_policy([rule])
        artifact = fit(data, policy)
        queries, contexts = [1, 31], [0, 2]
        self.assertLess(sum(queries) + sum(contexts), rule["max_total_tokens"])
        self.assertGreater(queries[1] + contexts[1], rule["max_sequence_tokens"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(artifact))
            model = RuntimeCostModel(replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path))))
            with self.assertRaises(SimulationError) as caught:
                model.estimate_step("mixed", queries, contexts, request_phases=["decode", "prefill"])
            self.assertEqual(caught.exception.code, "OUT_OF_CALIBRATION_DOMAIN")
        data["records"][0].update(query_tokens=queries, context_lengths=contexts)
        with self.assertRaisesRegex(SimulationError, "preregistered"):
            fit(data, policy)

    def test_sequence_budget_is_explicit_required_integer_policy_data(self):
        _, _, policy = fixture(selected=(("prefill", 1),))
        for change in ("missing", "float", "zero"):
            rule = deepcopy(policy["cells"][0])
            if change == "missing":
                rule.pop("max_sequence_tokens")
            else:
                rule["max_sequence_tokens"] = 160.0 if change == "float" else 0
            with self.subTest(change=change), self.assertRaises(SimulationError):
                freeze_runtime_cost_policy([rule])

    def test_training_must_obey_declared_mixed_and_active_scheduler_policy(self):
        experiment, data, policy = fixture(selected=(("mixed", 2),))
        for change in ({"mixed_batching": False}, {"max_active_requests": 1}, {"max_batch_tokens": 2}):
            changed = replace(experiment, strategy=replace(experiment.strategy, **change))
            data["domain"] = runtime_cost_domain(changed)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                fit(data, policy)

    def test_workspace_is_known_only_for_fully_observed_exact_shape(self):
        experiment, data, policy = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(fit(data, policy)))
            model = RuntimeCostModel(replace(experiment, calibration=replace(experiment.calibration, profile_path=str(path))))
            self.assertEqual(model.workspace_bytes([2], "prefill", context_lengths=[0]), 101)
            self.assertIsNone(model.workspace_bytes([3], "prefill", context_lengths=[0]))
            self.assertIsNone(model.workspace_bytes([1], "decode"))
            with self.assertRaises(SimulationError):
                model.workspace_bytes([2], "decode")

    def test_corruption_and_rehashed_semantic_tampering_are_rejected(self):
        _, data, policy = fixture()
        artifact = fit(data, policy)
        for change in ("raw", "parameter", "phase", "ids", "stationarity"):
            modified = deepcopy(artifact)
            if change == "raw":
                modified["raw_dataset"]["records"][0]["duration_ms"] += 1
            elif change == "parameter":
                modified["parameters"][0]["scale"] += 1
            elif change == "phase":
                modified["parameters"][0]["phase"] = "mixed"
            elif change == "ids":
                modified["training_record_ids"].reverse()
            else:
                modified["diagnostics"]["stationarity"]["verdict"] = "FAIL"
            rehash(modified)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                validate_runtime_cost_model(modified, allow_test_fixture=True)

    def test_measured_claims_fixture_permission_and_wrong_boundary_are_rejected(self):
        experiment, data, policy = fixture()
        with self.assertRaises(SimulationError) as caught:
            fit_runtime_costs(data, policy, calibration_id="fixture")
        self.assertEqual(caught.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        data["source"]["kind"] = "simulated"
        with self.assertRaises(SimulationError):
            fit(data, policy)
        data["source"]["kind"] = "measured"
        with self.assertRaises(SimulationError):
            fit(data, policy)
        for change in (replace(experiment.calibration, timing_boundary="step_includes_output_and_sampling_v1"),
                       replace(experiment.calibration, backend="profile"), replace(experiment.calibration, runtime_config_sha256="")):
            with self.assertRaises(SimulationError):
                runtime_cost_domain(replace(experiment, calibration=change))

    def test_loader_rejects_duplicate_json_and_rejection_artifacts(self):
        _, data, policy = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text('{"kind":"runtime_cost_model_v1","kind":"other"}')
            with self.assertRaises(SimulationError):
                load_runtime_cost(path, allow_test_fixture=True)
            for row in data["records"]:
                if row["session_id"] == "session-b":
                    row["duration_ms"] *= 2
            path.write_text(json.dumps(fit(data, policy)))
            with self.assertRaises(SimulationError):
                load_runtime_cost(path, allow_test_fixture=True)


if __name__ == "__main__":
    unittest.main()
