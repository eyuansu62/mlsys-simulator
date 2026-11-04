"""Complete Workspace ownership, exact per-rank bounds and memory precedence."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryPlanner,
                       ModelSpec, RequestSpec, SimulationError, StrategySpec, calibrate_dataset,
                       experiment_from_dict, load_experiment, profile_identity, replica_experiment,
                       simulate, validate_experiment)
from mlsys_sim.workspace import (ATTRIBUTION, WorkspaceModel, calibrate_workspace,
    load_workspace_dataset, load_workspace_profile, validate_workspace_dataset, validate_workspace_profile)


ROOT = Path(__file__).resolve().parents[1]


def experiment(tp=1, pp=1):
    return Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(tp * pp))),
        strategy=StrategySpec(model=ModelSpec(activation_dtype="float16"), tensor_parallel_size=tp,
                              pipeline_parallel_size=pp),
        workload=(RequestSpec("request", 0, 4, 1),), calibration=CalibrationSpec(backend="synthetic"))


def workspace_record(exp, *, stage=0, phase="prefill", queries=(4,), contexts=(0,), peaks=None,
                     roles=None, identifier="r"):
    if peaks is None:
        peaks = [0] * exp.strategy.tensor_parallel_size
    row = {"record_id": identifier, "group_id": identifier, "phase": phase, "stage_index": stage,
        "query_tokens": list(queries), "context_lengths": list(contexts),
        "workspace_bytes_per_rank": [{"tp_rank": rank, "peak_bytes": peak} for rank, peak in enumerate(peaks)]}
    if roles is not None:
        row["request_phases"] = list(roles)
    return row


def workspace_fixture(exp, rows=None):
    return {"schema_version": 1, "kind": "workspace_measurements", "dataset_id": "workspace-cpu-fixture",
        "identity": profile_identity(exp), "source": {"kind": "test_fixture",
            "reference": "Declared complete ownership fixture, not GPU allocator evidence"},
        "attribution": deepcopy(ATTRIBUTION), "records": rows if rows is not None else [workspace_record(exp)]}


def configured(exp, root, rows=None, *, version="workspace-test"):
    profile = calibrate_workspace(workspace_fixture(exp, rows), profile_version=version, allow_test_fixture=True)
    path = Path(root) / f"{version}.json"
    path.write_text(json.dumps(profile))
    return replace(exp, calibration=replace(exp.calibration, workspace_profile_path=str(path),
        workspace_profile_version=version, allow_test_fixture=True)), path


class WorkspaceContractTests(unittest.TestCase):
    def test_repeated_shapes_use_each_rank_max_and_preserve_true_zero(self):
        exp = experiment(tp=2)
        data = workspace_fixture(exp, [workspace_record(exp, peaks=(0, 20), identifier="first"),
            workspace_record(exp, peaks=(0, 10), identifier="second")])
        before = deepcopy(data)
        profile = calibrate_workspace(data, profile_version="max", allow_test_fixture=True)
        self.assertEqual(data, before)
        self.assertEqual(profile["samples"][0]["workspace_bytes_per_rank"],
                         [{"tp_rank": 0, "peak_bytes": 0}, {"tp_rank": 1, "peak_bytes": 20}])
        self.assertEqual(profile["samples"][0]["sample_count"], 2)
        model = WorkspaceModel(replace(exp, calibration=replace(exp.calibration, allow_test_fixture=True)), profile)
        self.assertEqual(model.workspace_bytes("prefill", [4], [0], 0, 0), 0)
        self.assertEqual(model.workspace_bytes("prefill", [4], [0], 0, 1), 20)

    def test_partial_or_wrong_attribution_never_becomes_full_workspace(self):
        exp = experiment()
        for field, value in (("complete", False), ("complete", 1), ("includes_non_torch", False),
                             ("method", "peak_minus_baseline"), ("excluded_categories", []),
                             ("scope", "managed_pool_only")):
            data = workspace_fixture(exp)
            data["attribution"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SimulationError):
                calibrate_workspace(data, profile_version="bad", allow_test_fixture=True)
        data = workspace_fixture(exp)
        data["attribution"]["unknown"] = True
        with self.assertRaises(SimulationError):
            validate_workspace_dataset(data, allow_test_fixture=True)

    def test_rank_coverage_exact_types_and_unknown_fields_are_strict(self):
        exp = experiment(tp=2)
        invalid = [[], [{"tp_rank": 0, "peak_bytes": 1}],
            [{"tp_rank": 0, "peak_bytes": 1}, {"tp_rank": 0, "peak_bytes": 2}],
            [{"tp_rank": False, "peak_bytes": 1}, {"tp_rank": 1, "peak_bytes": 2}],
            [{"tp_rank": 0, "peak_bytes": True}, {"tp_rank": 1, "peak_bytes": 2}],
            [{"tp_rank": 0, "peak_bytes": -1}, {"tp_rank": 1, "peak_bytes": 2}],
            [{"tp_rank": 0, "peak_bytes": None}, {"tp_rank": 1, "peak_bytes": 2}]]
        for ranks in invalid:
            data = workspace_fixture(exp)
            data["records"][0]["workspace_bytes_per_rank"] = ranks
            with self.subTest(ranks=ranks), self.assertRaises(SimulationError):
                validate_workspace_dataset(data, allow_test_fixture=True)
        for change in ({"duration_ms": 1}, {"stage_index": 1}, {"schema_version": 2},
                       {"request_phases": ["prefill"]}):
            data = workspace_fixture(exp)
            data["records"][0].update(change)
            with self.assertRaises(SimulationError):
                validate_workspace_dataset(data, allow_test_fixture=True)

    def test_zero_unknown_and_missing_cannot_be_conflated(self):
        exp = experiment()
        for value in (None, "0", 0.0, float("nan")):
            data = workspace_fixture(exp)
            data["records"][0]["workspace_bytes_per_rank"][0]["peak_bytes"] = value
            with self.subTest(value=value), self.assertRaises(SimulationError):
                validate_workspace_dataset(data, allow_test_fixture=True)
        normalized = validate_workspace_dataset(workspace_fixture(exp), allow_test_fixture=True)
        self.assertEqual(normalized["records"][0]["workspace_bytes_per_rank"][0]["peak_bytes"], 0)

    def test_shape_and_mixed_roles_are_exact_without_interpolation(self):
        exp = experiment(tp=2, pp=2)
        rows = [workspace_record(exp, stage=1, phase="mixed", queries=(1, 2), contexts=(8, 0),
                                roles=("decode", "prefill"), peaks=(8, 16))]
        profile = calibrate_workspace(workspace_fixture(exp, rows), profile_version="mixed", allow_test_fixture=True)
        model = WorkspaceModel(replace(exp, calibration=replace(exp.calibration, allow_test_fixture=True)), profile)
        self.assertEqual(model.workspace_bytes("mixed", [1, 2], [8, 0], 1, 1,
                                              request_phases=["decode", "prefill"]), 16)
        for phase, queries, contexts, stage, rank, roles in (
            ("mixed", [2, 1], [0, 8], 1, 1, ["prefill", "decode"]),
            ("mixed", [1, 2], [9, 0], 1, 1, ["decode", "prefill"]),
            ("mixed", [1, 2], [8, 0], 0, 1, ["decode", "prefill"]),
            ("prefill", [4], [0], 0, 0, None), ("decode", [1], [8], 0, 2, None)):
            with self.subTest(phase=phase, stage=stage, rank=rank), self.assertRaises(SimulationError) as raised:
                model.workspace_bytes(phase, queries, contexts, stage, rank, request_phases=roles)
            self.assertEqual(raised.exception.code, "OUT_OF_WORKSPACE_CALIBRATION_DOMAIN")

    def test_source_identity_and_versions_are_bound(self):
        exp = experiment()
        data = workspace_fixture(exp)
        with self.assertRaises(SimulationError) as raised:
            calibrate_workspace(data, profile_version="fixture")
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        profile = calibrate_workspace(data, profile_version="fixture", allow_test_fixture=True)
        with self.assertRaises(SimulationError):
            WorkspaceModel(exp, profile)
        changed = replace(exp, strategy=replace(exp.strategy, model=replace(exp.strategy.model, revision="other")),
            calibration=replace(exp.calibration, allow_test_fixture=True))
        with self.assertRaises(SimulationError) as raised:
            WorkspaceModel(changed, profile)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        data["source"]["kind"] = "measured"
        with self.assertRaises(SimulationError):
            validate_workspace_dataset(data, allow_test_fixture=True)

    def test_parent_lineage_is_new_version_and_does_not_reuse_values(self):
        exp = experiment()
        data = workspace_fixture(exp, [workspace_record(exp, peaks=(30,))])
        parent = calibrate_workspace(data, profile_version="parent", allow_test_fixture=True)
        with self.assertRaises(SimulationError) as raised:
            calibrate_workspace(data, profile_version="parent", parent_profile=parent, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "PROFILE_VERSION_CONFLICT")
        child_data = workspace_fixture(exp, [workspace_record(exp, peaks=(0,), identifier="new")])
        child = calibrate_workspace(child_data, profile_version="child", parent_profile=parent, allow_test_fixture=True)
        self.assertEqual(child["training"]["parent_profile_version"], "parent")
        self.assertEqual(child["samples"][0]["workspace_bytes_per_rank"][0]["peak_bytes"], 0)
        self.assertEqual(parent["samples"][0]["workspace_bytes_per_rank"][0]["peak_bytes"], 30)
        wrong = deepcopy(parent)
        wrong["identity"]["model"]["revision"] = "different"
        with self.assertRaises(SimulationError):
            calibrate_workspace(child_data, profile_version="child", parent_profile=wrong, allow_test_fixture=True)

    def test_profile_training_provenance_and_unknown_methods_are_strict(self):
        exp = experiment()
        profile = calibrate_workspace(workspace_fixture(exp), profile_version="v", allow_test_fixture=True)
        for field, value in (("record_ids", ["a", "b"]), ("group_ids", ["g", "g"]),
                             ("shape_keys", ["unobserved"]), ("dataset_sha256", "bad"),
                             ("parent_profile_version", "v")):
            bad = deepcopy(profile)
            bad["training"][field] = value
            with self.subTest(field=field), self.assertRaises(SimulationError):
                validate_workspace_profile(bad, allow_test_fixture=True)
        bad = deepcopy(profile)
        bad["method"] = "linear_fit"
        with self.assertRaises(SimulationError):
            validate_workspace_profile(bad, allow_test_fixture=True)
        data = workspace_fixture(exp)
        data["records"] *= 2
        with self.assertRaises(SimulationError):
            validate_workspace_dataset(data, allow_test_fixture=True)

    def test_file_hash_immutable_load_and_duplicate_json_rejection(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            exp, path = configured(exp, temporary)
            before = path.read_bytes()
            model = WorkspaceModel(exp)
            self.assertEqual(model.provenance["sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(model.provenance["sha256_scope"], "file_bytes")
            self.assertEqual(load_workspace_profile(path, allow_test_fixture=True)["kind"], "workspace_profile")
            path.write_text("broken after loading")
            self.assertEqual(model.workspace_bytes("prefill", [4], [0], 0, 0), 0)
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(SimulationError):
                load_workspace_dataset(path, allow_test_fixture=True)
            path.write_text('{"unknown": NaN}')
            with self.assertRaises(SimulationError):
                load_workspace_profile(path, allow_test_fixture=True)


class WorkspaceMemoryTests(unittest.TestCase):
    def test_explicit_profile_can_replace_a_larger_analytical_static_floor(self):
        exp = experiment()
        fixed = MemoryPlanner(exp).snapshot()[0]["fixed_budget_bytes"]
        exp = replace(exp, infra=replace(exp.infra, devices=(replace(exp.infra.devices[0], memory_bytes=fixed + 16),)))
        with self.assertRaises(SimulationError) as raised:
            MemoryPlanner(exp)
        self.assertEqual(raised.exception.code, "STATIC_MEMORY_INFEASIBLE")
        with tempfile.TemporaryDirectory() as temporary:
            exp, _ = configured(exp, temporary)
            planner = MemoryPlanner(exp)
            decision = planner.reserve({"request": 4}, [4], "prefill")
            self.assertTrue(decision.ok)
            self.assertEqual(decision.details["ranks"][0]["workspace_bytes"], 0)
            self.assertEqual(decision.details["ranks"][0]["workspace_source"], "workspace_profile")
            self.assertEqual(planner.snapshot()[0]["workspace_peak_bytes"], 0)

    def test_asymmetric_rank_bound_fails_atomically_and_reports_weak_rank(self):
        exp = experiment(tp=2)
        fixed = MemoryPlanner(exp).snapshot()[0]["fixed_budget_bytes"]
        exp = replace(exp, infra=replace(exp.infra, devices=tuple(replace(d, memory_bytes=fixed + 20) for d in exp.infra.devices)))
        with tempfile.TemporaryDirectory() as temporary:
            exp, _ = configured(exp, temporary, [workspace_record(exp, peaks=(10, 30))])
            planner = MemoryPlanner(exp)
            decision = planner.reserve({"request": 4}, [4], "prefill")
            self.assertFalse(decision.ok)
            self.assertEqual(decision.code, "WORKSPACE_BUDGET_EXCEEDED")
            self.assertEqual([r["physical_ok"] for r in decision.details["ranks"]], [True, False])
            self.assertFalse(planner.allocations)
            self.assertTrue(all(r["workspace_peak_bytes"] == 0 for r in planner.snapshot()))

    def test_explicit_profile_takes_precedence_over_legacy_stage_workspace(self):
        exp = experiment()
        stage_data = {"schema_version": 1, "dataset_id": "legacy-workspace", "identity": profile_identity(exp),
            "source": {"kind": "test_fixture", "reference": "Declared stage fixture"}, "records": [{
                "record_id": "prefill", "group_id": "prefill", "phase": "prefill", "stage_index": 0,
                "query_tokens": [4], "context_lengths": [0], "duration_ms": 1, "workspace_peak_bytes": 999}]}
        with tempfile.TemporaryDirectory() as temporary:
            exp, _ = configured(exp, temporary, [workspace_record(exp, peaks=(5,))])
            stage_path = Path(temporary) / "stage.json"
            stage_path.write_text(json.dumps(calibrate_dataset(stage_data, profile_version="stage", allow_test_fixture=True)))
            exp = replace(exp, calibration=replace(exp.calibration, backend="profile", profile_path=str(stage_path),
                                                   profile_version="stage"))
            planner = MemoryPlanner(exp)
            decision = planner.check({"request": 4}, [4], "prefill")
            self.assertEqual(decision.details["workspace_bytes"], 5)
            self.assertEqual(decision.details["ranks"][0]["workspace_source"], "workspace_profile")
            old = replace(exp, calibration=replace(exp.calibration, workspace_profile_path=None, workspace_profile_version=""))
            self.assertEqual(MemoryPlanner(old).check({"request": 4}, [4], "prefill").details["workspace_bytes"], 999)

    def test_outside_shape_is_not_an_admission_failure_or_analytical_fallback(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            exp, _ = configured(exp, temporary)
            planner = MemoryPlanner(exp)
            with self.assertRaises(SimulationError) as raised:
                planner.reserve({"request": 5}, [5], "prefill")
            self.assertEqual(raised.exception.code, "OUT_OF_WORKSPACE_CALIBRATION_DOMAIN")
            self.assertFalse(planner.allocations)
            self.assertTrue(planner.reserve_kv({"request": 5}).ok)
            self.assertEqual(planner.snapshot()[0]["workspace_peak_bytes"], 0)
            self.assertEqual(planner.check_kv({"request": 6}).details["ranks"][0]["workspace_source"], "not_required_for_copy")

    def test_snapshots_are_independent_and_loaded_profile_is_not_reread(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            exp, path = configured(exp, temporary, [workspace_record(exp, peaks=(7,))])
            planner = MemoryPlanner(exp)
            path.write_text("broken source")
            self.assertTrue(planner.reserve({"request": 4}, [4], "prefill").ok)
            snapshot = planner.snapshot()
            snapshot[0]["workspace_profile_provenance"]["attribution"]["excluded_categories"].clear()
            self.assertEqual(planner.snapshot()[0]["workspace_profile_provenance"]["attribution"]["excluded_categories"],
                             ATTRIBUTION["excluded_categories"])
            physical = planner.snapshot()[0]["physical_reserved_bytes"]
            planner.release("request")
            self.assertEqual(planner.snapshot()[0]["physical_reserved_bytes"], physical)
            self.assertEqual(planner.snapshot()[0]["kv_used_bytes"], 0)

    def test_paths_config_flags_and_replica_selection_preserve_stage_identity(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen, _ = configured(exp, root)
            self.assertEqual(profile_identity(chosen), profile_identity(exp))
            validate_experiment(chosen)
            for calibration in (replace(exp.calibration, workspace_profile_version="no-file"),
                                replace(exp.calibration, workspace_profile_path=""),
                                replace(exp.calibration, replica_workspace_profile_paths=("a", "b"))):
                with self.assertRaises(SimulationError):
                    validate_experiment(replace(exp, calibration=calibration))
            data = asdict(exp)
            data["calibration"].update(workspace_profile_path="workspace-test.json", allow_test_fixture=True)
            (root / "experiment.json").write_text(json.dumps(data))
            self.assertEqual(load_experiment(root / "experiment.json").calibration.workspace_profile_path,
                             str((root / "workspace-test.json").resolve()))
            with self.assertRaises(SimulationError):
                experiment_from_dict({"calibration": {"workspace_profile_path": True}})
            cluster = replace(exp, infra=replace(exp.infra, devices=(DeviceSpec("replica0"), DeviceSpec("replica1"))),
                              strategy=replace(exp.strategy, replicas=2))
            paths = []
            for i in range(2):
                local = replica_experiment(cluster, i)
                _, path = configured(local, root, version=f"replica-{i}")
                paths.append(str(path))
            cluster = replace(cluster, calibration=replace(cluster.calibration,
                replica_workspace_profile_paths=tuple(paths), allow_test_fixture=True))
            self.assertEqual(replica_experiment(cluster, 1).calibration.workspace_profile_path, paths[1])
            self.assertFalse(replica_experiment(cluster, 1).calibration.replica_workspace_profile_paths)
            self.assertEqual(simulate(cluster).result["counts"]["completed"], 1)
            wrong = replace(cluster, calibration=replace(cluster.calibration,
                replica_workspace_profile_paths=tuple(reversed(paths))))
            with self.assertRaises(SimulationError):
                simulate(wrong)

    def test_workspace_stage_and_communication_profiles_preserve_native_recovery_lifetimes(self):
        from mlsys_sim import calibrate_pp_communication
        from test_native_pp_transport import dataset as stage_dataset, record as stage_record
        from test_pp_communication_integration import communication_dataset

        exp = load_experiment(ROOT / "examples/v1_7/qwen3-native-r2-tp2-pp2.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            communication_paths = []
            for replica in range(2):
                local = replica_experiment(exp, replica)
                path = root / f"communication-{replica}.json"
                path.write_text(json.dumps(calibrate_pp_communication(communication_dataset(local, scale=.01),
                    profile_version="communication-combo", allow_test_fixture=True)))
                communication_paths.append(str(path))
            exp = replace(exp, calibration=replace(exp.calibration, allow_test_fixture=True,
                replica_native_pp_communication_profile_paths=tuple(communication_paths),
                native_pp_communication_profile_version="communication-combo"))
            baseline = simulate(exp)
            stage_paths, workspace_paths = [], []
            for replica in range(2):
                local = replica_experiment(exp, replica)
                stage_rows, workspace_rows = [], []
                for index, event in enumerate(baseline.events):
                    if event["event"] != "STAGE_START" or event["replica_index"] != replica:
                        continue
                    row = stage_record(local, event["stage"], phase=event["phase"], query=event["query_tokens"],
                        context=event["context_lengths"], padded=sum(event["query_tokens"]),
                        roles=event.get("request_phases"), duration=event["duration_ms"])
                    row.update(record_id=f"stage-{index}", group_id=f"shape-{index}")
                    stage_rows.append(row)
                    workspace_rows.append(workspace_record(local, stage=event["stage"], phase=event["phase"],
                        queries=event["query_tokens"], contexts=event["context_lengths"],
                        roles=event.get("request_phases"), peaks=(2, 3), identifier=f"workspace-{index}"))
                path = root / f"stage-{replica}.json"
                path.write_text(json.dumps(calibrate_dataset(stage_dataset(local, stage_rows),
                    profile_version="stage-combo", allow_test_fixture=True)))
                stage_paths.append(str(path))
                _, workspace_path = configured(local, root, workspace_rows, version=f"workspace-{replica}")
                workspace_paths.append(str(workspace_path))
            calibrated = replace(exp, calibration=replace(exp.calibration, backend="profile",
                replica_profile_paths=tuple(stage_paths), profile_version="stage-combo",
                replica_workspace_profile_paths=tuple(workspace_paths)))
            result = simulate(calibrated)
            self.assertEqual(result.requests, baseline.requests)
            self.assertEqual(result.result["counts"]["completed"], 10)
            self.assertEqual(result.result["prefix_cache"]["hit_tokens"], 24)
            self.assertGreater(result.result["batching"]["mixed_batches_completed"], 0)
            self.assertEqual(result.result["swap"]["completed_transfers_by_direction"], {"D2H": 2, "H2D": 2})
            self.assertEqual(result.result["swap"]["host_reserved_bytes"], 0)
            self.assertEqual(result.result["prefix_cache"]["attached_block_references"], 0)
            self.assertTrue(all(rank["workspace_source"] == "workspace_profile" for rank in result.result["memory"]))
            self.assertTrue(all(rank["workspace_peak_bytes"] == (2 if rank["tp_rank"] == 0 else 3)
                                for rank in result.result["memory"]))


if __name__ == "__main__":
    unittest.main()
