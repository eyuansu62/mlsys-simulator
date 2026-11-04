"""Exact Workspace bounds compose with existing timing and serving policies.

All observations here are declared CPU test fixtures, not GPU measurements.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (MemoryPlanner, SimulationError, calibrate_dataset, load_experiment,
                       predict_experiment, profile_identity, replica_experiment, simulate)
from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.workspace import ATTRIBUTION, WorkspaceModel, calibrate_workspace
from test_native_pp_transport import dataset as stage_dataset, experiment, record as stage_record
from test_pp_communication_integration import fit as fit_communication


ROOT = Path(__file__).resolve().parents[1]


def workspace_record(exp, stage, *, phase="prefill", queries=(4,), contexts=(0,), roles=None,
                     peaks=None, identifier=None):
    ranks = exp.strategy.tensor_parallel_size
    row = {"record_id": identifier or f"{phase}-stage-{stage}",
        "group_id": identifier or f"{phase}-stage-{stage}", "phase": phase, "stage_index": stage,
        "query_tokens": list(queries), "context_lengths": list(contexts),
        "workspace_bytes_per_rank": [{"tp_rank": rank, "peak_bytes": (peaks[rank] if peaks is not None else rank * 32)}
                                     for rank in range(ranks)]}
    if phase == "mixed":
        row["request_phases"] = list(roles)
    return row


def workspace_dataset(exp, rows):
    return {"schema_version": 1, "kind": "workspace_measurements", "dataset_id": "workspace-integration",
        "identity": profile_identity(exp), "source": {"kind": "test_fixture",
            "reference": "Declared per-rank CPU test bounds; no allocator or GPU measurement"},
        "attribution": deepcopy(ATTRIBUTION), "records": rows}


def fit_workspace(exp, path, rows=None, *, version="workspace-integration"):
    if rows is None:
        rows = [workspace_record(exp, stage) for stage in range(exp.strategy.pipeline_parallel_size)]
    profile = calibrate_workspace(workspace_dataset(exp, rows), profile_version=version, allow_test_fixture=True)
    path.write_text(json.dumps(profile))
    return replace(exp, calibration=replace(exp.calibration, workspace_profile_path=str(path),
        workspace_profile_version=version, allow_test_fixture=True))


class WorkspaceIntegrationTests(unittest.TestCase):
    def test_rank_specific_zero_replaces_stage_bound_without_changing_stage_identity(self):
        base = experiment()
        rows = [stage_record(base, stage, padded=4) for stage in range(2)]
        rows += [stage_record(base, stage, phase="decode", query=(1,), context=(4,), padded=1)
                 for stage in range(2)]
        for row in rows:
            row["workspace_peak_bytes"] = 4096
        profile = calibrate_dataset(stage_dataset(base, rows), profile_version="timing", allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stage_path = directory / "timing.json"
            stage_path.write_text(json.dumps(profile))
            timing = replace(base, calibration=replace(base.calibration, backend="profile",
                profile_path=str(stage_path), profile_version="timing", allow_test_fixture=True))
            legacy = MemoryPlanner(timing).check({"request": 4}, [4], "prefill")
            self.assertEqual([row["workspace_bytes"] for row in legacy.details["ranks"]], [4096] * 4)
            self.assertTrue(all(row["workspace_source"] == "profile" for row in legacy.details["ranks"]))
            workspace_rows = [workspace_record(timing, stage) for stage in range(2)]
            workspace_rows += [workspace_record(timing, stage, phase="decode", queries=(1,), contexts=(4,))
                               for stage in range(2)]
            configured = fit_workspace(timing, directory / "workspace.json", workspace_rows)
            self.assertEqual(profile_identity(configured), profile_identity(timing))
            decision = MemoryPlanner(configured).check({"request": 4}, [4], "prefill")
            self.assertTrue(decision.ok)
            self.assertEqual([row["workspace_bytes"] for row in decision.details["ranks"]], [0, 32, 0, 32])
            self.assertTrue(all(row["workspace_source"] == "workspace_profile" for row in decision.details["ranks"]))
            self.assertTrue(all(row["workspace_profile_provenance"]["source_kind"] == "test_fixture"
                                for row in decision.details["ranks"]))
            prediction = predict_experiment(configured)
            self.assertTrue(prediction["prefill_memory_feasible"])
            self.assertEqual([row["workspace_bytes"] for row in prediction["memory_check_details"]["ranks"]],
                             [0, 32, 0, 32])
            self.assertEqual(simulate(configured).requests[0]["completion_ms"], simulate(timing).requests[0]["completion_ms"])
        analytical = MemoryPlanner(base).check({"request": 4}, [4], "prefill")
        expected = base.strategy.memory.workspace_base_bytes + 4 * base.strategy.memory.workspace_bytes_per_token
        self.assertEqual(analytical.details["workspace_bytes"], expected)
        self.assertTrue(all(row["workspace_source"] == "analytical_assumption" for row in analytical.details["ranks"]))

    def test_selected_zero_removes_only_analytical_floor_and_preserves_fixed_allocations(self):
        reference = experiment()
        excessive = replace(reference, strategy=replace(reference.strategy,
            memory=replace(reference.strategy.memory, workspace_base_bytes=128 * 1024**2)))
        with self.assertRaises(SimulationError) as raised:
            MemoryPlanner(excessive)
        self.assertEqual(raised.exception.code, "STATIC_MEMORY_INFEASIBLE")
        with tempfile.TemporaryDirectory() as temporary:
            rows = [workspace_record(excessive, stage, peaks=(0, 0)) for stage in range(2)]
            configured = fit_workspace(excessive, Path(temporary) / "zero.json", rows)
            planner = MemoryPlanner(configured)
            original = MemoryPlanner(reference).snapshot()
            before = planner.snapshot()
            for actual, expected in zip(before, original):
                for field in ("weights_bytes", "kv_pool_reserved_bytes", "physical_reserved_bytes",
                              "fixed_budget_bytes", "pipeline_buffer_reserved_bytes"):
                    self.assertEqual(actual[field], expected[field], field)
            self.assertTrue(planner.reserve({"request": 4}, [4], "prefill").ok)
            allocated = planner.snapshot()
            self.assertTrue(all(row["workspace_peak_bytes"] == 0 for row in allocated))
            self.assertTrue(all(row["workspace_source"] == "workspace_profile" for row in allocated))
            planner.release("request")
            self.assertEqual([row["physical_reserved_bytes"] for row in planner.snapshot()],
                             [row["physical_reserved_bytes"] for row in before])
            self.assertTrue(all(row["used_blocks"] == 0 for row in planner.snapshot()))

    def test_mixed_context_roles_are_exact_and_missing_shape_does_not_commit_kv(self):
        exp = experiment(mixed_batching=True, chunked_prefill=True, prefill_chunk_tokens=4)
        rows = [workspace_record(exp, stage, phase="mixed", queries=(1, 1), contexts=(8, 4),
                 roles=("decode", "prefill"), identifier=f"mixed-{stage}") for stage in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit_workspace(exp, Path(temporary) / "mixed.json", rows)
            planner = MemoryPlanner(configured)
            decision = planner.check({"decoder": 9, "cached-prefill": 5}, [1, 1], "mixed",
                                     request_phases=["decode", "prefill"])
            self.assertTrue(decision.ok)
            for targets, roles in (({"decoder": 9, "cached-prefill": 5}, ["prefill", "decode"]),
                                   ({"decoder": 10, "cached-prefill": 5}, ["decode", "prefill"])):
                with self.subTest(targets=targets, roles=roles), self.assertRaises(SimulationError) as raised:
                    planner.check(targets, [1, 1], "mixed", request_phases=roles)
                self.assertEqual(raised.exception.code, "OUT_OF_WORKSPACE_CALIBRATION_DOMAIN")
            self.assertFalse(planner.allocations)
            engine = PipelineSimulation(configured)
            with self.assertRaises(SimulationError) as raised:
                engine.run()
            self.assertEqual(raised.exception.code, "OUT_OF_WORKSPACE_CALIBRATION_DOMAIN")
            self.assertFalse(engine.memory.allocations)
            self.assertFalse(engine.waves)
            self.assertFalse(engine.transfers)

    def test_kv_copy_destination_does_not_require_a_gpu_workspace_shape(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit_workspace(exp, Path(temporary) / "workspace.json")
            planner = MemoryPlanner(configured)
            self.assertTrue(planner.reserve_kv({"restored": 20}).ok)
            decision = planner.check_kv({"restored": 24})
            self.assertTrue(decision.ok)
            self.assertTrue(all(row["workspace_source"] == "not_required_for_copy" for row in decision.details["ranks"]))
            self.assertTrue(all(row["workspace_source"] == "not_executed" for row in planner.snapshot()))
            planner.release("restored")
            self.assertTrue(all(row["used_blocks"] == 0 for row in planner.snapshot()))

    def test_weak_rank_has_an_exact_one_byte_budget_boundary_and_atomic_failure(self):
        base = experiment()
        fixed = MemoryPlanner(base).snapshot()[1]["fixed_budget_bytes"]
        with tempfile.TemporaryDirectory() as temporary:
            for extra, expected in ((31, False), (32, True)):
                devices = list(base.infra.devices)
                devices[1] = replace(devices[1], memory_bytes=fixed + extra)
                exp = replace(base, infra=replace(base.infra, devices=tuple(devices)))
                configured = fit_workspace(exp, Path(temporary) / f"capacity-{extra}.json")
                planner = MemoryPlanner(configured)
                decision = planner.reserve({"request": 4}, [4], "prefill")
                with self.subTest(extra=extra):
                    self.assertEqual(decision.ok, expected)
                    self.assertEqual(decision.details["ranks"][1]["physical_ok"], expected)
                    self.assertTrue(all(row["physical_ok"] for index, row in
                                        enumerate(decision.details["ranks"]) if index != 1))
                    if expected:
                        self.assertEqual(planner.allocations, {"request": 4})
                    else:
                        self.assertEqual(decision.code, "WORKSPACE_BUDGET_EXCEEDED")
                        self.assertFalse(planner.allocations)

    def test_loaded_workspace_is_immutable_and_changed_runtime_is_rejected(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workspace.json"
            configured = fit_workspace(exp, path)
            planner = MemoryPlanner(configured)
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_text("invalid after immutable load")
            self.assertTrue(planner.reserve({"request": 4}, [4], "prefill").ok)
            self.assertTrue(all(row["workspace_profile_provenance"]["sha256"] == expected_hash
                                for row in planner.snapshot()))
            configured = fit_workspace(exp, path)
            changed = replace(configured, calibration=replace(configured.calibration, runtime_version="different-runtime"))
            with self.assertRaises(SimulationError) as raised:
                WorkspaceModel(changed)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_three_profile_kinds_compose_with_replicas_cache_chunk_swap_and_mixed(self):
        exp = load_experiment(ROOT / "examples/v1_7/qwen3-native-r2-tp2-pp2.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            communication_paths = []
            for replica in range(2):
                path = root / f"communication-{replica}.json"
                fit_communication(replica_experiment(exp, replica), path, scale=.01)
                communication_paths.append(str(path))
            configured = replace(exp, calibration=replace(exp.calibration,
                replica_native_pp_communication_profile_paths=tuple(communication_paths),
                native_pp_communication_profile_version="comm-integration", allow_test_fixture=True))
            baseline = simulate(configured)
            stage_paths, workspace_paths = [], []
            for replica in range(2):
                local = replica_experiment(configured, replica)
                stage_rows, workspace_rows = [], []
                for index, event in enumerate(baseline.events):
                    if event["event"] != "STAGE_START" or event["replica_index"] != replica:
                        continue
                    timing = stage_record(local, event["stage"], phase=event["phase"], query=event["query_tokens"],
                        context=event["context_lengths"], padded=sum(event["query_tokens"]),
                        roles=event.get("request_phases"), duration=event["duration_ms"])
                    timing.update(record_id=f"timing-fixture-{index}", group_id=f"timing-shape-{index}",
                                  workspace_peak_bytes=512 * 1024)
                    stage_rows.append(timing)
                    workspace_rows.append(workspace_record(local, event["stage"], phase=event["phase"],
                        queries=event["query_tokens"], contexts=event["context_lengths"],
                        roles=event.get("request_phases"), peaks=(0, 1024 * (event["stage"] + 1)),
                        identifier=f"owned-workspace-fixture-{index}"))
                stage_profile = calibrate_dataset(stage_dataset(local, stage_rows),
                    profile_version="triple-stage", allow_test_fixture=True)
                stage_path = root / f"stage-{replica}.json"
                stage_path.write_text(json.dumps(stage_profile))
                stage_paths.append(str(stage_path))
                workspace_path = root / f"workspace-{replica}.json"
                fit_workspace(local, workspace_path, workspace_rows, version="triple-workspace")
                workspace_paths.append(str(workspace_path))
            calibrated = replace(configured, calibration=replace(configured.calibration,
                backend="profile", profile_version="triple-stage", replica_profile_paths=tuple(stage_paths),
                workspace_profile_version="triple-workspace", replica_workspace_profile_paths=tuple(workspace_paths)))
            replay = simulate(calibrated)
            self.assertEqual(replay.requests, baseline.requests)
            self.assertEqual(replay.result["counts"]["completed"], 10)
            self.assertGreater(replay.result["batching"]["mixed_batches_completed"], 0)
            self.assertEqual(replay.result["prefix_cache"]["hit_tokens"], 24)
            self.assertEqual(replay.result["swap"]["completed_transfers_by_direction"], {"D2H": 2, "H2D": 2})
            self.assertEqual(replay.result["swap"]["host_reserved_bytes"], 0)
            self.assertEqual(replay.result["prefix_cache"]["attached_block_references"], 0)
            self.assertEqual(replay.result["prefix_cache"]["private_blocks"], 0)
            self.assertTrue(all(row["used_blocks"] == 0 for row in replay.result["memory"]))
            self.assertTrue(all(row["workspace_source"] == "workspace_profile" for row in replay.result["memory"]))
            self.assertTrue(all(row["workspace_profile_provenance"]["source_kind"] == "test_fixture"
                                for row in replay.result["memory"]))
            wrong = replace(calibrated, calibration=replace(calibrated.calibration,
                replica_workspace_profile_paths=tuple(reversed(workspace_paths))))
            with self.assertRaises(SimulationError) as raised:
                simulate(wrong)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")


if __name__ == "__main__":
    unittest.main()
