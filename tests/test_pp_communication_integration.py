"""Explicit primitive fits compose with native Profiles and serving lifetimes."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (SimulationError, calibrate_dataset, experiment_from_dict,
                       load_experiment, predict_experiment, profile_identity, replica_experiment,
                       simulate, validate_experiment)
from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.pp_communication import PPCommunicationModel, calibrate_pp_communication
from mlsys_sim.pp_transport import estimate_pipeline_transfer
from test_native_pp_transport import dataset as stage_dataset, experiment, record as stage_record


ROOT = Path(__file__).resolve().parents[1]


def communication_dataset(exp, *, sizes=(4, 1048576), label="integration", scale=1):
    tp, pp = exp.strategy.tensor_parallel_size, exp.strategy.pipeline_parallel_size
    rows = []
    for stage in range(1, pp):
        groups = [("p2p", rank, [(stage - 1) * tp + rank, stage * tp + rank]) for rank in range(tp)]
        if tp > 1:
            groups.append(("allgather", None, [stage * tp + rank for rank in range(tp)]))
        for primitive, rank, participants in groups:
            for size in sizes:
                factor = rank + 1 if rank is not None else 3
                duration = scale * (factor * .01 + factor * 1e-6 * size)
                identifier = f"{label}-{stage}-{primitive}-{rank}-{size}"
                rows.append({"record_id": identifier, "group_id": identifier,
                    "primitive": primitive, "stage_index": stage, "tp_rank": rank,
                    "participants": participants, "message_bytes": size, "duration_ms": duration,
                    "rank_durations_ms": [{"rank": participant, "duration_ms": duration}
                                          for participant in participants], "correctness_errors": 0})
    return {"schema_version": 1, "kind": "native_pp_communication_measurements", "dataset_id": label,
        "identity": profile_identity(exp), "source": {"kind": "test_fixture",
            "reference": "Hand-computed primitive curves, not hardware timings"}, "records": rows,
        "benchmark": {"protocol": "native_pp_primitives_v1",
            "timing_boundary": "synchronized_host_primitive_completion_v1", "p2p_direction": "unidirectional",
            "allgather_bytes": "full_output", "rank_reduction": "maximum_local_duration",
            "barrier_in_timing": False, "warmup_in_records": False, "startup_in_timing": False,
            "allocation_in_timing": False, "buffer_policy": "preallocated_out_of_place",
            "backend": "nccl", "stack": {"fixture": "CPU hand calculations"}, "launch": {}}}


def fit(exp, path, *, scale=1):
    profile = calibrate_pp_communication(communication_dataset(exp, scale=scale),
                                        profile_version="comm-integration", allow_test_fixture=True)
    path.write_text(json.dumps(profile))
    return replace(exp, calibration=replace(exp.calibration,
        native_pp_communication_profile_path=str(path), native_pp_communication_profile_version="comm-integration",
        allow_test_fixture=True))


class PPCommunicationIntegrationTests(unittest.TestCase):
    def test_explicit_selection_keeps_stage_identity_and_old_default_behavior(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit(exp, Path(temporary) / "comm.json")
            validate_experiment(configured)
            self.assertEqual(profile_identity(configured), profile_identity(exp))
            self.assertNotEqual(simulate(configured).requests, simulate(exp).requests)
            for changes in ({"native_pp_communication_profile_path": ""},
                            {"replica_native_pp_communication_profile_paths": ("a", "b")},
                            {"native_pp_communication_profile_version": "wrong"}):
                changed = replace(configured, calibration=replace(configured.calibration, **changes))
                with self.subTest(changes=changes), self.assertRaises(SimulationError):
                    simulate(changed)
            for strategy in (replace(configured.strategy, pipeline_transport="full_hidden"),
                             replace(configured.strategy, pipeline_p2p=False)):
                with self.assertRaises(SimulationError):
                    validate_experiment(replace(configured, strategy=strategy))
        for changes in ({"native_pp_communication_profile_version": "without-file"},
                        {"native_pp_communication_allow_extrapolation": True}):
            with self.assertRaises(SimulationError):
                validate_experiment(replace(exp, calibration=replace(exp.calibration, **changes)))
        with self.assertRaises(SimulationError):
            experiment_from_dict({"calibration": {"native_pp_communication_allow_extrapolation": 1}})

    def test_per_pair_max_then_full_output_allgather_is_charged_once(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit(exp, Path(temporary) / "comm.json")
            edge = estimate_pipeline_transfer(configured, [4])
            expected = 2 * ((.02 + 2e-6 * 512) + (.03 + 3e-6 * 1024))
            self.assertAlmostEqual(edge["duration_ms"], expected)
            self.assertEqual(edge["source"], "test_fixture_native_pp_communication_profile")
            self.assertTrue(edge["breakdown"]["communication_calibrated"])
            self.assertFalse(edge["breakdown"]["receiver_wait_used_as_network_service"])
            self.assertEqual([e["message_bytes"] for e in edge["breakdown"]["tensors"][0]["p2p_rank_estimates"]], [512, 512])
            self.assertEqual(edge["breakdown"]["tensors"][0]["receiver_allgather_estimate"]["message_bytes"], 1024)
            prediction = predict_experiment(configured)
            result = simulate(configured)
            self.assertAlmostEqual(prediction["isolated_prefill_ms"], 10 + expected)
            self.assertAlmostEqual(result.requests[0]["completion_ms"], 10 + expected)
            self.assertEqual(prediction["communication"]["p2p_source"], edge["source"])

    def test_observed_padding_in_stage_profile_controls_both_primitive_sizes(self):
        exp = experiment()
        rows = [stage_record(exp, stage, padded=8) for stage in range(2)]
        rows += [stage_record(exp, stage, phase="decode", query=(1,), context=(4,), padded=4)
                 for stage in range(2)]
        stage_profile = calibrate_dataset(stage_dataset(exp, rows), profile_version="native-step",
                                          allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit(exp, Path(temporary) / "comm.json")
            path = Path(temporary) / "stage.json"
            path.write_text(json.dumps(stage_profile))
            configured = replace(configured, calibration=replace(configured.calibration,
                backend="profile", profile_path=str(path), profile_version="native-step"))
            prediction = predict_experiment(configured)
            expected = 2 * ((.02 + 2e-6 * 1024) + (.03 + 3e-6 * 2048))
            self.assertAlmostEqual(prediction["isolated_prefill_ms"], 6 + expected)
            self.assertAlmostEqual(simulate(configured).requests[0]["completion_ms"], 6 + expected)
            self.assertEqual(prediction["communication"]["prefill_p2p_bytes_per_rank_per_boundary"], 2048)

    def test_missing_group_and_outside_coverage_fail_before_kv_reservation(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comm.json"
            data = communication_dataset(exp, sizes=(4, 8))
            profile = calibrate_pp_communication(data, profile_version="tiny", allow_test_fixture=True)
            path.write_text(json.dumps(profile))
            configured = replace(exp, calibration=replace(exp.calibration,
                native_pp_communication_profile_path=str(path), allow_test_fixture=True))
            engine = PipelineSimulation(configured)
            with self.assertRaises(SimulationError) as raised:
                engine.run()
            self.assertEqual(raised.exception.code, "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN")
            self.assertFalse(engine.memory.allocations)
            self.assertFalse(engine.waves)
            explicit = replace(configured, calibration=replace(configured.calibration,
                native_pp_communication_allow_extrapolation=True))
            self.assertEqual(simulate(explicit).result["counts"]["completed"], 1)
            data = communication_dataset(exp)
            data["records"] = [r for r in data["records"] if r["primitive"] == "p2p"]
            path.write_text(json.dumps(calibrate_pp_communication(data, profile_version="no-ag", allow_test_fixture=True)))
            with self.assertRaises(SimulationError):
                simulate(configured)

    def test_loaded_model_is_immutable_and_identity_rejects_reversed_devices(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comm.json"
            configured = fit(exp, path)
            engine = PipelineSimulation(configured)
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_text("broken after load")
            result = engine.run()
            self.assertEqual(result.result["pipeline"]["communication_profile_provenance"]["profile_sha256"], expected_hash)
            configured = fit(exp, path)
            wrong = replace(configured, infra=replace(configured.infra, devices=tuple(reversed(configured.infra.devices))))
            with self.assertRaises(SimulationError) as raised:
                PPCommunicationModel(wrong)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_native_communication_profiles_combine_with_replica_stage_profiles_and_recovery(self):
        exp = load_experiment(ROOT / "examples/v1_7/qwen3-native-r2-tp2-pp2.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            communication_paths = []
            for replica in range(2):
                local = replica_experiment(exp, replica)
                path = root / f"comm-{replica}.json"
                fit(local, path, scale=.01)
                communication_paths.append(str(path))
            configured = replace(exp, calibration=replace(exp.calibration,
                replica_native_pp_communication_profile_paths=tuple(communication_paths),
                native_pp_communication_profile_version="comm-integration", allow_test_fixture=True))
            baseline = simulate(configured)
            stage_paths = []
            for replica in range(2):
                local = replica_experiment(configured, replica)
                rows = []
                for index, event in enumerate(baseline.events):
                    if event["event"] != "STAGE_START" or event["replica_index"] != replica:
                        continue
                    row = stage_record(local, event["stage"], phase=event["phase"], query=event["query_tokens"],
                        context=event["context_lengths"], padded=sum(event["query_tokens"]),
                        roles=event.get("request_phases"), duration=event["duration_ms"])
                    row.update(record_id=f"fixture-{index}", group_id=f"shape-{index}")
                    rows.append(row)
                profile = calibrate_dataset(stage_dataset(local, rows), profile_version="combo", allow_test_fixture=True)
                path = root / f"stage-{replica}.json"
                path.write_text(json.dumps(profile))
                stage_paths.append(str(path))
            calibrated = replace(configured, calibration=replace(configured.calibration,
                backend="profile", replica_profile_paths=tuple(stage_paths), profile_version="combo"))
            replay = simulate(calibrated)
            self.assertEqual(replay.requests, baseline.requests)
            for result in (baseline, replay):
                self.assertEqual(result.result["counts"]["completed"], 10)
                self.assertGreater(result.result["batching"]["mixed_batches_completed"], 0)
                self.assertEqual(result.result["prefix_cache"]["hit_tokens"], 24)
                self.assertEqual(result.result["swap"]["completed_transfers_by_direction"], {"D2H": 2, "H2D": 2})
                self.assertEqual(result.result["swap"]["host_reserved_bytes"], 0)
                self.assertEqual(result.result["prefix_cache"]["attached_block_references"], 0)
                edges = [e for e in result.events if e["event"] == "TRANSFER_START"]
                self.assertTrue(edges)
                self.assertTrue(all(e["source"] == "test_fixture_native_pp_communication_profile" for e in edges))
            swapped = replace(configured, calibration=replace(configured.calibration,
                replica_native_pp_communication_profile_paths=tuple(reversed(communication_paths))))
            with self.assertRaises(SimulationError):
                simulate(swapped)

    def test_new_communication_paths_resolve_relative_to_calibration_section(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "settings").mkdir()
            calibration = asdict(exp.calibration)
            calibration.update(native_pp_communication_profile_path="comm.json", allow_test_fixture=True)
            (root / "settings/calibration.json").write_text(json.dumps(calibration))
            data = asdict(exp)
            data["calibration"] = "settings/calibration.json"
            (root / "experiment.json").write_text(json.dumps(data))
            loaded = load_experiment(root / "experiment.json")
            self.assertEqual(loaded.calibration.native_pp_communication_profile_path, str((root / "settings/comm.json").resolve()))
            self.assertEqual(exp.calibration.native_pp_communication_profile_path, None)


if __name__ == "__main__":
    unittest.main()
