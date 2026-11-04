"""Small-message lookup Profiles compose with native PP and serving lifetimes.

Every timing in this module is a declared CPU fixture. Virtual rank placements
exercise accounting and identity checks; they are not hardware validation.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import (SimulationError, calibrate_dataset, load_experiment,
                       predict_experiment, replica_experiment, simulate)
from mlsys_sim.pipeline import PipelineSimulation
from mlsys_sim.pp_communication import PPCommunicationModel, calibrate_pp_communication
from mlsys_sim.pp_transport import estimate_pipeline_transfer
from test_native_pp_transport import dataset as stage_dataset, experiment, record as stage_record
from test_pp_communication_integration import communication_dataset
from test_workspace_integration import fit_workspace, workspace_record


ROOT = Path(__file__).resolve().parents[1]
METHOD = "piecewise_size_lookup_v1"
VERSION = "small-message-fixture"


def lookup_dataset(exp, *, sizes=(128, 256, 512, 1024, 2048), label="small-message", scale=1):
    """Five independently identified observations per placement and size."""
    data = communication_dataset(exp, sizes=sizes, label=label, scale=scale)
    rows = []
    for base in data["records"]:
        for repeat, multiplier in enumerate((.8, .9, 1, 1.1, 1.2)):
            row = deepcopy(base)
            row["record_id"] += f"-repeat-{repeat}"
            row["group_id"] += f"-repeat-{repeat}"
            row["duration_ms"] *= multiplier
            for rank in row["rank_durations_ms"]:
                rank["duration_ms"] *= multiplier
            rows.append(row)
    data["records"] = rows
    return data


def fit_lookup(exp, path, *, data=None, sizes=(128, 256, 512, 1024, 2048), scale=1):
    data = lookup_dataset(exp, sizes=sizes, scale=scale) if data is None else data
    profile = calibrate_pp_communication(data, profile_version=VERSION,
                                        method=METHOD, allow_test_fixture=True)
    path.write_text(json.dumps(profile))
    return replace(exp, calibration=replace(exp.calibration,
        native_pp_communication_profile_path=str(path), native_pp_communication_profile_version=VERSION,
        allow_test_fixture=True))


class SmallMessageIntegrationTests(unittest.TestCase):
    def test_exact_and_interpolated_boundaries_charge_each_tensor_and_pair_max_once(self):
        exp = experiment()
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit_lookup(exp, Path(temporary) / "lookup.json")
            model = PPCommunicationModel(configured)
            for queries, wire, full, exact in (([4], 512, 1024, True), ([3], 384, 768, False)):
                edge = estimate_pipeline_transfer(configured, queries)
                # Two tensors, each with max(P2P rank0, rank1) plus one AllGather.
                expected = 2 * ((.02 + 2e-6 * wire) + (.03 + 3e-6 * full))
                self.assertAlmostEqual(edge["duration_ms"], expected)
                self.assertEqual(edge["source"], "test_fixture_native_pp_communication_profile")
                self.assertFalse(edge["breakdown"]["receiver_wait_used_as_network_service"])
                tensors = edge["breakdown"]["tensors"]
                self.assertEqual([row["name"] for row in tensors], ["hidden_states", "residual"])
                for tensor in tensors:
                    p2p = tensor["p2p_rank_estimates"]
                    self.assertEqual([row["message_bytes"] for row in p2p], [wire, wire])
                    self.assertAlmostEqual(tensor["p2p_ms"], .02 + 2e-6 * wire)
                    gather = tensor["receiver_allgather_estimate"]
                    self.assertEqual(gather["message_bytes"], full)
                    self.assertEqual([row["exact"] for row in p2p], [exact, exact])
                    self.assertEqual(gather["exact"], exact)
                    self.assertEqual(gather["model_method"], METHOD)
                    self.assertFalse(gather["extrapolated"])
                estimate = model.estimate("p2p", wire, 1, tp_rank=1)
                self.assertEqual(estimate["interpolation"], "exact" if exact else "linear_bytes")
                self.assertAlmostEqual(estimate["duration_ms"], .02 + 2e-6 * wire)
            edge = estimate_pipeline_transfer(configured, [4])
            self.assertAlmostEqual(predict_experiment(configured)["isolated_prefill_ms"], 10 + edge["duration_ms"])
            self.assertAlmostEqual(simulate(configured).requests[0]["completion_ms"], 10 + edge["duration_ms"])

    def test_stage_profile_padding_and_native_lookup_do_not_double_charge_communication(self):
        exp = experiment()
        rows = [stage_record(exp, stage, padded=8) for stage in range(2)]
        rows += [stage_record(exp, stage, phase="decode", query=(1,), context=(4,), padded=4)
                 for stage in range(2)]
        timing = calibrate_dataset(stage_dataset(exp, rows), profile_version="step-lookup",
                                   allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configured = fit_lookup(exp, root / "lookup.json")
            path = root / "timing.json"
            path.write_text(json.dumps(timing))
            configured = replace(configured, calibration=replace(configured.calibration,
                backend="profile", profile_path=str(path), profile_version="step-lookup"))
            # Observed padding is eight tokens: each tensor wires 1024 B and gathers 2048 B.
            expected = 2 * ((.02 + 2e-6 * 1024) + (.03 + 3e-6 * 2048))
            self.assertAlmostEqual(predict_experiment(configured)["isolated_prefill_ms"], 6 + expected)
            replay = simulate(configured)
            self.assertAlmostEqual(replay.requests[0]["completion_ms"], 6 + expected)
            edges = [row for row in replay.events if row["event"] == "TRANSFER_START"]
            self.assertEqual(len(edges), 1)
            self.assertAlmostEqual(edges[0]["breakdown"]["duration_ms"], expected)

    def test_placement_identity_and_fixture_source_are_preserved_without_cross_boundary_fallback(self):
        exp = experiment(pp=3)
        data = lookup_dataset(exp)
        for row in data["records"]:
            if row["stage_index"] == 2:
                row["duration_ms"] *= 7
                for rank in row["rank_durations_ms"]:
                    rank["duration_ms"] *= 7
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit_lookup(exp, Path(temporary) / "lookup.json", data=data)
            model = PPCommunicationModel(configured)
            first = model.estimate("p2p", 512, 1, tp_rank=1)
            second = model.estimate("p2p", 512, 2, tp_rank=1)
            self.assertAlmostEqual(second["duration_ms"], 7 * first["duration_ms"])
            self.assertEqual(second["source"], "test_fixture_native_pp_communication_profile")
            self.assertEqual(model.provenance["source_kind"], "test_fixture")
            self.assertAlmostEqual(estimate_pipeline_transfer(configured, [4], stage_index=1)["duration_ms"],
                                   7 * estimate_pipeline_transfer(configured, [4], stage_index=0)["duration_ms"])
            denied = replace(configured, calibration=replace(configured.calibration, allow_test_fixture=False))
            with self.assertRaises(SimulationError) as raised:
                PPCommunicationModel(denied)
            self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
            wrong = replace(configured, infra=replace(configured.infra,
                devices=tuple(reversed(configured.infra.devices))))
            with self.assertRaises(SimulationError) as raised:
                PPCommunicationModel(wrong)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_tp1_uses_full_wire_without_receiver_allgather(self):
        exp = experiment(tp=1)
        with tempfile.TemporaryDirectory() as temporary:
            configured = fit_lookup(exp, Path(temporary) / "lookup.json")
            edge = estimate_pipeline_transfer(configured, [4])
            self.assertAlmostEqual(edge["duration_ms"], 2 * (.01 + 1e-6 * 1024))
            for tensor in edge["breakdown"]["tensors"]:
                self.assertIsNone(tensor["receiver_allgather_estimate"])
                self.assertEqual(len(tensor["p2p_rank_estimates"]), 1)

    def test_discontinuity_and_size_gaps_fail_before_kv_even_with_extrapolation(self):
        exp = experiment()
        cases = []
        jumped = lookup_dataset(exp, sizes=(256, 1024, 2048), label="jump")
        for row in jumped["records"]:
            if row["message_bytes"] >= 1024:
                row["duration_ms"] *= 30
                for rank in row["rank_durations_ms"]:
                    rank["duration_ms"] *= 30
        cases.append(("jump", jumped))
        cases.append(("sparse", lookup_dataset(exp, sizes=(128, 1024, 2048), label="sparse")))
        cases.append(("outside", lookup_dataset(exp, sizes=(4, 8), label="outside")))
        missing = lookup_dataset(exp, label="missing-allgather")
        missing["records"] = [row for row in missing["records"] if row["primitive"] != "allgather"]
        cases.append(("missing", missing))
        with tempfile.TemporaryDirectory() as temporary:
            for label, data in cases:
                configured = fit_lookup(exp, Path(temporary) / f"{label}.json", data=data)
                for allow in (False, True):
                    configured = replace(configured, calibration=replace(configured.calibration,
                        native_pp_communication_allow_extrapolation=allow))
                    with self.subTest(case=label, allow_extrapolation=allow):
                        engine = PipelineSimulation(configured)
                        with self.assertRaises(SimulationError) as raised:
                            engine.run()
                        self.assertEqual(raised.exception.code, "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN")
                        self.assertFalse(engine.memory.allocations)
                        self.assertFalse(engine.waves)
                        self.assertFalse(engine.transfers)
                if label in ("jump", "sparse"):
                    model = PPCommunicationModel(configured)
                    self.assertTrue(model.estimate("p2p", 1024, 1, tp_rank=0)["exact"])
                    self.assertTrue(model.estimate("p2p", 2048, 1, tp_rank=0)["exact"])

    def test_dense_lookup_composes_with_three_profiles_replicas_cache_chunk_swap_and_mixed(self):
        exp = load_experiment(ROOT / "examples/v1_7/qwen3-native-r2-tp2-pp2.json")
        model = exp.strategy.model
        # Include every possible full/wire size for this fixture, not just endpoints.
        full = int(model.hidden_size * model.activation_bytes)
        sizes = sorted({tokens * size for tokens in range(1, exp.strategy.pipeline_transport_max_tokens + 1)
                        for size in (full, full // exp.strategy.tensor_parallel_size)})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            communication_paths = []
            for replica in range(2):
                path = root / f"communication-{replica}.json"
                local = replica_experiment(exp, replica)
                fit_lookup(local, path, sizes=sizes, scale=.01)
                communication_paths.append(str(path))
            configured = replace(exp, calibration=replace(exp.calibration,
                replica_native_pp_communication_profile_paths=tuple(communication_paths),
                native_pp_communication_profile_version=VERSION, allow_test_fixture=True))
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
                    timing.update(record_id=f"timing-{index}", group_id=f"timing-{index}",
                                  workspace_peak_bytes=512 * 1024)
                    stage_rows.append(timing)
                    workspace_rows.append(workspace_record(local, event["stage"], phase=event["phase"],
                        queries=event["query_tokens"], contexts=event["context_lengths"],
                        roles=event.get("request_phases"), peaks=(0, 1024 * (event["stage"] + 1)),
                        identifier=f"workspace-{index}"))
                profile = calibrate_dataset(stage_dataset(local, stage_rows),
                                            profile_version="triple-lookup-stage", allow_test_fixture=True)
                path = root / f"stage-{replica}.json"
                path.write_text(json.dumps(profile))
                stage_paths.append(str(path))
                workspace_path = root / f"workspace-{replica}.json"
                fit_workspace(local, workspace_path, workspace_rows, version="triple-lookup-workspace")
                workspace_paths.append(str(workspace_path))
            calibrated = replace(configured, calibration=replace(configured.calibration,
                backend="profile", profile_version="triple-lookup-stage", replica_profile_paths=tuple(stage_paths),
                workspace_profile_version="triple-lookup-workspace",
                replica_workspace_profile_paths=tuple(workspace_paths)))
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
            edges = [event for event in replay.events if event["event"] == "TRANSFER_START"]
            self.assertTrue(edges)
            for edge in edges:
                self.assertEqual(edge["source"], "test_fixture_native_pp_communication_profile")
                for tensor in edge["breakdown"]["tensors"]:
                    p2p = tensor["p2p_rank_estimates"]
                    wire = p2p[0]["message_bytes"]
                    gather = tensor["receiver_allgather_estimate"]
                    full_bytes = gather["message_bytes"]
                    expected = .01 * ((.02 + 2e-6 * wire) + (.03 + 3e-6 * full_bytes))
                    self.assertAlmostEqual(tensor["duration_ms"], expected)
                    self.assertTrue(all(row["exact"] for row in p2p))
                    self.assertTrue(gather["exact"])
                    self.assertEqual(gather["model_method"], METHOD)


class SmallMessageCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def cli(self, target, *arguments, expected=0):
        output = self.root / target
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", *map(str, arguments), "--out", str(output)],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return output, json.loads((output / "result.json").read_text())

    def test_default_and_explicit_affine_profiles_remain_identical(self):
        path = self.write("affine.json", communication_dataset(experiment()))
        arguments = ("calibrate", "--native-pp-communication", "--dataset", path,
                     "--profile-version", "legacy-affine", "--allow-test-fixture")
        default, default_result = self.cli("default", *arguments)
        explicit, explicit_result = self.cli("explicit", *arguments, "--native-pp-model", "affine")
        self.assertEqual((default / "profile.json").read_bytes(), (explicit / "profile.json").read_bytes())
        self.assertNotIn("profile_method", default_result)
        self.assertEqual(explicit_result.pop("profile_method"), "nonnegative_affine_ols_v1")
        self.assertEqual(default_result, explicit_result)
        profile = json.loads((default / "profile.json").read_text())
        self.assertNotIn("model_options", profile)
        self.assertNotIn("knots", profile["groups"][0])

    def test_cli_lookup_keeps_fixture_provenance_and_all_repetitions(self):
        data = lookup_dataset(experiment())
        path = self.write("lookup.json", data)
        before = path.read_bytes()
        output, result = self.cli("lookup", "calibrate", "--native-pp-communication",
            "--native-pp-model", "size-lookup", "--dataset", path, "--profile-version", VERSION,
            "--allow-test-fixture")
        profile = json.loads((output / "profile.json").read_text())
        self.assertEqual(result["profile_method"], METHOD)
        self.assertEqual(profile["method"], METHOD)
        self.assertEqual(profile["source"]["kind"], "test_fixture")
        self.assertFalse(result["hardware_validated"])
        self.assertEqual(result["training_record_count"], len(data["records"]))
        self.assertTrue(all(knot["sample_count"] == 5 for group in profile["groups"] for knot in group["knots"]))
        self.assertEqual(path.read_bytes(), before)
        heldout = self.write("independent-holdout.json", communication_dataset(
            experiment(), sizes=(384, 768), label="independent-holdout"))
        validated, report = self.cli("validated", "validate", "--native-pp-communication",
            "--profile", output / "profile.json", "--dataset", heldout, "--allow-test-fixture")
        self.assertEqual(report["coverage"]["fraction"], 1)
        self.assertLess(report["metrics"]["mae_ms"], 1e-15)
        self.assertFalse(report["hardware_validated"])
        self.assertTrue((validated / "holdout.json").exists())

    def test_model_flag_rejects_other_calibration_modes_and_preserves_diagnostics(self):
        path = self.write("input.json", lookup_dataset(experiment()))
        before = path.read_bytes()
        for index, flags in enumerate(((), ("--communication",), ("--workspace",),
                                       ("--native-pp-communication", "--communication"),
                                       ("--native-pp-communication", "--workspace"))):
            output, result = self.cli(f"invalid-{index}", "calibrate", *flags,
                "--native-pp-model", "size-lookup", "--dataset", path,
                "--profile-version", VERSION, "--allow-test-fixture", expected=2)
            self.assertEqual(result["error_code"], "INVALID_ARGUMENT")
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "logs.txt").read_text())
            self.assertFalse((output / "profile.json").exists())
        output, result = self.cli("invalid-choice", "calibrate", "--native-pp-communication",
            "--native-pp-model", "automatic", "--dataset", path, "--profile-version", VERSION, expected=2)
        self.assertEqual(result["error_code"], "INVALID_ARGUMENT")
        snapshot = {item.name: item.read_bytes() for item in output.iterdir()}
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", "calibrate",
            "--native-pp-model", "size-lookup", "--dataset", str(path), "--profile-version", VERSION,
            "--out", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2)
        self.assertIn("OUTPUT_EXISTS", process.stdout)
        self.assertEqual(snapshot, {item.name: item.read_bytes() for item in output.iterdir()})
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
