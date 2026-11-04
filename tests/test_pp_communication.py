"""Standalone native PP fits retain placement, rank proof and exact source."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec, SimulationError, StrategySpec
from mlsys_sim.pp_communication import (PPCommunicationModel, calibrate_pp_communication,
    load_pp_communication_dataset, load_pp_communication_profile, validate_pp_communication_dataset,
    validate_pp_communication_holdout, validate_pp_communication_profile)
from mlsys_sim.profiles import profile_identity


def experiment(tp=2, pp=2):
    model = ModelSpec(name="native-comm-fixture", revision="fixture-revision", num_layers=pp * 2,
        hidden_size=128, num_attention_heads=8, num_kv_heads=2, head_dim=16,
        activation_dtype="float16")
    return Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}") for i in range(tp * pp))),
        strategy=StrategySpec(model=model, tensor_parallel_size=tp, pipeline_parallel_size=pp,
            pipeline_transport="vllm_native", max_batch_tokens=64),
        calibration=CalibrationSpec(allow_test_fixture=True))


def dataset(exp=None, *, sizes=(512, 4096), dataset_id="train", repeats=1):
    exp = exp or experiment()
    tp, pp = exp.strategy.tensor_parallel_size, exp.strategy.pipeline_parallel_size
    records = []
    for stage in range(1, pp):
        placements = [("p2p", rank, [(stage - 1) * tp + rank, stage * tp + rank]) for rank in range(tp)]
        if tp > 1:
            placements.append(("allgather", None, list(range(stage * tp, (stage + 1) * tp))))
        for primitive, rank, participants in placements:
            for size in sizes:
                for repeat in range(repeats):
                    duration = (.010 * stage + .001 * rank + size / (10_000_000.0 * (rank + 1))
                                if primitive == "p2p" else .020 * stage + size / 5_000_000.0)
                    cell = f"{dataset_id}/{primitive}/s{stage}/tp{rank}/bytes{size}"
                    records.append({"record_id": f"{cell}/repeat{repeat}", "group_id": cell,
                        "primitive": primitive, "stage_index": stage, "tp_rank": rank,
                        "participants": participants, "message_bytes": size, "duration_ms": duration,
                        "rank_durations_ms": [{"rank": participant, "duration_ms": duration - index * .0001}
                            for index, participant in enumerate(participants)], "correctness_errors": 0})
    return {"schema_version": 1, "kind": "native_pp_communication_measurements", "dataset_id": dataset_id,
        "identity": profile_identity(exp), "source": {"kind": "test_fixture",
            "reference": "CPU hand-derived operation fits, no GPU or network measurement"},
        "records": records, "benchmark": {
            "protocol": "native_pp_primitives_v1", "timing_boundary": "synchronized_host_primitive_completion_v1",
            "p2p_direction": "unidirectional", "allgather_bytes": "full_output",
            "rank_reduction": "maximum_local_duration", "barrier_in_timing": False,
            "warmup_in_records": False, "startup_in_timing": False, "allocation_in_timing": False,
            "buffer_policy": "preallocated_out_of_place", "backend": "nccl",
            "stack": {"fixture": "CPU hand arithmetic"}, "launch": {"fixture": "no process launched"}}}


def fit(data=None, *, version="comm-v1", parent=None):
    return calibrate_pp_communication(dataset() if data is None else data, profile_version=version,
                                      parent_profile=parent, allow_test_fixture=True)


class NativePPCommunicationFitTests(unittest.TestCase):
    def test_direct_affine_fits_preserve_each_paired_link_and_allgather_full_output(self):
        data = dataset(repeats=2)
        before = deepcopy(data)
        profile = fit(data)
        self.assertEqual(data, before)
        groups = {(g["primitive"], g["stage_index"], g["tp_rank"]): g for g in profile["groups"]}
        self.assertEqual(len(groups), 3)
        for rank in (0, 1):
            group = groups["p2p", 1, rank]
            self.assertAlmostEqual(group["intercept_ms"], .010 + .001 * rank, places=14)
            self.assertAlmostEqual(group["slope_ms_per_byte"], 1 / (10_000_000.0 * (rank + 1)), places=20)
            self.assertEqual(group["participants"], [rank, 2 + rank])
            self.assertEqual(group["coverage"]["record_count"], 4)
        ag = groups["allgather", 1, None]
        self.assertEqual(ag["participants"], [2, 3])
        self.assertAlmostEqual(ag["intercept_ms"], .020, places=14)
        self.assertAlmostEqual(ag["slope_ms_per_byte"], 1 / 5_000_000.0, places=20)
        self.assertNotIn("infra_patch", ag)
        self.assertTrue(profile["application"]["no_local_stage_collectives"])
        self.assertTrue(profile["application"]["no_global_infra_patch"])
        self.assertEqual(validate_pp_communication_profile(profile, allow_test_fixture=True), profile)

    def test_adjacent_stages_do_not_share_fit_parameters(self):
        exp = experiment(tp=1, pp=3)
        profile = fit(dataset(exp))
        self.assertEqual(len(profile["groups"]), 2)
        model = PPCommunicationModel(exp, profile)
        self.assertAlmostEqual(model.estimate("p2p", 1024, 2, tp_rank=0)["duration_ms"]
                             - model.estimate("p2p", 1024, 1, tp_rank=0)["duration_ms"], .01, places=14)

    def test_nonnegative_intercept_refits_boundary_solution(self):
        data = dataset(sizes=(2, 4, 6))
        data["records"] = [r for r in data["records"] if r["primitive"] == "p2p" and r["tp_rank"] == 0]
        for row, duration in zip(data["records"], (1.0, 3.0, 5.0)):
            row["duration_ms"] = duration
            row["rank_durations_ms"] = [{"rank": rank, "duration_ms": duration} for rank in row["participants"]]
        group = fit(data)["groups"][0]
        self.assertEqual(group["intercept_ms"], 0)
        self.assertTrue(group["intercept_clamped"])
        self.assertAlmostEqual(group["slope_ms_per_byte"], (2 * 1 + 4 * 3 + 6 * 5) / (2**2 + 4**2 + 6**2))

    def test_each_group_requires_two_sizes_and_a_positive_slope(self):
        for data in (dataset(sizes=(512, 512)), dataset()):
            if len(set(r["message_bytes"] for r in data["records"])) > 1:
                for row in data["records"]:
                    row["duration_ms"] = 1
                    row["rank_durations_ms"] = [{"rank": rank, "duration_ms": 1} for rank in row["participants"]]
            with self.assertRaises(SimulationError):
                fit(data)

    def test_parent_version_lineage_never_reuses_samples_or_parameters(self):
        parent = fit()
        before = deepcopy(parent)
        data = dataset(dataset_id="new-capture")
        child = fit(data, version="comm-v2", parent=parent)
        self.assertEqual(parent, before)
        self.assertEqual(child["training"]["parent_profile_version"], "comm-v1")
        self.assertFalse(child["application"]["parent_parameters_reused"])
        self.assertEqual(child["training"]["record_ids"], sorted(r["record_id"] for r in data["records"]))
        with self.assertRaises(SimulationError) as raised:
            fit(data, version="comm-v1", parent=parent)
        self.assertEqual(raised.exception.code, "PROFILE_VERSION_CONFLICT")
        other = deepcopy(data)
        other["benchmark"]["stack"]["fixture"] = "other build"
        with self.assertRaises(SimulationError):
            fit(other, version="comm-v2", parent=parent)

    def test_profile_rejects_corrupt_parameters_coverage_and_training(self):
        good = fit()
        mutations = [
            lambda p: p["groups"][0].update(slope_ms_per_byte=0),
            lambda p: p["groups"][0].update(intercept_ms=-1),
            lambda p: p["groups"][0].update(intercept_clamped=True),
            lambda p: p["groups"][0]["coverage"].update(distinct_message_sizes=5),
            lambda p: p["groups"][0]["coverage"].update(record_count=True),
            lambda p: p["groups"][0]["coverage"].update(message_bytes_min=4096),
            lambda p: p["groups"].append(deepcopy(p["groups"][0])),
            lambda p: p["training"]["record_ids"].pop(),
            lambda p: p["application"].update(no_global_infra_patch=False),
            lambda p: p.update(kind="communication_profile"),
        ]
        for mutate in mutations:
            profile = deepcopy(good)
            mutate(profile)
            with self.subTest(mutation=mutate), self.assertRaises(SimulationError):
                validate_pp_communication_profile(profile, allow_test_fixture=True)


class NativePPCommunicationValidationTests(unittest.TestCase):
    def test_fixture_permission_source_and_benchmark_boundary_are_strict(self):
        with self.assertRaises(SimulationError) as raised:
            validate_pp_communication_dataset(dataset())
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        with self.assertRaises(SimulationError):
            validate_pp_communication_profile(fit())
        for field, value in (("p2p_direction", "pingpong_half"), ("allgather_bytes", "per_rank_input"),
                             ("barrier_in_timing", True), ("warmup_in_records", True),
                             ("allocation_in_timing", True), ("startup_in_timing", 0),
                             ("buffer_policy", "allocate_each_iteration"), ("backend", "gloo")):
            data = dataset()
            data["benchmark"][field] = value
            with self.subTest(field=field), self.assertRaises(SimulationError):
                validate_pp_communication_dataset(data, allow_test_fixture=True)

    def test_placement_bytes_correctness_and_rank_reduction_proof_are_required(self):
        mutations = [
            lambda r: r.update(participants=[2, 0]),
            lambda r: r.update(tp_rank=True),
            lambda r: r.update(stage_index=0),
            lambda r: r.update(message_bytes=3),
            lambda r: r.update(duration_ms=r["duration_ms"] + .001),
            lambda r: r.update(correctness_errors=True),
            lambda r: r.update(correctness_errors=1),
            lambda r: r["rank_durations_ms"].pop(),
            lambda r: r["rank_durations_ms"][0].update(rank=1),
            lambda r: r["rank_durations_ms"][0].update(duration_ms=True),
        ]
        for mutate in mutations:
            data = dataset()
            mutate(data["records"][0])
            with self.subTest(mutation=mutate), self.assertRaises(SimulationError):
                validate_pp_communication_dataset(data, allow_test_fixture=True)
        for size in (2, 5):
            data = dataset()
            ag = next(r for r in data["records"] if r["primitive"] == "allgather")
            ag["message_bytes"] = size
            with self.assertRaises(SimulationError):
                validate_pp_communication_dataset(data, allow_test_fixture=True)

    def test_schema_unknown_fields_duplicates_and_nonfinite_data_are_rejected(self):
        for value in (True, float("inf"), float("nan"), 10**1000):
            data = dataset()
            data["records"][0]["duration_ms"] = value
            with self.subTest(value=value), self.assertRaises(SimulationError):
                validate_pp_communication_dataset(data, allow_test_fixture=True)
        data = dataset()
        data["records"].append(deepcopy(data["records"][0]))
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data, allow_test_fixture=True)
        data = dataset()
        data["records"][0]["some_extra"] = 1
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data, allow_test_fixture=True)
        data = dataset()
        data["benchmark"]["launch"]["bad"] = float("inf")
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data, allow_test_fixture=True)

    def test_non_native_multi_replica_or_implicit_dtype_identity_is_rejected(self):
        data = dataset()
        data["identity"] = profile_identity(Experiment())
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data, allow_test_fixture=True)
        data = dataset()
        data["identity"]["parallelism"]["replicas"] = 2
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data, allow_test_fixture=True)
        data = dataset()
        data["identity"]["model"]["activation_dtype"] = "auto"
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data, allow_test_fixture=True)

    def test_measured_stack_needs_loaded_rccl_hash_and_matching_identity(self):
        base = experiment()
        exp = replace(base, infra=replace(base.infra,
                devices=tuple(replace(d, name="AMD schema-only device") for d in base.infra.devices)),
            strategy=replace(base.strategy, model=replace(base.strategy.model,
                weight_dtype="float16", kv_dtype="float16")),
            calibration=replace(base.calibration, runtime_name="vllm", runtime_version="declared-vllm",
                driver_version="declared-driver", rocm_version="declared-rocm", rccl_version="declared-rccl"))
        data = dataset(exp)
        data["source"] = {"kind": "measured", "reference": "Schema checks only; this is not a hardware run"}
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data)
        data["benchmark"]["stack"] = {"torch_version": "declared-torch", "hip_version": "declared-hip",
            "rccl_version": "different-rccl", "rccl_library_sha256": "a" * 64,
            "communication_environment_sha256": "b" * 64}
        with self.assertRaises(SimulationError) as raised:
            validate_pp_communication_dataset(data)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        data["benchmark"]["stack"]["rccl_version"] = "declared-rccl"
        self.assertEqual(validate_pp_communication_dataset(data)["source"]["kind"], "measured")
        data["benchmark"]["stack"]["rccl_library_sha256"] = "A" * 64
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data)
        data["benchmark"]["stack"]["rccl_library_sha256"] = "a" * 64
        data["benchmark"]["stack"].pop("communication_environment_sha256")
        with self.assertRaises(SimulationError):
            validate_pp_communication_dataset(data)

    def test_bounded_json_loader_preserves_input_and_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            raw = json.dumps(dataset()).encode()
            path.write_bytes(raw)
            self.assertEqual(load_pp_communication_dataset(path, allow_test_fixture=True)["dataset_id"], "train")
            self.assertEqual(path.read_bytes(), raw)
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(SimulationError):
                load_pp_communication_dataset(path, allow_test_fixture=True)
            path.write_text('{"value":NaN}')
            with self.assertRaises(SimulationError):
                load_pp_communication_profile(path, allow_test_fixture=True)
            with self.assertRaises(SimulationError):
                load_pp_communication_profile(Path(directory) / "missing")


class NativePPCommunicationLookupTests(unittest.TestCase):
    def test_model_preserves_bytes_hash_and_private_parameters(self):
        exp = experiment()
        profile = fit(dataset(exp))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            raw = (json.dumps(profile, indent=2) + "\n").encode()
            path.write_bytes(raw)
            configured = replace(exp, calibration=replace(exp.calibration,
                native_pp_communication_profile_path=str(path), native_pp_communication_profile_version="comm-v1"))
            model = PPCommunicationModel(configured)
            self.assertEqual(model.provenance["profile_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(model.provenance["sha256_scope"], "file_bytes")
            self.assertEqual(load_pp_communication_profile(path, allow_test_fixture=True), profile)
            path.write_text("corrupt after initial loading")
            result = model.estimate("p2p", 1024, 1, tp_rank=1)
            self.assertAlmostEqual(result["duration_ms"], .011 + 1024 / 20_000_000, places=14)
            self.assertEqual(result["source"], "test_fixture_native_pp_communication_profile")
            self.assertFalse(result["extrapolated"])
            self.assertEqual(result["provenance"]["profile_sha256"], model.provenance["profile_sha256"])
            self.assertNotIn("benchmark", result["provenance"])
            self.assertNotIn("application", result["provenance"])
            self.assertIn("benchmark", model.provenance)
        model = PPCommunicationModel(exp, profile)
        profile["groups"][0]["intercept_ms"] = 1000
        self.assertLess(model.estimate("allgather", 1024, 1)["duration_ms"], 1)

    def test_model_rejects_identity_version_fixture_gate_and_legacy_profile(self):
        exp = experiment()
        profile = fit(dataset(exp))
        for configured in (
            replace(exp, calibration=replace(exp.calibration, allow_test_fixture=False)),
            replace(exp, calibration=replace(exp.calibration, native_pp_communication_profile_version="wrong")),
            replace(exp, infra=replace(exp.infra, devices=tuple(replace(d, name="different") for d in exp.infra.devices))),
        ):
            with self.assertRaises(SimulationError):
                PPCommunicationModel(configured, profile)
        bad = deepcopy(profile)
        bad["kind"] = "communication_profile"
        with self.assertRaises(SimulationError):
            PPCommunicationModel(exp, bad)

    def test_lookup_never_falls_back_across_placements_or_outside_coverage(self):
        exp = experiment()
        model = PPCommunicationModel(exp, fit(dataset(exp)))
        for primitive, size, stage, rank in (("p2p", 2, 1, 0), ("p2p", 8192, 1, 0),
                                           ("allgather", 2, 1, None), ("p2p", 1024, 2, 0),
                                           ("p2p", 1024, True, 0), ("p2p", 1024, "1", 0),
                                           ("p2p", 1024, 1, None)):
            with self.subTest(primitive=primitive, size=size, stage=stage), self.assertRaises(SimulationError) as raised:
                model.estimate(primitive, size, stage, tp_rank=rank)
            self.assertEqual(raised.exception.code, "OUT_OF_COMMUNICATION_CALIBRATION_DOMAIN")
        partial = dataset(exp)
        partial["records"] = [r for r in partial["records"] if r["primitive"] == "allgather"]
        model = PPCommunicationModel(exp, fit(partial))
        with self.assertRaises(SimulationError):
            model.estimate("p2p", 1024, 1, tp_rank=0)

    def test_explicit_extrapolation_remains_labeled_and_group_specific(self):
        exp = experiment()
        exp = replace(exp, calibration=replace(exp.calibration, native_pp_communication_allow_extrapolation=True))
        result = PPCommunicationModel(exp, fit(dataset(exp))).estimate("p2p", 8192, 1, tp_rank=0)
        self.assertTrue(result["extrapolated"])
        self.assertAlmostEqual(result["duration_ms"], .01 + 8192 / 10_000_000, places=14)


class NativePPCommunicationHoldoutTests(unittest.TestCase):
    def test_independent_interior_holdout_reports_errors_without_hardware_pass(self):
        profile = fit()
        held = dataset(sizes=(1024, 2048), dataset_id="holdout")
        report = validate_pp_communication_holdout(profile, held, allow_test_fixture=True)
        self.assertEqual(report["coverage"]["fraction"], 1)
        self.assertEqual(report["coverage"]["evaluated_records"], 6)
        self.assertLess(report["metrics"]["mae_ms"], 1e-15)
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["source"]["kind"], "test_fixture")
        self.assertTrue(all(r["status"] == "EVALUATED" for r in report["records"]))

    def test_holdout_rejects_record_or_group_leakage_and_stack_mismatch(self):
        profile = fit()
        for mutate in (
            lambda d: d["records"][0].update(record_id=profile["training"]["record_ids"][0]),
            lambda d: d["records"][0].update(group_id=profile["training"]["group_ids"][0]),
            lambda d: d["benchmark"]["stack"].update(fixture="different software"),
        ):
            data = dataset(sizes=(1024,), dataset_id="holdout")
            mutate(data)
            with self.assertRaises(SimulationError):
                validate_pp_communication_holdout(profile, data, allow_test_fixture=True)

    def test_partial_coverage_keeps_rejections_instead_of_filling_zero(self):
        report = validate_pp_communication_holdout(fit(), dataset(sizes=(1024, 8192), dataset_id="holdout"),
                                                  allow_test_fixture=True)
        self.assertEqual(report["coverage"]["evaluated_records"], 3)
        self.assertEqual(report["coverage"]["rejected_records"], 3)
        self.assertEqual(report["coverage"]["fraction"], .5)
        self.assertTrue(all(r["predicted_ms"] is None for r in report["records"] if r["status"] == "OUT_OF_DOMAIN"))
        report = validate_pp_communication_holdout(fit(), dataset(sizes=(8192,), dataset_id="extra"),
            allow_test_fixture=True, allow_extrapolation=True)
        self.assertTrue(all(r["extrapolated"] for r in report["records"]))


if __name__ == "__main__":
    unittest.main()
