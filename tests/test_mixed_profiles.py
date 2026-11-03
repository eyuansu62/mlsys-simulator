"""Joint mixed timing, explicit per-member roles and leakage-safe observations."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (AnalyticalModel, CalibrationSpec, DeviceSpec, Experiment,
                       InfraSpec, ProfileModel, SimulationError, StrategySpec,
                       calibrate_dataset, estimate_profile, profile_identity,
                       split_dataset, validate_holdout, write_json_exclusive)
from mlsys_sim.layout import layer_parameters
from mlsys_sim.profiles import shape_key, validate_dataset, validate_profile


def record(name, *, queries=(1, 4), contexts=(8, 0), roles=("decode", "prefill"),
           duration=7, workspace=200, stage=0, phase="mixed", group=None):
    result = {"record_id": name, "group_id": group or name, "phase": phase,
              "stage_index": stage, "query_tokens": list(queries),
              "context_lengths": list(contexts), "duration_ms": duration,
              "workspace_peak_bytes": workspace}
    if roles is not None:
        result["request_phases"] = list(roles)
    return result


def dataset(records, exp=None):
    return {"schema_version": 1, "dataset_id": "mixed-stage-fixture",
            "identity": profile_identity(exp or Experiment()),
            "source": {"kind": "test_fixture", "reference": "Hand calculations, not GPU measurements"},
            "records": records}


def fit(records, exp=None):
    return calibrate_dataset(dataset(records, exp), profile_version="mixed-v1", allow_test_fixture=True)


def estimate(profile, queries=(1, 4), contexts=(8, 0), roles=("decode", "prefill"), **kwargs):
    return estimate_profile(profile, "mixed", queries, contexts, request_phases=roles,
                            allow_test_fixture=True, **kwargs)


class MixedPerformanceTests(unittest.TestCase):
    def test_joint_analytical_batch_reuses_weights_and_collectives_once(self):
        exp = Experiment(infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
                         strategy=StrategySpec(tensor_parallel_size=2, mixed_batching=True))
        model = AnalyticalModel(exp)
        mixed = model.estimate_step("mixed", [1, 4], [8, 0], request_phases=["decode", "prefill"])
        decode = model.estimate_step("decode", [1], [8])
        prefill = model.estimate_step("prefill", [4], [0])
        parts, left, right = mixed.breakdown, decode.breakdown, prefill.breakdown
        for component in ("linear", "attention", "misc"):
            self.assertEqual(parts[component]["flops_per_rank"],
                             left[component]["flops_per_rank"] + right[component]["flops_per_rank"])
        one_weight_read = (exp.strategy.model.num_layers
                           * layer_parameters(exp.strategy.model, 2)["linear"]
                           * exp.strategy.model.weight_bytes)
        self.assertEqual(left["linear"]["hbm_bytes_per_rank"] + right["linear"]["hbm_bytes_per_rank"]
                         - parts["linear"]["hbm_bytes_per_rank"], one_weight_read)
        self.assertEqual(parts["communication"]["allreduce_count"], 8)
        self.assertEqual(parts["communication"]["allreduce_payload_bytes"], 5 * 128 * 2)
        self.assertLess(mixed.duration_ms, decode.duration_ms + prefill.duration_ms)
        self.assertEqual(parts["batch"]["request_phases"], ["decode", "prefill"])
        self.assertTrue(parts["batch"]["shared_macro_execution"])

    def test_synthetic_mixed_duration_is_independent_and_stage_proportional(self):
        exp = Experiment(infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
                         strategy=StrategySpec(pipeline_parallel_size=2, stage_layers=(1, 3)),
                         calibration=CalibrationSpec(backend="synthetic", prefill_ms=2,
                                                      decode_ms=3, mixed_ms=14))
        model = AnalyticalModel(exp)
        self.assertEqual([model.estimate_step("mixed", [1, 4], [8, 0], stage,
            request_phases=["decode", "prefill"]).duration_ms for stage in range(2)], [3.5, 10.5])
        self.assertEqual(model.estimate_step("prefill", [4], [0]).duration_ms, .5)
        self.assertEqual(model.estimate_step("decode", [1], [8]).duration_ms, .75)

    def test_invalid_role_vectors_are_rejected_by_analytical_model(self):
        model = AnalyticalModel(Experiment())
        for roles in (None, "decode,prefill", [], ["decode"], ["decode", "decode"],
                      ["prefill", "prefill"], ["decode", "recompute"], [[], "prefill"],
                      [True, "prefill"]):
            with self.subTest(roles=roles), self.assertRaises(SimulationError) as raised:
                model.estimate_step("mixed", [1, 4], [8, 0], request_phases=roles)
            self.assertEqual(raised.exception.code, "INVALID_CONFIG")
        with self.assertRaises(SimulationError):
            model.estimate_step("mixed", [2, 4], [8, 0], request_phases=["decode", "prefill"])
        with self.assertRaises(SimulationError):
            model.estimate_step("prefill", [4], [0], request_phases=["prefill"])


class MixedProfileTests(unittest.TestCase):
    def test_legacy_keys_are_unchanged_and_mixed_roles_extend_identity(self):
        self.assertEqual(shape_key("prefill", [4], [0]), '["prefill",0,[4],[0]]')
        key = shape_key("mixed", [1, 1], [4, 4], request_phases=["decode", "prefill"])
        other = shape_key("mixed", [1, 1], [4, 4], request_phases=["prefill", "decode"])
        self.assertEqual(json.loads(key), ["mixed", 0, [1, 1], [4, 4], ["decode", "prefill"]])
        self.assertNotEqual(key, other)

    def test_one_token_prefill_and_decode_are_distinct_exact_samples(self):
        profile = fit([record("a", queries=(1, 1), contexts=(4, 4), duration=7),
                       record("b", queries=(1, 1), contexts=(4, 4), roles=("prefill", "decode"), duration=11)])
        self.assertEqual(len(profile["samples"]), 2)
        self.assertEqual(estimate(profile, (1, 1), (4, 4)).duration_ms, 7)
        self.assertEqual(estimate(profile, (1, 1), (4, 4), ("prefill", "decode")).duration_ms, 11)
        self.assertEqual(estimate(profile, (1, 1), (4, 4)).breakdown["method"], "exact")

    def test_no_interpolation_extrapolation_or_homogeneous_fallback(self):
        profile = fit([record("low", queries=(1, 2)), record("high", queries=(1, 6)),
                       record("prefill", phase="prefill", queries=(1, 4), contexts=(8, 0), roles=None)])
        for extrapolation in (False, True):
            for queries in ((1, 4), (1, 8)):
                with self.subTest(extrapolation=extrapolation, queries=queries), self.assertRaises(SimulationError) as raised:
                    estimate(profile, queries, allow_extrapolation=extrapolation)
                self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_roles_query_context_and_stage_must_all_match(self):
        exp = Experiment(infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
                         strategy=StrategySpec(pipeline_parallel_size=2))
        profile = fit([record("a", duration=5), record("b", duration=9, stage=1)], exp)
        self.assertEqual(estimate(profile, stage_index=0).duration_ms, 5)
        self.assertEqual(estimate(profile, stage_index=1).duration_ms, 9)
        for queries, contexts, roles in [((1, 4), (9, 0), ("decode", "prefill")),
                                         ((1, 1), (8, 0), ("decode", "prefill")),
                                         ((1, 1), (8, 0), ("prefill", "decode"))]:
            with self.subTest(queries=queries, contexts=contexts, roles=roles), self.assertRaises(SimulationError):
                estimate(profile, queries, contexts, roles)

    def test_regular_mixed_query_shape_does_not_interpolate_context_grid(self):
        profile = fit([record("low", queries=(1, 1), contexts=(4, 4)),
                       record("high", queries=(1, 1), contexts=(8, 8))])
        for extrapolation in (False, True):
            with self.subTest(extrapolation=extrapolation), self.assertRaises(SimulationError) as raised:
                estimate(profile, (1, 1), (6, 6), allow_extrapolation=extrapolation)
            self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_profile_model_workspace_uses_same_mixed_shape(self):
        exp = Experiment()
        profile = fit([record("mixed", workspace=9876)], exp)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            write_json_exclusive(path, profile)
            exp = replace(exp, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                                           allow_test_fixture=True))
            model = ProfileModel(exp)
            self.assertEqual(model.estimate_step("mixed", [1, 4], [8, 0],
                request_phases=["decode", "prefill"]).duration_ms, 7)
            self.assertEqual(model.workspace_bytes([1, 4], "mixed", 0, [8, 0],
                request_phases=["decode", "prefill"]), 9876)
            with self.assertRaises(SimulationError):
                model.workspace_bytes([1, 4], "mixed", request_phases=["decode", "prefill"])
            with self.assertRaises(SimulationError):
                model.workspace_bytes([1], "decode", request_phases=["decode"])

    def test_malformed_role_metadata_is_rejected_for_both_evidence_contracts(self):
        for roles in (None, "decode,prefill", [], ["decode"], ["decode", "decode"],
                      ["decode", "recompute"], [[], "prefill"], [True, "prefill"]):
            row = record("bad")
            row["request_phases"] = roles
            with self.subTest(roles=roles), self.assertRaises(SimulationError) as raised:
                validate_dataset(dataset([row]), allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "INVALID_DATASET")
        missing = record("missing", roles=None)
        with self.assertRaises(SimulationError):
            fit([missing])
        for role_metadata in (None, ["prefill"]):
            row = record("homogeneous", phase="prefill", queries=(4,), contexts=(0,), roles=None)
            row["request_phases"] = role_metadata
            with self.subTest(metadata=role_metadata), self.assertRaises(SimulationError):
                fit([row])
        profile = fit([record("valid")])
        broken = deepcopy(profile)
        broken["samples"][0]["request_phases"] = ["decode", "decode"]
        with self.assertRaises(SimulationError):
            validate_profile(broken, allow_test_fixture=True)


class MixedCalibrationTests(unittest.TestCase):
    def test_repeats_aggregate_by_ordered_roles_and_preserve_input(self):
        data = dataset([record("a", queries=(1, 1), contexts=(4, 4), duration=6, workspace=100),
                        record("b", queries=(1, 1), contexts=(4, 4), duration=10, workspace=200),
                        record("c", queries=(1, 1), contexts=(4, 4), roles=("prefill", "decode"), duration=12)])
        before = deepcopy(data)
        profile = calibrate_dataset(data, profile_version="role-fit", allow_test_fixture=True)
        self.assertEqual(data, before)
        self.assertEqual(len(profile["samples"]), 2)
        sample = next(row for row in profile["samples"] if row["request_phases"] == ["decode", "prefill"])
        self.assertEqual((sample["duration_ms"], sample["workspace_peak_bytes"], sample["sample_count"]), (8, 200, 2))
        self.assertEqual(profile["source"]["kind"], "test_fixture")
        self.assertEqual(len(set(profile["training"]["shape_keys"])), 2)

    def test_split_keeps_repeated_roles_together_and_other_roles_independent(self):
        data = dataset([record("a", queries=(1, 1), contexts=(4, 4), group="same-a"),
                        record("b", queries=(1, 1), contexts=(4, 4), group="same-b"),
                        record("c", queries=(1, 1), contexts=(4, 4), roles=("prefill", "decode"), group="other")])
        training, heldout = split_dataset(data, holdout_groups=["same-a"], allow_test_fixture=True)
        self.assertEqual([row["record_id"] for row in heldout["records"]], ["a", "b"])
        self.assertEqual([row["record_id"] for row in training["records"]], ["c"])
        report = validate_holdout(fit(training["records"]), heldout, allow_test_fixture=True,
                                  allow_extrapolation=True)
        self.assertEqual(report["summary"]["rejected_count"], 2)
        self.assertEqual(report["summary"]["coverage"], 0)
        self.assertTrue(report["test_fixture_only"])
        self.assertEqual(report["hardware_validation"]["status"], "not_asserted")

    def test_same_role_shape_leakage_is_detected_even_with_new_ids_and_group(self):
        profile = fit([record("train", group="train")])
        with self.assertRaises(SimulationError) as raised:
            validate_holdout(profile, dataset([record("test", group="test")]), allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")


if __name__ == "__main__":
    unittest.main()
