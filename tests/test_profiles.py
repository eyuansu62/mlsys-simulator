"""Profile evidence, rectangular interpolation and strict identity regressions."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (DeviceSpec, Experiment, InfraSpec,
                       RequestSpec, SimulationError, StrategySpec)
from mlsys_sim.performance import AnalyticalModel, predict_experiment
from mlsys_sim.profiles import (ProfileModel, create_performance_model, estimate_profile,
                                load_dataset, load_profile, profile_identity, shape_key,
                                validate_dataset, validate_identity, validate_profile)


def observation(phase="prefill", queries=(4,), contexts=(0,), stage=0, duration=10.0,
                workspace=100):
    return {"phase": phase, "stage_index": stage, "query_tokens": list(queries),
            "context_lengths": list(contexts), "duration_ms": duration,
            "workspace_peak_bytes": workspace, "sample_count": 1}


def profile(samples=None, experiment=None, source="test_fixture"):
    samples = samples if samples is not None else [observation()]
    return {"schema_version": 1, "profile_id": "unit-profile", "profile_version": "v1",
            "identity": profile_identity(experiment or Experiment()),
            "source": {"kind": source, "reference": "unit-test-only fixture, not hardware evidence"},
            "method": "lookup", "timing_includes_communication": True, "samples": samples,
            "training": {"dataset_sha256": "a" * 64,
                         "record_ids": [f"r{i}" for i in range(len(samples))],
                         "group_ids": [f"g{i}" for i in range(len(samples))],
                         "shape_keys": [shape_key(row["phase"], row["query_tokens"],
                                                  row["context_lengths"], row["stage_index"])
                                        for row in samples], "parent_profile_version": None}}


def fixture_estimate(data, phase="prefill", queries=(4,), contexts=(0,), stage=0, **kwargs):
    return estimate_profile(data, phase, queries, contexts, stage,
                            allow_test_fixture=True, **kwargs)


class ProfileLookupTests(unittest.TestCase):
    def test_exact_ragged_lookup_does_not_collapse_total_tokens(self):
        data = profile([observation(queries=(2, 6), contexts=(0, 0), duration=17)])
        result = fixture_estimate(data, queries=(2, 6), contexts=(0, 0))
        self.assertEqual(result.duration_ms, 17)
        self.assertEqual(result.breakdown["method"], "exact")
        self.assertEqual(result.source, "test_fixture_profile")
        with self.assertRaises(SimulationError) as raised:
            fixture_estimate(data, queries=(4, 4), contexts=(0, 0))
        self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_multilinear_prefill_grid_and_conservative_workspace(self):
        samples = [observation(queries=(query,) * batch, contexts=(0,) * batch,
                               duration=2 * batch + 3 * query,
                               workspace=100 * batch + 10 * query)
                   for batch, query in product([1, 3], [2, 6])]
        result = fixture_estimate(profile(samples), queries=(4, 4), contexts=(0, 0))
        self.assertAlmostEqual(result.duration_ms, 16)
        self.assertEqual(result.breakdown["method"], "interpolation")
        self.assertFalse(result.breakdown["extrapolated"])
        self.assertEqual(result.breakdown["workspace_peak_bytes"], 360)

    def test_decode_interpolates_batch_and_actual_context(self):
        samples = [observation("decode", (1,) * batch, (context,) * batch,
                               duration=2 * batch + 3 + 5 * context)
                   for batch, context in product([1, 3], [2, 6])]
        self.assertEqual(fixture_estimate(profile(samples), "decode", (1, 1), (4, 4)).duration_ms, 27)

    def test_missing_corner_is_not_nearest_neighbour(self):
        samples = [observation(queries=(query,) * batch, contexts=(0,) * batch)
                   for batch, query in [(1, 2), (1, 6), (3, 2)]]
        with self.assertRaises(SimulationError) as raised:
            fixture_estimate(profile(samples), queries=(4, 4), contexts=(0, 0))
        self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_extrapolation_requires_flag_and_has_unknown_workspace(self):
        data = profile([observation(queries=(2,), duration=4), observation(queries=(6,), duration=12)])
        with self.assertRaises(SimulationError):
            fixture_estimate(data, queries=(8,))
        result = fixture_estimate(data, queries=(8,), allow_extrapolation=True)
        self.assertEqual(result.duration_ms, 16)
        self.assertTrue(result.breakdown["extrapolated"])
        self.assertIsNone(result.breakdown["workspace_peak_bytes"])
        with self.assertRaises(SimulationError):
            fixture_estimate(profile(), queries=(8,), allow_extrapolation=True)

    def test_nonpositive_extrapolation_is_rejected(self):
        data = profile([observation(queries=(2,), duration=4), observation(queries=(6,), duration=1)])
        with self.assertRaises(SimulationError) as raised:
            fixture_estimate(data, queries=(10,), allow_extrapolation=True)
        self.assertEqual(raised.exception.code, "OUT_OF_CALIBRATION_DOMAIN")

    def test_fixture_requires_explicit_permission_in_public_entry_points(self):
        data = profile()
        with self.assertRaises(SimulationError) as raised:
            estimate_profile(data, "prefill", [4], [0])
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        with self.assertRaises(SimulationError):
            validate_profile(data)
        self.assertEqual(validate_profile(data, allow_test_fixture=True)["source"]["kind"], "test_fixture")

    def test_source_and_domain_overrides_must_be_actual_booleans(self):
        with self.assertRaises(SimulationError):
            validate_profile(profile(), allow_test_fixture="false")
        with self.assertRaises(SimulationError):
            fixture_estimate(profile(), allow_extrapolation="false")


class ProfileValidationTests(unittest.TestCase):
    def test_identity_validation_rejects_partial_identity(self):
        with self.assertRaises(SimulationError):
            validate_identity({"model": "fixture"})
        identity = profile_identity(Experiment())
        self.assertEqual(validate_identity(identity), identity)
        with self.assertRaises(SimulationError):
            validate_identity(identity, measured=True)

    def test_measured_sources_require_explicit_hardware_model_and_runtime_identity(self):
        data = profile(source="measured")
        with self.assertRaises(SimulationError):
            validate_profile(data)
        identity = data["identity"]
        identity["model"].update(revision="unit-test-revision", weight_dtype="float16",
                                  kv_dtype="float16", activation_dtype="float16")
        identity["devices"][0]["name"] = "unit-test-explicit-device"
        identity["runtime"].update(name="unit-runtime", version="1.0", driver_version="1.2.3",
                                    rocm_version="not-applicable", rccl_version="not-applicable")
        self.assertEqual(validate_profile(data)["source"]["kind"], "measured")
        validated_identity = validate_identity(identity, measured=True)
        self.assertEqual(validated_identity["runtime"]["rccl_version"], "not-applicable")
        paths = [("model", "revision"), ("runtime", "name"), ("runtime", "version"),
                 ("runtime", "driver_version"), ("runtime", "rocm_version"), ("runtime", "rccl_version")]
        for section, field in paths:
            invalid = deepcopy(data)
            invalid["identity"][section][field] = "unspecified"
            with self.subTest(field=field), self.assertRaises(SimulationError):
                validate_profile(invalid)
        invalid = deepcopy(data)
        invalid["identity"]["devices"][0]["name"] = "toy-device"
        with self.assertRaises(SimulationError):
            validate_profile(invalid)
        row = {key: value for key, value in observation().items() if key != "sample_count"}
        row.update(record_id="measured-record", group_id="measured-config")
        dataset = {"schema_version": 1, "dataset_id": "measurement", "identity": identity,
                   "source": data["source"], "records": [row]}
        self.assertEqual(validate_dataset(dataset)["source"]["kind"], "measured")
        dataset["identity"] = profile_identity(Experiment())
        with self.assertRaises(SimulationError) as raised:
            validate_dataset(dataset)
        self.assertEqual(raised.exception.code, "INVALID_DATASET")

    def test_measured_identity_cannot_leave_equal_width_dtypes_implicit(self):
        identity = profile_identity(Experiment())
        identity["model"].update(revision="unit-revision", weight_dtype="float16",
                                  kv_dtype="bfloat16", activation_dtype="float16")
        identity["devices"][0]["name"] = "unit-explicit-device"
        identity["runtime"].update(name="unit-runtime", version="1.0", driver_version="1.2.3",
                                    rocm_version="not-applicable", rccl_version="not-applicable")
        self.assertEqual(validate_identity(identity, measured=True)["model"]["kv_dtype"], "bfloat16")
        for kind in ("weight_dtype", "kv_dtype", "activation_dtype"):
            implicit = deepcopy(identity)
            implicit["model"][kind] = "auto"
            with self.subTest(kind=kind), self.assertRaises(SimulationError):
                validate_identity(implicit, measured=True)

    def test_dataset_validation_repeats_allowed_but_ids_unique(self):
        sample = observation()
        record = {key: value for key, value in sample.items() if key != "sample_count"}
        record.update(record_id="r1", group_id="configuration")
        data = {"schema_version": 1, "dataset_id": "fixture", "source": profile()["source"],
                "identity": profile_identity(Experiment()),
                "records": [record, {**record, "record_id": "r2"}]}
        validated = validate_dataset(data, allow_test_fixture=True)
        self.assertEqual(len(validated["records"]), 2)
        data["records"][1]["record_id"] = "r1"
        with self.assertRaises(SimulationError) as raised:
            validate_dataset(data, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "INVALID_DATASET")

    def test_invalid_shapes_durations_sources_and_unknown_fields(self):
        changes = [lambda p: p.update(backend="simulated"),
                   lambda p: p["samples"][0].update(duration_ms=float("nan")),
                   lambda p: p["samples"][0].update(sample_count=True),
                   lambda p: p["samples"][0].update(query_tokens=[True]),
                   lambda p: p["samples"][0].update(context_lengths=[1]),
                   lambda p: p["samples"].append(deepcopy(p["samples"][0])),
                   lambda p: p.update(timing_includes_communication=False),
                   lambda p: p["source"].update(kind="simulated"),
                   lambda p: p["training"].update(dataset_sha256="not-a-digest"),
                   lambda p: p["identity"]["runtime"].update(version=""),
                   lambda p: p["identity"]["model"].pop("revision")]
        for change in changes:
            data = profile()
            change(data)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                validate_profile(data, allow_test_fixture=True)

    def test_notes_and_json_duplicate_keys_are_checked(self):
        data = profile()
        data["notes"] = {"invalid": float("inf")}
        with self.assertRaises(SimulationError):
            validate_profile(data, allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(SimulationError):
                load_profile(path, allow_test_fixture=True)
            with self.assertRaises(SimulationError) as raised:
                load_dataset(path, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "INVALID_DATASET")

    def test_training_counts_shapes_and_groups_cannot_fabricate_observations(self):
        changes = [lambda p: p["samples"][0].update(sample_count=1000000),
                   lambda p: p["training"]["shape_keys"].append(shape_key("prefill", [8], [0])),
                   lambda p: p["training"]["group_ids"].append("fabricated-second-group")]
        for change in changes:
            data = profile()
            change(data)
            with self.subTest(change=change), self.assertRaises(SimulationError) as raised:
                validate_profile(data, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "INVALID_PROFILE")
        genuine_repeats = profile()
        genuine_repeats["samples"][0]["sample_count"] = 2
        genuine_repeats["training"]["record_ids"].append("second-observation")
        self.assertEqual(validate_profile(genuine_repeats, allow_test_fixture=True)
                         ["samples"][0]["sample_count"], 2)


class ProfileModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.json"

    def experiment(self, base=None, data=None):
        base = base or Experiment()
        self.path.write_text(json.dumps(data or profile(experiment=base)))
        return replace(base, calibration=replace(base.calibration, backend="profile",
                                                 profile_path=str(self.path), allow_test_fixture=True))

    def test_identity_matches_shape_hardware_software_and_runtime_assumptions(self):
        exp = self.experiment()
        self.assertIsInstance(create_performance_model(exp), ProfileModel)
        changed = [replace(exp, infra=replace(exp.infra, topology="different")),
                   replace(exp, calibration=replace(exp.calibration, runtime_version="new")),
                   replace(exp, strategy=replace(exp.strategy, model=replace(exp.strategy.model, revision="new"))),
                   replace(exp, infra=replace(exp.infra, devices=(DeviceSpec(name="different"),))),
                   replace(exp, strategy=replace(exp.strategy, memory=replace(exp.strategy.memory, graph_bytes=1)))]
        for candidate in changed:
            with self.subTest(candidate=candidate), self.assertRaises(SimulationError) as raised:
                ProfileModel(candidate)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_measured_profile_rejects_fp16_bf16_reuse_despite_equal_bytes(self):
        base = Experiment()
        base = replace(base,
                       strategy=replace(base.strategy, model=replace(
                           base.strategy.model, revision="unit-revision", weight_dtype="float16",
                           kv_dtype="float16", activation_dtype="float16")),
                       infra=InfraSpec(devices=(DeviceSpec(name="unit-explicit-device"),)),
                       calibration=replace(base.calibration, runtime_name="unit-runtime", runtime_version="1.0",
                                           driver_version="1.2.3", rocm_version="not-applicable",
                                           rccl_version="not-applicable"))
        exp = self.experiment(base, profile(experiment=base, source="measured"))
        self.assertEqual(ProfileModel(exp).estimate_step("prefill", [4], [0]).source, "measured_profile")
        for kind in ("weight_dtype", "kv_dtype", "activation_dtype"):
            model = replace(exp.strategy.model, **{kind: "bfloat16"})
            candidate = replace(exp, strategy=replace(exp.strategy, model=model))
            with self.subTest(kind=kind), self.assertRaises(SimulationError) as raised:
                ProfileModel(candidate)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_file_snapshot_and_exposed_copy_cannot_mutate_a_running_model(self):
        exp = self.experiment()
        raw = self.path.read_bytes()
        model = ProfileModel(exp)
        exposed = model.profile
        exposed["samples"][0]["duration_ms"] = 999
        self.path.write_text("invalid after initialization")
        self.assertEqual(model.estimate_step("prefill", [4], [0]).duration_ms, 10)
        self.assertEqual(model.provenance["sha256"], hashlib.sha256(raw).hexdigest())

    def test_explicit_profile_version_must_match_file_default_is_inferred(self):
        exp = self.experiment()
        self.assertEqual(ProfileModel(exp).provenance["profile_version"], "v1")
        matching = replace(exp, calibration=replace(exp.calibration, profile_version="v1"))
        self.assertEqual(ProfileModel(matching).provenance["profile_version"], "v1")
        mismatch = replace(exp, calibration=replace(exp.calibration, profile_version="different"))
        with self.assertRaises(SimulationError) as raised:
            ProfileModel(mismatch)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_workspace_decode_context_and_unknown_values_are_explicit(self):
        samples = [observation("decode", (1,), (4,), workspace=700)]
        model = ProfileModel(self.experiment(data=profile(samples)))
        self.assertIsNone(model.workspace_bytes([1], "decode"))
        self.assertEqual(model.workspace_bytes([1], "decode", context_lengths=[4]), 700)
        with self.assertRaises(SimulationError):
            model.workspace_bytes([1], "decode", context_lengths=[5])

    def test_analytical_constructor_rejects_profile_instead_of_silent_fallback(self):
        with self.assertRaises(SimulationError) as raised:
            AnalyticalModel(self.experiment())
        self.assertEqual(raised.exception.code, "UNSUPPORTED_MECHANISM")

    def test_tp_full_step_time_is_not_charged_communication_again(self):
        base = Experiment(workload=(RequestSpec("r", 0.0, 4, 2),),
                          infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1")),
                                          link_latency_us=10000),
                          strategy=StrategySpec(tensor_parallel_size=2))
        samples = [observation(duration=7), observation("decode", (1,), (4,), duration=3)]
        exp = self.experiment(base, profile(samples, base))
        report = predict_experiment(exp)
        self.assertEqual(report["isolated_prefill_ms"], 7)
        self.assertEqual(report["isolated_decode_step_ms"], 3)
        self.assertEqual(report["profile_version"], "v1")
        self.assertFalse(report["hardware_validated"])

    def test_pp_stage_totals_add_only_explicit_analytical_boundary(self):
        base = Experiment(workload=(RequestSpec("r", 0.0, 4, 2),),
                          infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
                          strategy=StrategySpec(pipeline_parallel_size=2))
        samples = [observation(phase, (query,), (context,), stage, duration=10 * (stage + 1))
                   for stage in range(2)
                   for phase, query, context in [("prefill", 4, 0), ("decode", 1, 4)]]
        report = predict_experiment(self.experiment(base, profile(samples, base)))
        self.assertAlmostEqual(report["isolated_prefill_ms"], 30 + report["communication"]["prefill_p2p_ms"])
        self.assertEqual(report["online_metrics_reason"], "SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING")


if __name__ == "__main__":
    unittest.main()
