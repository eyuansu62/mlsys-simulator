"""Independent exact repeats use CPU fixtures, never claimed GPU validation."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (SimulationError, calibrate_dataset, load_vllm_captures,
                       normalize_vllm_capture, validate_holdout, validate_profile_repeats)
from mlsys_sim.cli import main
from test_calibration import dataset, record
from test_native_pp_transport import dataset as native_dataset, experiment, record as native_record
from test_vllm_capture import capture, experiment as capture_experiment


def acquisition(session):
    return {"session_id": session, "clock_domain": "cpu-fixture-monotonic",
            "capture_input_sha256": [hashlib.sha256((session + "-raw").encode()).hexdigest()]}


def observations(session, points=((4, 10), (8, 20))):
    data = dataset([record(f"{session}-{index}", query, duration, f"{session}-group-{index}")
                    for index, (query, duration) in enumerate(points)])
    data["dataset_id"] = session
    data["notes"] = {"acquisition": acquisition(session)}
    return data


def fit(data):
    return calibrate_dataset(data, profile_version="repeat-fixture", allow_test_fixture=True)


class ExactStageRepeatTests(unittest.TestCase):
    def test_exact_repeat_errors_and_acquisition_are_preserved_without_mutating_inputs(self):
        training = observations("training", ((4, 10), (4, 14), (8, 20)))
        repeated = observations("independent-repeat", ((4, 10), (8, 20)))
        profile = fit(training)
        before = deepcopy((training, repeated, profile))
        report = validate_profile_repeats(profile, training, repeated, allow_test_fixture=True)
        self.assertEqual((training, repeated, profile), before)
        self.assertEqual(profile["notes"]["acquisition"], training["notes"]["acquisition"])
        self.assertEqual(report["summary"]["coverage"], 1)
        self.assertEqual(report["summary"]["mae_ms"], 1)
        self.assertEqual(report["summary"]["bias_ms"], 1)
        self.assertEqual(report["summary"]["mape_percent"], 10)
        self.assertEqual(report["summary"]["extrapolated_count"], 0)
        self.assertTrue(all(row["method"] == "exact" for row in report["records"]))
        self.assertEqual(report["validation_type"], "independent_exact_stage_repeats")
        self.assertFalse(report["unseen_shape_generalization"])
        self.assertFalse(report["hardware_validated"])
        self.assertTrue(report["test_fixture_only"])
        self.assertEqual(report["hardware_validation"]["status"], "not_asserted")
        self.assertIn("fresh Runtime", report["hardware_validation"]["reason"])
        with self.assertRaises(SimulationError) as raised:
            validate_holdout(profile, repeated, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")
        self.assertIn("shape_keys", str(raised.exception))

    def test_interior_and_exterior_unretained_shapes_never_interpolate_or_extrapolate(self):
        training = observations("train")
        repeated = observations("repeat", ((4, 10), (6, 15), (12, 30)))
        report = validate_profile_repeats(fit(training), training, repeated, allow_test_fixture=True)
        self.assertEqual(report["summary"]["coverage"], 1 / 3)
        self.assertEqual(report["summary"]["rejected_count"], 2)
        self.assertEqual([row["status"] for row in report["records"]],
                         ["PREDICTED", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN"])
        self.assertFalse(report["allow_extrapolation"])

    def test_canonical_training_sha_and_fitted_provenance_are_strict(self):
        training, repeated = observations("train"), observations("repeat")
        profile = fit(training)
        for change in (
            lambda value: value["records"][0].update(duration_ms=11),
            lambda value: value["source"].update(reference="Different declared acquisition"),
            lambda value: value["notes"].update(extra="Canonical JSON changed"),
        ):
            changed = deepcopy(training)
            change(changed)
            with self.subTest(change=change), self.assertRaises(SimulationError) as raised:
                validate_profile_repeats(profile, changed, repeated, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        for change in (
            lambda value: value["samples"][0].update(duration_ms=11),
            lambda value: value["training"]["record_ids"].__setitem__(0, "fabricated-id"),
            lambda value: value["training"]["group_ids"].__setitem__(0, "fabricated-group"),
            lambda value: value["source"].update(reference="Different fitted provenance"),
            lambda value: value["notes"].update(record_count=99),
            lambda value: value["notes"]["acquisition"].update(session_id="fabricated-session"),
        ):
            changed = deepcopy(profile)
            change(changed)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                validate_profile_repeats(changed, training, repeated, allow_test_fixture=True)

    def test_record_group_session_or_raw_capture_reuse_is_rejected(self):
        training, repeated = observations("train"), observations("repeat")
        profile = fit(training)
        for change in (
            lambda value: value["records"][0].update(record_id=training["records"][0]["record_id"]),
            lambda value: value["records"][0].update(group_id=training["records"][0]["group_id"]),
            lambda value: value["notes"]["acquisition"].update(session_id="train"),
            lambda value: value["notes"]["acquisition"].update(
                capture_input_sha256=training["notes"]["acquisition"]["capture_input_sha256"]),
        ):
            changed = deepcopy(repeated)
            change(changed)
            with self.subTest(change=change), self.assertRaises(SimulationError) as raised:
                validate_profile_repeats(profile, training, changed, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")

    def test_missing_or_malformed_capture_facts_cannot_be_inferred_from_opaque_ids(self):
        training, repeated = observations("train"), observations("repeat")
        for value in (None, {}, {"session_id": "repeat"},
                      {**acquisition("repeat"), "clock_domain": ""},
                      {**acquisition("repeat"), "capture_input_sha256": []},
                      {**acquisition("repeat"), "capture_input_sha256": ["not-a-digest"]},
                      {**acquisition("repeat"), "capture_input_sha256": ["a" * 64, "a" * 64]},
                      {**acquisition("repeat"), "unknown": True}):
            changed = deepcopy(repeated)
            changed["notes"]["acquisition"] = value
            with self.subTest(value=value), self.assertRaises(SimulationError) as raised:
                validate_profile_repeats(fit(training), training, changed, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "INVALID_ACQUISITION_PROVENANCE")
        legacy = deepcopy(training)
        legacy.pop("notes")
        legacy_profile = fit(legacy)
        self.assertNotIn("acquisition", legacy_profile["notes"])
        with self.assertRaises(SimulationError):
            validate_profile_repeats(legacy_profile, legacy, repeated, allow_test_fixture=True)

    def test_identity_and_fixture_gate_are_unchanged(self):
        training, repeated = observations("train"), observations("repeat")
        changed = deepcopy(repeated)
        changed["identity"]["runtime"]["version"] = "another-fixture-runtime"
        with self.assertRaises(SimulationError) as raised:
            validate_profile_repeats(fit(training), training, changed, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        with self.assertRaises(SimulationError) as raised:
            validate_profile_repeats(fit(training), training, repeated)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")

    def test_mixed_roles_are_part_of_the_exact_domain(self):
        training = observations("mixed-train", ((4, 10),))
        training["records"][0].update(phase="mixed", query_tokens=[1, 1], context_lengths=[4, 0],
                                      request_phases=["decode", "prefill"])
        repeated = deepcopy(training)
        repeated["dataset_id"] = "mixed-repeat"
        repeated["notes"]["acquisition"] = acquisition("mixed-repeat")
        repeated["records"][0].update(record_id="mixed-repeat", group_id="mixed-repeat",
                                      request_phases=["prefill", "decode"])
        report = validate_profile_repeats(fit(training), training, repeated, allow_test_fixture=True)
        self.assertEqual(report["summary"]["coverage"], 0)
        self.assertEqual(report["records"][0]["status"], "OUT_OF_DOMAIN")

    def test_native_payload_must_equal_the_retained_shape_and_all_stages_are_checked(self):
        exp = experiment()
        training = native_dataset(exp, [native_record(exp, stage, padded=4) for stage in range(2)])
        training["notes"] = {"acquisition": acquisition("native-train")}
        repeated = deepcopy(training)
        repeated["dataset_id"] = "native-repeat"
        repeated["notes"]["acquisition"] = acquisition("native-repeat")
        for row in repeated["records"]:
            row["record_id"] += "-independent"
            row["group_id"] += "-independent"
        profile = fit(training)
        report = validate_profile_repeats(profile, training, repeated, allow_test_fixture=True)
        self.assertEqual(report["summary"]["predicted_count"], 2)
        for stage in range(2):
            changed = deepcopy(repeated)
            replacement = native_record(exp, stage, padded=8)
            changed["records"][stage]["pp_transport"] = replacement["pp_transport"]
            with self.subTest(stage=stage), self.assertRaises(SimulationError) as raised:
                validate_profile_repeats(profile, training, changed, allow_test_fixture=True)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")


class AcquisitionNormalizerTests(unittest.TestCase):
    def test_validated_capture_headers_supply_session_clock_and_actual_input_hashes(self):
        exp = capture_experiment()
        raw = capture(exp)
        raw["session_id"] = "explicit-capture-session"
        expected = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        bundle = normalize_vllm_capture(exp, [raw], allow_test_fixture=True)
        facts = bundle["dataset"]["notes"]["acquisition"]
        self.assertEqual(facts, {"session_id": raw["session_id"], "clock_domain": raw["clock_domain"],
                                 "capture_input_sha256": [expected]})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps(raw, indent=2) + "\n")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_vllm_captures([path])
            facts = normalize_vllm_capture(exp, loaded, allow_test_fixture=True)["dataset"]["notes"]["acquisition"]
            self.assertEqual(facts["capture_input_sha256"], [expected])

    def test_independent_hook_sessions_can_repeat_exact_shapes_but_same_capture_is_rejected(self):
        exp = capture_experiment()
        data = []
        for session in ("first-session", "independent-session"):
            raw = capture(exp)
            raw["session_id"] = session
            data.append(normalize_vllm_capture(exp, [raw], allow_test_fixture=True)["dataset"])
        report = validate_profile_repeats(fit(data[0]), *data, allow_test_fixture=True)
        self.assertEqual(report["summary"]["coverage"], 1)
        self.assertEqual(report["summary"]["mae_ms"], 0)
        with self.assertRaises(SimulationError) as raised:
            validate_profile_repeats(fit(data[0]), data[0], data[0], allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")


class ExactRepeatCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        training, repeated = observations("train"), observations("repeat")
        for name, value in (("training", training), ("repeat", repeated), ("profile", fit(training))):
            (self.root / f"{name}.json").write_text(json.dumps(value))

    def cli(self, target, *extra, expected=0):
        output = self.root / target
        args = ["validate", "--dataset", str(self.root / "repeat.json"), "--out", str(output),
                "--allow-test-fixture", *map(str, extra)]
        with redirect_stdout(io.StringIO()) as summary:
            code = main(args)
        self.assertEqual(code, expected, summary.getvalue())
        return output, json.loads((output / "result.json").read_text())

    def test_explicit_cli_snapshots_training_profile_and_repeat_without_hardware_claims(self):
        output, report = self.cli("repeat-run", "--exact-repeats", "--profile", self.root / "profile.json",
                                 "--training-dataset", self.root / "training.json")
        self.assertEqual(report["summary"]["coverage"], 1)
        self.assertFalse(report["hardware_validated"])
        self.assertTrue(all((output / name).exists()
                            for name in ("training.json", "repeat.json", "profile.json", "manifest.json", "logs.txt")))
        _, legacy = self.cli("legacy-holdout", "--profile", self.root / "profile.json", expected=2)
        self.assertEqual(legacy["error_code"], "VALIDATION_DATA_LEAKAGE")

    def test_explicit_mode_rejects_missing_training_split_other_domains_and_extrapolation(self):
        base = ("--exact-repeats", "--profile", self.root / "profile.json",
                "--training-dataset", self.root / "training.json")
        for index, extra in enumerate((("--allow-extrapolation",), ("--seed", "1"),
                                      ("--holdout-fraction", ".5"), ("--holdout-group", "train"),
                                      ("--native-pp-communication",), ("--memory-capacity",))):
            output, report = self.cli(f"invalid-{index}", *base, *extra, expected=2)
            self.assertEqual(report["error_code"], "INVALID_ARGUMENT")
            self.assertTrue((output / "logs.txt").read_text())
        for index, options in enumerate((("--exact-repeats", "--profile", self.root / "profile.json"),
                ("--exact-repeats", "--training-dataset", self.root / "training.json"),
                ("--profile", self.root / "profile.json", "--training-dataset", self.root / "training.json"))):
            _, report = self.cli(f"missing-{index}", *options, expected=2)
            self.assertEqual(report["error_code"], "INVALID_ARGUMENT")


if __name__ == "__main__":
    unittest.main()
