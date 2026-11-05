"""Frozen exact Workspace repeats use CPU-built contracts, never GPU evidence.

Even source-kind="measured" below exercises metadata/schema only. No allocator,
Runtime process, actual hardware attribution or timing measurement is performed.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import unittest

from mlsys_sim import CalibrationSpec, SimulationError, profile_identity
from mlsys_sim.workspace import calibrate_workspace, validate_workspace_repeats
from test_workspace import experiment, workspace_fixture, workspace_record


def acquisition(session):
    return {"session_id": session, "clock_domain": "cpu-fixture-monotonic",
            "capture_input_sha256": [hashlib.sha256((session + "-raw").encode()).hexdigest()]}


def observations(session, exp, rows):
    data = workspace_fixture(exp, deepcopy(rows))
    data["dataset_id"] = session
    data["notes"] = {"acquisition": acquisition(session)}
    for index, row in enumerate(data["records"]):
        row.update(record_id=f"{session}-record-{index}", group_id=f"{session}-group-{index}")
    return data


def fit(data, **options):
    return calibrate_workspace(data, profile_version="workspace-repeat-fixture", allow_test_fixture=True, **options)


def validate(profile, training, repeated):
    return validate_workspace_repeats(profile, training, repeated, allow_test_fixture=True)


class WorkspaceRepeatTests(unittest.TestCase):
    def test_multistage_tp_ranks_keep_distinct_errors_and_observed_zero(self):
        exp = experiment(tp=2, pp=2)
        training_rows, repeat_rows = [], []
        for stage, a, b, c, d in ((0, (0, 20), (10, 40), (0, 20), (15, 30)),
                                  (1, (0, 30), (5, 60), (0, 34), (5, 60))):
            for phase, query, context, peaks, repeated_peaks in (
                ("prefill", (4,), (0,), a, c), ("decode", (1,), (4,), b, d)):
                training_rows.append(workspace_record(exp, stage=stage, phase=phase, queries=query,
                    contexts=context, peaks=peaks, identifier=f"{stage}-{phase}"))
                repeat_rows.append(workspace_record(exp, stage=stage, phase=phase, queries=query,
                    contexts=context, peaks=repeated_peaks, identifier=f"{stage}-{phase}"))
        training = observations("training", exp, training_rows)
        repeated = observations("fresh-repeat", exp, repeat_rows)
        # Rank arrays may arrive out of order; explicit rank IDs decide comparison.
        repeated["records"][1]["workspace_bytes_per_rank"].reverse()
        profile = fit(training)
        before = deepcopy((profile, training, repeated))
        report = validate(profile, training, repeated)
        self.assertEqual((profile, training, repeated), before)
        self.assertEqual(report["summary"], {
            "sample_count": 4, "predicted_count": 4, "rejected_count": 0, "coverage": 1,
            "rank_sample_count": 8, "rank_predicted_count": 8, "rank_rejected_count": 0,
            "rank_coverage": 1, "underprediction_count": 2, "underpredicted_record_count": 2,
            "max_underprediction_bytes": 5, "max_absolute_error_bytes": 10, "extrapolated_count": 0,
            "profile_shape_count": 4, "repeated_retained_shape_count": 4, "retained_shape_coverage": 1})
        self.assertEqual(report["records"][0]["ranks"][0]["observed_bytes"], 0)
        self.assertEqual(report["records"][0]["ranks"][0]["profile_bytes"], 0)
        self.assertEqual(report["records"][1]["ranks"][0]["underprediction_bytes"], 5)
        self.assertEqual(report["records"][1]["ranks"][1]["absolute_error_bytes"], 10)
        by_rank = {(row["stage_index"], row["tp_rank"]): row for row in report["by_stage_and_rank"]}
        self.assertEqual(by_rank[0, 0]["underprediction_count"], 1)
        self.assertEqual(by_rank[0, 1]["underprediction_count"], 0)
        self.assertEqual(by_rank[1, 1]["max_underprediction_bytes"], 4)
        self.assertFalse(report["hardware_validated"])
        self.assertFalse(report["unseen_shape_generalization"])
        self.assertTrue(report["test_fixture_only"])
        self.assertEqual(report["hardware_validation"]["status"], "not_asserted")
        self.assertIn("fresh Runtime", report["hardware_validation"]["reason"])
        self.assertIsNone(report["online_sla"])
        json.dumps(report, allow_nan=False)

    def test_unretained_interior_exterior_and_decode_contexts_are_not_interpolated(self):
        exp = experiment(tp=2)
        training = observations("train", exp, [workspace_record(exp, queries=(q,), peaks=(0, q)) for q in (4, 8)])
        repeated = observations("repeat", exp, [workspace_record(exp, queries=(q,), peaks=(0, q)) for q in (4, 6, 12)])
        report = validate(fit(training), training, repeated)
        self.assertEqual(report["summary"]["coverage"], 1 / 3)
        self.assertEqual(report["summary"]["rank_coverage"], 1 / 3)
        self.assertEqual(report["summary"]["rank_rejected_count"], 4)
        self.assertEqual(report["summary"]["retained_shape_coverage"], 1 / 2)
        self.assertEqual([row["status"] for row in report["records"]],
                         ["PREDICTED", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN"])
        self.assertIsNone(report["records"][1]["ranks"][0]["profile_bytes"])
        self.assertIsNone(report["records"][1]["ranks"][0]["underprediction_bytes"])
        self.assertFalse(report["interpolation"])
        self.assertFalse(report["allow_extrapolation"])
        training = observations("decode-train", exp, [workspace_record(exp, phase="decode", queries=(1,), contexts=(4,))])
        repeated = observations("decode-repeat", exp, [workspace_record(exp, phase="decode", queries=(1,), contexts=(5,))])
        result = validate(fit(training), training, repeated)
        self.assertEqual(result["summary"]["coverage"], 0)
        self.assertIsNone(result["summary"]["max_underprediction_bytes"])

    def test_subset_repeat_does_not_imply_all_profile_stages_were_observed(self):
        exp = experiment(pp=2)
        training = observations("train", exp, [workspace_record(exp, stage=s) for s in (0, 1)])
        repeated = observations("repeat", exp, [workspace_record(exp, stage=1)])
        report = validate(fit(training), training, repeated)
        self.assertEqual(report["summary"]["coverage"], 1)
        self.assertEqual(report["summary"]["retained_shape_coverage"], .5)
        self.assertEqual(len(report["by_stage_and_rank"]), 1)
        self.assertEqual(report["by_stage_and_rank"][0]["stage_index"], 1)

    def test_missing_duplicated_or_unknown_rank_is_rejected_before_comparison(self):
        exp = experiment(tp=2)
        training = observations("train", exp, [workspace_record(exp, peaks=(0, 20))])
        repeated = observations("repeat", exp, [workspace_record(exp, peaks=(0, 20))])
        profile = fit(training)
        for ranks in ([], [{"tp_rank": 0, "peak_bytes": 0}],
                      [{"tp_rank": 0, "peak_bytes": 0}, {"tp_rank": 0, "peak_bytes": 20}],
                      [{"tp_rank": 0, "peak_bytes": 0}, {"tp_rank": 2, "peak_bytes": 20}]):
            changed = deepcopy(repeated)
            changed["records"][0]["workspace_bytes_per_rank"] = ranks
            with self.subTest(ranks=ranks), self.assertRaises(SimulationError) as raised:
                validate(profile, training, changed)
            self.assertEqual(raised.exception.code, "INVALID_WORKSPACE_DATASET")

    def test_session_capture_record_and_group_reuse_are_all_rejected(self):
        exp = experiment()
        training = observations("train", exp, [workspace_record(exp)])
        repeated = observations("repeat", exp, [workspace_record(exp)])
        profile = fit(training)
        changes = (
            lambda x: x["notes"]["acquisition"].update(session_id="train"),
            lambda x: x["notes"]["acquisition"].update(capture_input_sha256=acquisition("train")["capture_input_sha256"]),
            lambda x: x["records"][0].update(record_id=training["records"][0]["record_id"]),
            lambda x: x["records"][0].update(group_id=training["records"][0]["group_id"]),
        )
        for change in changes:
            bad = deepcopy(repeated)
            change(bad)
            with self.subTest(change=change), self.assertRaises(SimulationError) as raised:
                validate(profile, training, bad)
            self.assertEqual(raised.exception.code, "VALIDATION_DATA_LEAKAGE")

    def test_training_canonical_sha_rejects_different_values_metadata_or_source(self):
        exp = experiment()
        training = observations("train", exp, [workspace_record(exp, peaks=(2,))])
        repeated = observations("repeat", exp, [workspace_record(exp, peaks=(2,))])
        profile = fit(training)
        for change in (
            lambda x: x["records"][0]["workspace_bytes_per_rank"][0].update(peak_bytes=3),
            lambda x: x["source"].update(reference="Another declared acquisition"),
            lambda x: x["notes"].update(extra="Changed canonical data"),
        ):
            bad = deepcopy(training)
            change(bad)
            with self.subTest(change=change), self.assertRaises(SimulationError) as raised:
                validate(profile, bad, repeated)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_frozen_values_counts_and_provenance_must_recompute_exactly(self):
        exp = experiment()
        training = observations("train", exp, [workspace_record(exp, peaks=(2,))])
        repeated = observations("repeat", exp, [workspace_record(exp, peaks=(2,))])
        profile = fit(training)
        for change in (
            lambda x: x["samples"][0]["workspace_bytes_per_rank"][0].update(peak_bytes=3),
            lambda x: x["samples"][0].update(sample_count=2),
            lambda x: x["training"]["record_ids"].__setitem__(0, "invented-id"),
            lambda x: x["training"]["group_ids"].__setitem__(0, "invented-group"),
            lambda x: x["source"].update(reference="Invented fitting source"),
            lambda x: x["notes"].update(parent_samples_reused=0),
            lambda x: x["notes"]["acquisition"].update(session_id="invented-session"),
        ):
            bad = deepcopy(profile)
            change(bad)
            with self.subTest(change=change), self.assertRaises(SimulationError):
                validate(bad, training, repeated)

    def test_parent_version_is_context_and_never_supplies_missing_or_larger_bounds(self):
        exp = experiment()
        parent_data = observations("parent", exp, [workspace_record(exp, peaks=(1000,))])
        parent = calibrate_workspace(parent_data, profile_version="parent", allow_test_fixture=True)
        training = observations("child-train", exp, [workspace_record(exp, peaks=(0,))])
        repeated = observations("child-repeat", exp, [workspace_record(exp, peaks=(5,))])
        child = fit(training, parent_profile=parent)
        report = validate(child, training, repeated)
        self.assertEqual(report["parent_profile_version"], "parent")
        self.assertFalse(report["parent_samples_reused"])
        self.assertEqual(report["summary"]["underprediction_count"], 1)
        self.assertEqual(report["records"][0]["ranks"][0]["profile_bytes"], 0)
        bad = deepcopy(child)
        bad["samples"][0]["workspace_bytes_per_rank"][0]["peak_bytes"] = 1000
        with self.assertRaises(SimulationError):
            validate(bad, training, repeated)

    def test_missing_malformed_facts_cannot_be_inferred_from_ids(self):
        exp = experiment()
        training = observations("train", exp, [workspace_record(exp)])
        repeated = observations("repeat", exp, [workspace_record(exp)])
        profile = fit(training)
        for facts in (None, {}, {"session_id": "repeat"},
                      {**acquisition("repeat"), "capture_input_sha256": []},
                      {**acquisition("repeat"), "capture_input_sha256": ["not-sha256"]},
                      {**acquisition("repeat"), "clock_domain": ""},
                      {**acquisition("repeat"), "capture_input_sha256": ["a" * 64, "a" * 64]},
                      {**acquisition("repeat"), "unexpected": True}):
            bad = deepcopy(repeated)
            bad["notes"]["acquisition"] = facts
            with self.subTest(facts=facts), self.assertRaises(SimulationError) as raised:
                validate(profile, training, bad)
            self.assertEqual(raised.exception.code, "INVALID_ACQUISITION_PROVENANCE")

    def test_acquisition_is_conditionally_copied_and_legacy_fit_notes_stay_unchanged(self):
        exp = experiment()
        legacy = workspace_fixture(exp)
        profile = fit(legacy)
        self.assertEqual(profile["notes"], {
            "dataset_id": legacy["dataset_id"], "fit": "maximum_observed_bytes_per_exact_shape_and_rank",
            "parent_samples_reused": False, "scope": "replacement_for_dynamic_workspace_only",
            "interpolation": False, "extrapolation": False})
        training = observations("train", exp, [workspace_record(exp)])
        self.assertEqual(fit(training)["notes"]["acquisition"], training["notes"]["acquisition"])
        repeated = observations("repeat", exp, [workspace_record(exp)])
        with self.assertRaises(SimulationError) as raised:
            validate(profile, legacy, repeated)
        self.assertEqual(raised.exception.code, "INVALID_ACQUISITION_PROVENANCE")
        bad = deepcopy(training)
        bad["notes"]["acquisition"] = {}
        with self.assertRaises(SimulationError):
            fit(bad)

    def test_mixed_member_roles_and_order_remain_exact(self):
        exp = experiment(tp=2)
        row = workspace_record(exp, phase="mixed", queries=(1, 1), contexts=(4, 0),
                               roles=("decode", "prefill"), peaks=(0, 4))
        training = observations("train", exp, [row])
        repeated = observations("repeat", exp, [row])
        profile = fit(training)
        changed = deepcopy(repeated)
        changed["records"][0].update(query_tokens=[1, 1], context_lengths=[0, 4],
            request_phases=["prefill", "decode"])
        report = validate(profile, training, changed)
        self.assertEqual(report["summary"]["coverage"], 0)
        self.assertEqual(report["summary"]["rank_coverage"], 0)
        self.assertEqual(report["records"][0]["request_phases"], ["prefill", "decode"])

    def test_identity_and_fixture_optin_are_not_relaxed(self):
        exp = experiment()
        training = observations("train", exp, [workspace_record(exp)])
        repeated = observations("repeat", exp, [workspace_record(exp)])
        profile = fit(training)
        changed = deepcopy(repeated)
        changed["identity"]["runtime"]["version"] = "different-runtime"
        with self.assertRaises(SimulationError) as raised:
            validate(profile, training, changed)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")
        with self.assertRaises(SimulationError) as raised:
            validate_workspace_repeats(profile, training, repeated)
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")

    def test_valid_measured_metadata_still_does_not_establish_hardware_acceptance(self):
        exp = experiment()
        exp = replace(exp, infra=replace(exp.infra, devices=tuple(replace(device, name="Schema-only GPU")
            for device in exp.infra.devices)), strategy=replace(exp.strategy, model=replace(exp.strategy.model,
            revision="schema-only-revision", weight_dtype="float16", kv_dtype="float16", activation_dtype="float16")),
            calibration=CalibrationSpec(runtime_name="schema-only-vllm", runtime_version="schema-only-version",
                driver_version="schema-only-driver", rocm_version="schema-only-rocm", rccl_version="schema-only-rccl"))
        training = observations("measured-label-train", exp, [workspace_record(exp, peaks=(0,))])
        repeated = observations("measured-label-repeat", exp, [workspace_record(exp, peaks=(0,))])
        for value in (training, repeated):
            value["identity"] = profile_identity(exp)
            value["source"] = {"kind": "measured", "reference": "Schema-only metadata test, no actual measurement"}
        profile = calibrate_workspace(training, profile_version="schema-only")
        report = validate_workspace_repeats(profile, training, repeated)
        self.assertFalse(report["test_fixture_only"])
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["summary"]["max_absolute_error_bytes"], 0)
        self.assertEqual(report["summary"]["max_underprediction_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
