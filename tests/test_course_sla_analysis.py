"""CPU-only commit/reveal checks; fixture observations are never GPU evidence."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import experiment_from_dict
from mlsys_sim.load_validation import predict_closed_loop
from mlsys_sim.schema import CalibrationSpec, Experiment, StrategySpec
from mlsys_sim.workload import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("baseline", "short", "long")
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "fixture_course_sla_analysis", ROOT / "scripts/analyze_course_sla_study.py")
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(ANALYSIS)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slower_observation(run):
    """Keep the same policy/stream with independently slower terminal feedback."""
    observed = deepcopy(run)
    for row in observed["requests"]:
        for name in ("arrival_ms", "first_token_ms", "completion_ms", "ttft_ms", "tpot_ms", "end_to_end_ms"):
            if row.get(name) is not None:
                row[name] *= 2
        for name in ("token_times_ms", "itl_ms"):
            row[name] = [time * 2 for time in row[name]]
    observed["result"]["simulated_ms"] *= 2
    for values in observed["result"]["latency_ms"].values():
        for key, value in values.items():
            if value is not None:
                values[key] = value * 2
    return observed


class FrozenStudy:
    """A complete three-cell directory, independent of ignored real study runs."""

    def __init__(self, root):
        self.root = root
        self.frozen = root / "frozen-commitment"
        self.frozen.mkdir()
        runtime_sha = canonical_sha256({"kind": "test_fixture", "runtime": "CPU-only"})
        experiment = Experiment(strategy=StrategySpec(scheduler="fifo"), calibration=CalibrationSpec(
            backend="synthetic", prefill_ms=10, decode_ms=1, runtime_config_sha256=runtime_sha))
        write_json(self.frozen / "experiment.json", asdict(experiment))
        experiment = experiment_from_dict(json.loads((self.frozen / "experiment.json").read_text()))
        self.predictions = {}
        self.summaries = {}
        jobs = []
        canonical_commitments = {}
        raw_commitments = {}
        for index, cohort in enumerate(COHORTS):
            spec = {"schema_version": 1, "study_id": "analysis_fixture", "cohort": cohort,
                    "concurrency_grid": [1], "requests_per_run": 2, "replications": 1,
                    "seed": 42, "think_ms": 0, "input_lengths": [[2], [1], [4]][index],
                    "output_tokens": 2, "timeout_ms": 1000, "max_run_ms": 10_000}
            write_json(self.frozen / f"{cohort}.spec.json", spec)
            prediction = predict_closed_loop(experiment, spec)
            self.predictions[cohort] = prediction
            canonical_commitments[cohort] = canonical_sha256(prediction)
            raw = (json.dumps(prediction, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            raw_commitments[cohort] = hashlib.sha256(raw).hexdigest()
            contract = {"schema_version": 1, "prediction_frozen_at": "2026-11-11T01:00:00Z",
                        "prediction_payload_sha256": canonical_commitments[cohort], "min_completed_samples": 2,
                        "timing_boundary": "server_token_ready_v1", "allow_test_fixture": True,
                        "max_relative_error": {"ttft_p99_ms": 0.5, "tpot_p99_ms": 0.5,
                            "end_to_end_p99_ms": 0.5, "elapsed_output_tokens_per_s": 1},
                        "constraints": {"schema_version": 1, "min_completed_samples": 2,
                            "latency": {"p99_ttft_ms": 1000, "p99_tpot_ms": 1000, "p99_e2e_ms": 1000},
                            "throughput": {"basis": "elapsed", "min_output_tokens_per_s": 1}}}
            write_json(self.frozen / f"{cohort}.validation-contract.json", contract)
            job = {"label": f"{cohort}-c1-r0", "cohort": cohort, "concurrency": 1, "replication": 0}
            jobs.append(job)
            folder = root / "formal" / job["label"]
            folder.mkdir(parents=True)
            observed = slower_observation(prediction["runs"][0])
            write_json(folder / "observed-run.json", {"result": observed["result"], "requests": observed["requests"]})
            observed_sha = file_sha256(folder / "observed-run.json")
            write_json(folder / "api-feedback.json", {
                "formal_start_ns": 1001 + index, "resources_drained": observed["resources_drained"],
                "offered_count": observed["offered_count"], "not_offered_count": observed["not_offered_count"]})
            self.summaries[cohort] = {
                "observed_run_sha256": observed_sha, "identity": prediction["identity"],
                "runtime_config_sha256": runtime_sha, "session_id": f"fixture-session-{cohort}",
                "policy_sha256": observed["policy_sha256"],
                "request_stream_sha256": observed["request_stream_sha256"],
                "source": {"kind": "test_fixture", "reference": "CPU commit-reveal fixture, never GPU evidence"},
                "collection_started_at": f"2026-11-11T02:{index}0:00Z",
                "collection_ended_at": f"2026-11-11T02:{index}1:00Z",
                "timing_boundary": observed["timing_boundary"]}
            receipt = {**job, "returncode": 0, "worker_status": "COMPLETED", "error_code": None,
                       "observed_run_sha256": observed_sha}
            write_json(folder / "job-receipt.json", receipt)
        write_json(self.frozen / "jobs.json", jobs)
        write_json(self.frozen / "selection.json", {"kind": "test_fixture", "selection": "frozen_before_collection"})
        self.receipt = {
            "schema_version": 1, "study_id": "analysis_fixture", "frozen_at": "2026-11-11T01:00:00Z",
            "remote_freeze_monotonic_ns": 1000, "runtime_config_sha256": runtime_sha,
            "scheduler_config_sha256": canonical_sha256({"scheduler": "fixture"}),
            "checkpoint_manifest_sha256": canonical_sha256({"checkpoint": "fixture"}),
            "local_manifest_sha256": canonical_sha256({"manifest": "fixture"}),
            "canonical_prediction_payload_sha256": canonical_commitments,
            "prediction_artifacts_sha256": raw_commitments,
            "files_sha256": {path.name: file_sha256(path) for path in sorted(self.frozen.iterdir())},
            "code_files_sha256": {name: file_sha256(ROOT / "mlsys_sim" / name) for name in
                                  ("load_validation.py", "vllm_sla_worker.py", "schema.py", "performance.py")}}
        self.state = {"status": "COMPLETED", "completed_jobs": [json.loads(
            (root / "formal" / job["label"] / "job-receipt.json").read_text()) for job in jobs]}
        self.seal()

    def seal(self):
        """Commit a receipt before binding the independent worker summaries to it."""
        write_json(self.frozen / "freeze-receipt.json", self.receipt)
        self.receipt_sha = file_sha256(self.frozen / "freeze-receipt.json")
        self.state["receipt_sha256"] = self.receipt_sha
        write_json(self.root / "formal-status.json", self.state)
        for cohort, summary in self.summaries.items():
            summary["freeze_receipt"] = deepcopy(self.receipt)
            summary["source_sha256"] = self.receipt["code_files_sha256"]["vllm_sla_worker.py"]
            write_json(self.folder(cohort) / "summary.json", summary)

    def folder(self, cohort):
        return self.root / "formal" / f"{cohort}-c1-r0"

    def analyze(self):
        return ANALYSIS.analyze(self.root, self.receipt_sha, self.root / "analysis")


class CourseSlaAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = FrozenStudy(self.root)

    def test_reveals_both_commitments_and_validates_all_fixture_cohorts(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        report = self.study.analyze()
        self.assertTrue(report["prediction_reveal_verified"])
        self.assertFalse(report["hardware_validated"])
        self.assertEqual(report["receipt_sha256"], self.study.receipt_sha)
        self.assertEqual(report["fresh_runtime_sessions"], 3)
        self.assertEqual(set(report["cohorts"]), set(COHORTS))
        self.assertEqual(len(report["attempts"]), 3)
        for cohort, validation in report["cohorts"].items():
            with self.subTest(cohort=cohort):
                self.assertEqual(validation["verdict"], "PASS")
                self.assertFalse(validation["hardware_validated"])
                self.assertEqual(validation["predictions_sha256"], canonical_sha256(self.study.predictions[cohort]))
                self.assertEqual(len(validation["runs"]), 1)
                run = validation["runs"][0]
                self.assertEqual(run["observation"]["source_kind"], "test_fixture")
                self.assertEqual(run["metrics"]["ttft_p99_ms"]["relative_error"], 0.5)
                self.assertEqual(run["metrics"]["elapsed_output_tokens_per_s"]["relative_error"], 1)
                saved = json.loads((self.root / "analysis" / f"{cohort}.validation.json").read_text())
                self.assertEqual(saved, validation)
        saved = json.loads((self.root / "analysis/study-analysis.json").read_text())
        self.assertEqual(canonical_sha256(saved), canonical_sha256(report))
        self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))

    def test_changed_canonical_prediction_commitment_is_rejected(self):
        self.study.receipt["canonical_prediction_payload_sha256"]["baseline"] = "0" * 64
        self.study.seal()
        with self.assertRaisesRegex(ValueError, "Canonical prediction reveal failed: baseline"):
            self.study.analyze()

    def test_raw_prediction_commitment_requires_the_exact_terminal_newline(self):
        prediction = self.study.predictions["baseline"]
        raw_without_newline = json.dumps(prediction, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.study.receipt["prediction_artifacts_sha256"]["baseline"] = hashlib.sha256(raw_without_newline).hexdigest()
        self.study.seal()
        with self.assertRaisesRegex(ValueError, "Raw prediction reveal failed: baseline"):
            self.study.analyze()

    def test_observed_bytes_change_is_rejected_even_when_json_is_unchanged(self):
        path = self.study.folder("baseline") / "observed-run.json"
        before = json.loads(path.read_text())
        path.write_bytes(path.read_bytes() + b"\n")
        self.assertEqual(json.loads(path.read_text()), before)
        with self.assertRaisesRegex(ValueError, r"Evidence bytes differ.*observed-run\.json"):
            self.study.analyze()

    def test_reused_runtime_session_across_cohorts_is_rejected(self):
        self.study.summaries["short"]["session_id"] = self.study.summaries["baseline"]["session_id"]
        self.study.seal()
        with self.assertRaisesRegex(ValueError, "Distinct fresh Runtime sessions"):
            self.study.analyze()

    def test_incomplete_controller_status_is_rejected_before_analysis(self):
        self.study.state["status"] = "RUNNING"
        self.study.seal()
        with self.assertRaisesRegex(ValueError, "Every declared GPU cell must be attempted"):
            self.study.analyze()
        self.assertFalse((self.root / "analysis").exists())

    def test_missing_completed_cell_is_rejected(self):
        self.study.state["completed_jobs"].pop()
        self.study.seal()
        with self.assertRaisesRegex(ValueError, "Every declared GPU cell must be attempted"):
            self.study.analyze()

    def test_wrong_external_receipt_digest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"Evidence bytes differ.*freeze-receipt\.json"):
            ANALYSIS.analyze(self.root, "0" * 64, self.root / "analysis")
        self.assertFalse((self.root / "analysis").exists())

    def test_changed_frozen_spec_is_rejected_without_recomputing_its_commitment(self):
        path = self.study.frozen / "baseline.spec.json"
        spec = json.loads(path.read_text())
        spec["seed"] += 1
        write_json(path, spec)
        with self.assertRaisesRegex(ValueError, r"Evidence bytes differ.*baseline\.spec\.json"):
            self.study.analyze()

    def test_current_worker_source_must_match_frozen_code_digest(self):
        self.study.receipt["code_files_sha256"]["vllm_sla_worker.py"] = "0" * 64
        self.study.seal()
        with self.assertRaisesRegex(ValueError, "Frozen collection/prediction source changed: vllm_sla_worker.py"):
            self.study.analyze()

    def test_controller_receipt_binding_is_checked(self):
        self.study.state["receipt_sha256"] = "0" * 64
        write_json(self.root / "formal-status.json", self.study.state)
        with self.assertRaisesRegex(ValueError, "Controller did not use the same freeze receipt"):
            self.study.analyze()

    def test_formal_monotonic_start_must_follow_remote_freeze(self):
        path = self.study.folder("baseline") / "api-feedback.json"
        feedback = json.loads(path.read_text())
        feedback["formal_start_ns"] = self.study.receipt["remote_freeze_monotonic_ns"]
        write_json(path, feedback)
        with self.assertRaisesRegex(ValueError, "Formal monotonic start does not follow remote freeze"):
            self.study.analyze()


if __name__ == "__main__":
    unittest.main()
