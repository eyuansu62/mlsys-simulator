"""CPU portability and historical-snapshot safety for the PR10 tools."""

import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim.schema import SimulationError
from mlsys_sim.workload import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SHA256 = {
    "docs/course-sla-9070xt-20261111/HOLDOUT-CONSUMED.json": "21bdda6a5611c075fb2428db20a8777732e70efa94515378d854d5fbc1f6568f",
    "examples/course/9070-sla-20261116/frozen-commitment/baseline.spec.json": "0e05bad13df374f1e888c075a3f33a4bc6482b464c7fe8bdd0bd77241242cb70",
    "examples/course/9070-sla-20261116/frozen-commitment/experiment.json": "89ec0d9378014c979bca50fe0bce6de55573c9a1c68fbe61d30c252dcbb91739",
    "examples/course/9070-sla-20261116/frozen-commitment/jobs.json": "a70472359b313eeaea59f27dea1e3652ebd1685fbce14eb63fb6b360721c94ef",
    "examples/course/9070-sla-20261116/frozen-commitment/long.spec.json": "5ded87004097c24771b5ba6a8d99a7ceef6be0b72cb5b40ba032fd360f99f59f",
    "examples/course/9070-sla-20261116/frozen-commitment/manifest.json": "abf998642b823575378d407f7dd1eedfbc57eb021caa2462c5c5549fb28bc127",
    "examples/course/9070-sla-20261116/frozen-commitment/predictions-summary.json": "dbb99bb00d9571ad455ab6518cbfda502b4bfcc92e49f63871b45f9e9f9e45bf",
    "examples/course/9070-sla-20261116/frozen-commitment/selection.json": "a357fa8cbfd0bbaff7f54b4fb50e31d3bb7ed8b3b343b02115b6df9dba4ffa92",
    "examples/course/9070-sla-20261116/frozen-commitment/short.spec.json": "48821cb956c0bc4c4319382d0fa917ad8cb03cca0a2ea8449c6b9b3acb220243",
    "examples/course/9070-sla-20261116/frozen-commitment/validation-contract.template.json": "8f0bb4eeacf5cf3876d66134d472f230e67d9d44cdbc668a9722fe60a1855b1c",
}


def script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("pr10_" + name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(name, *arguments, cwd):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / (name + ".py")), *map(str, arguments)],
                          cwd=cwd, env=environment, capture_output=True, text=True, timeout=30)


class HistoricalSnapshotTests(unittest.TestCase):
    def test_all_ten_protected_historical_files_keep_their_original_bytes(self):
        self.assertEqual(len(HISTORICAL_SHA256), 10)
        for name, expected in HISTORICAL_SHA256.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)


class DecodeAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = script("analyze_decode_domain_gap")

    def test_real_cpu_numbers_are_diagnostic_proxies_and_actual_domains_are_rejected(self):
        with patch("mlsys_sim.cost_calibration.fit_analytical_costs", side_effect=AssertionError("No fit")) as fit:
            report = self.tool.analyze()
        fit.assert_not_called()
        self.assertFalse(report["hardware_validated"])
        self.assertFalse(report["fitted"])
        self.assertEqual(report["context_proxy_tokens"], list(range(257, 272)))
        expected = {1: (14.326440689198144, 19.108692581402078),
                    2: (14.400916059200354, 19.1973825950772),
                    4: (14.54986679920477, 19.37476262242744),
                    8: (14.847768279213607, 19.729522677127918)}
        for row in report["rows"]:
            default, proxy = expected[row["batch_size_proxy"]]
            self.assertAlmostEqual(row["default_decode_median_ms"], default, places=10)
            self.assertAlmostEqual(row["out_of_domain_cost_proxy_ms"], proxy, places=10)
        self.assertEqual(report["rows"][0]["historical_closed_loop_tpot_reference_ms"], 21.204)
        self.assertAlmostEqual(report["model_internal_components"][0]["linear_fraction_model_internal"],
                               .8804606581169891, places=10)
        self.assertEqual({row["error_code"] for row in report["scope"].values()}, {"CALIBRATION_MISMATCH"})
        self.assertIn("bandwidth_proxy_gbps", report["model_internal_components"][0])
        self.assertTrue(any("not isolated" in line for line in report["interpretation"]))
        self.assertTrue(any("not profiler evidence" in line for line in report["interpretation"]))

    def test_only_expected_calibration_mismatch_is_caught(self):
        failure = SimulationError("OTHER_CPU_ERROR", "Unexpected problem")
        with patch.object(self.tool, "corrected_analytical_step", side_effect=failure):
            with self.assertRaises(SimulationError) as caught:
                self.tool.analyze()
        self.assertIs(caught.exception, failure)
        with patch.object(self.tool, "corrected_analytical_step", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "unexpectedly accepted"):
                self.tool.analyze()

    def test_json_cli_and_both_help_commands_work_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory(prefix="pr10-cwd-") as temporary:
            result = run_cli("analyze_decode_domain_gap", "--json", cwd=temporary)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["rows"][0]["historical_closed_loop_tpot_reference_ms"], 21.204)
            for name in ("analyze_decode_domain_gap", "freeze_course_sla_20261116_local"):
                result = run_cli(name, "--help", cwd=temporary)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_imports_have_no_analysis_file_reads_prints_or_gpu_imports(self):
        code = '''
import contextlib, importlib.util, io, pathlib, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import mlsys_sim
from mlsys_sim.load_validation import predict_closed_loop
output=io.StringIO()
with patch('mlsys_sim.load_experiment',side_effect=AssertionError('import-time analysis')) as load:
 with patch('mlsys_sim.load_validation.predict_closed_loop',side_effect=AssertionError('import-time prediction')) as predict:
  with patch.object(pathlib.Path,'read_bytes',side_effect=AssertionError('import-time file read')):
   with contextlib.redirect_stdout(output):
    for index,filename in enumerate(sys.argv[2:]):
     spec=importlib.util.spec_from_file_location('pr10_quiet_'+str(index),filename)
     module=importlib.util.module_from_spec(spec)
     spec.loader.exec_module(module)
  load.assert_not_called()
  predict.assert_not_called()
assert output.getvalue()==''
assert not any(name.split('.')[0] in {'torch','vllm'} for name in sys.modules)
print('CPU_IMPORT_OK')
'''
        with tempfile.TemporaryDirectory(prefix="pr10-import-") as temporary:
            result = subprocess.run([sys.executable, "-c", code, str(ROOT),
                str(ROOT / "scripts/analyze_decode_domain_gap.py"), str(ROOT / "scripts/freeze_course_sla_20261116_local.py")],
                cwd=temporary, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CPU_IMPORT_OK")


class LocalDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = script("freeze_course_sla_20261116_local")
        cls.temporary = tempfile.TemporaryDirectory(prefix="pr10-draft-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.folder = Path(cls.temporary.name)
        cls.out = cls.folder / "new-draft"
        # Exercise real predictions/CLI from another cwd, then inspect the
        # process's loaded modules. No predictor stubs or injected latencies.
        code = '''
import runpy,sys
path=sys.argv.pop(1)
try:
 runpy.run_path(path,run_name='__main__')
except SystemExit as exc:
 if exc.code:raise
assert not any(name.split('.')[0] in {'torch','vllm'} for name in sys.modules)
print('CPU_MODULES_OK')
'''
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        environment.pop("PYTHONPATH", None)
        process = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/freeze_course_sla_20261116_local.py"),
            "--out", str(cls.out), "--study-id", "pr10-cpu-draft-regression", "--requests-per-run", "2", "--replications", "1"],
            cwd=cls.folder, env=environment, capture_output=True, text=True, timeout=30)
        if process.returncode:
            raise AssertionError(process.stdout + process.stderr)
        if "CPU_MODULES_OK" not in process.stdout:
            raise AssertionError("CPU-only import inspection was not completed")
        cls.manifest = json.loads((cls.out / "manifest.json").read_bytes())

    def test_default_recipe_and_small_draft_metadata_are_truthful(self):
        spec = self.tool.make_spec("baseline", [256], study_id="new-default-recipe")
        self.assertEqual((spec["requests_per_run"], spec["replications"], spec["concurrency_grid"]),
                         (1000, 3, [1, 2, 4]))
        self.assertEqual(len(self.tool.COHORTS) * len(spec["concurrency_grid"]) * spec["replications"], 27)
        manifest = self.manifest
        self.assertEqual(manifest["kind"], "course_sla_current_code_local_draft_v1")
        self.assertEqual(manifest["status"], "LOCAL_DRAFT")
        self.assertEqual((manifest["requests_per_run"], manifest["replications"], manifest["jobs_count"]), (2, 1, 9))
        self.assertFalse(manifest["hardware_validated"])
        self.assertFalse(manifest["historical_reproduction"])
        self.assertTrue(manifest["remote_freeze_pending"])
        self.assertEqual((self.out / "experiment.json").read_bytes(), (self.tool.SRC / "experiment.json").read_bytes())
        template = json.loads((self.out / "validation-contract.template.json").read_bytes())
        self.assertEqual(template["min_completed_samples"], 2)
        self.assertEqual(template["constraints"]["min_completed_samples"], 2)
        self.assertEqual(template["prediction_frozen_at"], "REMOTE_RECEIPT_PENDING")
        selection = json.loads((self.out / "selection.json").read_bytes())
        self.assertEqual(selection["baseline_configuration_role"], "historical_retest")
        self.assertFalse(selection["new_blind_holdout_claim"])
        self.assertFalse(selection["grid_disjoint_from_consumed_1111_holdout"])
        self.assertEqual((selection["requests_per_run"], selection["replications"], selection["jobs_count"]), (2, 1, 9))
        self.assertTrue(all(len(row["replication_verdicts"]) == 1
                            for row in selection["baseline_predicted_verdicts_record_only"]))
        self.assertTrue(all(row["replication_statuses"] == ["COMPLETED"]
                            for row in selection["baseline_predicted_verdicts_record_only"]))

    def test_complete_real_payloads_and_current_source_hashes_are_retained(self):
        jobs = json.loads((self.out / "jobs.json").read_bytes())
        self.assertEqual(len(jobs), 9)
        self.assertEqual({row["replication"] for row in jobs}, {0})
        total_requests = 0
        for name in self.tool.COHORTS:
            spec = json.loads((self.out / f"{name}.spec.json").read_bytes())
            self.assertEqual(spec["study_id"], "pr10-cpu-draft-regression")
            self.assertEqual((spec["requests_per_run"], spec["replications"]), (2, 1))
            relative = self.manifest["prediction_artifact_paths"][name]
            self.assertEqual(relative, f"predictions/{name}.prediction-payload.json")
            raw = (self.out / relative).read_bytes()
            payload = json.loads(raw)
            expected_raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            self.assertEqual(raw, expected_raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), self.manifest["prediction_artifacts_sha256"][name])
            self.assertEqual(canonical_sha256(payload), self.manifest["canonical_prediction_payload_sha256"][name])
            self.assertEqual(payload["status"], "COMPLETED")
            self.assertEqual(payload["spec"], spec)
            self.assertEqual(len(payload["runs"]), 3)
            self.assertFalse(payload["hardware_validated"])
            for run in payload["runs"]:
                self.assertEqual((run["expected_requests"], run["offered_count"], len(run["requests"])), (2, 2, 2))
                self.assertIn("events", run)
                self.assertIn("offered_requests", run)
                total_requests += run["offered_count"]
        self.assertEqual(total_requests, 18)
        actual_files = {str(path.relative_to(self.out)) for path in self.out.rglob("*")
                        if path.is_file() and path.name != "manifest.json"}
        self.assertEqual(actual_files, set(self.manifest["files_sha256"]))
        for name, digest in self.manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(), digest)
        code_files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (ROOT / "mlsys_sim").rglob("*.py")}
        self.assertEqual(self.manifest["code_files_sha256"], code_files)
        self.assertEqual(self.manifest["local_manifest_sha256"], canonical_sha256(
            {key: value for key, value in self.manifest.items() if key != "local_manifest_sha256"}))

    def test_existing_outputs_and_symlink_aliases_are_never_overwritten(self):
        existing = self.folder / "existing"
        existing.mkdir()
        sentinel = existing / "keep.bin"
        sentinel.write_bytes(b"preserve existing bytes")
        alias = self.folder / "existing-alias"
        alias.symlink_to(existing, target_is_directory=True)
        dangling = self.folder / "dangling-alias"
        missing = self.folder / "must-not-be-created"
        dangling.symlink_to(missing, target_is_directory=True)
        with patch.object(self.tool, "predict_closed_loop", side_effect=AssertionError("Must reject before prediction")):
            for path in (existing, sentinel, alias, dangling):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.tool.freeze_local(path, study_id="pr10-safety-new", requests_per_run=2, replications=1)
        self.assertEqual(sentinel.read_bytes(), b"preserve existing bytes")
        self.assertFalse(missing.exists())
        process = run_cli("freeze_course_sla_20261116_local", "--out", existing, "--study-id", "pr10-cli-existing",
                          "--requests-per-run", "2", "--replications", "1", cwd=self.folder)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(sentinel.read_bytes(), b"preserve existing bytes")

    def test_historical_snapshots_and_symlinked_children_are_protected(self):
        with patch.object(self.tool, "predict_closed_loop", side_effect=AssertionError("Must reject before prediction")):
            for index, protected in enumerate((self.tool.SRC, self.tool.HISTORICAL)):
                alias = self.folder / f"history-alias-{index}"
                alias.symlink_to(protected, target_is_directory=True)
                child = protected / "pr10-child-must-not-exist"
                self.assertFalse(child.exists())
                for out in (protected, child, alias, alias / child.name):
                    with self.subTest(out=out), self.assertRaises(ValueError):
                        self.tool.freeze_local(out, study_id="pr10-protected-new", requests_per_run=2, replications=1)
                self.assertFalse(child.exists())
        for name, digest in HISTORICAL_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def _assert_case_alias_protected(self, protected):
        alias = protected.with_name(protected.name.upper())
        try:
            aliases_original = alias.samefile(protected)
        except FileNotFoundError:
            self.skipTest("This filesystem does not provide the differently cased directory alias")
        if not aliases_original:
            self.skipTest("Differently cased paths have distinct filesystem identities")
        child = alias / "pr10-case-child-must-not-exist"
        self.assertFalse(child.exists())
        # A failing old guard must never create a child in the real snapshot.
        with patch.object(Path, "mkdir", side_effect=AssertionError("Unsafe historical mkdir reached")):
            with self.assertRaisesRegex(ValueError, "protected"):
                self.tool._new_output(child)
        self.assertFalse((protected / child.name).exists())

    def test_source_snapshot_case_alias_cannot_create_a_new_child(self):
        self._assert_case_alias_protected(self.tool.SRC)

    def test_november16_snapshot_case_alias_cannot_create_a_new_child(self):
        self._assert_case_alias_protected(self.tool.HISTORICAL)

    def test_existing_ancestor_identity_alias_is_rejected_cross_platform(self):
        alias = self.folder / "directory-identity-alias"
        alias.mkdir()
        identity_alias = alias.resolve()
        protected = self.tool.SRC.resolve()
        original_samefile = Path.samefile
        checked = []
        def samefile(candidate, other):
            checked.append(candidate)
            return True if candidate == identity_alias and other == protected else original_samefile(candidate, other)
        target = alias / "missing-parent" / "new-draft"
        with (patch.object(Path, "samefile", new=samefile),
              patch.object(Path, "mkdir", side_effect=AssertionError("Unsafe alias mkdir reached"))):
            with self.assertRaisesRegex(ValueError, "protected"):
                self.tool._new_output(target)
        self.assertIn(identity_alias, checked)
        self.assertFalse(target.exists())

    def test_directory_identity_errors_fail_closed_before_mkdir(self):
        target = self.folder / "identity-unreadable-new-draft"
        with (patch.object(Path, "samefile", side_effect=PermissionError("Cannot inspect directory identity")),
              patch.object(Path, "mkdir", side_effect=AssertionError("Unchecked mkdir reached"))):
            with self.assertRaises(PermissionError):
                self.tool._new_output(target)
        self.assertFalse(target.exists())

    def test_partial_real_cpu_run_cannot_pass_sla_from_the_completed_subset(self):
        experiment = self.tool.load_experiment(self.tool.SRC / "experiment.json")
        spec = self.tool.make_spec("baseline", [256], study_id="pr10-partial-real-cpu",
                                   requests_per_run=2, replications=1)
        spec["concurrency_grid"] = [1]
        limited = replace(experiment, execution=replace(experiment.execution, max_events=20))
        partial = self.tool.predict_closed_loop(limited, spec)["runs"][0]
        self.assertEqual(partial["status"], "UNRESOLVED")
        self.assertEqual(partial["result"]["counts"]["completed"], 1)
        self.assertEqual(partial["result"]["counts"]["unfinished"], 1)
        self.assertFalse(partial["resources_drained"])
        # The completed subset has finite, apparently passing latency/rate
        # metrics. Overall cohort completeness must veto that apparent PASS.
        self.assertLess(partial["result"]["latency_ms"]["end_to_end"]["p99"], 500)
        self.assertGreater(partial["result"]["throughput"]["output_tokens_per_s"], 20)
        self.assertFalse(self.tool.sla_verdict(partial, expected_requests=2))
        complete = self.tool.predict_closed_loop(experiment, spec)["runs"][0]
        self.assertEqual(complete["status"], "COMPLETED")
        self.assertTrue(self.tool.sla_verdict(complete, expected_requests=2))
        self.assertFalse(self.tool.sla_verdict(complete, expected_requests=3))

    def test_historical_ids_and_invalid_counts_do_not_create_outputs(self):
        for index, (study, requests, replications) in enumerate((("course9070sla20261111", 2, 1),
            ("course9070sla20261116", 2, 1), ("pr10-new", 0, 1), ("pr10-new", 2, 0))):
            out = self.folder / f"invalid-{index}"
            with self.subTest(study=study, requests=requests, replications=replications), self.assertRaises(ValueError):
                self.tool.freeze_local(out, study_id=study, requests_per_run=requests, replications=replications)
            self.assertFalse(out.exists())
        result = run_cli("freeze_course_sla_20261116_local", cwd=self.folder)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--out", result.stderr)
        self.assertIn("--study-id", result.stderr)


if __name__ == "__main__":
    unittest.main()
