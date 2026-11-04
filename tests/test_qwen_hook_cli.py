"""CLI model and internal-capture artifacts preserve fixture and domain boundaries."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/v1_6"


class QwenHookCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_cli(self, name, *arguments, expected=0):
        output = self.root / name
        process = subprocess.run([sys.executable, "-m", "mlsys_sim", *map(str, arguments), "--out", str(output)],
                                 cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return output, json.loads((output / "result.json").read_text())

    def test_internal_fixture_cli_writes_usable_stage_dataset_without_forging_online_trace(self):
        source = EXAMPLES / "runner-capture.fixture.json"
        output, result = self.run_cli("capture", "collect", "--kind", "vllm-hook",
            "--experiment", EXAMPLES / "hook-experiment.fixture.json", "--raw", source,
            "--allow-test-fixture")
        self.assertTrue(result["canonical_stage_ready"])
        self.assertFalse(result["canonical_comparable"])
        self.assertFalse(result["hardware_validated"])
        self.assertFalse((output / "observed.json").exists())
        dataset = json.loads((output / "datasets/replica-0.json").read_text())
        self.assertEqual(dataset["source"]["kind"], "test_fixture")
        self.assertEqual(dataset, json.loads((output / "dataset.json").read_text()))
        self.assertTrue(dataset["identity"]["runtime"]["runtime_config_sha256"])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(),
                      {entry["sha256"] for entry in manifest["input_files"]})
        self.run_cli("fit", "calibrate", "--dataset", output / "dataset.json",
                     "--profile-version", "hook-cli-fixture", "--allow-test-fixture")

    def test_hook_plan_resolves_inputs_relative_to_itself(self):
        output, result = self.run_cli("plan-capture", "collect", "--kind", "vllm-hook",
            "--experiment", EXAMPLES / "hook-experiment.fixture.json",
            "--plan", EXAMPLES / "hook-import.fixture.plan.json", "--allow-test-fixture")
        self.assertTrue(result["canonical_stage_ready"])
        self.assertTrue((output / "datasets/replica-0.json").is_file())
        self.assertEqual(result["capture_summary"]["accepted_groups"], 1)

    def test_fixture_permission_and_ambiguous_input_are_rejected(self):
        _, result = self.run_cli("not-permitted", "collect", "--kind", "vllm-hook",
            "--experiment", EXAMPLES / "hook-experiment.fixture.json",
            "--raw", EXAMPLES / "runner-capture.fixture.json", expected=2)
        self.assertEqual(result["error_code"], "TEST_FIXTURE_NOT_ALLOWED")
        _, result = self.run_cli("ambiguous", "collect", "--kind", "vllm-hook",
            "--experiment", EXAMPLES / "hook-experiment.fixture.json",
            "--raw", EXAMPLES / "runner-capture.fixture.json",
            "--plan", EXAMPLES / "hook-import.fixture.plan.json", "--allow-test-fixture", expected=2)
        self.assertEqual(result["error_code"], "INVALID_ARGUMENT")

    def test_quantized_config_import_does_not_emit_a_model(self):
        config = json.loads((EXAMPLES / "qwen3-4b.hf.json").read_text())
        config["quantization_config"] = {"quant_method": "fp8"}
        path = self.root / "unsupported.json"
        path.write_text(json.dumps(config))
        output, result = self.run_cli("unsupported", "import-model", "--kind", "qwen3", "--config", path,
                                     "--name", "Qwen3-4B", "--revision", "fixture", expected=2)
        self.assertEqual(result["error_code"], "UNSUPPORTED_MECHANISM")
        self.assertFalse((output / "model.json").exists())
        self.assertTrue((output / "logs.txt").read_text())


if __name__ == "__main__":
    unittest.main()
