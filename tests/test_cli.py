"""End-to-end checks on the public commands and reproducible run artifacts."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_cli(self, example, name="run", command="simulate.py"):
        output = self.directory / name
        process = subprocess.run(
            [sys.executable, str(ROOT / command), "--experiment", str(ROOT / "examples" / example),
             "--out", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
        return process, output

    def test_fifo_hand_calculation_and_manifest(self):
        process, output = self.run_cli("fifo.json")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "COMPLETED")
        requests = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        rows = {row["request_id"]: row for row in requests}
        self.assertEqual([rows[r]["first_token_ms"] for r in ["r1", "r2", "r3"]], [10, 24, 34])
        self.assertEqual([rows[r]["completion_ms"] for r in ["r1", "r2", "r3"]], [14, 24, 36])
        self.assertIsNone(rows["r2"]["tpot_ms"])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["calibration_source"], "synthetic")
        self.assertFalse(manifest["hardware_validated"])
        self.assertEqual(len(manifest["config_sha256"]), 64)
        self.assertTrue((output / "events.jsonl").exists())

    def test_existing_run_cannot_be_overwritten(self):
        process, output = self.run_cli("fifo.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        previous = (output / "result.json").read_bytes()
        process, _ = self.run_cli("tp2.json")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stdout)["error_code"], "OUTPUT_EXISTS")
        self.assertEqual((output / "result.json").read_bytes(), previous)

    def test_repeated_runs_have_identical_results_and_events(self):
        first, first_out = self.run_cli("memory_pressure.json", "first")
        second, second_out = self.run_cli("memory_pressure.json", "second")
        self.assertEqual((first.returncode, second.returncode), (0, 0), first.stderr + second.stderr)
        for name in ["result.json", "requests.jsonl", "events.jsonl"]:
            self.assertEqual((first_out / name).read_bytes(), (second_out / name).read_bytes(), name)

    def test_memory_pressure_finishes_without_losing_output(self):
        process, output = self.run_cli("memory_pressure.json")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["counts"]["completed"], 2)
        self.assertGreater(result["preemption_count"], 0)
        self.assertGreater(result["recomputed_tokens"], 0)
        rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([row["output_tokens_actual"] for row in rows], [24, 24])
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in result["memory"]))

    def test_pp_supports_analysis_and_online_simulation(self):
        prediction, output = self.run_cli("pp2_analytical.json", "prediction", "predict.py")
        self.assertEqual(prediction.returncode, 0, prediction.stdout + prediction.stderr)
        self.assertTrue((output / "result.json").exists())
        simulation, online = self.run_cli("pp2_analytical.json", "simulation")
        self.assertEqual(simulation.returncode, 0, simulation.stdout + simulation.stderr)
        result = json.loads((online / "result.json").read_text())
        self.assertEqual(result["counts"]["completed"], 3)
        self.assertIn("pipeline", result)
        self.assertTrue((online / "manifest.json").exists())

    def test_invalid_config_is_machine_readable(self):
        config = self.directory / "bad.json"
        config.write_text('{"strategy":{"prefix_cache":true}}')
        output = self.directory / "bad-run"
        process = subprocess.run([sys.executable, str(ROOT / "simulate.py"), "--experiment",
                                  str(config), "--out", str(output)], cwd=ROOT,
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout)["status"], "FAILED")
        self.assertTrue((output / "logs.txt").read_text())


if __name__ == "__main__":
    unittest.main()
