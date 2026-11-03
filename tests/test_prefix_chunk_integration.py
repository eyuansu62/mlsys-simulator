"""Packaged examples, replay artifacts and profile-driven TP/PP prefix reuse."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryConfig,
                       RequestSpec, StrategySpec, calibrate_dataset, profile_identity,
                       simulate, write_json_exclusive)


ROOT = Path(__file__).resolve().parents[1]


class PrefixChunkIntegrationTests(unittest.TestCase):
    def test_public_cli_examples_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            for example, chunks, computed, reused in [("prefix_cache", 2, 16, 8),
                                                       ("chunked_prefill", 6, 24, 0),
                                                       ("pp_prefix_chunk", 4, 16, 8)]:
                outputs = []
                for repetition in range(2):
                    output = Path(temporary) / f"{example}-{repetition}"
                    process = subprocess.run([sys.executable, "-m", "mlsys_sim", "simulate",
                        "--experiment", str(ROOT / "examples" / "v1_2" / f"{example}.json"),
                        "--out", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    result = json.loads((output / "result.json").read_text())
                    self.assertEqual(result["counts"]["completed"], 2)
                    self.assertEqual(result["prefill"]["completed_chunks"], chunks)
                    self.assertEqual(result["prefill"]["computed_tokens"], computed)
                    self.assertEqual(result["prefill"]["reused_prefix_tokens"], reused)
                    self.assertEqual(result["prefix_cache"]["pinned_blocks"], 0)
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertEqual(manifest["comparison_contract"]["strategy"]["prefix_cache"],
                                     result["prefix_cache"]["enabled"])
                    outputs.append(output)
                for artifact in ("result.json", "requests.jsonl", "events.jsonl"):
                    self.assertEqual((outputs[0] / artifact).read_bytes(),
                                     (outputs[1] / artifact).read_bytes(), (example, artifact))

    def test_tp_pp_profile_models_each_chunk_and_cached_suffix_context(self):
        exp = Experiment(
            workload=(RequestSpec("first", 0, 8, 1, prompt_token_ids=tuple(range(8))),
                      RequestSpec("reuse", 10, 8, 1, prompt_token_ids=tuple(range(8)))),
            infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(4))),
            strategy=StrategySpec(prefix_cache=True, chunked_prefill=True, prefill_chunk_tokens=4,
                max_batch_tokens=4, tensor_parallel_size=2, pipeline_parallel_size=2,
                pipeline_p2p=False, memory=MemoryConfig(kv_block_tokens=2)))
        data = {"schema_version": 1, "dataset_id": "tp-pp-cached-context-fixture",
                "identity": profile_identity(exp), "source": {"kind": "test_fixture",
                "reference": "Hand-calculated stage times, not device measurements"},
                "records": [{"record_id": f"s{stage}-c{context}", "group_id": f"s{stage}-c{context}",
                    "phase": "prefill", "stage_index": stage, "query_tokens": [query],
                    "context_lengths": [context], "duration_ms": 1,
                    "workspace_peak_bytes": 100 + 10 * context + stage}
                    for stage in range(2) for query, context in [(4, 0), (4, 4), (2, 6)]]}
        profile = calibrate_dataset(data, profile_version="tp-pp-chunks", allow_test_fixture=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            write_json_exclusive(path, profile)
            result = simulate(replace(exp, calibration=CalibrationSpec(backend="profile",
                profile_path=str(path), profile_version="tp-pp-chunks", allow_test_fixture=True)))
        self.assertEqual(result.result["counts"]["completed"], 2)
        self.assertEqual([row["token_times_ms"] for row in result.requests], [[4], [12]])
        self.assertEqual(result.requests[1]["prefix_cached_tokens"], 6)
        self.assertEqual(result.requests[1]["prefill_computed_tokens"], 2)
        self.assertEqual([rank["workspace_peak_bytes"] for rank in result.result["memory"]],
                         [160, 160, 161, 161])
        stage_starts = [event for event in result.events if event["event"] == "STAGE_START"]
        self.assertTrue(all(event["source"] == "test_fixture_profile" for event in stage_starts))
        self.assertEqual([event["context_lengths"] for event in stage_starts],
                         [[0], [0], [4], [4], [6], [6]])


if __name__ == "__main__":
    unittest.main()
