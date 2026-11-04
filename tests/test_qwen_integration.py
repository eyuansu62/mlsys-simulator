"""Public model imports and existing serving mechanisms with true Qwen3 widths."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryConfig, ModelSpec,
                       RequestSpec, SimulationError, StrategySpec, compare_runs, comparison_contract,
                       load_experiment, model_from_qwen3_config, simulate)
from mlsys_sim.layout import rank_weight_bytes


ROOT = Path(__file__).resolve().parents[1]


class QwenIntegrationTests(unittest.TestCase):
    def model(self, name="qwen3-4b"):
        config = json.loads((ROOT / "examples/v1_6" / f"{name}.hf.json").read_text())
        return model_from_qwen3_config(config, name=name, revision="fixture-integration-revision")

    def test_import_model_cli_preserves_raw_config_and_independent_dimensions(self):
        config = ROOT / "examples/v1_6/qwen3-4b.hf.json"
        original = config.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model"
            args = [sys.executable, "-m", "mlsys_sim", "import-model", "--kind", "qwen3",
                    "--config", str(config), "--name", "Qwen3-4B", "--revision", "fixture-revision",
                    "--out", str(output)]
            process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads((output / "result.json").read_text())
            model = json.loads((output / "model.json").read_text())
            self.assertEqual(result["dimensions"]["hidden_size"], 2560)
            self.assertEqual(result["dimensions"]["query_projection_size"], 4096)
            self.assertEqual(model["head_dim"], 128)
            self.assertTrue(all(model[key] for key in ("tie_word_embeddings", "qk_norm", "final_norm")))
            self.assertEqual(result["provenance"]["source_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(model["architecture_config_sha256"],
                             result["provenance"]["architecture_config_sha256"])
            self.assertFalse(result["hardware_validated"])
            previous = (output / "model.json").read_bytes()
            repeated = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual(json.loads(repeated.stdout)["error_code"], "OUTPUT_EXISTS")
            self.assertEqual((output / "model.json").read_bytes(), previous)
        self.assertEqual(config.read_bytes(), original)

    def test_tied_vocabulary_changes_static_capacity_without_freeing_lm_head_work(self):
        model = self.model()
        memory = MemoryConfig(kv_pool_bytes=147456 * 16, workspace_base_bytes=0,
                              workspace_bytes_per_token=0, runtime_bytes=0, safety_margin_bytes=0)
        base = Experiment(strategy=StrategySpec(model=model, memory=memory),
            workload=(RequestSpec("r", 0, 1, 1),),
            calibration=CalibrationSpec(backend="synthetic", prefill_ms=2))
        capacity = rank_weight_bytes(base, 0) + memory.kv_pool_bytes
        exp = replace(base, infra=InfraSpec(devices=(DeviceSpec("gpu", memory_bytes=capacity),)))
        run = simulate(exp)
        self.assertEqual(run.requests[0]["token_times_ms"], [2])
        self.assertEqual(run.result["memory"][0]["physical_peak_bytes"], capacity)
        untied = replace(exp, strategy=replace(exp.strategy, model=replace(model,
                         tie_word_embeddings=False, architecture_config_sha256="")))
        with self.assertRaises(SimulationError) as raised:
            simulate(untied)
        self.assertEqual(raised.exception.code, "STATIC_MEMORY_INFEASIBLE")

    def test_qwen_r2_tp2_pp2_cache_swap_and_mixed_public_example_replays(self):
        exp = load_experiment(ROOT / "examples/v1_6/qwen3-r2-tp2-pp2.json")
        self.assertEqual(exp.strategy.model.hidden_size, 2560)
        self.assertEqual(exp.strategy.model.query_projection_size, 4096)
        first = simulate(exp)
        second = simulate(replace(exp, execution=replace(exp.execution, detailed_events=False)))
        self.assertEqual(first.result, second.result)
        self.assertEqual(first.requests, second.requests)
        self.assertEqual(second.events, [])
        self.assertEqual(first.result["counts"]["completed"], 10)
        self.assertEqual(first.result["batching"]["mixed_batches_completed"], 4)
        self.assertEqual(first.result["prefix_cache"]["hit_tokens"], 24)
        self.assertEqual(first.result["swap"]["completed_transfers_by_direction"], {"D2H": 2, "H2D": 2})
        self.assertEqual(first.result["recomputed_tokens"], 0)
        self.assertEqual(first.result["prefix_cache"]["attached_block_references"], 0)
        self.assertEqual(first.result["prefix_cache"]["private_blocks"], 0)
        self.assertEqual(first.result["swap"]["host_reserved_bytes"], 0)
        self.assertEqual(len(first.result["memory"]), 8)
        for row in first.requests:
            self.assertEqual(len(row["stage_computed_tokens"]), 2)
            self.assertEqual(row["stage_computed_tokens"], [row["num_computed_tokens"]] * 2)

    def test_runtime_execution_hash_is_part_of_online_comparison(self):
        model = ModelSpec(hidden_size=8, num_attention_heads=4, num_kv_heads=2, head_dim=4,
                          intermediate_size=16, vocab_size=32, tie_word_embeddings=True,
                          qk_norm=True, final_norm=True, architecture_config_sha256="b" * 64)
        base = Experiment(strategy=StrategySpec(model=model),
                          calibration=CalibrationSpec(runtime_config_sha256="a" * 64))
        current = comparison_contract(base)
        observed = {"schema_version": 1, "kind": "measured_run", "comparison_contract": deepcopy(current),
                    "source": {"kind": "test_fixture", "reference": "Runtime policy identity fixture"},
                    "metrics": {"request_count": 0}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predicted = root / "predicted"
            predicted.mkdir()
            (predicted / "manifest.json").write_text(json.dumps({"backend": "simulated", "comparison_contract": current}))
            (predicted / "result.json").write_text(json.dumps({"backend": "simulated", "status": "COMPLETED",
                                                               "counts": {"total": 0}}))
            path = root / "observed.json"
            path.write_text(json.dumps(observed))
            self.assertEqual(compare_runs(predicted, path, allow_test_fixture=True)
                             ["metrics"]["request_count"]["absolute_error"], 0)
            for replacement in ("c" * 64, None):
                changed = deepcopy(observed)
                runtime = changed["comparison_contract"]["identity"]["runtime"]
                if replacement is None:
                    runtime.pop("runtime_config_sha256")
                else:
                    runtime["runtime_config_sha256"] = replacement
                path.write_text(json.dumps(changed))
                with self.subTest(replacement=replacement), self.assertRaises(SimulationError) as raised:
                    compare_runs(predicted, path, allow_test_fixture=True)
                self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")


if __name__ == "__main__":
    unittest.main()
