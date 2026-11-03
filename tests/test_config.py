"""Input contracts and unsupported mechanism boundaries."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim.config import experiment_from_dict, load_experiment, validate_experiment
from mlsys_sim.schema import DeviceSpec, Experiment, InfraSpec, SimulationError


class ConfigTests(unittest.TestCase):
    def assert_invalid(self, value, code="INVALID_CONFIG"):
        with self.assertRaises(SimulationError) as caught:
            experiment_from_dict(value)
        self.assertEqual(caught.exception.code, code)

    def test_defaults_and_empty_workload_are_valid(self):
        exp = experiment_from_dict({"workload": [], "infra": {}, "strategy": {}})
        self.assertEqual(exp, Experiment())

    def test_unknown_fields_do_not_silently_change_model(self):
        self.assert_invalid({"strategy": {"prefix_cache_typo": True}})
        self.assert_invalid({"calibration": {"compute_effeciency": 0.3}})

    def test_nonfinite_bool_and_wrong_types_are_rejected(self):
        self.assert_invalid({"infra": {"link_latency_us": float("nan")}})
        self.assert_invalid({"strategy": {"max_batch_size": True}})
        self.assert_invalid({"execution": {"detailed_events": 1}})
        self.assert_invalid({"workload": [{"request_id": "x", "arrival_ms": 0,
                                           "input_tokens": 1, "output_tokens": "4"}]})

    def test_oversized_integers_are_rejected_before_layout_arithmetic(self):
        huge = 10 ** 200
        model = {"num_layers": 1, "hidden_size": huge, "intermediate_size": huge,
                 "head_dim": huge, "num_attention_heads": 1, "num_kv_heads": 1,
                 "vocab_size": 1}
        self.assert_invalid({"strategy": {"model": model}})
        for seed in (2 ** 63, -(2 ** 63) - 1):
            with self.subTest(seed=seed):
                self.assert_invalid({"execution": {"seed": seed}})
        for seed in (2 ** 63 - 1, -(2 ** 63)):
            self.assertEqual(experiment_from_dict({"execution": {"seed": seed}})
                             .execution.seed, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.json"
            path.write_text(json.dumps({"strategy": {"model": model}}))
            with self.assertRaises(SimulationError) as caught:
                load_experiment(path)
            self.assertEqual(caught.exception.code, "INVALID_CONFIG")
            self.assertIn("signed 64-bit", str(caught.exception))

    def test_direct_dataclass_construction_is_validated(self):
        with self.assertRaises(SimulationError):
            validate_experiment(replace(Experiment(), infra=InfraSpec(
                devices=(DeviceSpec(compute_tflops=float("inf")),))))
        with self.assertRaises(SimulationError):
            validate_experiment(replace(Experiment(), strategy={}))

    def test_duplicate_requests_and_negative_lengths_are_invalid(self):
        req = {"request_id": "x", "arrival_ms": 0, "input_tokens": 2, "output_tokens": 3}
        self.assert_invalid({"workload": [req, req]})
        self.assert_invalid({"workload": [{**req, "input_tokens": -1}]})

    def test_sharding_and_placement_constraints(self):
        valid = {"strategy": {"tensor_parallel_size": 4},
                 "infra": {"devices": [{"id": f"gpu{i}"} for i in range(4)]}}
        exp = experiment_from_dict(valid)
        self.assertEqual(exp.strategy.tensor_parallel_size, 4)
        self.assert_invalid({**valid, "strategy": {"tensor_parallel_size": 4,
                                                   "model": {"num_kv_heads": 3}}})
        self.assert_invalid({"strategy": {"tensor_parallel_size": 2}})
        self.assert_invalid({"strategy": {"pipeline_parallel_size": 2, "stage_layers": [3, 2]},
                             "infra": {"devices": [{"id": "a"}, {"id": "b"}]}})

    def test_unsupported_backends_and_modes_are_explicit(self):
        self.assert_invalid({"calibration": {"backend": "measured"}}, "UNSUPPORTED_MECHANISM")
        self.assert_invalid({"strategy": {"preemption": "nvme_swap"}}, "UNSUPPORTED_MECHANISM")
        self.assert_invalid({"strategy": {"replicas": 2}}, "UNSUPPORTED_MECHANISM")

    def test_quantized_storage_does_not_silently_omit_metadata(self):
        for field in ("weight_bytes", "kv_bytes", "activation_bytes"):
            for width in (-1.0, 0.0, 0.5, 1.0, 1.25, 3.0):
                with self.subTest(field=field, width=width):
                    self.assert_invalid({"strategy": {"model": {field: width}}},
                                        "UNSUPPORTED_MECHANISM")
            for width in (2.0, 4.0, 8.0):
                with self.subTest(field=field, width=width):
                    exp = experiment_from_dict({"strategy": {"model": {field: width}}})
                    self.assertEqual(getattr(exp.strategy.model, field), width)

    def test_jsonl_and_toml_section_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requests.jsonl").write_text(json.dumps(
                {"request_id": "r", "arrival_ms": 0, "input_tokens": 2, "output_tokens": 3}) + "\n")
            (root / "strategy.toml").write_text('scheduler = "fifo"\n')
            (root / "experiment.toml").write_text(
                'schema_version = 1\nworkload = "requests.jsonl"\nstrategy = "strategy.toml"\n')
            exp = load_experiment(root / "experiment.toml")
            self.assertEqual(exp.workload[0].input_tokens, 2)
            self.assertEqual(exp.strategy.scheduler, "fifo")

    def test_duplicate_json_keys_and_missing_files_are_config_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema_version":1,"schema_version":2}')
            with self.assertRaises(SimulationError):
                load_experiment(path)
            with self.assertRaises(SimulationError):
                load_experiment(path.with_name("missing.json"))

    def test_model_reference_resolves_relative_to_strategy_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "specs").mkdir()
            (root / "specs/model.json").write_text('{"name":"referenced-model"}')
            (root / "specs/strategy.json").write_text('{"model":"model.json"}')
            (root / "experiment.json").write_text('{"strategy":"specs/strategy.json"}')
            self.assertEqual(load_experiment(root / "experiment.json").strategy.model.name,
                             "referenced-model")

    def test_optional_yaml_parser_rejects_malformed_and_duplicate_keys(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is optional")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for content in ['schema_version: [broken', 'schema_version: 1\nschema_version: 1\n']:
                path.write_text(content)
                with self.assertRaises(SimulationError):
                    load_experiment(path)


if __name__ == "__main__":
    unittest.main()
