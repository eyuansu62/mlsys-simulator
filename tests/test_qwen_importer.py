"""Exact Qwen3 config mapping, behavior fingerprints and reference compatibility."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlsys_sim import Experiment, SimulationError, StrategySpec
from mlsys_sim.layout import rank_kv_bytes_per_token, rank_weight_bytes
from mlsys_sim.profiles import profile_identity
from mlsys_sim.qwen3 import (load_qwen3_model, model_from_qwen3_config,
                            normalized_qwen3_architecture, qwen3_architecture_fingerprint)
from mlsys_sim.release_adapter import _model_summary


def config(size="1.7b"):
    large = size == "4b"
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "num_hidden_layers": 36 if large else 28, "hidden_size": 2560 if large else 2048,
        "num_attention_heads": 32 if large else 16, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 9728 if large else 6144, "vocab_size": 151936,
        "tie_word_embeddings": True, "torch_dtype": "bfloat16", "hidden_act": "silu",
        "attention_bias": False, "attention_dropout": 0.0, "use_cache": True,
        "use_sliding_window": False, "sliding_window": None, "max_position_embeddings": 40960,
        "max_window_layers": 36 if large else 28, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000, "rope_scaling": None, "transformers_version": "4.51.0"}


def imported(value=None):
    return model_from_qwen3_config(config() if value is None else value,
                                   name="Qwen3-reference", revision="explicit-model-revision")


class Qwen3ImporterTests(unittest.TestCase):
    def test_actual_model_shapes_and_logical_weight_counts_are_preserved(self):
        for size, query, weights, kv in (("1.7b", 2048, 3441149952, 114688),
                                         ("4b", 4096, 8044936192, 147456)):
            with self.subTest(size=size):
                raw = config(size)
                model = imported(raw)
                exp = Experiment(strategy=StrategySpec(model=model))
                self.assertEqual(model.hidden_size, raw["hidden_size"])
                self.assertEqual(model.query_projection_size, query)
                self.assertEqual(rank_weight_bytes(exp, 0), weights)
                self.assertEqual(rank_kv_bytes_per_token(exp, 0), kv)
                self.assertTrue(model.tie_word_embeddings and model.qk_norm and model.final_norm)
                self.assertEqual(model.weight_dtype, "bfloat16")

    def test_explicit_untied_model_and_kv_override_remain_distinct(self):
        tied = imported()
        raw = config()
        raw["tie_word_embeddings"] = False
        untied = imported(raw)
        delta = rank_weight_bytes(Experiment(strategy=StrategySpec(model=untied)), 0)
        delta -= rank_weight_bytes(Experiment(strategy=StrategySpec(model=tied)), 0)
        self.assertEqual(delta, tied.vocab_size * tied.hidden_size * 2)
        overridden = model_from_qwen3_config(config(), name=tied.name, revision=tied.revision, kv_dtype="float32")
        self.assertEqual(overridden.kv_bytes, 4)
        self.assertEqual(overridden.weight_bytes, 2)
        self.assertEqual(overridden.architecture_config_sha256, tied.architecture_config_sha256)
        self.assertNotEqual(profile_identity(Experiment(strategy=StrategySpec(model=overridden))),
                            profile_identity(Experiment(strategy=StrategySpec(model=tied))))

    def test_family_and_required_geometry_are_explicit(self):
        changes = [{"model_type": "qwen3_moe"}, {"architectures": ["Qwen3MoeForCausalLM"]},
                   {"architectures": "Qwen3ForCausalLM"}, {"num_hidden_layers": True},
                   {"head_dim": None}, {"hidden_size": 0}, {"tie_word_embeddings": 1}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(SimulationError):
                imported({**config(), **change})
        for missing in ("head_dim", "tie_word_embeddings", "model_type", "architectures"):
            raw = config()
            raw.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(SimulationError):
                imported(raw)

    def test_no_revision_or_current_runtime_version_is_inferred(self):
        raw = config()
        raw["transformers_version"] = "version-from-an-old-model-file"
        for name, revision in (("", "rev"), ("name", "unknown"), ("name", ""), (" name", "rev")):
            with self.subTest(name=name, revision=revision), self.assertRaises(SimulationError):
                model_from_qwen3_config(raw, name=name, revision=revision)
        model = imported(raw)
        self.assertEqual(model.revision, "explicit-model-revision")
        self.assertNotEqual(model.revision, raw["transformers_version"])

    def test_full_rotary_attention_rejects_odd_head_dimensions(self):
        for dimension in (1, 3, 127):
            with self.subTest(dimension=dimension), self.assertRaises(SimulationError) as raised:
                imported({**config(), "head_dim": dimension})
            self.assertEqual(raised.exception.code, "UNSUPPORTED_MECHANISM")

    def test_storage_dtype_is_explicit_and_conflicts_cannot_silently_cast(self):
        for dtype, width in (("float16", 2), ("bfloat16", 2), ("float32", 4), ("float64", 8)):
            raw = {**config(), "dtype": dtype, "torch_dtype": dtype}
            model = imported(raw)
            self.assertEqual(model.weight_dtype, dtype)
            self.assertEqual(model.weight_bytes, width)
        for change in ({"dtype": "float16"}, {"torch_dtype": "auto"}, {"torch_dtype": "float8_e4m3fn"},
                       {"torch_dtype": None}, {"torch_dtype": 2}):
            with self.subTest(change=change), self.assertRaises(SimulationError):
                imported({**config(), **change})
        raw = config()
        raw["dtype"] = raw.pop("torch_dtype")
        self.assertEqual(imported(raw).weight_dtype, "bfloat16")
        for kv in ("auto", "int8", 2):
            with self.subTest(kv=kv), self.assertRaises(SimulationError):
                model_from_qwen3_config(config(), name="model", revision="revision", kv_dtype=kv)

    def test_unsupported_attention_moe_quantization_and_code_are_rejected(self):
        changes = [{"attention_bias": True}, {"hidden_act": "gelu"}, {"attention_dropout": .1},
                   {"attention_dropout": False}, {"quantization_config": {}},
                   {"quantization_config": {"quant_method": "fp8"}}, {"num_experts": 0},
                   {"use_sliding_window": True, "sliding_window": 4096},
                   {"layer_types": ["sliding_attention"] * 28},
                   {"rope_scaling": {"rope_type": "yarn", "factor": 2}},
                   {"qk_norm": False}, {"final_norm": False}, {"use_cache": False},
                   {"auto_map": {"AutoModel": "execute_this_module"}}, {"add_cross_attention": True}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(SimulationError):
                imported({**config(), **change})
        with self.assertRaises(SimulationError) as raised:
            imported({**config(), "quantization_config": {"quant_method": "fp8"}})
        self.assertEqual(raised.exception.code, "UNSUPPORTED_MECHANISM")
        self.assertIn("weight_dequant", str(raised.exception))
        self.assertIn("docs/quantization.md", str(raised.exception))

    def test_unknown_fields_are_rejected_and_known_metadata_is_fingerprinted_separately(self):
        with self.assertRaises(SimulationError):
            imported({**config(), "invented_attention_backend": "custom"})
        first = config()
        second = {**first, "_name_or_path": "PRIVATE_PATH_METADATA", "_commit_hash": "metadata-commit",
                  "transformers_version": "new-metadata-version", "temperature": .5, "num_beams": 1}
        self.assertEqual(qwen3_architecture_fingerprint(first), qwen3_architecture_fingerprint(second))
        self.assertEqual(imported(first), imported(second))

    def test_behavior_fingerprint_preserves_norm_rope_context_and_storage(self):
        base = qwen3_architecture_fingerprint(config())
        for change in ({"rms_norm_eps": 1e-5}, {"rope_theta": 2000000},
                       {"max_position_embeddings": 8192}, {"torch_dtype": "float16"},
                       {"tie_word_embeddings": False}, {"hidden_size": 2560}):
            with self.subTest(change=change):
                self.assertNotEqual(qwen3_architecture_fingerprint({**config(), **change}), base)
        equivalent = {**config(), "rms_norm_eps": .000001, "rope_theta": 1000000.0,
                      "layer_types": ["full_attention"] * 28, "sliding_window": 4096}
        self.assertEqual(qwen3_architecture_fingerprint(equivalent), base)

    def test_default_normalization_matches_resolved_hf_values(self):
        raw = config()
        for key in ("rms_norm_eps", "rope_theta", "max_position_embeddings", "max_window_layers"):
            raw.pop(key)
        resolved = {**raw, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                    "max_position_embeddings": 32768, "max_window_layers": 28}
        self.assertEqual(normalized_qwen3_architecture(raw), normalized_qwen3_architecture(resolved))
        nested = config()
        theta = nested.pop("rope_theta")
        nested["rope_parameters"] = {"rope_type": "default", "rope_theta": theta}
        self.assertEqual(qwen3_architecture_fingerprint(config()), qwen3_architecture_fingerprint(nested))
        with self.assertRaises(SimulationError):
            imported({**config(), "rope_parameters": {"rope_type": "default", "rope_theta": 10000}})

    def test_import_does_not_mutate_the_config(self):
        raw = config("4b")
        before = deepcopy(raw)
        imported(raw)
        self.assertEqual(raw, before)

    def test_load_preserves_raw_hash_and_does_not_copy_metadata_values(self):
        raw = {**config("4b"), "_name_or_path": "PRIVATE_PATH_METADATA"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            content = json.dumps(raw, indent=2).encode()
            path.write_bytes(content)
            bundle = load_qwen3_model(path, name="Qwen3-4B", revision="revision-4b")
        self.assertEqual(bundle["provenance"]["source_sha256"], hashlib.sha256(content).hexdigest())
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(bundle["provenance"]["canonical_config_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(bundle["provenance"]["architecture_config_sha256"], bundle["model"]["architecture_config_sha256"])
        self.assertIn("_name_or_path", bundle["provenance"]["ignored_metadata_fields"])
        self.assertNotIn("PRIVATE_PATH_METADATA", json.dumps(bundle))
        self.assertFalse(bundle["hardware_validated"])

    def test_nonfinite_wrong_json_duplicate_keys_and_size_are_rejected(self):
        for change in ({"rms_norm_eps": float("nan")}, {"rope_theta": float("inf")},
                       {"rms_norm_eps": True}, {"initializer_range": -1}, {"_name_or_path": "\ud800"}):
            with self.subTest(change=repr(change)), self.assertRaises(SimulationError):
                imported({**config(), **change})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for raw in (b'{"model_type":"qwen3","model_type":"qwen3"}', b'{"bad":NaN}', b'[]', b'\xff'):
                path.write_bytes(raw)
                with self.subTest(raw=raw), self.assertRaises(SimulationError):
                    load_qwen3_model(path, name="model", revision="revision")
            path.write_text(json.dumps(config()))
            with patch("mlsys_sim.qwen3._MAX_BYTES", 100), self.assertRaises(SimulationError):
                load_qwen3_model(path, name="model", revision="revision")

    def test_reference_summary_accepts_tied_and_independent_query_geometry(self):
        for size in ("1.7b", "4b"):
            with self.subTest(size=size):
                summary = _model_summary("qwen-reference", {"served_name": "Qwen3-reference",
                    "hf_revision": "reference-revision"}, config(size))
                self.assertTrue(summary["simulator_model_compatible"])
                self.assertTrue(summary["geometry_supported"])
                self.assertEqual(summary["compatibility_reasons"], [])
                self.assertEqual(summary["query_projection_size"], imported(config(size)).query_projection_size)
                self.assertEqual(summary["model_spec"]["tie_word_embeddings"], True)

    def test_reference_summary_does_not_hide_unsupported_storage_or_missing_identity(self):
        summary = _model_summary("qwen-reference", {"hf_revision": "reference-revision"},
                                {**config(), "quantization_config": {"quant_method": "fp8"}})
        self.assertFalse(summary["geometry_supported"])
        self.assertIn("QUANTIZED_FORMAT_NOT_SUPPORTED", summary["compatibility_reasons"])
        self.assertNotIn("model_spec", summary)
        incomplete = _model_summary("qwen-reference", {}, config())
        self.assertFalse(incomplete["geometry_supported"])
        self.assertIn("MODEL_REVISION_NOT_PROVIDED", incomplete["compatibility_reasons"])

    def test_model_architecture_hash_participates_in_profile_identity(self):
        first = imported()
        second = imported({**config(), "rms_norm_eps": 1e-5})
        identity_a = profile_identity(Experiment(strategy=StrategySpec(model=first)))
        identity_b = profile_identity(Experiment(strategy=StrategySpec(model=second)))
        self.assertNotEqual(identity_a, identity_b)
        self.assertEqual(identity_a["model"]["architecture_config_sha256"], first.architecture_config_sha256)
        self.assertEqual(asdict(first)["hidden_size"], 2048)


if __name__ == "__main__":
    unittest.main()
