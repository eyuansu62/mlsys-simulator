"""Loaded storage, not transient checkpoint flags, establishes local tying.

All tensors/runners below are source-shaped CPU fixtures; no Torch or GPU is
loaded and no weight contents are compared to manufacture alias evidence.
"""

from copy import deepcopy
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlsys_sim import DeviceSpec, Experiment, InfraSpec, SimulationError, StrategySpec
from mlsys_sim.qwen3 import model_from_qwen3_config
from mlsys_sim.vllm_hooks import (_config_digest, _execution_config, _shape_model_matches,
    install_runner_hook, SUPPORTED_VERSION_PREFIX)
from test_vllm_hooks import Clock, Runner, schedule, state


class Device:
    type = "cuda"
    index = 0
    def __str__(self):
        return "cuda:0"


class Storage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size
    def data_ptr(self):
        return self.pointer
    def nbytes(self):
        return self.size


class Weight:
    def __init__(self, rows=256, hidden=128, pointer=4096):
        self.shape = (rows, hidden)
        self.dtype = "torch.bfloat16"
        self.device = Device()
        self.storage = Storage(pointer, rows * hidden * 2)
        self.offset = 0
    def stride(self):
        return (self.shape[1], 1)
    def untyped_storage(self):
        return self.storage
    def data_ptr(self):
        return self.storage.pointer + self.offset * 2
    def storage_offset(self):
        return self.offset
    def element_size(self):
        return 2
    def cpu(self):
        raise AssertionError("Storage proof must not copy weight contents")


class HFConfig(SimpleNamespace):
    def to_dict(self):
        # Deliberately return live storage to verify the matcher makes its own
        # copy before applying an observed effective tying policy.
        return self.raw


def configured(*, declared_tied=True, hf_tied=False, checkpoint_flag=True, pp=1):
    raw = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 4, "hidden_size": 128, "intermediate_size": 512,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 256, "tie_word_embeddings": declared_tied, "hidden_act": "silu",
        "attention_bias": False, "use_sliding_window": False, "sliding_window": None,
        "torch_dtype": "bfloat16", "rms_norm_eps": 1e-6, "rope_theta": 10000,
        "max_position_embeddings": 32768}
    model = model_from_qwen3_config(raw, name="checkpoint-qwen", revision="fixture-weights")
    exp = Experiment(strategy=StrategySpec(model=model, pipeline_parallel_size=pp),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}") for i in range(pp))))
    actual = deepcopy(raw)
    actual["tie_word_embeddings"] = hf_tied
    hf = HFConfig(**actual)
    hf.raw = actual
    runner = Runner()
    runner.model_config = SimpleNamespace(hf_config=hf, dtype="torch.bfloat16", revision=None,
        word_embeddings_untied_by_checkpoint=checkpoint_flag)
    runner.kv_cache_dtype = "torch.bfloat16"
    runner.device = Device()
    first, second = Weight(), Weight(pointer=131072)
    runner.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=first),
        start_layer=0, end_layer=4), lm_head=SimpleNamespace(weight=second))
    runner.requests = {"r": state(4)}
    return runner, exp


def alias_weights(runner, *, same_parameter=True):
    first = runner.model.model.embed_tokens.weight
    runner.model.lm_head.weight = first if same_parameter else Weight(pointer=first.storage.pointer)


def with_default_rope_alias():
    runner, exp = configured()
    hf = runner.model_config.hf_config
    original = deepcopy(hf.raw)
    original.update(tie_word_embeddings=True, rope_theta=1000000)
    model = model_from_qwen3_config(original, name=exp.strategy.model.name, revision=exp.strategy.model.revision)
    exp = replace(exp, strategy=replace(exp.strategy, model=model))
    parameters = {"rope_type": "default", "rope_theta": 1000000}
    hf.raw.pop("rope_theta")
    hf.raw["rope_parameters"] = deepcopy(parameters)
    hf.rope_theta = None
    hf.rope_parameters = parameters
    hf.rope_scaling = parameters  # HF 5.18 exposes an attribute-only alias.
    return runner, exp


class CheckpointTyingTests(unittest.TestCase):
    def hook(self, runner, exp):
        hook = install_runner_hook(runner, exp, session_id="checkpoint-fixture", runtime_version=SUPPORTED_VERSION_PREFIX,
                                   clock_ns=Clock())
        self.addCleanup(hook.close)
        return hook

    def test_default_direct_matcher_remains_strict_before_loading(self):
        runner, exp = configured()
        with self.assertRaises(SimulationError) as raised:
            _shape_model_matches(runner, exp)
        self.assertEqual(raised.exception.code, "HARDWARE_IDENTITY_MISMATCH")
        pending = _shape_model_matches(runner, exp, _phase="construction")["word_embedding_tying"]
        self.assertTrue(pending["pending_loaded_storage_proof"])
        self.assertIsNone(pending["effective_tie_word_embeddings"])
        self.assertEqual(pending["architecture_fingerprint_status"], "pending_loaded_storage_proof")

    def test_checkpoint_retie_is_accepted_only_after_parameter_alias_is_observed(self):
        runner, exp = configured()
        original = deepcopy(runner.model_config.hf_config.raw)
        hook = self.hook(runner, exp)
        self.assertTrue(hook.header["metadata"]["model_binding"]["word_embedding_tying"]["pending_loaded_storage_proof"])
        alias_weights(runner)
        runner.execute_model(schedule({"r": 4}))
        runner.sample_tokens(None)
        raw = hook.export_capture()
        self.assertEqual(raw["source"]["kind"], "test_fixture")
        self.assertEqual(raw["records"][0]["status"], "COMPLETED")
        self.assertTrue(raw["records"][0]["profile_eligible"])
        tying = raw["metadata"]["model_binding"]["word_embedding_tying"]
        self.assertFalse(tying["startup_hf_tie_word_embeddings"])
        self.assertTrue(tying["startup_word_embeddings_untied_by_checkpoint"])
        self.assertTrue(tying["constructor_pending_loaded_storage_proof"])
        self.assertFalse(tying["pending_loaded_storage_proof"])
        self.assertEqual(tying["verification_phase"], "prepared")
        self.assertEqual(tying["architecture_fingerprint_status"], "verified")
        self.assertTrue(tying["local_alias"]["same_parameter"])
        self.assertTrue(tying["local_alias"]["same_tensor_view"])
        self.assertEqual(runner.model_config.hf_config.raw, original)
        self.assertFalse(runner.model_config.hf_config.tie_word_embeddings)

    def test_identical_tensor_view_alias_does_not_require_same_python_parameter(self):
        runner, exp = configured()
        alias_weights(runner, same_parameter=False)
        tying = _shape_model_matches(runner, exp, _phase="prepared")["word_embedding_tying"]
        self.assertFalse(tying["local_alias"]["same_parameter"])
        self.assertTrue(tying["local_alias"]["same_storage"])
        self.assertTrue(tying["effective_tie_word_embeddings"])

    def test_still_untied_checkpoint_and_missing_storage_proof_are_not_canonical(self):
        for missing in (False, True):
            runner, exp = configured()
            hook = self.hook(runner, exp)
            if missing:
                runner.model = None
            runner.execute_model(schedule({"r": 4}))
            runner.sample_tokens(None)
            row = hook.export_capture()["records"][0]
            with self.subTest(missing=missing):
                self.assertEqual(row["status"], "UNSUPPORTED")
                self.assertFalse(row["profile_eligible"])
                self.assertFalse(row["forward_complete"])
                self.assertEqual(runner.calls, ["execute", "prepare", "sample"])

    def test_false_hf_flag_without_checkpoint_transition_is_not_a_pending_tie(self):
        for flag in (False, 1, "true"):
            runner, exp = configured(checkpoint_flag=flag)
            original_execute = runner.execute_model.__func__
            with self.subTest(flag=flag), self.assertRaises(SimulationError):
                self.hook(runner, exp)
            self.assertIs(runner.execute_model.__func__, original_execute)
            self.assertNotIn("_mlsys_internal_capture", vars(runner))
            self.assertNotIn("execute_model", vars(runner))
            self.assertNotIn("_prepare_inputs", vars(runner))

    def test_unexpected_alias_of_declared_untied_weights_is_rejected(self):
        runner, exp = configured(declared_tied=False, hf_tied=False, checkpoint_flag=False)
        alias_weights(runner)
        hook = self.hook(runner, exp)
        runner.execute_model(schedule({"r": 4}))
        self.assertFalse(hook.records[0]["profile_eligible"])
        self.assertEqual(hook.records[0]["status"], "UNSUPPORTED")
        self.assertEqual(hook.records[0]["error_code"], "HARDWARE_IDENTITY_MISMATCH")

    def test_shared_backing_with_a_different_tensor_view_is_not_effective_tying(self):
        runner, exp = configured()
        alias_weights(runner, same_parameter=False)
        runner.model.lm_head.weight.offset = 1
        runner.model.lm_head.weight.storage.size += 2
        with self.assertRaises(SimulationError):
            _shape_model_matches(runner, exp, _phase="prepared")

    def test_pending_pipeline_endpoints_cannot_claim_cross_gpu_alias(self):
        runner, exp = configured(pp=2)
        _shape_model_matches(runner, exp, _phase="construction")
        alias_weights(runner)
        with self.assertRaises(SimulationError) as raised:
            _shape_model_matches(runner, exp, _phase="prepared")
        self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")

    def test_other_model_guards_remain_strict_during_checkpoint_transition(self):
        for change in ("geometry", "dtype", "revision", "rope"):
            runner, exp = configured()
            if change == "geometry":
                runner.model_config.hf_config.head_dim += 2
            elif change == "dtype":
                runner.kv_cache_dtype = "torch.float16"
            elif change == "revision":
                runner.model_config.revision = "different-weights"
            else:
                runner.model_config.hf_config.rope_scaling = {"rope_type": "yarn", "factor": 2}
            with self.subTest(change=change), self.assertRaises(SimulationError):
                _shape_model_matches(runner, exp, _phase="construction")

    def test_loaded_binding_cannot_change_mid_capture(self):
        runner, exp = configured()
        alias_weights(runner)
        hook = self.hook(runner, exp)
        runner.execute_model(schedule({"r": 4}))
        runner.sample_tokens(None)
        replacement = Weight(pointer=262144)
        runner.model.model.embed_tokens.weight = replacement
        runner.model.lm_head.weight = replacement
        runner.execute_model(schedule({"r": 4}))
        self.assertFalse(hook.records[1]["profile_eligible"])
        self.assertEqual(hook.records[1]["status"], "UNSUPPORTED")
        self.assertIn("changed", hook.records[1]["message"])

    def test_prepared_alias_normalization_does_not_hide_other_architecture_changes(self):
        runner, exp = configured()
        alias_weights(runner)
        original = deepcopy(runner.model_config.hf_config.raw)
        runner.model_config.hf_config.raw["rope_theta"] *= 2
        with self.assertRaises(SimulationError):
            _shape_model_matches(runner, exp, _phase="prepared")
        self.assertFalse(runner.model_config.hf_config.raw["tie_word_embeddings"])
        self.assertNotEqual(runner.model_config.hf_config.raw["rope_theta"], original["rope_theta"])

    def test_transport_policy_changes_the_actual_runtime_digest_and_restores_stably(self):
        runner, exp = configured()
        names = ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET",
                 "TORCH_NCCL_BLOCKING_WAIT")
        with patch.dict(os.environ, clear=False):
            for name in names:
                os.environ.pop(name, None)
            original = _execution_config(runner.vllm_config, exp, False)
            self.assertTrue(all(name not in original["environment"] for name in names))
            os.environ["NCCL_P2P_DISABLE"] = "1"
            first = _config_digest(_execution_config(runner.vllm_config, exp, False))
            os.environ["NCCL_P2P_DISABLE"] = "0"
            other = _config_digest(_execution_config(runner.vllm_config, exp, False))
            os.environ["NCCL_P2P_DISABLE"] = "1"
            restored = _execution_config(runner.vllm_config, exp, False)
            self.assertNotEqual(first, other)
            self.assertEqual(first, _config_digest(restored))
            self.assertEqual(restored["environment"]["NCCL_P2P_DISABLE"], "1")

    def test_hf_default_rope_alias_keeps_original_theta_fingerprint_after_retie(self):
        runner, exp = with_default_rope_alias()
        original = deepcopy(runner.model_config.hf_config.to_dict())
        hook = self.hook(runner, exp)
        alias_weights(runner)
        runner.execute_model(schedule({"r": 4}))
        runner.sample_tokens(None)
        raw = hook.export_capture()
        self.assertEqual(raw["records"][0]["status"], "COMPLETED")
        tying = raw["metadata"]["model_binding"]["word_embedding_tying"]
        self.assertEqual(tying["architecture_fingerprint_status"], "verified")
        self.assertTrue(tying["local_alias"]["same_parameter"])
        self.assertNotIn("rope_scaling", runner.model_config.hf_config.to_dict())
        self.assertEqual(runner.model_config.hf_config.to_dict(), original)
        self.assertEqual(runner.model_config.hf_config.rope_parameters["rope_theta"], 1000000)
        self.assertIsNone(runner.model_config.hf_config.rope_theta)

    def test_default_rope_alias_never_accepts_scaling_or_unknown_parameters(self):
        for parameters in ({"rope_type": "linear", "rope_theta": 1000000},
                           {"rope_type": "dynamic", "rope_theta": 1000000},
                           {"rope_type": "longrope", "rope_theta": 1000000},
                           {"rope_type": "default", "factor": 2},
                           {"rope_type": "default", "unknown": 1},
                           {"rope_theta": 1000000}, "default"):
            runner, exp = with_default_rope_alias()
            runner.model_config.hf_config.rope_scaling = parameters
            with self.subTest(parameters=parameters), self.assertRaises(SimulationError):
                _shape_model_matches(runner, exp, _phase="construction")

    def test_default_rope_theta_requires_finite_positive_numeric_and_consistent_aliases(self):
        for value in (True, None, 0, -1, float("nan"), float("inf")):
            runner, exp = with_default_rope_alias()
            runner.model_config.hf_config.rope_scaling = {"rope_type": "default", "rope_theta": value}
            with self.subTest(value=value), self.assertRaises(SimulationError):
                _shape_model_matches(runner, exp, _phase="construction")
        for field in ("rope_parameters", "rope_theta"):
            runner, exp = with_default_rope_alias()
            if field == "rope_parameters":
                runner.model_config.hf_config.rope_parameters = {"rope_type": "default", "rope_theta": 2000000}
            else:
                runner.model_config.hf_config.rope_theta = 2000000
            with self.subTest(field=field), self.assertRaises(SimulationError) as raised:
                _shape_model_matches(runner, exp, _phase="construction")
            self.assertEqual(raised.exception.code, "HARDWARE_IDENTITY_MISMATCH")


if __name__ == "__main__":
    unittest.main()
