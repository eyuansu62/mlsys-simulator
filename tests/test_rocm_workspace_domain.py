"""CPU-only guards for the optional pinned Workspace acquisition/merger."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import unittest

from mlsys_sim import (
    CalibrationSpec,
    DeviceSpec,
    Experiment,
    InfraSpec,
    MemoryConfig,
    ModelSpec,
    StrategySpec,
)
from mlsys_sim.vllm_hooks import _config_digest

_PATH = Path(__file__).resolve().parents[1] / "scripts/rocm_workspace_audit/workspace_domain.py"
_SPEC = importlib.util.spec_from_file_location("workspace_domain", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DOMAIN = importlib.util.module_from_spec(_SPEC)
sys.modules["workspace_domain"] = _DOMAIN
_SPEC.loader.exec_module(_DOMAIN)


def fixture():
    model = ModelSpec(
        name="Qwen3-1.7B",
        num_layers=28,
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=151936,
        weight_dtype="bfloat16",
        kv_dtype="bfloat16",
        activation_dtype="bfloat16",
        tie_word_embeddings=True,
        qk_norm=True,
        final_norm=True,
    )
    exp = asdict(
        Experiment(
            infra=InfraSpec(
                devices=(DeviceSpec("gpu0", name=_DOMAIN.DEVICE_NAME, memory_bytes=17095983104),)
            ),
            strategy=StrategySpec(
                model=model,
                max_batch_size=1,
                max_batch_tokens=512,
                memory=MemoryConfig(kv_pool_bytes=_DOMAIN.KV_BYTES),
            ),
            calibration=CalibrationSpec(
                runtime_name="vllm",
                runtime_version=_DOMAIN.RUNTIME_VERSION,
                driver_version="6.16.13",
                rocm_version="10.1.0",
                rccl_version="2.30.7",
            ),
        )
    )
    obs = dict(
        schema_version=1,
        runtime_name="vllm",
        runtime_version=_DOMAIN.RUNTIME_VERSION,
        torch_version=_DOMAIN.TORCH_VERSION,
        torch_git=_DOMAIN.TORCH_GIT,
        hip_build_version=_DOMAIN.HIP_VERSION,
        visible_device_count=1,
        allocator_backend="native",
        device=dict(index=0, name=_DOMAIN.DEVICE_NAME, memory_bytes=17095983104, architecture="gfx1201"),
        runner=dict(
            class_name=_DOMAIN.RUNNER_CLASS,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            config_tensor_parallel_size=1,
            config_pipeline_parallel_size=1,
            data_parallel_size=1,
            use_async_scheduling=False,
        ),
        hardware=dict(
            source_kind="measured",
            status="READY",
            driver_version="6.16.13",
            rocm_version="10.1.0",
            rccl_version="2.30.7",
            rccl_library_sha256="a" * 64,
        ),
    )
    config = dict(
        api_revision="98dff2a81",
        async_scheduling=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        model_config=dict(enforce_eager=True, max_model_len=512),
        scheduler_config=dict(
            policy="fcfs", max_num_batched_tokens=512, max_num_seqs=1, enable_chunked_prefill=False
        ),
        cache_config=dict(
            cache_dtype="auto",
            block_size=16,
            enable_prefix_caching=False,
            kv_sharing_fast_prefill=False,
            kv_cache_memory_bytes=_DOMAIN.KV_BYTES,
            swap_space=None,
            cpu_offload_gb=None,
        ),
        parallel_config=dict(
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
            enable_expert_parallel=False,
        ),
        compilation_config=dict(mode=0, cudagraph_mode="NONE"),
        environment=dict(VLLM_USE_V2_MODEL_RUNNER="0"),
    )
    exp["calibration"]["runtime_config_sha256"] = _DOMAIN.config_sha256(config)
    session = dict(
        experiment=exp,
        observed_runtime_domain=obs,
        observed_runtime_version=_DOMAIN.RUNTIME_VERSION,
        torch_version=_DOMAIN.TORCH_VERSION,
        torch_git=_DOMAIN.TORCH_GIT,
        runtime_config=config,
    )
    return exp, obs, config, session


class WorkspaceDomainTests(unittest.TestCase):
    def test_supported_actual_domain_and_digest_pass_without_mutating_inputs(self):
        exp, obs, config, session = fixture()
        before = deepcopy(session)
        self.assertEqual(_DOMAIN.declared_domain_issues(exp), [])
        self.assertEqual(_DOMAIN.observed_domain_issues(exp, obs, config), [])
        self.assertEqual(_DOMAIN.recorded_session_domain_issues(session), [])
        self.assertEqual(_DOMAIN.config_sha256(config), _config_digest(config))
        self.assertEqual(session, before)

    def test_wrong_runtime_declaration_never_becomes_complete(self):
        exp, obs, config, session = fixture()
        exp["calibration"]["runtime_version"] = "0.30.0"
        self.assertTrue(_DOMAIN.declared_domain_issues(exp))
        self.assertTrue(_DOMAIN.observed_domain_issues(exp, obs, config))
        self.assertTrue(_DOMAIN.recorded_session_domain_issues(session))

    def test_parallelism_budget_and_scheduling_switches_are_rejected_upfront(self):
        changes = [
            ("tensor_parallel_size", 2),
            ("pipeline_parallel_size", 2),
            ("replicas", 2),
            ("max_batch_tokens", 1024),
            ("max_batch_size", 2),
            ("prefix_cache", True),
            ("chunked_prefill", True),
            ("mixed_batching", True),
        ]
        for field, value in changes:
            exp, _, _, _ = fixture()
            exp["strategy"][field] = value
            with self.subTest(field=field):
                self.assertTrue(_DOMAIN.declared_domain_issues(exp))
        for field, value in (
            ("kv_pool_bytes", 1234),
            ("kv_block_tokens", 32),
            ("graph_bytes", 1),
            ("cpu_swap_bytes", 4096),
        ):
            exp, _, _, _ = fixture()
            exp["strategy"]["memory"][field] = value
            self.assertTrue(_DOMAIN.declared_domain_issues(exp))

    def test_actual_runner_parallelism_async_and_device_capacity_cannot_use_declared_values(self):
        for section, field, value in (
            ("runner", "tensor_parallel_size", 2),
            ("runner", "config_pipeline_parallel_size", 2),
            ("runner", "use_async_scheduling", True),
            ("runner", "class_name", "other.Runner"),
            ("device", "memory_bytes", 1),
            ("device", "architecture", "gfx1100"),
        ):
            exp, obs, config, _ = fixture()
            obs[section][field] = value
            self.assertTrue(_DOMAIN.observed_domain_issues(exp, obs, config))

    def test_actual_runtime_and_torch_build_must_match_observations(self):
        for field, value in (
            ("runtime_version", "0.30.0"),
            ("torch_git", "b" * 40),
            ("torch_version", "2.12"),
            ("hip_build_version", "unknown"),
            ("allocator_backend", "cudaMallocAsync"),
        ):
            exp, obs, config, _ = fixture()
            obs[field] = value
            self.assertTrue(_DOMAIN.observed_domain_issues(exp, obs, config))

    def test_actual_config_switches_and_explicit_null_fields_are_checked(self):
        for section, field, value in (
            ("scheduler_config", "max_num_seqs", 2),
            ("scheduler_config", "enable_chunked_prefill", True),
            ("cache_config", "enable_prefix_caching", True),
            ("cache_config", "kv_cache_memory_bytes", 1),
            ("compilation_config", "mode", 1),
            ("compilation_config", "cudagraph_mode", "FULL"),
            ("model_config", "enforce_eager", False),
        ):
            exp, obs, config, _ = fixture()
            config[section][field] = value
            self.assertTrue(_DOMAIN.observed_domain_issues(exp, obs, config))
        exp, obs, config, _ = fixture()
        del config["cache_config"]["cpu_offload_gb"]
        self.assertTrue(_DOMAIN.observed_domain_issues(exp, obs, config))

    def test_nonempty_hash_mismatch_is_rejected_and_empty_can_bind_actual(self):
        exp, _, config, session = fixture()
        exp["calibration"]["runtime_config_sha256"] = "b" * 64
        with self.assertRaises(_DOMAIN.WorkspaceDomainError):
            _DOMAIN.bound_config_sha256(exp, config)
        self.assertTrue(_DOMAIN.recorded_session_domain_issues(session))
        exp["calibration"]["runtime_config_sha256"] = ""
        self.assertEqual(_DOMAIN.bound_config_sha256(exp, config), _DOMAIN.config_sha256(config))
        self.assertTrue(_DOMAIN.recorded_session_domain_issues(session))

    def test_legacy_missing_observed_facts_stays_partial(self):
        _, _, _, session = fixture()
        del session["observed_runtime_domain"]
        self.assertIn(
            "RECORDED_DOMAIN:MISSING_OBSERVED_RUNTIME_DOMAIN", _DOMAIN.recorded_session_domain_issues(session)
        )
        _, _, _, session = fixture()
        del session["observed_runtime_version"]
        self.assertTrue(_DOMAIN.recorded_session_domain_issues(session))

    def test_missing_or_fixture_hardware_and_wrong_actual_build_are_rejected(self):
        for field, value in (
            ("source_kind", "test_fixture"),
            ("status", "UNAVAILABLE"),
            ("rccl_version", "2.29.0"),
            ("rccl_library_sha256", None),
        ):
            exp, obs, config, _ = fixture()
            obs["hardware"][field] = value
            self.assertTrue(_DOMAIN.observed_domain_issues(exp, obs, config))
        exp, obs, config, _ = fixture()
        del obs["runner"]["use_async_scheduling"]
        self.assertTrue(_DOMAIN.observed_domain_issues(exp, obs, config))


if __name__ == "__main__":
    unittest.main()
