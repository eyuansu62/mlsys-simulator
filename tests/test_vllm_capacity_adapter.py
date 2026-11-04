"""CPU mocks for a narrow native-KV adapter, never hardware/OOM evidence."""

from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryConfig,
                       RequestSpec, SimulationError, StrategySpec)
from mlsys_sim.capacity import config_sha256, validate_memory_capacity
from mlsys_sim.capacity_worker import _kv_bindings, run_capacity_trial
from mlsys_sim.qwen3 import model_from_qwen3_config
from mlsys_sim.vllm_capacity_adapter import MODEL_PATH_ENV, VllmKVPoolAdapter, kv_pool_factory
from mlsys_sim.vllm_hooks import SUPPORTED_VERSION_PREFIX

from test_capacity import FakeOOM, FakeTorch as CounterTorch, PoolTensor, worker_config


QUANTUM = 1835008
VERSION = SUPPORTED_VERSION_PREFIX + ".d20261002"
ALLOCATOR = '''def allocate_kv_cache(kv_cache_config, device, layout, kernel_block_sizes=None):
    warmup()
    raw_size = kv_cache_config.kv_cache_tensors[0].size
    buf_size = (raw_size + 4095) // 4096 * 4096
    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)
    after_allocation()
    return {"layer0": buf, "layer1": buf}
'''
INITIALIZER = '''class GPUModelRunner:
    def initialize_kv_cache(self, kv_cache_config, is_profiling=False, kv_cache_allocation_context=None):
        state.profiling = is_profiling
        values = allocate_kv_cache(kv_cache_config, self.device, None)
        self.kv_caches = list(values.values())
'''


def config():
    return {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 28, "hidden_size": 2048, "intermediate_size": 6144,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": True, "hidden_act": "silu",
        "attention_bias": False, "torch_dtype": "bfloat16"}


def experiment(model, pool=QUANTUM * 4):
    return Experiment(workload=(RequestSpec("real-cohort", 0.0, 16, 8, prompt_token_ids=(100,) * 16),),
        infra=InfraSpec(devices=(DeviceSpec("g0", memory_bytes=17_095_983_104, name="CPU HIP fixture"),)),
        strategy=StrategySpec(model=model, scheduler="fifo", max_batch_size=1, max_batch_tokens=256,
            memory=MemoryConfig(kv_pool_bytes=pool, graph_bytes=0, cpu_swap_bytes=0)),
        calibration=CalibrationSpec(runtime_name="vllm", runtime_version=VERSION))


class FakeTorch(CounterTorch):
    int8 = "fixture-int8"
    def __init__(self, exp, state):
        super().__init__(exp)
        self.state = state
        self.zeros_calls = []
    def zeros(self, size, *, dtype, device):
        self.zeros_calls.append((size, dtype, device))
        if self.state.mode == "kv_oom" and not self.state.profiling:
            self.fail("native zeros failed")
        if self.state.mode == "profile_oom" and self.state.profiling:
            self.fail("profiling allocation failed")
        return PoolTensor(0, size, 123 + len(self.zeros_calls))
    def fail(self, message):
        self.cuda.ooms[0] += 1
        self.state.exception = FakeOOM(message)
        raise self.state.exception


class RuntimeFixture:
    def __init__(self, directory, exp, *, mode="complete"):
        self.exp = exp
        self.state = SimpleNamespace(mode=mode, profiling=False, exception=None, shutdowns=0, llm_calls=0,
            bad_policy=None, output_tokens=exp.workload[0].output_tokens, generates=[])
        self.torch = FakeTorch(exp, self.state)
        self.utils = ModuleType("vllm.v1.worker.utils")
        path = directory / "fixture_allocator.py"
        path.write_text(ALLOCATOR)
        self.utils.__file__ = str(path)
        def warmup():
            if self.state.mode == "warmup_oom" and not self.state.profiling:
                self.torch.fail("startup GEMM warmup failed")
        def after_allocation():
            if self.state.mode == "after_oom" and not self.state.profiling:
                self.torch.fail("metadata allocation failed")
        self.utils.__dict__.update(torch=self.torch, warmup=warmup, after_allocation=after_allocation)
        exec(compile(ALLOCATOR, str(path), "exec"), self.utils.__dict__)
        self.runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
        self.runner_module.__dict__.update(allocate_kv_cache=self.utils.allocate_kv_cache, state=self.state)
        exec(INITIALIZER, self.runner_module.__dict__)
        self.vllm = SimpleNamespace(__version__=VERSION, LLM=self.llm, SamplingParams=lambda **kwargs: kwargs)
        self.modules = {"torch": self.torch, "vllm": self.vllm,
            "vllm.v1.worker.gpu_model_runner": self.runner_module}

    def llm(self, path, **kwargs):
        self.state.llm_calls += 1
        self.state.kwargs = kwargs
        if self.state.mode == "startup_oom":
            self.torch.fail("model startup failed")
        if self.state.mode == "assertion":
            raise AssertionError("workspace manager is locked")
        runner = self.runner_module.GPUModelRunner()
        runner.device = SimpleNamespace(type="cuda", index=0)
        runner.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True, enable_sleep_mode=False, max_model_len=kwargs["max_model_len"]),
            scheduler_config=SimpleNamespace(policy="fcfs", max_num_seqs=1,
                max_num_batched_tokens=kwargs["max_num_batched_tokens"], enable_chunked_prefill=False, async_scheduling=False),
            cache_config=SimpleNamespace(enable_prefix_caching=False, block_size=16,
                kv_cache_memory_bytes=kwargs["kv_cache_memory_bytes"], num_gpu_blocks_override=None),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                decode_context_parallel_size=1, prefill_context_parallel_size=1, use_ubatching=False,
                enable_expert_parallel=False, disable_custom_all_reduce=True),
            compilation_config=SimpleNamespace(mode=0, cudagraph_mode=0),
            offload_config=SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0),
                prefetch=SimpleNamespace(offload_group_size=0)))
        if self.state.bad_policy:
            section, field, value = self.state.bad_policy
            setattr(getattr(runner.vllm_config, section), field, value)
        def pool(size):
            blocks = size // QUANTUM
            return SimpleNamespace(num_blocks=blocks, kv_cache_tensors=[SimpleNamespace(size=blocks * QUANTUM)])
        runner.initialize_kv_cache(pool(QUANTUM), is_profiling=True)
        runner.initialize_kv_cache(pool(kwargs["kv_cache_memory_bytes"]))
        core = SimpleNamespace(model_executor=SimpleNamespace(driver_worker=SimpleNamespace(
            worker=SimpleNamespace(model_runner=runner))))
        def shutdown():
            self.state.shutdowns += 1
        def generate(prompts, params, *, use_tqdm):
            self.state.generates.append((prompts, params, use_tqdm))
            if self.state.mode == "run_oom":
                self.torch.fail("execution failed after actual pool binding")
            return [SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=[5] * self.state.output_tokens)])]
        return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(engine_core=core, shutdown=shutdown)),
                               generate=generate)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {self.utils.__name__: self.utils}))
        stack.enter_context(patch("mlsys_sim.vllm_capacity_adapter.import_module",
                                  side_effect=lambda name: self.modules[name]))
        # Explicit source-shaped CPU fixture; production retains the fixed upstream digest.
        digest = hashlib.sha256(inspect.getsource(self.utils.allocate_kv_cache).encode()).hexdigest()
        stack.enter_context(patch("mlsys_sim.vllm_capacity_adapter._ALLOCATOR_SOURCE_SHA256", digest))
        stack.enter_context(patch("mlsys_sim.vllm_capacity_adapter._shape_model_matches",
                                  return_value={"fixture": True, "hardware_validated": False}))
        return stack


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        raw = json.dumps(config()).encode()
        (self.directory / "config.json").write_bytes(raw)
        self.model = model_from_qwen3_config(config(), name="Qwen3-CPU-fixture",
            revision="local-config-sha256:" + hashlib.sha256(raw).hexdigest())
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def adapter(self, exp=None):
        return VllmKVPoolAdapter(experiment(self.model) if exp is None else exp, model_path=self.directory)

    def test_module_and_factory_are_lazy_and_path_is_explicit(self):
        result = subprocess.run([sys.executable, "-c", "import sys;import mlsys_sim.vllm_capacity_adapter;"
            "assert 'torch' not in sys.modules;assert 'vllm' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(SimulationError):
            kv_pool_factory(experiment(self.model))
        with patch.dict(os.environ, {MODEL_PATH_ENV: str(self.directory)}), patch(
                "mlsys_sim.vllm_capacity_adapter.import_module") as imported:
            adapter = kv_pool_factory(experiment(self.model))
        imported.assert_not_called()
        self.assertIsInstance(adapter, VllmKVPoolAdapter)

    def test_unsupported_declared_policies_fail_before_vllm_import(self):
        base = experiment(self.model)
        variants = [replace(base, strategy=replace(base.strategy, scheduler="continuous_batching")),
            replace(base, infra=InfraSpec(devices=base.infra.devices * 2),
                    strategy=replace(base.strategy, tensor_parallel_size=2)),
            replace(base, infra=InfraSpec(devices=tuple(replace(base.infra.devices[0], id=f"g{index}")
                for index in range(2))), strategy=replace(base.strategy, pipeline_parallel_size=2)),
            replace(base, strategy=replace(base.strategy, max_batch_size=2)),
            replace(base, strategy=replace(base.strategy, prefix_cache=True)),
            replace(base, strategy=replace(base.strategy, chunked_prefill=True)),
            replace(base, strategy=replace(base.strategy, mixed_batching=True)),
            replace(base, strategy=replace(base.strategy, memory=replace(base.strategy.memory, graph_bytes=1))),
            replace(base, strategy=replace(base.strategy, memory=replace(base.strategy.memory, cpu_swap_bytes=1))),
            replace(base, strategy=replace(base.strategy, memory=replace(base.strategy.memory, kv_block_tokens=8))),
            replace(base, workload=()), replace(base, workload=(replace(base.workload[0], prompt_token_ids=()),)),
            replace(base, workload=(replace(base.workload[0], arrival_ms=1),)),
            replace(base, execution=replace(base.execution, warmup=base.workload)),
            replace(base, calibration=replace(base.calibration, runtime_name="other-runtime")),
            replace(base, calibration=replace(base.calibration, runtime_config_sha256="a" * 64))]
        for variant in variants:
            with self.subTest(variant=variant), patch("mlsys_sim.vllm_capacity_adapter.import_module") as imported:
                with self.assertRaises(SimulationError):
                    self.adapter(variant)
                imported.assert_not_called()
        for env in ({"VLLM_USE_V2_MODEL_RUNNER": "1"}, {"VLLM_ENABLE_V1_MULTIPROCESSING": "1"},
                    {"MLSYS_VLLM_CAPTURE_CONFIG": "another-session"}):
            with self.subTest(env=env), patch.dict(os.environ, env), self.assertRaises(SimulationError):
                self.adapter()

    def test_local_identity_and_actual_version_are_bound(self):
        base = experiment(self.model)
        with self.assertRaises(SimulationError):
            self.adapter(replace(base, strategy=replace(base.strategy, model=replace(self.model, revision="unbound-revision"))))
        runtime = RuntimeFixture(self.directory, base)
        runtime.vllm.__version__ = "0.30.0"
        with runtime.patches(), self.assertRaises(SimulationError):
            self.adapter().metadata()
        self.assertEqual(runtime.state.llm_calls, 0)
        runtime = RuntimeFixture(self.directory, base)
        with runtime.patches():
            adapter = self.adapter()
            with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}), self.assertRaises(SimulationError):
                adapter.metadata()

    def test_complete_runtime_retains_real_pool_aliases_and_declared_tokens(self):
        exp = experiment(self.model)
        runtime = RuntimeFixture(self.directory, exp)
        original = runtime.runner_module.allocate_kv_cache
        with runtime.patches():
            adapter = self.adapter()
            metadata = adapter.metadata()
            self.assertEqual(metadata["effective_config_sha256"], config_sha256(exp))
            self.assertEqual(metadata["applied_axis"], {"name": "kv_pool_bytes", "value": exp.strategy.memory.kv_pool_bytes})
            adapter.prepare()
            self.assertIs(runtime.runner_module.allocate_kv_cache, original)
            bindings = _kv_bindings(adapter, exp)
            self.assertEqual(bindings[0]["allocation_count"], 1)
            self.assertEqual(bindings[0]["storage_bytes"], exp.strategy.memory.kv_pool_bytes)
            result = adapter.run()
            self.assertEqual(result["completed_request_ids"], [exp.workload[0].request_id])
            prompts, sampling, use_tqdm = runtime.state.generates[0]
            self.assertEqual(prompts, [{"prompt_token_ids": [100] * 16}])
            self.assertEqual(sampling["max_tokens"], 8)
            self.assertTrue(sampling["ignore_eos"])
            self.assertFalse(use_tqdm)
            with self.assertRaises(SimulationError):
                adapter.run()
            adapter.close()
            adapter.close()
            self.assertEqual(runtime.state.shutdowns, 1)
            with self.assertRaises(SimulationError):
                adapter.prepare()

    def test_native_zeros_oom_proof_is_scoped_and_exception_is_unchanged(self):
        runtime = RuntimeFixture(self.directory, experiment(self.model), mode="kv_oom")
        original = runtime.runner_module.allocate_kv_cache
        with runtime.patches():
            adapter = self.adapter()
            with self.assertRaises(FakeOOM) as raised:
                adapter.prepare()
            self.assertIs(raised.exception, runtime.state.exception)
            proof = adapter.capacity_oom_evidence()
            self.assertEqual(proof["attempts"], [{"device_index": 0, "shape": [QUANTUM * 4], "element_size": 1,
                "target_pool_bytes": QUANTUM * 4, "operation": "torch.zeros"}])
            self.assertEqual(proof["effective_config_sha256"], config_sha256(experiment(self.model)))
            self.assertEqual(runtime.torch.cuda.ooms[0], 1)
            self.assertIs(runtime.runner_module.allocate_kv_cache, original)
            adapter.close()
            self.assertEqual(runtime.state.shutdowns, 0)
            with self.assertRaises(SimulationError):
                adapter.run()

    def test_startup_profiling_warmup_metadata_and_run_oom_never_become_pool_proof(self):
        for mode in ("startup_oom", "profile_oom", "warmup_oom", "after_oom", "run_oom"):
            runtime = RuntimeFixture(self.directory, experiment(self.model), mode=mode)
            with self.subTest(mode=mode), runtime.patches():
                adapter = self.adapter()
                with self.assertRaises(FakeOOM):
                    adapter.prepare()
                    adapter.run()
                self.assertIsNone(adapter.capacity_oom_evidence())
                adapter.close()
                self.assertEqual(runtime.state.shutdowns, 0)

    def test_source_signature_actual_policy_and_alignment_guards_precede_target_allocation(self):
        runtime = RuntimeFixture(self.directory, experiment(self.model))
        with runtime.patches(), patch("mlsys_sim.vllm_capacity_adapter._ALLOCATOR_SOURCE_SHA256", "0" * 64):
            with self.assertRaises(SimulationError):
                self.adapter().prepare()
        self.assertEqual(runtime.state.llm_calls, 0)
        runtime = RuntimeFixture(self.directory, experiment(self.model))
        runtime.runner_module.allocate_kv_cache = lambda unexpected: None
        with runtime.patches(), self.assertRaises(SimulationError):
            self.adapter().prepare()
        self.assertEqual(runtime.state.llm_calls, 0)
        for policy in (("compilation_config", "mode", 1), ("cache_config", "enable_prefix_caching", True),
                       ("scheduler_config", "async_scheduling", True), ("parallel_config", "tensor_parallel_size", 2)):
            runtime = RuntimeFixture(self.directory, experiment(self.model))
            runtime.state.bad_policy = policy
            with self.subTest(policy=policy), runtime.patches(), self.assertRaises(SimulationError):
                self.adapter().prepare()
            self.assertEqual(len(runtime.torch.zeros_calls), 1)  # Profiling only.
        exp = experiment(self.model, QUANTUM * 4 + 1)
        runtime = RuntimeFixture(self.directory, exp)
        with runtime.patches(), self.assertRaises(SimulationError) as raised:
            self.adapter(exp).prepare()
        self.assertEqual(raised.exception.code, "KV_POOL_BINDING_MISMATCH")
        self.assertEqual(len(runtime.torch.zeros_calls), 1)

    def test_short_generation_and_workspace_assertions_are_not_oom(self):
        runtime = RuntimeFixture(self.directory, experiment(self.model))
        runtime.state.output_tokens = 7
        with runtime.patches():
            adapter = self.adapter()
            adapter.prepare()
            with self.assertRaises(SimulationError) as raised:
                adapter.run()
            self.assertEqual(raised.exception.code, "INCOMPLETE_CAPACITY_WORKLOAD")
        runtime = RuntimeFixture(self.directory, experiment(self.model), mode="assertion")
        with runtime.patches(), self.assertRaises(AssertionError):
            self.adapter().prepare()

    def test_owned_capacity_worker_closes_a_fixture_bracket_with_native_zeros(self):
        exp = experiment(self.model)
        trials = []
        for point, size in enumerate((QUANTUM * 3, QUANTUM * 4)):
            current = replace(exp, strategy=replace(exp.strategy, memory=replace(exp.strategy.memory, kv_pool_bytes=size)))
            runtime = RuntimeFixture(self.directory, current, mode="complete" if point == 0 else "kv_oom")
            settings = worker_config(current, axis="kv_pool_bytes", value=size, point=point, identifier=f"worker-{point}")
            with runtime.patches():
                observation = run_capacity_trial(settings, torch_module=runtime.torch,
                    factory=lambda effective: self.adapter(effective))
            self.assertEqual(observation["source"]["kind"], "test_fixture")
            trials.append({"point_index": point, "repeat": 0, "worker_id": settings["worker_id"], "axis_name": "kv_pool_bytes",
                "axis_value": size, "effective_experiment": asdict(current), "effective_config_sha256": config_sha256(current),
                "observation": observation, "prediction": {}})
        dataset = {"schema_version": 1, "kind": "memory_capacity_measurements", "protocol": settings["protocol"],
            "dataset_id": "native-zero-cpu-fixture", "source": {"kind": "test_fixture", "reference": "CPU mocks only"},
            "session_id": settings["session_id"], "base_experiment": asdict(exp),
            "axis": {"name": "kv_pool_bytes", "values": [QUANTUM * 3, QUANTUM * 4]}, "repeats": 1,
            "profile_inputs": [], "trials": trials}
        report = validate_memory_capacity(dataset, allow_test_fixture=True)
        self.assertEqual(report["threshold_status"], "BRACKETED", report)
        self.assertFalse(report["hardware_validated"])


if __name__ == "__main__":
    unittest.main()
