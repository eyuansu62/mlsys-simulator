"""Source-shaped pinned runner/Scheduler fixtures; no vLLM or GPU dependency."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from mlsys_sim import DeviceSpec, Experiment, InfraSpec, ModelSpec, SimulationError, StrategySpec
from mlsys_sim.vllm_hooks import (SUPPORTED_VERSION_PREFIX, install_runner_hook,
    install_scheduler_hook, register_vllm_hooks, uninstall_vllm_hooks_plugin, VllmCaptureWorkerExtension,
    _shape_model_matches, _stage_layer_binding)


class Clock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        self.value += 1_000_000
        return self.value


def experiment(pp=1, tp=1):
    return Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}") for i in range(pp * tp))),
        strategy=StrategySpec(pipeline_parallel_size=pp, tensor_parallel_size=tp))


def state(prompt, context=0, history=0):
    return SimpleNamespace(num_prompt_tokens=prompt, num_computed_tokens=context,
        output_token_ids=[7] * history, mm_features=[], lora_request=None, prompt_embeds=None,
        pooling_params=None, sampling_params=None)


def schedule(tokens, resumed=()):
    return SimpleNamespace(num_scheduled_tokens=dict(tokens), scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={}, kv_connector_metadata=None, ec_connector_metadata=None,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)))


class Future:
    def __init__(self, output):
        self.output = output
        self.marker = True
        self.calls = 0
    def get_output(self):
        self.calls += 1
        del self.marker  # The real pinned implementation consumes buffers once.
        return self.output


class Runner:
    def __init__(self, *, last=True, asynchronous=False):
        self.requests = {}
        self.input_batch = SimpleNamespace(num_reqs=0, req_ids=[], num_computed_tokens_cpu=[])
        self.execute_model_state = None
        self.use_async_scheduling = asynchronous
        self.broadcast_pp_output = False
        self.speculative_config = None
        self.is_pooling_model = False
        self.last = last
        self.failure = None
        self.calls = []
        self.prepared = None
        self.intermediate = object()
        self.future_class = Future
        self.parallel_config = SimpleNamespace(data_parallel_size=1, use_ubatching=False,
            enable_expert_parallel=False, decode_context_parallel_size=1, prefill_context_parallel_size=1)
        self.vllm_config = SimpleNamespace(parallel_config=self.parallel_config,
            scheduler_config=SimpleNamespace(policy="fcfs", max_num_batched_tokens=32, max_num_seqs=8,
                                             enable_chunked_prefill=True),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=16, enable_prefix_caching=True),
            compilation_config=SimpleNamespace(mode="eager", cudagraph_mode="NONE", backend="inductor"))
    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        self.calls.append("prepare")
        self.prepared = (list(self.input_batch.req_ids), list(num_scheduled_tokens),
                         list(self.input_batch.num_computed_tokens_cpu))
        return object()
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self.calls.append("execute")
        ids = sorted(scheduler_output.num_scheduled_tokens, reverse=True)
        self.input_batch = SimpleNamespace(num_reqs=len(ids), req_ids=ids,
            num_computed_tokens_cpu=[self.requests[name].num_computed_tokens for name in ids])
        self._prepare_inputs(scheduler_output, [scheduler_output.num_scheduled_tokens[name] for name in ids])
        if self.failure is not None:
            raise self.failure
        if not self.last:
            return self.intermediate
        self.execute_model_state = scheduler_output
        return None
    def sample_tokens(self, grammar_output):
        self.calls.append("sample")
        if self.execute_model_state is None:
            return SimpleNamespace(req_ids=[], sampled_token_ids=[])
        self.execute_model_state = None
        ids, queries, contexts = self.prepared
        output = SimpleNamespace(req_ids=ids, sampled_token_ids=[
            [7] if context + query >= self.requests[name].num_prompt_tokens else []
            for name, query, context in zip(ids, queries, contexts)])
        return self.future_class(output) if self.use_async_scheduling else output


class RunnerHookTests(unittest.TestCase):
    def hook(self, runner, **kwargs):
        clock = Clock()
        hook = install_runner_hook(runner, kwargs.pop("experiment", experiment()), session_id="cpu-fixture",
            runtime_version=SUPPORTED_VERSION_PREFIX + ".d20261002", clock_ns=clock, **kwargs)
        self.addCleanup(hook.close)
        return hook

    def test_actual_prepared_order_mixed_roles_and_complete_sampling_boundary(self):
        runner = Runner()
        runner.requests = {"a-prefill": state(1), "z-decode": state(4, 4, 1)}
        hook = self.hook(runner)
        self.assertIsNone(runner.execute_model(schedule({"a-prefill": 1, "z-decode": 1})))
        self.assertEqual(hook.export_capture()["records"][0]["status"], "INCOMPLETE")
        output = runner.sample_tokens(None)
        self.assertEqual(output.req_ids, ["z-decode", "a-prefill"])
        raw = hook.export_capture()
        record = raw["records"][0]
        self.assertEqual(record["phase"], "mixed")
        self.assertEqual(record["request_phases"], ["decode", "prefill"])
        self.assertEqual(record["query_tokens"], [1, 1])
        self.assertEqual(record["context_lengths"], [4, 0])
        self.assertEqual(record["sampled_token_counts"], [1, 1])
        self.assertEqual(record["duration_ms"], 1)
        self.assertEqual(record["status"], "COMPLETED")
        self.assertTrue(record["sampling_complete"])
        self.assertIsNone(record["workspace_peak_bytes"])
        self.assertEqual(raw["source"]["kind"], "test_fixture")
        self.assertEqual(len(raw["identity"]["runtime"]["runtime_config_sha256"]), 64)

    def test_cached_partial_prefill_retains_context_and_does_not_sample(self):
        runner = Runner()
        runner.requests = {"cached": state(12, 4)}
        hook = self.hook(runner)
        runner.execute_model(schedule({"cached": 4}))
        runner.sample_tokens(None)
        record = hook.export_capture()["records"][0]
        self.assertEqual(record["phase"], "prefill")
        self.assertEqual(record["context_lengths"], [4])
        self.assertEqual(record["sampled_token_counts"], [0])
        self.assertEqual(record["status"], "COMPLETED")

    def test_nonfinal_pp_records_forward_without_inventing_sampling(self):
        runner = Runner(last=False)
        runner.requests = {"r": state(4)}
        hook = self.hook(runner, experiment=experiment(pp=2), stage_index=0)
        self.assertIs(runner.execute_model(schedule({"r": 4})), runner.intermediate)
        runner.sample_tokens(None)  # Real nonfinal sample call is bookkeeping.
        records = hook.export_capture()["records"]
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["forward_complete"])
        self.assertFalse(records[0]["sampling_required"])
        self.assertFalse(records[0]["sampling_complete"])

    def test_async_output_is_consumed_only_by_caller_and_remains_raw(self):
        runner = Runner(asynchronous=True)
        runner.requests = {"r": state(4)}
        hook = self.hook(runner)
        runner.execute_model(schedule({"r": 4}))
        future = runner.sample_tokens(None)
        self.assertEqual(future.calls, 0)
        self.assertEqual(hook.records[0]["status"], "INCOMPLETE")
        self.assertIs(future.get_output(), future.output)
        self.assertTrue(hook.records[0]["deferred_output_resolved"])
        self.assertEqual(hook.records[0]["status"], "COMPLETED")
        self.assertFalse(hook.records[0]["profile_eligible"])
        before = deepcopy(hook.records)
        with self.assertRaises(AttributeError):
            future.get_output()
        self.assertEqual(hook.records, before)

    def test_consumed_async_future_is_not_retained_until_capture_close(self):
        runner = Runner(asynchronous=True)
        runner.requests = {"r": state(4)}
        hook = self.hook(runner)
        runner.execute_model(schedule({"r": 4}))
        future = runner.sample_tokens(None)
        self.assertEqual(len(hook._future_methods), 1)
        reference = weakref.ref(future)
        future.get_output()
        self.assertEqual(hook._future_methods, [])
        self.assertNotIn("get_output", vars(future))
        del future
        self.assertIsNone(reference())

    def test_close_while_real_async_consumer_is_blocked_seals_the_raw_capture(self):
        entered, resume = threading.Event(), threading.Event()
        class BlockingFuture(Future):
            def get_output(self):
                entered.set()
                if not resume.wait(3):
                    raise RuntimeError("Fixture consumer was not released")
                return super().get_output()
        runner = Runner(asynchronous=True)
        runner.future_class = BlockingFuture
        runner.requests = {"r": state(4)}
        hook = self.hook(runner)
        runner.execute_model(schedule({"r": 4}))
        future = runner.sample_tokens(None)
        consume = future.get_output
        outputs = []
        thread = threading.Thread(target=lambda: outputs.append(consume()))
        thread.start()
        self.assertTrue(entered.wait(3))
        hook.close()
        sealed = hook.export_capture()
        resume.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIs(outputs[0], future.output)
        self.assertEqual(hook.export_capture(), sealed)
        self.assertEqual(hook._future_methods, [])

    def test_original_return_exception_and_method_restoration_are_preserved(self):
        runner = Runner()
        runner.requests = {"r": state(4)}
        original = runner.execute_model
        hook = self.hook(runner)
        failure = RuntimeError("fixture-runtime-error")
        runner.failure = failure
        with self.assertRaises(RuntimeError) as raised:
            runner.execute_model(schedule({"r": 4}))
        self.assertIs(raised.exception, failure)
        self.assertEqual(hook.records[0]["status"], "FAILED")
        hook.close()
        hook.close()
        self.assertEqual(runner.execute_model, original)
        self.assertFalse(hasattr(runner, "_mlsys_internal_capture"))

    def test_unknown_version_or_abi_never_partially_installs(self):
        runner = Runner()
        original = runner.execute_model
        with self.assertRaises(SimulationError) as raised:
            install_runner_hook(runner, experiment(), session_id="fixture", runtime_version="0.30.0", clock_ns=Clock())
        self.assertEqual(raised.exception.code, "UNSUPPORTED_VLLM_VERSION")
        self.assertEqual(runner.execute_model, original)
        runner._prepare_inputs = lambda scheduler_output: None
        with self.assertRaises(SimulationError):
            self.hook(runner)
        self.assertEqual(runner.execute_model, original)

    def test_resumed_history_speculation_and_native_parallel_modes_are_raw_only(self):
        for mode in ("resumed", "history", "speculative", "dbo", "native-dp", "broadcast", "sp"):
            runner = Runner()
            runner.requests = {"r": state(4, 0, 1 if mode == "history" else 0)}
            if mode == "speculative":
                runner.speculative_config = object()
            if mode == "dbo":
                runner.parallel_config.use_ubatching = True
            if mode == "native-dp":
                runner.parallel_config.data_parallel_size = 2
            if mode == "broadcast":
                runner.broadcast_pp_output = True
            if mode == "sp":
                runner.vllm_config.compilation_config.pass_config = SimpleNamespace(enable_sp=True)
            with self.subTest(mode=mode):
                hook = self.hook(runner)
                runner.execute_model(schedule({"r": 4}, resumed=("r",) if mode == "resumed" else ()))
                runner.sample_tokens(None)
                self.assertEqual(hook.records[0]["status"], "UNSUPPORTED")
                self.assertFalse(hook.records[0]["profile_eligible"])

    def test_lazy_or_sliced_pp_transport_is_raw_without_touching_lazy_tensors(self):
        class AsyncIntermediateTensors:
            accesses = 0
            @property
            def tensors(self):
                self.accesses += 1
                raise RuntimeError("The hook must not trigger receive wait/AllGather")
        for tp in (1, 2):
            with self.subTest(tp=tp):
                runner = Runner()
                runner.requests = {"r": state(4)}
                incoming = AsyncIntermediateTensors()
                hook = self.hook(runner, experiment=experiment(pp=2, tp=tp), stage_index=1)
                runner.execute_model(schedule({"r": 4}), incoming)
                runner.sample_tokens(None)
                record = hook.records[0]
                self.assertEqual(record["status"], "COMPLETED")
                self.assertFalse(record["profile_eligible"])
                self.assertTrue(record["transport_metadata"]["incoming"]["lazy_incoming"])
                self.assertEqual(incoming.accesses, 0)

    def test_materialized_double_tensor_pp_payload_is_not_one_hidden_payload(self):
        runner = Runner(last=False)
        runner.requests = {"r": state(4)}
        runner.intermediate = SimpleNamespace(tensors={"hidden_states": SimpleNamespace(shape=(4, 128)),
                                                       "residual": SimpleNamespace(shape=(4, 128))})
        hook = self.hook(runner, experiment=experiment(pp=2), stage_index=0)
        self.assertIs(runner.execute_model(schedule({"r": 4})), runner.intermediate)
        self.assertFalse(hook.records[0]["profile_eligible"])
        self.assertEqual(hook.records[0]["canonical_rejection_reason"], "PP_INTERMEDIATE_PAYLOAD_NOT_MODELED")

    def test_execution_policy_hash_changes_quarantine_later_samples(self):
        runner = Runner()
        runner.requests = {"r": state(4)}
        hook = self.hook(runner)
        runner.execute_model(schedule({"r": 4}))
        runner.sample_tokens(None)
        first = hook.header["metadata"]["runtime_config_sha256"]
        runner.vllm_config.scheduler_config.max_num_batched_tokens = 64
        runner.execute_model(schedule({"r": 4}))
        runner.sample_tokens(None)
        self.assertEqual(hook.header["metadata"]["runtime_config_sha256"], first)
        self.assertEqual(hook.records[1]["error_code"], "HARDWARE_IDENTITY_MISMATCH")

    def test_record_limit_is_bounded_without_affecting_runtime_calls(self):
        runner = Runner()
        runner.requests = {"r": state(4)}
        hook = self.hook(runner, max_records=1)
        for _ in range(20):
            runner.execute_model(schedule({"r": 4}))
            runner.sample_tokens(None)
        self.assertEqual(len(hook.records), 1)
        self.assertEqual(len(hook.errors), 1)
        self.assertEqual(runner.calls.count("execute"), 20)

    def test_close_and_exclusive_flush_preserve_pending_as_incomplete(self):
        runner = Runner()
        runner.requests = {"r": state(4)}
        with tempfile.TemporaryDirectory() as temporary:
            hook = self.hook(runner, output_dir=temporary)
            runner.execute_model(schedule({"r": 4}))
            filename = hook.flush()
            self.assertEqual(hook.flush(), filename)
            payload = json.loads(Path(filename).read_text())
            self.assertTrue(payload["capture_closed"])
            self.assertEqual(payload["records"][0]["status"], "INCOMPLETE")
            self.assertEqual(len(list(Path(temporary).glob("*.json"))), 1)


class Scheduler:
    def __init__(self):
        self.requests = {}
        self.outputs = {}
        self.failure = None
    def add_request(self, request):
        if self.failure:
            raise self.failure
        self.requests[request.request_id] = request
    def update_from_output(self, scheduler_output, model_runner_output):
        return self.outputs
    def finish_requests(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids or self.requests)
        return [self.requests.pop(name) for name in ids if name in self.requests]


def request(name="r", priority=0):
    return SimpleNamespace(request_id=name, prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SimpleNamespace(max_tokens=2), priority=priority, resumable=False,
        mm_features=[], lora_request=None)


class SchedulerHookTests(unittest.TestCase):
    def hook(self, scheduler, **kwargs):
        hook = install_scheduler_hook(scheduler, experiment(), session_id="scheduler-fixture",
            runtime_version=SUPPORTED_VERSION_PREFIX, clock_ns=Clock(), **kwargs)
        self.addCleanup(hook.close)
        return hook

    def test_actual_published_outputs_define_tokens_and_terminal_events(self):
        scheduler = Scheduler()
        hook = self.hook(scheduler, record_prompt_token_ids=True)
        scheduler.add_request(request(priority=5))
        scheduler.outputs = {0: SimpleNamespace(outputs=[SimpleNamespace(request_id="r", new_token_ids=[7], finish_reason=None)])}
        output = scheduler.update_from_output(schedule({"r": 4}), object())
        self.assertIs(output, scheduler.outputs)
        scheduler.outputs = {0: SimpleNamespace(outputs=[SimpleNamespace(request_id="r", new_token_ids=[8], finish_reason="LENGTH")])}
        scheduler.update_from_output(schedule({"r": 1}), object())
        hook.close()
        raw = hook.export_capture()
        self.assertEqual([event["event"] for event in raw["events"]],
            ["REQUEST_ARRIVAL", "TOKEN_READY", "TOKEN_READY", "REQUEST_FINISH"])
        self.assertEqual(raw["requests"][0]["priority"], -5)
        self.assertEqual(raw["requests"][0]["native_priority"], 5)
        self.assertEqual(raw["requests"][0]["output_tokens_actual"], 2)
        self.assertEqual(raw["requests"][0]["prompt_token_ids"], [1, 2, 3, 4])
        self.assertTrue(raw["capture_closed"])
        self.assertEqual(raw["source"]["kind"], "test_fixture")
        self.assertGreaterEqual(raw["observed_until_ms"], raw["events"][-1]["timestamp_ms"])

    def test_unaccepted_runner_samples_do_not_become_published_tokens(self):
        scheduler = Scheduler()
        hook = self.hook(scheduler)
        scheduler.add_request(request())
        # A worker can propose a token discarded by the Scheduler. Only return
        # values from update_from_output establish publication.
        scheduler.outputs = {}
        scheduler.update_from_output(schedule({"r": 4}), SimpleNamespace(sampled_token_ids=[[9]]))
        self.assertEqual([event["event"] for event in hook.events], ["REQUEST_ARRIVAL"])
        self.assertNotIn("prompt_token_ids", hook.requests["r"])

    def test_external_abort_and_original_enqueue_error_are_preserved(self):
        scheduler = Scheduler()
        original = scheduler.add_request
        hook = self.hook(scheduler)
        failure = RuntimeError("enqueue-failed")
        scheduler.failure = failure
        with self.assertRaises(RuntimeError) as raised:
            scheduler.add_request(request())
        self.assertIs(raised.exception, failure)
        self.assertEqual(hook.events, [])
        scheduler.failure = None
        scheduler.add_request(request())
        scheduler.finish_requests("r", "FINISHED_ABORTED")
        self.assertEqual(hook.events[-1]["status"], "FAILED")
        hook.close()
        self.assertEqual(scheduler.add_request, original)

    def test_multi_token_output_reused_id_and_unknown_reason_are_quarantined(self):
        for mode in ("multi", "reuse", "reason"):
            scheduler = Scheduler()
            hook = self.hook(scheduler)
            scheduler.add_request(request())
            if mode == "reuse":
                scheduler.add_request(request())
            else:
                scheduler.outputs = {0: SimpleNamespace(outputs=[SimpleNamespace(request_id="r",
                    new_token_ids=[7, 8] if mode == "multi" else [], finish_reason="UNKNOWN" if mode == "reason" else None)])}
                scheduler.update_from_output(schedule({"r": 1}), object())
            with self.subTest(mode=mode):
                self.assertFalse(hook.export_capture()["metadata"]["canonical_eligible"])
                self.assertTrue(hook.errors)

    def test_existing_state_or_policy_mismatch_is_not_empty_cache_online_evidence(self):
        existing = Scheduler()
        existing.requests["prior"] = request("prior")
        hook = self.hook(existing)
        self.assertFalse(hook.header["metadata"]["fresh_state_verified"])
        self.assertFalse(hook.header["metadata"]["canonical_eligible"])
        mismatched = Scheduler()
        mismatched.block_size = 16
        mismatched.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(policy="fcfs", max_num_batched_tokens=1, max_num_seqs=2,
                                             enable_chunked_prefill=False),
            cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto", block_size=16),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1))
        other = self.hook(mismatched)
        self.assertIn("max_num_batched_tokens", other.header["metadata"]["policy_mismatches"])
        self.assertFalse(other.header["metadata"]["canonical_eligible"])

    def test_prompt_hash_is_retained_without_payload_and_distinguishes_same_length_ids(self):
        scheduler = Scheduler()
        hook = self.hook(scheduler)
        first = request("a")
        second = request("b")
        second.prompt_token_ids = [4, 3, 2, 1]
        scheduler.add_request(first)
        scheduler.add_request(second)
        rows = hook.export_capture()["requests"]
        self.assertNotIn("prompt_token_ids", rows[0])
        digest = hashlib.sha256(json.dumps(first.prompt_token_ids, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(rows[0]["prompt_token_ids_sha256"], digest)
        self.assertNotEqual(rows[0]["prompt_token_ids_sha256"], rows[1]["prompt_token_ids_sha256"])

    def test_native_cache_salt_is_recorded_independently_of_prompt_hash(self):
        scheduler = Scheduler()
        hook = self.hook(scheduler)
        first, second = request("a"), request("b")
        first.cache_salt, second.cache_salt = "partition-a", "partition-b"
        scheduler.add_request(first)
        scheduler.add_request(second)
        rows = hook.export_capture()["requests"]
        self.assertEqual(rows[0]["prompt_token_ids_sha256"], rows[1]["prompt_token_ids_sha256"])
        self.assertNotEqual(rows[0]["cache_namespace"], rows[1]["cache_namespace"])
        self.assertTrue(rows[0]["cache_namespace"].startswith("vllm-salt-sha256:"))
        self.assertNotIn("partition-a", json.dumps(hook.export_capture()))

    def test_scheduler_event_limit_also_bounds_request_descriptors(self):
        scheduler = Scheduler()
        hook = self.hook(scheduler, max_events=1)
        for index in range(100):
            scheduler.add_request(request(str(index)))
        self.assertEqual(len(hook.events), 1)
        self.assertEqual(len(hook.requests), 1)
        self.assertEqual(len(scheduler.requests), 100)
        self.assertEqual(len(hook.errors), 1)
        self.assertFalse(hook.header["metadata"]["canonical_eligible"])

    def test_native_priority_clock_reversal_is_preserved_as_raw_only(self):
        scheduler = Scheduler()
        scheduler.block_size = 16
        scheduler.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(policy="priority", max_num_batched_tokens=4096,
                max_num_seqs=8, enable_chunked_prefill=False),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=16, enable_prefix_caching=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1))
        exp = replace(experiment(), strategy=replace(experiment().strategy, mixed_batching=True))
        hook = install_scheduler_hook(scheduler, exp, session_id="priority-fixture",
            runtime_version=SUPPORTED_VERSION_PREFIX, clock_ns=Clock())
        self.addCleanup(hook.close)
        first, second = request("a"), request("b")
        first.arrival_time, second.arrival_time = 2.0, 1.0
        scheduler.add_request(first)
        scheduler.add_request(second)
        self.assertEqual(hook.requests["b"]["native_arrival_time"], 1.0)
        self.assertFalse(hook.header["metadata"]["canonical_eligible"])
        self.assertTrue(any(error["error_code"] == "RUNTIME_POLICY_MISMATCH" for error in hook.errors))


class LiveModelContractTests(unittest.TestCase):
    def runner(self):
        model = ModelSpec(name="declared-qwen3", revision="revision-a", tie_word_embeddings=True,
            qk_norm=True, final_norm=True, weight_dtype="bfloat16", kv_dtype="bfloat16", activation_dtype="bfloat16")
        hf = SimpleNamespace(model_type="qwen3", num_hidden_layers=model.num_layers,
            hidden_size=model.hidden_size, intermediate_size=model.intermediate_size,
            num_attention_heads=model.num_attention_heads, num_key_value_heads=model.num_kv_heads,
            head_dim=model.head_dim, vocab_size=model.vocab_size, tie_word_embeddings=True,
            attention_bias=False, use_sliding_window=False, sliding_window=None, hidden_act="silu")
        runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf, dtype="torch.bfloat16", revision=None),
                                 kv_cache_dtype="torch.bfloat16")
        return runner, Experiment(strategy=StrategySpec(model=model))

    def test_actual_kv_dtype_and_visible_revision_are_checked_separately(self):
        runner, exp = self.runner()
        facts = _shape_model_matches(runner, exp)
        self.assertFalse(facts["revision_verified"])
        self.assertFalse(facts["weights_revision_independently_verified"])
        runner.kv_cache_dtype = "torch.float8_e4m3fn"
        with self.assertRaises(SimulationError):
            _shape_model_matches(runner, exp)
        runner.kv_cache_dtype = "torch.bfloat16"
        runner.model_config.hf_config._commit_hash = "other-revision"
        with self.assertRaises(SimulationError):
            _shape_model_matches(runner, exp)
        runner.model_config.hf_config._commit_hash = "revision-a"
        self.assertTrue(_shape_model_matches(runner, exp)["configuration_revision_verified"])

    def test_live_matcher_rejects_active_rope_or_layer_changes_even_without_hash(self):
        runner, exp = self.runner()
        runner.model_config.hf_config.sliding_window = 4096
        self.assertFalse(_shape_model_matches(runner, exp)["revision_verified"])
        runner.model_config.hf_config.rope_scaling = {"rope_type": "yarn", "factor": 4}
        with self.assertRaises(SimulationError):
            _shape_model_matches(runner, exp)
        runner.model_config.hf_config.rope_scaling = None
        runner.model_config.hf_config.layer_types = ["sliding_attention"] * exp.strategy.model.num_layers
        with self.assertRaises(SimulationError):
            _shape_model_matches(runner, exp)

    def test_loaded_pp_layer_range_is_not_inferred_from_global_layer_count(self):
        runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(start_layer=0, end_layer=2)))
        exp = experiment(pp=2)
        self.assertEqual(_stage_layer_binding(runner, exp, 0)["layer_count"], 2)
        mismatched = replace(exp, strategy=replace(exp.strategy, stage_layers=(1, 3)))
        with self.assertRaises(SimulationError) as raised:
            _stage_layer_binding(runner, mismatched, 0)
        self.assertEqual(raised.exception.code, "HARDWARE_IDENTITY_MISMATCH")
        with self.assertRaises(SimulationError) as raised:
            _stage_layer_binding(SimpleNamespace(model=None), exp, 0)
        self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")


class PluginTests(unittest.TestCase):
    def tearDown(self):
        uninstall_vllm_hooks_plugin()

    def test_default_entrypoint_is_dormant_before_any_optional_import(self):
        with patch.dict(os.environ, {}, clear=True), patch("mlsys_sim.vllm_hooks.importlib.import_module",
                side_effect=AssertionError("No optional imports while dormant")):
            register_vllm_hooks()

    def test_unknown_active_version_does_not_patch_any_constructor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps({"schema_version": 1, "experiment": asdict(experiment()),
                "session_id": "fixture", "output_dir": "capture"}))
            with patch.dict(os.environ, {"MLSYS_VLLM_CAPTURE_CONFIG": str(path)}), \
                    patch("mlsys_sim.vllm_hooks._runtime_version", side_effect=SimulationError("UNSUPPORTED_VLLM_VERSION", "unknown")):
                with self.assertRaises(SimulationError):
                    register_vllm_hooks()

    def test_explicit_plugin_installs_both_internal_channels_and_restores_constructors(self):
        class RunnerType:
            def __init__(self, vllm_config, device):
                self.config, self.device = vllm_config, device
        class SchedulerType:
            def __init__(self, vllm_config, kv_cache_config, structured_output_manager, block_size,
                         hash_block_size=None, mm_registry=None, include_finished_set=False, log_stats=False):
                self.config = vllm_config
        class Handle:
            def close(self):
                self.closed = True
        original_runner, original_scheduler = RunnerType.__init__, SchedulerType.__init__
        modules = {"vllm.v1.worker.gpu_model_runner": SimpleNamespace(GPUModelRunner=RunnerType),
            "vllm.v1.core.sched.scheduler": SimpleNamespace(Scheduler=SchedulerType),
            "vllm.distributed": SimpleNamespace(get_pp_group=lambda: SimpleNamespace(rank_in_group=0),
                                                get_tp_group=lambda: SimpleNamespace(rank_in_group=0))}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps({"schema_version": 1, "experiment": asdict(experiment()),
                "session_id": "plugin-fixture", "output_dir": "capture", "record_prompt_token_ids": True}))
            with patch.dict(os.environ, {"MLSYS_VLLM_CAPTURE_CONFIG": str(path)}), \
                    patch("mlsys_sim.vllm_hooks._runtime_version", return_value=SUPPORTED_VERSION_PREFIX), \
                    patch("mlsys_sim.vllm_hooks.install_runner_hook", return_value=Handle()) as install_runner, \
                    patch("mlsys_sim.vllm_hooks.install_scheduler_hook", return_value=Handle()) as install_scheduler, \
                    patch("mlsys_sim.vllm_hooks.importlib.import_module", side_effect=lambda name: modules[name]):
                register_vllm_hooks()
                register_vllm_hooks()
                runner = RunnerType("actual-config", "device")
                scheduler = SchedulerType("actual-config", None, None, 16)
                self.assertEqual(install_runner.call_count, 1)
                self.assertEqual(install_scheduler.call_count, 1)
                self.assertIs(install_runner.call_args.args[0], runner)
                self.assertIs(install_scheduler.call_args.args[0], scheduler)
                self.assertTrue(install_scheduler.call_args.kwargs["_constructor_origin"])
                self.assertTrue(install_scheduler.call_args.kwargs["record_prompt_token_ids"])
                uninstall_vllm_hooks_plugin()
                self.assertIs(RunnerType.__init__, original_runner)
                self.assertIs(SchedulerType.__init__, original_scheduler)

    def test_worker_extension_adds_rpc_methods_and_uses_real_group_coordinates(self):
        worker = VllmCaptureWorkerExtension()
        worker.model_runner = object()
        handle = SimpleNamespace(stage_index=1, tp_rank=0, header={"runtime_version": SUPPORTED_VERSION_PREFIX},
            export_capture=lambda: {"kind": "fixture-capture"}, flush=lambda: "fixture.json", close=lambda: None)
        groups = SimpleNamespace(get_pp_group=lambda: SimpleNamespace(rank_in_group=1),
                                 get_tp_group=lambda: SimpleNamespace(rank_in_group=0))
        with patch("mlsys_sim.vllm_hooks.install_runner_hook", return_value=handle) as install, \
                patch("mlsys_sim.vllm_hooks.importlib.import_module", return_value=groups):
            result = worker.mlsys_capture_install({"schema_version": 1, "experiment": asdict(experiment(pp=2)),
                                                  "session_id": "worker-fixture"})
        self.assertEqual(result["stage_index"], 1)
        self.assertEqual(install.call_args.kwargs["stage_index"], 1)
        self.assertEqual(install.call_args.kwargs["tp_rank"], 0)
        self.assertEqual(worker.mlsys_capture_export()["kind"], "fixture-capture")
        self.assertEqual(worker.mlsys_capture_close()["capture_file"], "fixture.json")


if __name__ == "__main__":
    unittest.main()
