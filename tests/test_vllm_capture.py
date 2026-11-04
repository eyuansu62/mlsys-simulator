"""Safe rank aggregation without fabricated stage or lifecycle evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, ModelSpec,
                       RequestSpec, SimulationError, StrategySpec, calibrate_dataset,
                       replica_experiment)
from mlsys_sim.profiles import profile_identity
from mlsys_sim.vllm_capture import load_vllm_captures, normalize_vllm_capture


VERSION = "0.29.1.dev0+g98dff2a81.d20261002"
EXECUTION = {"async_scheduling": False, "cudagraph_mode": "NONE", "kv_cache_dtype": "float16"}
EXECUTION_HASH = hashlib.sha256(json.dumps(EXECUTION, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def experiment(tp=1, pp=1, replicas=1, *, runtime_hash=EXECUTION_HASH):
    model = ModelSpec(name="qwen-capture-fixture", revision="fixture-revision", num_layers=4,
        hidden_size=48, intermediate_size=96, num_attention_heads=4, num_kv_heads=2,
        head_dim=16, vocab_size=64, tie_word_embeddings=True, qk_norm=True, final_norm=True)
    return Experiment(workload=(RequestSpec("r", 0, 4, 1),),
        infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}") for i in range(tp * pp * replicas))),
        strategy=StrategySpec(model=model, tensor_parallel_size=tp, pipeline_parallel_size=pp, replicas=replicas),
        calibration=CalibrationSpec(runtime_name="vllm", runtime_version=VERSION,
                                    runtime_config_sha256=runtime_hash))


def record(stage=0, pp=1, *, duration=1.0, iteration=1, phase="prefill", queries=None,
           contexts=None, prompts=None, roles=None, ids=None):
    queries = [4] if queries is None else queries
    contexts = [0] if contexts is None else contexts
    prompts = [4] if prompts is None else prompts
    ids = ["r"] if ids is None else ids
    start = iteration * 10_000_000
    result = {"sample_id": f"iteration-{iteration}", "iteration_id": iteration,
        "request_ids": ids, "phase": phase, "query_tokens": queries,
        "context_lengths": contexts, "prompt_lengths": prompts,
        "start_monotonic_ns": start, "end_monotonic_ns": start + int(duration * 1e6),
        "duration_ms": duration, "status": "COMPLETED",
        "timing_boundary": "step_includes_output_and_sampling_v1",
        "capture_boundary": "runner_execute_to_sample_host_ready" if stage == pp - 1 else "runner_execute_host_ready",
        "forward_complete": True, "sampling_required": stage == pp - 1,
        "sampling_complete": stage == pp - 1, "profile_eligible": True,
        "workspace_peak_bytes": None,
        "allocator_baseline": [{"allocated_bytes": 1000, "reserved_bytes": 2000}],
        "allocator_after": [{"allocated_bytes": 1400, "reserved_bytes": 3000,
                             "peak_allocated_bytes": 2500, "peak_reserved_bytes": 4000}]}
    if roles is not None:
        result["request_phases"] = roles
    if pp > 1:
        common = {"object_type": "fixture.MaterializedIntermediateTensors", "lazy_incoming": False,
            "materialized_tensor_keys": ["hidden_states"], "materialized_shape": [sum(queries), 48],
            "modeled_transport": "one_unpadded_full_hidden_tensor_per_rank",
            "default_tp_sliced_p2p_receiver_allgather": False,
            "profile_transport_compatible": True, "incoming_communication_excluded": True}
        incoming = ({**common, "object_type": "builtins.NoneType", "materialized_tensor_keys": None}
                    if stage == 0 else dict(common))
        result["transport_metadata"] = {"incoming": incoming}
        if stage < pp - 1:
            result["transport_metadata"]["outgoing"] = dict(common)
    return result


def capture(exp=None, *, replica=0, stage=0, rank=0, records=None):
    exp = exp or experiment()
    local = replica_experiment(exp, replica)
    return {"schema_version": 1, "kind": "vllm_internal_capture",
        "source": {"kind": "test_fixture", "reference": "CPU source-shaped runner fixture, not a GPU run"},
        "session_id": "unit-session", "runtime_version": VERSION, "api_revision": "98dff2a81",
        "clock_domain": "unit-host-monotonic", "replica_index": replica, "stage_index": stage, "tp_rank": rank,
        "tensor_parallel_size": exp.strategy.tensor_parallel_size,
        "pipeline_parallel_size": exp.strategy.pipeline_parallel_size,
        "identity": profile_identity(local), "experiment": asdict(local),
        "records": records if records is not None else [record(stage, exp.strategy.pipeline_parallel_size)],
        "events": [], "metadata": {"instrumentation_source": "vllm_internal_runner", "synchronized": True,
                                      "runtime_execution_config": deepcopy(EXECUTION)},
        "hardware_validated": False}


def normalize(exp, values, **kwargs):
    return normalize_vllm_capture(exp, values, allow_test_fixture=True, **kwargs)


def actual_runner_capture(exp, clock, request_id="r", session_id="actual-hook-unit", *, stage=0, rank=0,
                          intermediate=None):
    from mlsys_sim.vllm_hooks import install_runner_hook

    class Runner:
        def __init__(self):
            self.input_batch = SimpleNamespace(num_reqs=1, req_ids=[request_id], num_computed_tokens_cpu=[0])
            self.requests = {request_id: SimpleNamespace(num_prompt_tokens=4, output_token_ids=[])}
            self.execute_model_state = None
            self.use_async_scheduling = False
        def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
            return None
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            self._prepare_inputs(scheduler_output, [scheduler_output.num_scheduled_tokens[request_id]])
            if stage < exp.strategy.pipeline_parallel_size - 1:
                return SimpleNamespace(tensors={"hidden_states": SimpleNamespace(shape=(4, exp.strategy.model.hidden_size))})
            self.execute_model_state = SimpleNamespace(scheduler_output=scheduler_output)
            return None
        def sample_tokens(self, grammar_output):
            self.execute_model_state = None
            return SimpleNamespace(req_ids=[request_id], sampled_token_ids=[[3]])

    runner = Runner()
    hook = install_runner_hook(runner, exp, session_id=session_id, runtime_version=VERSION,
                               clock_ns=clock, synchronize=lambda: None, stage_index=stage, tp_rank=rank)
    output = SimpleNamespace(num_scheduled_tokens={request_id: 4}, total_num_scheduled_tokens=4)
    runner.execute_model(output, intermediate)
    if stage == exp.strategy.pipeline_parallel_size - 1:
        runner.sample_tokens(None)
    hook.close()
    return hook.export_capture()


def actual_scheduler_and_runner(exp, *, record_ids=False, cache_salt=None):
    from mlsys_sim.vllm_hooks import install_scheduler_hook

    class Clock:
        value = 0
        def __call__(self):
            self.value += 1_000_000
            return self.value

    class Scheduler:
        def add_request(self, request):
            return None
        def update_from_output(self, scheduler_output, model_runner_output):
            return {0: SimpleNamespace(outputs=[SimpleNamespace(request_id="server-r", new_token_ids=[3],
                                                                 finish_reason="length")])}
        def finish_requests(self, request_ids, finished_status):
            return []

    clock = Clock()
    scheduler = Scheduler()
    hook = install_scheduler_hook(scheduler, exp, session_id="actual-hook-unit", runtime_version=VERSION,
                                  clock_ns=clock, record_prompt_token_ids=record_ids)
    scheduler.add_request(SimpleNamespace(request_id="server-r", prompt_token_ids=[1, 2, 3, 4],
                                          sampling_params=SimpleNamespace(max_tokens=1), priority=0,
                                          arrival_time=1_700_000_000.0, cache_salt=cache_salt))
    runner_capture = actual_runner_capture(exp, clock, "server-r")
    scheduler.update_from_output(SimpleNamespace(), SimpleNamespace())
    hook.close()
    return runner_capture, hook.export_capture()


class VllmCaptureTests(unittest.TestCase):
    def test_complete_tp_pp_groups_use_max_rank_duration_and_keep_workspace_unknown(self):
        exp = experiment(tp=2, pp=2)
        captures = [capture(exp, stage=stage, rank=rank,
            records=[record(stage, 2, duration=stage + rank + 1)])
            for stage in range(2) for rank in range(2)]
        before = deepcopy(captures)
        result = normalize(exp, captures)
        self.assertEqual(captures, before)
        self.assertTrue(result["canonical_stage_ready"])
        self.assertFalse(result["canonical_comparable"])
        self.assertFalse(result["hardware_validated"])
        rows = result["dataset"]["records"]
        self.assertEqual([row["duration_ms"] for row in rows], [2, 3])
        self.assertEqual([row["workspace_peak_bytes"] for row in rows], [None, None])
        fit = calibrate_dataset(result["dataset"], profile_version="capture-fixture", allow_test_fixture=True)
        self.assertEqual(fit["source"]["kind"], "test_fixture")

    def test_missing_rank_and_failed_iteration_preserve_raw_without_dataset(self):
        exp = experiment(tp=2)
        incomplete = normalize(exp, [capture(exp)])
        self.assertEqual(incomplete["datasets"], [])
        self.assertFalse(incomplete["canonical_stage_ready"])
        self.assertIn("missing_tp_rank", incomplete["quarantined"][0]["reasons"])
        broken = record()
        broken.update(status="FAILED", end_monotonic_ns=None, duration_ms=None, message="GPU error")
        result = normalize(experiment(), [capture(records=[broken])])
        self.assertEqual(result["datasets"], [])
        self.assertEqual(result["raw_records"][0]["records"][0]["message"], "GPU error")

    def test_async_resolved_or_execute_only_does_not_claim_complete_stage_service(self):
        for change, reason in (({"async_output": True, "deferred_output_resolved": True},
                                "async_consumption_is_not_exclusive_stage_service"),
                               ({"sampling_complete": False}, "sampling_not_complete"),
                               ({"profile_eligible": False}, "hook_marked_sample_ineligible"),
                               ({"capture_boundary": "execute_only"}, "unsupported_capture_boundary")):
            value = record()
            value.update(change)
            with self.subTest(change=change):
                result = normalize(experiment(), [capture(records=[value])])
                self.assertEqual(result["datasets"], [])
                self.assertIn(reason, result["quarantined"][0]["reasons"])

    def test_overlapping_host_intervals_are_not_serial_stage_costs(self):
        first = record(duration=4)
        second = record(iteration=2, duration=4)
        second["start_monotonic_ns"] = first["start_monotonic_ns"] + 1_000_000
        second["end_monotonic_ns"] = second["start_monotonic_ns"] + 4_000_000
        result = normalize(experiment(), [capture(records=[first, second])])
        self.assertEqual(result["datasets"], [])
        self.assertEqual(result["capture_summary"]["quarantined_groups"], 2)

    def test_rank_and_stage_shape_disagreement_is_rejected(self):
        exp = experiment(tp=2, pp=2)
        values = [capture(exp, stage=stage, rank=rank) for stage in range(2) for rank in range(2)]
        for position in (1, 2):
            changed = deepcopy(values)
            changed[position]["records"][0]["request_ids"] = ["other"]
            with self.subTest(position=position), self.assertRaises(SimulationError) as raised:
                normalize(exp, changed)
            self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_session_clock_version_and_duplicate_channels_are_rejected(self):
        exp = experiment(tp=2)
        original = [capture(exp, rank=0), capture(exp, rank=1)]
        for field, value in (("session_id", "other"), ("clock_domain", "other"),
                             ("runtime_version", "0.29.1.dev0+g98dff2a81.other-date"),
                             ("api_revision", "unknown")):
            changed = deepcopy(original)
            changed[1][field] = value
            with self.subTest(field=field), self.assertRaises(SimulationError):
                normalize(exp, changed)
        with self.assertRaises(SimulationError):
            normalize(exp, [original[0], deepcopy(original[0])])

    def test_cached_last_prompt_token_keeps_prefill_role_and_mixed_order(self):
        exp = experiment()
        mixed = record(phase="mixed", queries=[1, 1], contexts=[8, 3], prompts=[4, 4],
                       roles=["decode", "prefill"], ids=["decoder", "cached-suffix"])
        result = normalize(exp, [capture(records=[mixed])])
        row = result["dataset"]["records"][0]
        self.assertEqual(row["request_phases"], ["decode", "prefill"])
        self.assertEqual(row["context_lengths"], [8, 3])
        relabeled = deepcopy(mixed)
        relabeled["phase"] = "decode"
        relabeled.pop("request_phases")
        with self.assertRaises(SimulationError):
            normalize(exp, [capture(records=[relabeled])])

    def test_output_history_or_resume_does_not_become_a_new_prefill(self):
        for change, reason in (({"output_history_lengths": [1]}, "prefill_with_existing_output_history"),
                               ({"resumed_request_ids": ["r"]}, "resumed_recompute_boundary_unsupported")):
            value = record()
            value.update(change)
            result = normalize(experiment(), [capture(records=[value])])
            self.assertEqual(result["datasets"], [])
            self.assertIn(reason, result["quarantined"][0]["reasons"])

    def test_raw_peak_minus_baseline_cannot_be_submitted_as_workspace(self):
        value = record()
        value["workspace_peak_bytes"] = 1500
        result = normalize(experiment(), [capture(records=[value])])
        self.assertEqual(result["datasets"], [])
        self.assertIn("workspace_has_no_runtime_attribution", result["quarantined"][0]["reasons"])

    def test_actual_runtime_hash_is_bound_explicitly_and_nonempty_expectation_must_match(self):
        actual = experiment()
        declared = experiment(runtime_hash="")
        result = normalize(declared, [capture(actual)])
        self.assertTrue(result["provenance"]["runtime_config_binding"]["derived_from_capture"])
        self.assertEqual(result["experiment"]["calibration"]["runtime_config_sha256"], EXECUTION_HASH)
        self.assertEqual(result["dataset"]["identity"]["runtime"]["runtime_config_sha256"], EXECUTION_HASH)
        with self.assertRaises(SimulationError):
            normalize(experiment(runtime_hash="a" * 64), [capture(actual)])
        bad = capture(actual)
        bad["metadata"]["runtime_execution_config"]["async_scheduling"] = True
        with self.assertRaises(SimulationError):
            normalize(actual, [bad])

    def test_per_replica_datasets_keep_device_identity(self):
        exp = experiment(replicas=2)
        result = normalize(exp, [capture(exp, replica=0), capture(exp, replica=1)])
        self.assertNotIn("dataset", result)
        self.assertEqual([item["replica_index"] for item in result["datasets"]], [0, 1])
        self.assertEqual([item["dataset"]["identity"]["devices"][0]["id"] for item in result["datasets"]], ["g0", "g1"])
        self.assertTrue(result["canonical_stage_ready"])

    def test_explicit_server_lifecycle_can_normalize_but_runner_events_alone_cannot(self):
        exp = experiment()
        trace = {"schema_version": 1, "kind": "runtime_trace", "source": capture(exp)["source"],
            "timing_boundary": "server_token_ready_v1", "observed_until_ms": 2,
            "events": [{"event": "REQUEST_ARRIVAL", "request_id": "r", "timestamp_ms": 0},
                {"event": "TOKEN_READY", "request_id": "r", "timestamp_ms": 2,
                 "token_count": 1, "output_token_index": 1},
                {"event": "REQUEST_FINISH", "request_id": "r", "timestamp_ms": 2, "status": "COMPLETED"}]}
        raw = capture(exp)
        raw["events"] = deepcopy(trace["events"])
        self.assertFalse(normalize(exp, [raw])["canonical_comparable"])
        result = normalize(exp, [raw], runtime_trace=trace)
        self.assertTrue(result["canonical_comparable"])
        self.assertEqual(result["observation"]["metrics"]["ttft_p50_ms"], 2)
        self.assertEqual(result["observation"]["source"]["kind"], "test_fixture")

    def test_loader_preserves_file_hash_and_rejects_ambiguous_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps(capture(), indent=2))
            loaded = load_vllm_captures([path])
            self.assertEqual(loaded[0]["_capture_input"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(normalize(experiment(), loaded)["canonical_stage_ready"])
            for content in ('{"kind":"a","kind":"b"}', '{"value":NaN}', '[]'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(SimulationError):
                    load_vllm_captures([path])
        for values in ([], [None], [1], "capture.json"):
            with self.subTest(paths=values), self.assertRaises(SimulationError):
                load_vllm_captures(values)

    def test_source_permission_and_duration_integrity_are_enforced(self):
        with self.assertRaises(SimulationError) as raised:
            normalize_vllm_capture(experiment(), [capture()])
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        value = record()
        value["duration_ms"] = 3
        with self.assertRaises(SimulationError):
            normalize(experiment(), [capture(records=[value])])
        raw = capture()
        raw["identity"]["model"]["tie_word_embeddings"] = False
        with self.assertRaises(SimulationError):
            normalize(experiment(), [raw])

    def test_actual_runner_hook_export_can_be_normalized_without_schema_translation(self):
        values = iter((1_000_000_000, 1_002_000_000))
        exp = experiment(runtime_hash="")
        raw = actual_runner_capture(exp, lambda: next(values))
        self.assertEqual(raw["records"][0]["status"], "COMPLETED")
        result = normalize(exp, [raw])
        self.assertTrue(result["canonical_stage_ready"])
        self.assertEqual(result["dataset"]["records"][0]["duration_ms"], 2)
        self.assertTrue(result["provenance"]["runtime_config_binding"]["derived_from_capture"])

    def test_actual_scheduler_published_lifecycle_requires_explicit_replay_binding(self):
        exp = experiment(runtime_hash="")
        raw = list(actual_scheduler_and_runner(exp))
        unchanged = normalize(exp, raw)
        self.assertFalse(unchanged["canonical_comparable"])
        self.assertEqual(unchanged["experiment"]["workload"][0]["request_id"], "r")
        result = normalize(exp, raw, bind_runtime_workload=True)
        self.assertTrue(result["canonical_comparable"])
        request = result["experiment"]["workload"][0]
        self.assertEqual((request["request_id"], request["arrival_ms"], request["output_tokens"]), ("server-r", 1, 1))
        self.assertEqual(result["observation"]["metrics"]["ttft_p50_ms"], 3)
        self.assertIsNone(result["observation"]["metrics"]["throughput_requests_per_s"])
        self.assertTrue(result["provenance"]["runtime_workload_binding"]["applied"])
        self.assertFalse(result["hardware_validated"])

    def test_scheduler_prefix_replay_needs_actual_or_matched_prompt_identity(self):
        base = experiment(runtime_hash="")
        exp = replace(base, workload=(), strategy=replace(base.strategy, prefix_cache=True))
        result = normalize(exp, list(actual_scheduler_and_runner(exp)), bind_runtime_workload=True)
        self.assertFalse(result["canonical_comparable"])
        self.assertIn("prompt IDs", result["online_readiness"]["reason"])
        result = normalize(exp, list(actual_scheduler_and_runner(exp, record_ids=True)), bind_runtime_workload=True)
        self.assertTrue(result["canonical_comparable"])
        self.assertEqual(result["experiment"]["workload"][0]["prompt_token_ids"], (1, 2, 3, 4))

    def test_scheduler_missing_terminal_and_conflicting_identity_do_not_make_online_evidence(self):
        exp = experiment(runtime_hash="")
        raw = list(actual_scheduler_and_runner(exp))
        raw[1]["events"].pop()
        result = normalize(exp, raw, bind_runtime_workload=True)
        self.assertFalse(result["canonical_comparable"])
        self.assertIn("terminal", result["online_readiness"]["reason"])
        raw = list(actual_scheduler_and_runner(exp))
        raw[1]["identity"]["model"]["head_dim"] = 8
        with self.assertRaises(SimulationError):
            normalize(exp, raw, bind_runtime_workload=True)

    def test_any_fixture_channel_keeps_every_dataset_a_fixture(self):
        base = experiment(replicas=2)
        exp = replace(base,
            infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}", name=f"AMD declared device {i}") for i in range(2))),
            strategy=replace(base.strategy, model=replace(base.strategy.model,
                weight_dtype="float16", kv_dtype="float16", activation_dtype="float16")),
            calibration=replace(base.calibration, driver_version="declared-driver-1",
                                rocm_version="declared-rocm-1", rccl_version="not-applicable"))
        values = [capture(exp, replica=0), capture(exp, replica=1)]
        values[0]["source"] = {"kind": "measured", "reference": "Declared-source unit contract; no GPU was run"}
        result = normalize(exp, values)
        self.assertEqual(result["source"]["kind"], "test_fixture")
        self.assertTrue(all(item["dataset"]["source"]["kind"] == "test_fixture" for item in result["datasets"]))
        self.assertFalse(result["hardware_validated"])

    def test_missing_whole_pp_channel_does_not_claim_complete_layout_readiness(self):
        exp = experiment(pp=2)
        result = normalize(exp, [capture(exp, stage=1)])
        self.assertFalse(result["canonical_stage_ready"])
        self.assertTrue(result["canonical_stage_records_available"])
        self.assertEqual(result["capture_summary"]["missing_channels"],
                         [{"replica_index": 0, "stage_index": 0, "tp_rank": 0}])

    def test_measured_pp_requires_observed_loaded_layer_ranges(self):
        base = experiment(pp=2)
        exp = replace(base,
            infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}", name=f"AMD declared device {i}") for i in range(2))),
            strategy=replace(base.strategy, model=replace(base.strategy.model,
                weight_dtype="float16", kv_dtype="float16", activation_dtype="float16")),
            calibration=replace(base.calibration, driver_version="declared-driver-1",
                                rocm_version="declared-rocm-1", rccl_version="not-applicable"))
        captures = [capture(exp, stage=stage) for stage in range(2)]
        for stage, value in enumerate(captures):
            value["source"] = {"kind": "measured", "reference": "Declared-source schema test, no GPU execution"}
            value["metadata"]["stage_layer_binding"] = {"start_layer": stage * 2, "end_layer": stage * 2 + 2,
                "layer_count": 2, "method": "loaded_qwen_model_start_end_layer"}
        self.assertTrue(normalize(exp, captures)["canonical_stage_ready"])
        missing = deepcopy(captures)
        missing[1]["metadata"].pop("stage_layer_binding")
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, missing)
        self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")
        wrong = deepcopy(captures)
        wrong[1]["metadata"]["stage_layer_binding"].update(start_layer=0, end_layer=2)
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, wrong)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_borrowed_prompt_identity_requires_matching_actual_token_hash(self):
        base = experiment(runtime_hash="")
        exp = replace(base, workload=(RequestSpec("server-r", 1, 4, 1, prompt_token_ids=(1, 2, 3, 4)),))
        captures = list(actual_scheduler_and_runner(exp))
        self.assertTrue(normalize(exp, captures)["canonical_comparable"])
        wrong_exp = replace(exp, workload=(replace(exp.workload[0], prompt_token_ids=(8, 8, 8, 8)),))
        with self.assertRaises(SimulationError) as raised:
            normalize(wrong_exp, captures)
        self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")
        missing = deepcopy(captures)
        missing[1]["requests"][0].pop("prompt_token_ids_sha256")
        result = normalize(exp, missing)
        self.assertFalse(result["canonical_comparable"])
        self.assertIn("hash", result["online_readiness"]["reason"])
        raw_ids = list(actual_scheduler_and_runner(exp, record_ids=True))
        raw_ids[1]["requests"][0]["prompt_token_ids"][0] = 9
        with self.assertRaises(SimulationError):
            normalize(exp, raw_ids)

    def test_native_arrival_order_is_not_substituted_with_wall_minus_monotonic_time(self):
        exp = experiment(runtime_hash="")
        captures = list(actual_scheduler_and_runner(exp))
        self.assertIn("native_arrival_time", captures[1]["requests"][0])
        self.assertGreater(captures[1]["requests"][0]["native_arrival_time"], 1_000_000_000)
        baseline = normalize(exp, captures, bind_runtime_workload=True)
        self.assertEqual(baseline["experiment"]["workload"][0]["arrival_ms"], 1)
        second = deepcopy(captures[1]["requests"][0])
        second.update(request_id="backdated", arrival_ms=2, native_arrival_time=1_600_000_000.0)
        captures[1]["requests"].append(second)
        result = normalize(exp, captures, bind_runtime_workload=True)
        self.assertFalse(result["canonical_comparable"])
        self.assertIn("Native arrival order", result["online_readiness"]["reason"])

    def test_actual_cache_namespace_requires_match_or_explicit_conditional_binding(self):
        from mlsys_sim.vllm_hooks import cache_namespace_from_vllm_salt
        base = experiment(runtime_hash="")
        exp = replace(base, strategy=replace(base.strategy, prefix_cache=True),
            workload=(RequestSpec("server-r", 1, 4, 1, prompt_token_ids=(1, 2, 3, 4), cache_namespace="planned"),))
        values = list(actual_scheduler_and_runner(exp, cache_salt="actual"))
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, values)
        self.assertEqual(raised.exception.code, "COMPARISON_MISMATCH")
        result = normalize(exp, values, bind_runtime_workload=True)
        self.assertTrue(result["canonical_comparable"])
        self.assertEqual(result["experiment"]["workload"][0]["cache_namespace"], cache_namespace_from_vllm_salt("actual"))
        self.assertEqual(result["provenance"]["runtime_workload_binding"]["cache_namespace_binding"],
                         "vllm_nonempty_salt_sha256_v1")
        from mlsys_sim.config import experiment_from_dict
        effective = experiment_from_dict(result["experiment"])
        with self.assertRaises(SimulationError):
            normalize(effective, values)
        self.assertTrue(normalize(effective, values, bind_runtime_workload=True)["canonical_comparable"])
        planned = replace(exp, workload=(replace(exp.workload[0], cache_namespace="actual"),))
        self.assertTrue(normalize(planned, values)["canonical_comparable"])
        self.assertNotEqual(cache_namespace_from_vllm_salt(cache_namespace_from_vllm_salt("actual")),
                            cache_namespace_from_vllm_salt("actual"))
        missing = deepcopy(values)
        missing[1]["requests"][0].pop("cache_namespace")
        result = normalize(exp, missing, bind_runtime_workload=True)
        self.assertFalse(result["canonical_comparable"])
        self.assertIn("namespace", result["online_readiness"]["reason"])

    def test_actual_tp_pp_hook_transport_is_preserved_as_raw_without_canonical_service(self):
        class AsyncIntermediateTensors:
            @property
            def tensors(self):
                raise AssertionError("Capture must not wait on or access a lazy incoming tensor")

        exp = experiment(tp=2, pp=2, runtime_hash="")
        captures = []
        for stage in range(2):
            for rank in range(2):
                ticks = iter((1_000_000, 3_000_000))
                captures.append(actual_runner_capture(exp, lambda: next(ticks), stage=stage, rank=rank,
                    intermediate=AsyncIntermediateTensors() if stage else None))
        result = normalize(exp, captures)
        self.assertFalse(result["canonical_stage_ready"])
        self.assertEqual(result["datasets"], [])
        self.assertEqual(result["capture_summary"]["quarantined_groups"], 2)
        incoming = result["raw_records"][-1]["records"][0]["transport_metadata"]["incoming"]
        self.assertTrue(incoming["lazy_incoming"])
        self.assertTrue(incoming["default_tp_sliced_p2p_receiver_allgather"])


if __name__ == "__main__":
    unittest.main()
