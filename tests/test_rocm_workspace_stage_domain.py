"""CPU-built Worker-domain metadata; no GPU, SDK or Runtime is executed.

Measured source labels here test data gates only. They never authenticate the
producer or establish actual storage ownership, hardware use or accuracy.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec, MemoryConfig,
                       ModelSpec, StrategySpec, profile_identity)
from mlsys_sim.pp_transport import native_pp_descriptor
from test_vllm_capture import capture, record

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/rocm_workspace_audit"
_SPEC = importlib.util.spec_from_file_location("workspace_stage_domain", _SCRIPTS / "workspace_stage_domain.py")
assert _SPEC is not None and _SPEC.loader is not None
_DOMAIN = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_SCRIPTS))
try:
    _SPEC.loader.exec_module(_DOMAIN)
finally:
    sys.path.remove(str(_SCRIPTS))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def actual_config(tp, pp):
    return {"api_revision": "98dff2a81", "async_scheduling": False,
        "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
        "model_config": {"enforce_eager": True, "max_model_len": 512},
        "scheduler_config": {"policy": "fcfs", "max_num_batched_tokens": 512, "max_num_seqs": 1,
                             "enable_chunked_prefill": False},
        "cache_config": {"cache_dtype": "auto", "block_size": 16, "enable_prefix_caching": False,
            "kv_sharing_fast_prefill": False, "kv_cache_memory_bytes": 268435456, "swap_space": None,
            "cpu_offload_gb": None},
        "parallel_config": {"data_parallel_size": 1, "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1, "use_ubatching": False, "enable_expert_parallel": False},
        "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
        "torch_build": {"torch_version": _DOMAIN.TORCH_VERSION, "hip_build_version": _DOMAIN.HIP_VERSION},
        "actual_attention_backends": ["TRITON_ATTN"],
        "sampling_policy": {"temperature": 0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
            "logprobs": None, "prompt_logprobs": None},
        "environment": {"VLLM_USE_V2_MODEL_RUNNER": "0"}}


def prepared_record(exp, stage, index, cohort, request, prompt, step):
    query, context = (prompt, 0) if step == 0 else (1, prompt + step - 1)
    phase = "prefill" if step == 0 else "decode"
    row = record(stage, exp.strategy.pipeline_parallel_size, iteration=index + 1,
        duration=4, phase=phase, queries=[query], contexts=[context], prompts=[prompt], ids=[request])
    row.update(iteration_id=index, sample_id=f"opaque-sample-{index}", output_history_lengths=[step],
               resumed_request_ids=[], sampled_token_counts=[1] if stage == exp.strategy.pipeline_parallel_size - 1 else [])
    if exp.strategy.pipeline_parallel_size > 1:
        descriptor = native_pp_descriptor(exp, query)
        common = {"object_type": "fixture.IntermediateTensors", "lazy_incoming": False,
            "materialized_tensor_keys": ["hidden_states", "residual"], "materialized_shape": [query, 2048],
            "modeled_transport": "vllm_native_pp_v1", "profile_transport_compatible": True,
            "incoming_communication_excluded": True, "native_descriptor": descriptor,
            "default_tp_sliced_p2p_receiver_allgather": False}
        incoming = ({"object_type": "builtins.NoneType"} if stage == 0 else {
            **deepcopy(common), "lazy_incoming": True, "natural_receive_observed": True,
            "receive_wait_completed": True, "receive_completion_scope": "natural_wait_for_comm",
            "pre_wait_compute_stream_synchronized": True})
        row["transport_metadata"] = {"incoming": incoming}
        if stage == 0:
            row["transport_metadata"]["outgoing"] = deepcopy(common)
        else:
            left = row["start_monotonic_ns"]
            row["pp_receive_timing"] = {"method": "natural_wait_for_comm_v1", "excluded_ns": 2000000,
                "wall_duration_ms": 4.0,
                "intervals": [{"start_monotonic_ns": left + 1000000, "end_monotonic_ns": left + 3000000}]}
            row["duration_ms"] = 2.0
    raw = {"epoch_id": 10001 + index, "capture_iteration": index + 1,
        "iteration_id": index, "cohort": cohort, "process_id": 1001 + stage * exp.strategy.tensor_parallel_size,
        "stage_record_index": index, "stage_record": deepcopy(row),
        "phase": phase, "query_tokens": [query], "context_lengths": [context],
        "native_epoch_id": 10001 + index if cohort == "formal" else None,
        "status": "RAW_STEP_CAPTURED", "start_monotonic_ns": row["start_monotonic_ns"] - 1,
        "end_monotonic_ns": row["end_monotonic_ns"] + 1,
        "return_witness": {"type": "vllm.v1.outputs.ModelRunnerOutput"
                           if stage == exp.strategy.pipeline_parallel_size - 1 else "builtins.NoneType",
            "gpu_tensor_count": 0, "execute_state_is_none": True, "prev_sampled_is_none": True}}
    return row, raw


def inputs(mode="single", repeats=1):
    tp, pp = {"single": (1, 1), "tp2": (2, 1), "pp2": (1, 2)}[mode]
    names = ["AMD Radeon RX 9070 XT"] if mode == "single" else ["AMD Radeon RX 7900 XTX", "AMD Radeon RX 9070 XT"]
    devices = tuple(DeviceSpec(f"gpu{i}", name=name, memory_bytes=_DOMAIN._DEVICES[name][0])
                    for i, name in enumerate(names))
    model = ModelSpec(name="Qwen3-1.7B", revision="cpu-schema-revision", num_layers=28, hidden_size=2048,
        intermediate_size=6144, num_attention_heads=16, num_kv_heads=8, head_dim=128, vocab_size=151936,
        weight_dtype="bfloat16", kv_dtype="bfloat16", activation_dtype="bfloat16", tie_word_embeddings=pp == 1,
        qk_norm=True, final_norm=True)
    exp = Experiment(infra=InfraSpec(devices=devices), strategy=StrategySpec(model=model,
        tensor_parallel_size=tp, pipeline_parallel_size=pp, scheduler="fifo", max_batch_size=1, max_batch_tokens=512,
        pipeline_transport="vllm_native" if pp > 1 else "full_hidden", pipeline_transport_max_tokens=512 if pp > 1 else 0,
        memory=MemoryConfig(kv_pool_bytes=268435456)),
        calibration=CalibrationSpec(runtime_name="vllm", runtime_version=_DOMAIN.RUNTIME_VERSION,
            driver_version="6.16.13", rocm_version="10.1.0", rccl_version="2.30.7"))
    config = actual_config(tp, pp)
    bound = replace(exp, calibration=replace(exp.calibration, runtime_config_sha256=digest(config)))
    plan = {"schema_version": 1, "mode": mode, "cohort": "pilot", "repeats": repeats,
        "session_id": "cpu-domain-fixture", "clock_domain": "linux-boot:cpu-fixture-only",
        "clock": "CLOCK_MONOTONIC", "native_clock": "CLOCK_MONOTONIC_RAW",
        "expected_formal_records_per_rank": 8 * repeats,
        "worker_source": "/app/work/tools/worker_workspace_capture.py", "worker_source_sha256": "b" * 64,
        "visible_devices": "1" if mode == "single" else "0,1"}
    inventory_devices = [{"index": i, "name": dev.name, "memory_bytes": dev.memory_bytes,
                          "architecture": _DOMAIN._DEVICES[dev.name][1]} for i, dev in enumerate(devices)]
    loaded = {"verified": True, "runtime_version": "2.30.7", "library_sha256": "c" * 64,
              "verification_scope": "local_proc_maps_loaded_library_ncclGetVersion",
              "path": "/fixture/librccl.so", "version_code": 23007}
    workers = []
    for stage in range(pp):
        for rank in range(tp):
            global_rank = stage * tp + rank
            pid = 1001 + global_rank
            data_rows, epochs = [], []
            for cohort, count in (("warmup", 1), ("formal", repeats)):
                for repetition in range(count):
                    for prompt in (16, 32):
                        identifier = f"{cohort}-request-{repetition}-{prompt}"
                        for step in range(4):
                            row, raw = prepared_record(bound, stage, len(data_rows), cohort, identifier, prompt, step)
                            raw["process_id"] = pid
                            data_rows.append(row)
                            epochs.append(raw)
            channel = capture(bound, stage=stage, rank=rank, records=data_rows)
            channel.update(source={"kind": "measured", "reference": "CPU-only metadata fixture, no physical run"},
                session_id=plan["session_id"], clock_domain=plan["clock_domain"], capture_errors=[], capture_closed=False)
            tying = {"verification_phase": "prepared", "pending_loaded_storage_proof": False,
                     "effective_tie_word_embeddings": pp == 1, "architecture_fingerprint_status": "verified"}
            if pp == 1:
                tying["local_alias"] = {"same_parameter": True, "same_storage": True, "same_tensor_view": True,
                                        "effective_tie_word_embeddings": True}
            else:
                tying["verification_scope"] = "configured_distinct_pp_endpoint_copies_no_cross_rank_alias_claim"
            visibility = {"HIP_VISIBLE_DEVICES": plan["visible_devices"], "ROCR_VISIBLE_DEVICES": None,
                          "CUDA_VISIBLE_DEVICES": None if mode == "single" else "0,1"}
            actual_device = {**deepcopy(inventory_devices[global_rank]), "visibility": visibility}
            metadata = channel["metadata"]
            metadata.update(runtime_execution_config=deepcopy(config), runtime_config_sha256=digest(config),
                torch_version=_DOMAIN.TORCH_VERSION, hip_build_version=_DOMAIN.HIP_VERSION,
                model_binding={"word_embedding_tying": tying},
                stage_layer_binding={"start_layer": stage * 28 // pp, "end_layer": (stage + 1) * 28 // pp,
                    "layer_count": 28 // pp, "method": "loaded_qwen_model_start_end_layer"},
                device={**actual_device, "visible_index": global_rank})
            if tp > 1 or pp > 1:
                metadata["rccl_binding"] = deepcopy(loaded)
            if pp > 1:
                metadata["native_pp_backend_binding"] = {"backend": "nccl", "use_cpu_custom_send_recv": False,
                    "api_revision": "98dff2a81", "torch_version": _DOMAIN.TORCH_VERSION}
                metadata["native_pp_stream_binding"] = {"standard_process_group_nccl": True,
                    "actual_backend": "nccl", "group_type": "vllm.distributed.parallel_state.GroupCoordinator",
                    "contract": "current_to_communication_stream_until_work_wait_v1"}
                if stage:
                    metadata["native_pp_persistent_binding"] = {"max_num_tokens": 512,
                        "descriptor": native_pp_descriptor(bound, 512), "scope": "loaded_runner_intermediate_buffers"}
            placement = {"process_id": pid, "global_rank": global_rank, "stage_index": stage, "tp_rank": rank,
                "tensor_parallel_size": tp, "pipeline_parallel_size": pp, "world_size": tp * pp,
                "logical_device_index": global_rank}
            worker = {"schema_version": 1, "kind": "vllm_worker_workspace_raw", **placement,
                "placement": deepcopy(placement), "session_id": plan["session_id"], "plan_sha256": "a" * 64,
                "capture_closed": True, "capture_errors": [], "complete_workspace": False, "hardware_validated": False,
                "native_sdk_stop_status": 0, "native_sdk_finalize_status": 0,
                "source_bindings": {"worker_source_reference": plan["worker_source"], "worker_source_sha256": "b" * 64},
                "runtime_version": _DOMAIN.RUNTIME_VERSION, "torch_version": _DOMAIN.TORCH_VERSION,
                "torch_git": _DOMAIN.TORCH_GIT, "hip_build_version": _DOMAIN.HIP_VERSION,
                "observed_device": actual_device, "loaded_rccl": deepcopy(loaded),
                "observed_hardware_inventory": {"schema_version": 1, "kind": "rocm_inventory",
                    "status": "READY", "source": {"kind": "measured"},
                    "versions": {"driver_version": "6.16.13", "rocm_version": "10.1.0", "rccl_version": "2.30.7"},
                    "torch": {"version": _DOMAIN.TORCH_VERSION, "hip_version": _DOMAIN.HIP_VERSION, "cuda_available": True},
                    "devices": deepcopy(inventory_devices), "rccl_library": {"version": "2.30.7", "sha256": "c" * 64}},
                "stage_capture": channel, "records": epochs,
                "runtime_config": deepcopy(config), "runtime_config_sha256": digest(config),
                "identity": profile_identity(bound), "experiment": deepcopy(channel["experiment"]),
                "model_binding": deepcopy(metadata["model_binding"])}
            workers.append(worker)
    return plan, workers, asdict(exp)


def run(plan, workers, exp):
    return _DOMAIN.validate_stage_workspace_domain(plan, workers, experiment=exp, plan_sha256="a" * 64)


def change_record(worker, index, values):
    worker["stage_capture"]["records"][index].update(deepcopy(values))
    worker["records"][index]["stage_record"].update(deepcopy(values))
    for key in ("phase", "query_tokens", "context_lengths", "iteration_id", "request_phases"):
        if key in values:
            worker["records"][index][key] = deepcopy(values[key])


class WorkspaceStageDomainTests(unittest.TestCase):
    def test_additional_exported_rccl_mapping_facts_do_not_change_prepared_library_identity(self):
        plan, workers, exp = inputs("tp2")
        for worker in workers:
            worker["loaded_rccl"]["mapped_segments"] = [{"path": worker["loaded_rccl"]["path"],
                "start": 4096, "end": 8192, "offset": 0, "permissions": "r-xp"}]
        result = run(plan, workers, exp)
        self.assertEqual(result["status"], "SUPPORTED_WORKSPACE_STAGE_DOMAIN")
        workers[0]["stage_capture"]["metadata"]["rccl_binding"]["version_code"] = 1
        with self.assertRaises(ValueError):
            run(plan, workers, exp)

    def test_single_tp2_pp2_exact_domains_use_actual_prepared_shapes_without_promoting_attribution(self):
        for mode in ("single", "tp2", "pp2"):
            for repeats in (1, 3):
                plan, workers, exp = inputs(mode, repeats)
                workers.reverse()
                before = deepcopy((plan, workers, exp))
                result = run(plan, workers, exp)
                self.assertEqual((plan, workers, exp), before)
                self.assertTrue(result["normalization"]["canonical_stage_ready"])
                self.assertFalse(result["hardware_validated"])
                self.assertFalse(result["attribution_complete"])
                self.assertEqual(result["formal_full_three_repetitions"], repeats == 3)
                self.assertEqual(result["identity"]["runtime"]["runtime_config_sha256"], digest(actual_config(
                    exp["strategy"]["tensor_parallel_size"], exp["strategy"]["pipeline_parallel_size"])))
                for global_rank, placement in enumerate(result["placements"]):
                    self.assertEqual(placement["device_index"], global_rank)
                    self.assertEqual(len(placement), 7)
                    formal = result["formal_by_rank"][str(global_rank)]
                    self.assertEqual(len(formal), 8 * repeats)
                    self.assertEqual(sum(row["phase"] == "decode" for row in formal), 6 * repeats)
                self.assertTrue(all(row["workspace_peak_bytes"] is None
                    for row in result["normalization"]["dataset"]["records"]))

    def test_plan_layout_model_and_actual_declared_stack_are_strict(self):
        for change in (
            lambda p, w, e: p.update(mode="four-gpu"),
            lambda p, w, e: p.update(repeats=4),
            lambda p, w, e: p.update(expected_formal_records_per_rank=7),
            lambda p, w, e: e["strategy"].update(tensor_parallel_size=2, pipeline_parallel_size=2),
            lambda p, w, e: e["strategy"]["model"].update(hidden_size=1024),
            lambda p, w, e: e["strategy"]["model"].update(tie_word_embeddings=False),
            lambda p, w, e: e["calibration"].update(runtime_version="another-build"),
            lambda p, w, e: e["calibration"].update(timing_boundary="execute-only"),
            lambda p, w, e: e["strategy"].update(prefix_cache=True),
            lambda p, w, e: e["strategy"]["memory"].update(cpu_swap_bytes=4096),
        ):
            plan, workers, exp = inputs()
            change(plan, workers, exp)
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(plan, workers, exp)

    def test_worker_process_rank_and_device_cohort_cannot_be_missing_or_duplicated(self):
        plan, workers, exp = inputs("tp2")
        for change in (
            lambda w: w.pop(), lambda w: w.append(deepcopy(w[0])),
            lambda w: w[1].update(process_id=w[0]["process_id"]),
            lambda w: w[1]["placement"].update(tp_rank=0),
            lambda w: w[1]["placement"].update(global_rank=0),
            lambda w: w[1]["placement"].update(logical_device_index=0),
        ):
            bad = deepcopy(workers)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(plan, bad, exp)

    def test_runtime_hardware_source_versions_device_capacity_and_loaded_rccl_are_observed(self):
        for change in (
            lambda w: w.update(torch_git="different"),
            lambda w: w.update(hip_build_version="different"),
            lambda w: w["observed_hardware_inventory"].update(status="ERROR"),
            lambda w: w["observed_hardware_inventory"].update(kind="invented"),
            lambda w: w["observed_hardware_inventory"]["source"].update(kind="test_fixture"),
            lambda w: w["observed_hardware_inventory"]["versions"].update(rccl_version="2.99.0"),
            lambda w: w["observed_hardware_inventory"]["torch"].update(cuda_available=False),
            lambda w: w["observed_hardware_inventory"]["devices"][0].update(index=False),
            lambda w: w["observed_device"].update(memory_bytes=123),
            lambda w: w["observed_device"].update(architecture="gfx9999"),
            lambda w: w["loaded_rccl"].update(verified=False),
            lambda w: w["observed_hardware_inventory"]["rccl_library"].update(sha256="d" * 64),
            lambda w: w["stage_capture"]["source"].update(kind="test_fixture"),
        ):
            plan, workers, exp = inputs()
            change(workers[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(plan, workers, exp)
        plan, workers, exp = inputs("tp2")
        workers[1]["stage_capture"]["metadata"]["rccl_binding"]["library_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            run(plan, workers, exp)

    def test_open_or_failed_capture_and_ambiguous_native_teardown_are_rejected(self):
        for update in ({"capture_closed": False}, {"capture_errors": ["failed"]},
                       {"native_sdk_stop_status": 1}, {"native_sdk_finalize_status": False},
                       {"native_sdk_finalize_status": 2}, {"complete_workspace": True},
                       {"hardware_validated": True}):
            plan, workers, exp = inputs()
            workers[0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                run(plan, workers, exp)

    def test_source_and_immutable_plan_bindings_are_required(self):
        for change in (
            lambda w: w.update(plan_sha256="e" * 64),
            lambda w: w["source_bindings"].update(worker_source_sha256="e" * 64),
            lambda w: w["source_bindings"].update(worker_source_reference="another-source.py"),
            lambda w: w.update(session_id="another-session"),
            lambda w: w["stage_capture"].update(clock_domain="linux-boot:another"),
        ):
            plan, workers, exp = inputs()
            change(workers[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(plan, workers, exp)

    def test_prepared_config_requires_backend_build_policy_and_unchanged_hash(self):
        for change in (
            lambda w: w["runtime_config"].pop("actual_attention_backends"),
            lambda w: w["runtime_config"]["torch_build"].update(hip_build_version="other"),
            lambda w: w["runtime_config"]["sampling_policy"].update(temperature=1),
            lambda w: w["runtime_config"]["sampling_policy"].update(temperature=False),
            lambda w: w["runtime_config"]["cache_config"].update(enable_prefix_caching=True),
            lambda w: w["runtime_config"].pop("sampling_policy"),
            lambda w: w.update(runtime_config_sha256="f" * 64),
        ):
            plan, workers, exp = inputs()
            change(workers[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(plan, workers, exp)
        plan, workers, exp = inputs()
        exp["calibration"]["runtime_config_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            run(plan, workers, exp)

    def test_epoch_index_payload_process_and_interval_are_actual_not_opaque_id_parsing(self):
        for change in (
            lambda w: w["records"][8].update(stage_record_index=0),
            lambda w: w["records"][8].update(capture_iteration=0),
            lambda w: w["records"][8].update(process_id=999),
            lambda w: w["records"][8].update(native_epoch_id=999),
            lambda w: w["records"][8]["stage_record"].update(sample_id="changed-in-one-copy"),
            lambda w: w["records"][8].update(start_monotonic_ns=999999999999),
            lambda w: w["records"][8].update(epoch_id=w["records"][7]["epoch_id"]),
        ):
            plan, workers, exp = inputs()
            change(workers[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(plan, workers, exp)
        plan, workers, exp = inputs()
        for index in range(len(workers[0]["records"])):
            change_record(workers[0], index, {"sample_id": "arbitrary-label-" + str(index)})
        self.assertEqual(run(plan, workers, exp)["formal_repetitions"], 1)

    def test_formal_missing_shapes_partial_prefill_or_missing_sample_do_not_enter_domain(self):
        for update in ({"phase": "recompute"}, {"query_tokens": [8]}, {"context_lengths": [1]},
                       {"sampled_token_counts": [0]}, {"output_history_lengths": [9]},
                       {"prompt_lengths": [64]}):
            plan, workers, exp = inputs()
            change_record(workers[0], 8, update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                run(plan, workers, exp)
        plan, workers, exp = inputs()
        workers[0]["records"].pop()
        workers[0]["stage_capture"]["records"].pop()
        with self.assertRaises(ValueError):
            run(plan, workers, exp)

    def test_pp_loaded_layers_endpoint_tying_and_native_payloads_are_not_relabelled(self):
        plan, workers, exp = inputs("pp2")
        for change in (
            lambda w: w["stage_capture"]["metadata"]["stage_layer_binding"].update(start_layer=0),
            lambda w: w["model_binding"]["word_embedding_tying"].update(effective_tie_word_embeddings=True),
            lambda w: w["stage_capture"]["metadata"].pop("native_pp_persistent_binding"),
        ):
            bad = deepcopy(workers)
            change(bad[1])
            with self.subTest(change=change), self.assertRaises(ValueError):
                run(plan, bad, exp)
        bad = deepcopy(workers)
        # A complete local descriptor with a different physical token count
        # must still fail its actual adjacent sender/receiver agreement.
        original = bad[1]["records"][8]["stage_record"]["transport_metadata"]["incoming"]["native_descriptor"]
        from mlsys_sim.config import experiment_from_dict
        replacement = native_pp_descriptor(experiment_from_dict(exp), original["padded_tokens"] + 1)
        for row in (bad[1]["records"][8]["stage_record"], bad[1]["stage_capture"]["records"][8]):
            row["transport_metadata"]["incoming"]["native_descriptor"] = deepcopy(replacement)
        with self.assertRaises(ValueError):
            run(plan, bad, exp)

    def test_last_stage_cpu_return_and_nonfinal_async_handoff_stay_distinct(self):
        plan, workers, exp = inputs("pp2")
        self.assertEqual(run(plan, workers, exp)["placements"][0]["stage_index"], 0)
        for update in ({"type": "vllm.v1.outputs.AsyncModelRunnerOutput"},
                       {"gpu_tensor_count": 1}, {"gpu_tensor_count": False},
                       {"execute_state_is_none": False}, {"prev_sampled_is_none": False}):
            bad = deepcopy(workers)
            bad[1]["records"][8]["return_witness"].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                run(plan, bad, exp)


if __name__ == "__main__":
    unittest.main()
