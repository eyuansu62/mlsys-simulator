"""Native PP evidence, interval isolation and actual Hook-to-Profile workflows."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from mlsys_sim import (DeviceSpec, InfraSpec, ProfileModel, SimulationError, calibrate_dataset, profile_identity,
                       predict_experiment, simulate, write_json_exclusive)
from mlsys_sim.pp_transport import native_pp_descriptor
from mlsys_sim.vllm_capture import normalize_vllm_capture
from mlsys_sim.vllm_hooks import install_runner_hook
from test_vllm_capture import VERSION, capture, experiment, record


ROOT = Path(__file__).resolve().parents[1]


def native_experiment(tp=2, pp=2):
    base = experiment(tp=tp, pp=pp, runtime_hash="")
    model = replace(base.strategy.model, activation_dtype="float16")
    return replace(base, strategy=replace(base.strategy, model=model,
        pipeline_transport="vllm_native", pipeline_transport_max_tokens=32,
        max_batch_tokens=16, max_batch_size=1))


def native_record(exp, stage, *, padded=4, iteration=1):
    row = record(stage, exp.strategy.pipeline_parallel_size, iteration=iteration, duration=4)
    descriptor = native_pp_descriptor(exp, padded)
    common = {"object_type": "fixture.IntermediateTensors", "lazy_incoming": False,
        "materialized_tensor_keys": ["hidden_states", "residual"],
        "modeled_transport": "vllm_native_pp_v1", "profile_transport_compatible": True,
        "incoming_communication_excluded": True, "native_descriptor": descriptor,
        "default_tp_sliced_p2p_receiver_allgather": any(t["tp_sliced"] for t in descriptor["tensors"])}
    incoming = ({"object_type": "builtins.NoneType"} if stage == 0 else {
        **deepcopy(common), "lazy_incoming": True, "natural_receive_observed": True,
        "receive_wait_completed": True, "receive_completion_scope": "natural_wait_for_comm",
        "pre_wait_compute_stream_synchronized": True})
    row["transport_metadata"] = {"incoming": incoming}
    if stage < exp.strategy.pipeline_parallel_size - 1:
        row["transport_metadata"]["outgoing"] = deepcopy(common)
    if stage:
        start = row["start_monotonic_ns"]
        row["pp_receive_timing"] = {"method": "natural_wait_for_comm_v1",
            "intervals": [{"start_monotonic_ns": start + 1_000_000,
                           "end_monotonic_ns": start + 3_000_000}],
            "excluded_ns": 2_000_000, "wall_duration_ms": 4.0}
        row["duration_ms"] = 2.0
    return row


def native_channels(exp, *, padded=4):
    # Hand-authored schema fixtures retain the established explicit Runtime hash.
    base = experiment(runtime_hash="a" * 64)
    from test_vllm_capture import EXECUTION_HASH
    exp = replace(exp, calibration=replace(base.calibration, runtime_config_sha256=EXECUTION_HASH))
    return exp, [capture(exp, stage=s, rank=r, records=[native_record(exp, s, padded=padded)])
                 for s in range(exp.strategy.pipeline_parallel_size)
                 for r in range(exp.strategy.tensor_parallel_size)]


def normalize(exp, channels):
    return normalize_vllm_capture(exp, channels, allow_test_fixture=True)


class Clock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        self.value += 1_000_000
        return self.value


class Tensor:
    def __init__(self, tokens, hidden):
        self.shape = (tokens, hidden)
        self.dtype = "torch.float16"
    def element_size(self):
        return 2
    def numel(self):
        return self.shape[0] * self.shape[1]
    def is_contiguous(self):
        return True


class AsyncIntermediateTensors:
    """Mirror the pinned lazy attribute contract; the test never imports vLLM."""
    def __init__(self, tensors):
        self.tensors = tensors
        self._comm_handles = []
        self._comm_postprocess = []
        self._comm_waited = False
        self.calls = 0
    def wait_for_comm(self):
        if self._comm_waited:
            return
        self.calls += 1
        for handle in self._comm_handles:
            handle.wait()
        for callback in self._comm_postprocess:
            callback()
        self._comm_waited = True
    def __getattribute__(self, name):
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


class Runner:
    def __init__(self, exp, stage):
        self.exp, self.stage = exp, stage
        self.requests = {"r": SimpleNamespace(num_prompt_tokens=4, output_token_ids=[])}
        self.input_batch = SimpleNamespace(num_reqs=1, req_ids=["r"], num_computed_tokens_cpu=[0])
        self.execute_model_state = None
        self.use_async_scheduling = False
    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        return None
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self._prepare_inputs(scheduler_output, list(scheduler_output.num_scheduled_tokens.values()))
        if intermediate_tensors is not None:
            # This is the real caller's natural consumption, not the capture's.
            assert set(intermediate_tensors.tensors) == {"hidden_states", "residual"}
        if self.stage < self.exp.strategy.pipeline_parallel_size - 1:
            n = sum(scheduler_output.num_scheduled_tokens.values())
            return SimpleNamespace(tensors={name: Tensor(n, self.exp.strategy.model.hidden_size)
                                           for name in ("hidden_states", "residual")})
        self.execute_model_state = scheduler_output
        return None
    def sample_tokens(self, grammar_output):
        self.execute_model_state = None
        return SimpleNamespace(req_ids=["r"], sampled_token_ids=[[7]])


def actual_native_channels(exp):
    channels = []
    for stage in range(exp.strategy.pipeline_parallel_size):
        for rank in range(exp.strategy.tensor_parallel_size):
            runner = Runner(exp, stage)
            hook = install_runner_hook(runner, exp, session_id="native-actual-fixture", stage_index=stage,
                tp_rank=rank, runtime_version=VERSION, clock_ns=Clock(), synchronize=lambda: None)
            try:
                for query, context, history in ((4, 0, 0), (1, 4, 1), (1, 5, 2)):
                    runner.input_batch.num_computed_tokens_cpu = [context]
                    runner.requests["r"].output_token_ids = [7] * history
                    schedule = SimpleNamespace(num_scheduled_tokens={"r": query})
                    incoming = (AsyncIntermediateTensors({name: Tensor(query, exp.strategy.model.hidden_size)
                                for name in ("hidden_states", "residual")}) if stage else None)
                    runner.execute_model(schedule, incoming)
                    if stage == exp.strategy.pipeline_parallel_size - 1:
                        runner.sample_tokens(None)
                    if incoming is not None:
                        assert incoming.calls == 1
                        assert "wait_for_comm" not in vars(incoming)
            finally:
                hook.close()
            channels.append(hook.export_capture())
    return channels


class NativePPCaptureTests(unittest.TestCase):
    def test_native_wait_is_diagnostic_and_profile_service_excludes_it(self):
        exp, channels = native_channels(native_experiment(), padded=8)
        before = deepcopy(channels)
        result = normalize(exp, channels)
        self.assertEqual(channels, before)
        self.assertTrue(result["canonical_stage_ready"])
        self.assertFalse(result["hardware_validated"])
        rows = result["dataset"]["records"]
        self.assertEqual([r["duration_ms"] for r in rows], [4.0, 2.0])
        self.assertEqual(rows[0]["pp_transport"]["outgoing"]["padded_tokens"], 8)
        self.assertEqual(rows[0]["pp_transport"]["outgoing"], rows[1]["pp_transport"]["incoming"])
        self.assertFalse(result["dataset"]["notes"]["pp_transport_policy"]["receive_wait_is_network_latency_sample"])
        self.assertEqual(result["raw_records"][2]["records"][0]["pp_receive_timing"]["wall_duration_ms"], 4.0)

    def test_receive_interval_arithmetic_is_strict(self):
        exp, original = native_channels(native_experiment())
        for field, value in (("excluded_ns", 1), ("excluded_ns", True), ("wall_duration_ms", 2.0),
                             ("method", "network_service"), ("intervals", [])):
            channels = deepcopy(original)
            channels[2]["records"][0]["pp_receive_timing"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SimulationError):
                normalize(exp, channels)
        channels = deepcopy(original)
        channels[2]["records"][0]["pp_receive_timing"]["intervals"][0]["end_monotonic_ns"] += 10_000_000
        with self.assertRaises(SimulationError):
            normalize(exp, channels)

    def test_complete_adjacent_rank_geometry_and_natural_consumption_are_required(self):
        exp, channels = native_channels(native_experiment())
        missing = normalize(exp, channels[:2])
        self.assertFalse(missing["canonical_stage_ready"])
        self.assertEqual(missing["datasets"], [])
        for path in ("natural_receive_observed", "receive_wait_completed", "pre_wait_compute_stream_synchronized"):
            changed = deepcopy(channels)
            changed[2]["records"][0]["transport_metadata"]["incoming"][path] = False
            result = normalize(exp, changed)
            self.assertFalse(result["canonical_stage_ready"])
            self.assertTrue(any("native_pp_natural_receive_unobserved" in r["reasons"] for r in result["quarantined"]))
        mismatch = deepcopy(channels)
        mismatch[2]["records"][0]["transport_metadata"]["incoming"]["native_descriptor"] = native_pp_descriptor(exp, 8)
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, mismatch)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

    def test_fabricated_tensor_bytes_or_slice_branch_cannot_be_calibrated(self):
        exp, channels = native_channels(native_experiment())
        for field, value in (("wire_bytes_per_rank", 1), ("tp_sliced", False), ("element_size", True)):
            changed = deepcopy(channels)
            tensor = changed[0]["records"][0]["transport_metadata"]["outgoing"]["native_descriptor"]["tensors"][0]
            tensor[field] = value
            with self.subTest(field=field), self.assertRaises(SimulationError):
                normalize(exp, changed)

    def test_external_completed_receive_does_not_claim_a_natural_wait(self):
        exp, original = native_channels(native_experiment())
        for lazy in (False, True):
            channels = deepcopy(original)
            for channel in channels[2:]:
                row = channel["records"][0]
                row.pop("pp_receive_timing")
                row["duration_ms"] = 4.0
                info = row["transport_metadata"]["incoming"]
                info.update(lazy_incoming=lazy, natural_receive_observed=False,
                            receive_completion_scope="before_runner_entry", already_waited_external=lazy,
                            external_receive_synchronized=True)
            with self.subTest(lazy=lazy):
                self.assertTrue(normalize(exp, channels)["canonical_stage_ready"])
                if lazy:
                    channels[2]["records"][0]["transport_metadata"]["incoming"].pop("already_waited_external")
                    self.assertFalse(normalize(exp, channels)["canonical_stage_ready"])

    def test_native_tp_one_still_requires_loaded_rccl_for_measured_pp(self):
        base = native_experiment(tp=1)
        from test_vllm_capture import EXECUTION_HASH
        exp = replace(base, infra=InfraSpec(devices=tuple(DeviceSpec(f"d{i}", name="AMD schema-only device")
                                                        for i in range(2))),
            strategy=replace(base.strategy, model=replace(base.strategy.model,
                weight_dtype="float16", kv_dtype="float16")),
            calibration=replace(base.calibration, driver_version="declared-driver", rocm_version="declared-rocm",
                rccl_version="declared-rccl", runtime_config_sha256=EXECUTION_HASH))
        channels = [capture(exp, stage=s, records=[native_record(exp, s)]) for s in range(2)]
        for stage, channel in enumerate(channels):
            channel["source"] = {"kind": "measured", "reference": "Schema rejection test only; no GPU evidence"}
            channel["metadata"]["stage_layer_binding"] = {"start_layer": stage * 2,
                "end_layer": stage * 2 + 2, "layer_count": 2, "method": "loaded_qwen_model_start_end_layer"}
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, channels)
        self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")
        self.assertIn("RCCL", str(raised.exception))
        for channel in channels:
            channel["metadata"]["rccl_binding"] = {"verified": True, "runtime_version": "declared-rccl",
                "library_sha256": "a" * 64, "verification_scope": "local_proc_maps_loaded_library_ncclGetVersion"}
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, channels)
        self.assertIn("transport binding", str(raised.exception))
        for channel in channels:
            channel["metadata"]["native_pp_backend_binding"] = {"backend": "nccl",
                "use_cpu_custom_send_recv": False, "api_revision": "98dff2a81", "torch_version": "declared-torch"}
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, channels)
        self.assertEqual(raised.exception.code, "CALIBRATION_MISMATCH")

        execution = {"torch_build": {"torch_version": "declared-torch"}}
        digest = hashlib.sha256(json.dumps(execution, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        exp = replace(exp, calibration=replace(exp.calibration, runtime_config_sha256=digest))
        for channel in channels:
            channel["identity"] = profile_identity(exp)
            channel["experiment"] = asdict(exp)
            channel["metadata"].update(runtime_execution_config=execution, torch_version="declared-torch",
                native_pp_stream_binding={"standard_process_group_nccl": True, "actual_backend": "nccl",
                    "group_type": "vllm.distributed.parallel_state.GroupCoordinator",
                    "contract": "current_to_communication_stream_until_work_wait_v1"})
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, channels)
        self.assertIn("persistent", str(raised.exception))
        larger = replace(exp, strategy=replace(exp.strategy, pipeline_transport_max_tokens=32768))
        channels[1]["metadata"]["native_pp_persistent_binding"] = {"max_num_tokens": 32768,
            "descriptor": native_pp_descriptor(larger, 32768), "scope": "loaded_runner_intermediate_buffers"}
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, channels)
        self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")
        channels[1]["metadata"]["native_pp_persistent_binding"].update(max_num_tokens=1,
            descriptor=native_pp_descriptor(exp, 1))
        with self.assertRaises(SimulationError) as raised:
            normalize(exp, channels)
        self.assertIn("exceeds loaded persistent", str(raised.exception))

    def test_actual_hook_exports_merge_fit_and_run_exact_native_profiles(self):
        base = native_experiment()
        exp = replace(base, workload=(replace(base.workload[0], output_tokens=3),))
        channels = actual_native_channels(exp)
        result = normalize(exp, channels)
        self.assertTrue(result["canonical_stage_ready"], result["quarantined"])
        self.assertEqual(result["capture_summary"]["accepted_groups"], 6)
        profile = calibrate_dataset(result["dataset"], profile_version="native-e2e", allow_test_fixture=True)
        self.assertTrue(all("pp_transport" in sample for sample in profile["samples"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            write_json_exclusive(path, profile)
            configured = replace(exp, calibration=replace(exp.calibration,
                runtime_config_sha256=result["provenance"]["runtime_config_binding"]["observed_sha256"],
                backend="profile", profile_path=str(path), profile_version="native-e2e", allow_test_fixture=True))
            prediction = predict_experiment(configured)
            simulated = simulate(configured)
            self.assertEqual(simulated.result["counts"]["completed"], 1)
            self.assertEqual(simulated.requests[0]["output_tokens_actual"], 3)
            self.assertGreater(prediction["communication"]["prefill_p2p_ms"], 0)
            self.assertEqual(ProfileModel(configured).estimate_step("prefill", [4], [0], 1).source,
                             "test_fixture_profile")
            config = Path(directory) / "experiment.json"
            config.write_text(json.dumps(asdict(exp)))
            paths = []
            for i, channel in enumerate(channels):
                channel_path = Path(directory) / f"rank-{i}.json"
                channel_path.write_text(json.dumps(channel))
                paths.append(channel_path.name)
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"schema_version": 1, "captures": paths}))
            out = Path(directory) / "collected"
            process = subprocess.run([sys.executable, "-m", "mlsys_sim", "collect", "--kind", "vllm-hook",
                "--experiment", str(config), "--plan", str(plan), "--allow-test-fixture", "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            self.assertTrue(json.loads((out / "result.json").read_text())["canonical_stage_ready"])
            self.assertTrue((out / "dataset.json").exists())


if __name__ == "__main__":
    unittest.main()
