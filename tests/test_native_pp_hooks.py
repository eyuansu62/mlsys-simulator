"""Native PP lazy receive fixtures preserve source-shaped consumer boundaries."""

from copy import deepcopy
from dataclasses import replace
import gc
import unittest
import weakref

from mlsys_sim import ModelSpec, SimulationError
from mlsys_sim.vllm_hooks import SUPPORTED_VERSION_PREFIX, install_runner_hook
from test_vllm_hooks import Runner, experiment, schedule, state


class ManualClock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        return self.value
    def advance(self, milliseconds):
        self.value += int(milliseconds * 1_000_000)


class Tensor:
    def __init__(self, shape, dtype="bfloat16", element_size=2):
        self.shape, self.dtype, self.bytes = tuple(shape), "torch." + dtype, element_size
    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result
    def element_size(self):
        return self.bytes


class IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors
    def items(self):
        return self.tensors.items()


class Handle:
    def __init__(self, clock, error=None):
        self.clock, self.error, self.calls = clock, error, 0
    def wait(self):
        self.calls += 1
        self.clock.advance(20)
        if self.error is not None:
            raise self.error


class AsyncIntermediateTensors(IntermediateTensors):
    """Mirror the pinned lazy attribute access and mutable postprocess dict."""
    def __init__(self, tensors, handles, postprocess):
        super().__init__(tensors)
        self._comm_handles, self._comm_postprocess = handles, postprocess
        self._comm_waited, self.calls, self.tensor_accesses = False, 0, 0
    def wait_for_comm(self):
        self.calls += 1
        if self._comm_waited:
            return
        for handle in self._comm_handles:
            handle.wait()
        for callback in self._comm_postprocess:
            callback()
        self._comm_waited = True
    def __getattribute__(self, name):
        if name == "tensors":
            object.__setattr__(self, "tensor_accesses", object.__getattribute__(self, "tensor_accesses") + 1)
            if not object.__getattribute__(self, "_comm_waited"):
                object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


def native_experiment(pp=2, tp=2, hidden=128):
    original = experiment(pp=pp, tp=tp)
    model = replace(ModelSpec(), hidden_size=hidden, activation_dtype="bfloat16")
    return replace(original, strategy=replace(original.strategy, model=model,
        pipeline_transport="vllm_native", pipeline_transport_max_tokens=16, max_batch_tokens=16))


def payload(rows=8, hidden=128, dtype="bfloat16", element_size=2):
    return {name: Tensor((rows, hidden), dtype, element_size) for name in ("hidden_states", "residual")}


def lazy_payload(clock, *, rows=8, hidden=128, tp=2, wait_error=None, post_error=None, after_wait=None):
    full = payload(rows, hidden)
    tensors = {name: Tensor((rows * hidden // tp,)) for name in full}
    handle = Handle(clock, wait_error)
    def postprocess():
        clock.advance(7)
        if post_error is not None:
            raise post_error
        tensors.update(full)
        if after_wait is not None:
            after_wait()
    return AsyncIntermediateTensors(tensors, [handle], [postprocess]), handle


class NativeRunner(Runner):
    def __init__(self, clock, *, last=True):
        super().__init__(last=last)
        self.clock, self.consume, self.repeat_wait = clock, True, False
        self.intermediate = IntermediateTensors(payload())
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        result = super().execute_model(scheduler_output, intermediate_tensors)
        self.clock.advance(3)  # Prepared batch work before natural input use.
        if self.consume and intermediate_tensors is not None:
            for _key, _tensor in intermediate_tensors.items():
                self.clock.advance(5)  # Local persistent-buffer copy remains stage work.
            if self.repeat_wait and hasattr(intermediate_tensors, "wait_for_comm"):
                intermediate_tensors.wait_for_comm()
        self.clock.advance(11)
        return result
    def sample_tokens(self, grammar_output):
        result = super().sample_tokens(grammar_output)
        self.clock.advance(2)
        return result


class NativePPHookTests(unittest.TestCase):
    def setup_capture(self, *, stage=1, tp=2, pp=2, clock=None, synchronize=None, **kwargs):
        clock = clock or ManualClock()
        runner = NativeRunner(clock, last=stage == pp - 1)
        runner.requests = {"r": state(4)}
        hook = install_runner_hook(runner, kwargs.pop("experiment", native_experiment(pp, tp)),
            session_id="native-pp-cpu", stage_index=stage, clock_ns=clock,
            runtime_version=SUPPORTED_VERSION_PREFIX + ".d20261002",
            synchronize=synchronize or (lambda: clock.advance(1)), **kwargs)
        self.addCleanup(hook.close)
        return clock, runner, hook

    def run_final(self, runner, incoming, tokens=None):
        self.assertIsNone(runner.execute_model(schedule(tokens or {"r": 4}), incoming))
        return runner.sample_tokens(None)

    def test_natural_wait_excluded_copy_retained_and_actual_mixed_order_preserved(self):
        clock = ManualClock()
        incoming, handle = lazy_payload(clock)
        synchronizations = []
        def synchronize():
            self.assertTrue(incoming._comm_waited, "capture synchronized posted receive before natural use")
            synchronizations.append(clock.value)
            clock.advance(1)
        _, runner, hook = self.setup_capture(clock=clock, synchronize=synchronize)
        runner.requests = {"a-prefill": state(12, 4), "z-decode": state(8, 8, 1)}
        output = self.run_final(runner, incoming, {"a-prefill": 4, "z-decode": 1})
        self.assertEqual(output.req_ids, ["z-decode", "a-prefill"])
        record = hook.records[0]
        self.assertEqual(record["status"], "COMPLETED")
        self.assertTrue(record["profile_eligible"])
        self.assertEqual(record["request_phases"], ["decode", "prefill"])
        self.assertEqual(record["context_lengths"], [8, 4])
        self.assertEqual(record["duration_ms"], 27)
        self.assertEqual(record["pp_receive_timing"], {"method": "natural_wait_for_comm_v1",
            "intervals": [{"start_monotonic_ns": 3_000_000, "end_monotonic_ns": 31_000_000}],
            "excluded_ns": 28_000_000, "wall_duration_ms": 55})
        info = record["transport_metadata"]["incoming"]
        self.assertEqual(info["receive_completion_scope"], "natural_wait_for_comm")
        self.assertTrue(info["natural_receive_observed"])
        self.assertTrue(info["pre_wait_compute_stream_synchronized"])
        self.assertTrue(info["receive_wait_completed"])
        self.assertTrue(info["incoming_communication_excluded"])
        self.assertEqual(info["native_descriptor"]["padded_tokens"], 8)
        self.assertEqual(info["native_descriptor"]["wire_bytes_per_rank"], 2048)
        self.assertEqual(info["native_descriptor"]["resident_bytes_per_rank"], 4096)
        self.assertEqual(handle.calls, 1)
        self.assertEqual(incoming.calls, 1)
        self.assertEqual(incoming.tensor_accesses, 1)
        self.assertEqual(len(synchronizations), 2)
        self.assertNotIn("wait_for_comm", vars(incoming))
        self.assertEqual(hook._receive_methods, [])
        self.assertEqual(hook.header["source"]["kind"], "test_fixture")
        self.assertFalse(hook.header["hardware_validated"])

    def test_nonfinal_native_output_uses_observed_two_tensor_padded_payload(self):
        _, runner, hook = self.setup_capture(stage=0)
        output = runner.execute_model(schedule({"r": 4}))
        self.assertIs(output, runner.intermediate)
        record = hook.records[0]
        self.assertTrue(record["profile_eligible"])
        self.assertNotIn("pp_receive_timing", record)
        self.assertFalse(record["sampling_required"])
        descriptor = record["transport_metadata"]["outgoing"]["native_descriptor"]
        self.assertEqual(descriptor["padded_tokens"], 8)
        self.assertEqual([row["name"] for row in descriptor["tensors"]], ["hidden_states", "residual"])
        self.assertTrue(all(row["tp_sliced"] for row in descriptor["tensors"]))
        self.assertEqual(record["transport_metadata"]["incoming"]["modeled_transport"], "vllm_native_pp_v1")

    def test_unconsumed_lazy_receive_is_raw_without_instrumentation_wait_or_sync(self):
        clock = ManualClock()
        incoming, handle = lazy_payload(clock)
        synchronizations = []
        _, runner, hook = self.setup_capture(clock=clock, synchronize=lambda: synchronizations.append(True))
        runner.consume = False
        output = self.run_final(runner, incoming)
        self.assertEqual(output.req_ids, ["r"])
        self.assertFalse(hook.records[0]["profile_eligible"])
        self.assertEqual(handle.calls, 0)
        self.assertEqual(incoming.calls, 0)
        self.assertEqual(incoming.tensor_accesses, 0)
        self.assertEqual(synchronizations, [])
        self.assertEqual(hook._receive_methods, [])

    def test_real_wait_and_postprocess_exceptions_propagate_identically_and_restore(self):
        for kind in ("wait", "postprocess"):
            with self.subTest(kind=kind):
                clock = ManualClock()
                failure = RuntimeError(kind + " fixture failure")
                incoming, _ = lazy_payload(clock, wait_error=failure if kind == "wait" else None,
                                           post_error=failure if kind == "postprocess" else None)
                _, runner, hook = self.setup_capture(clock=clock)
                with self.assertRaises(RuntimeError) as raised:
                    runner.execute_model(schedule({"r": 4}), incoming)
                self.assertIs(raised.exception, failure)
                self.assertEqual(hook.records[0]["status"], "FAILED")
                self.assertFalse(hook.records[0]["profile_eligible"])
                self.assertNotIn("wait_for_comm", vars(incoming))
                self.assertEqual(hook._receive_methods, [])

    def test_repeated_natural_wait_remains_idempotent_and_records_one_interval(self):
        clock, runner, hook = self.setup_capture()
        incoming, handle = lazy_payload(clock)
        runner.repeat_wait = True
        self.run_final(runner, incoming)
        self.assertEqual(handle.calls, 1)
        self.assertEqual(incoming.calls, 2)
        self.assertEqual(len(hook.records[0]["pp_receive_timing"]["intervals"]), 1)
        self.assertEqual(hook.records[0]["pp_receive_timing"]["excluded_ns"], 28_000_000)
        self.assertTrue(hook.records[0]["profile_eligible"])

    def test_already_waited_lazy_input_declares_external_completion_without_new_interval(self):
        clock, runner, hook = self.setup_capture()
        incoming, handle = lazy_payload(clock)
        incoming.wait_for_comm()
        self.run_final(runner, incoming)
        record = hook.records[0]
        self.assertTrue(record["profile_eligible"])
        self.assertNotIn("pp_receive_timing", record)
        info = record["transport_metadata"]["incoming"]
        self.assertTrue(info["already_waited_external"])
        self.assertFalse(info["natural_receive_observed"])
        self.assertEqual(info["receive_completion_scope"], "before_runner_entry")
        self.assertTrue(info["external_receive_synchronized"])
        self.assertEqual(handle.calls, 1)
        self.assertEqual(incoming.calls, 1)

    def test_materialized_input_declares_external_completion(self):
        _, runner, hook = self.setup_capture()
        self.run_final(runner, IntermediateTensors(payload()))
        record = hook.records[0]
        self.assertTrue(record["profile_eligible"])
        self.assertNotIn("pp_receive_timing", record)
        info = record["transport_metadata"]["incoming"]
        self.assertFalse(info["lazy_incoming"])
        self.assertFalse(info["natural_receive_observed"])
        self.assertEqual(info["receive_completion_scope"], "before_runner_entry")
        self.assertTrue(info["external_receive_synchronized"])

    def test_pending_preparation_is_completed_before_receive_clock_and_stays_in_stage_service(self):
        clock = ManualClock()
        incoming, handle = lazy_payload(clock)
        completed_preparation = []
        def synchronize_current_stream():
            self.assertFalse(incoming._comm_waited)
            self.assertEqual(handle.calls, 0)
            completed_preparation.append(True)
            clock.advance(13)  # H2D work queued by _prepare_inputs, not receive work.
        _, runner, hook = self.setup_capture(clock=clock,
            synchronize_current_stream=synchronize_current_stream)
        self.run_final(runner, incoming)
        record = hook.records[0]
        self.assertEqual(completed_preparation, [True])
        self.assertTrue(record["profile_eligible"])
        self.assertEqual(record["duration_ms"], 40)  # 27ms service plus 13ms preparation.
        self.assertEqual(record["pp_receive_timing"]["wall_duration_ms"], 68)
        self.assertEqual(record["pp_receive_timing"]["excluded_ns"], 28_000_000)
        self.assertEqual(record["pp_receive_timing"]["intervals"][0]["start_monotonic_ns"], 16_000_000)

    def test_external_communication_gpu_tail_finishes_before_reset_stage_start(self):
        for kind in ("lazy", "materialized"):
            with self.subTest(kind=kind):
                clock = ManualClock()
                calls = []
                def synchronize():
                    calls.append(True)
                    clock.advance(47 if len(calls) == 1 else 1)
                _, runner, hook = self.setup_capture(clock=clock, synchronize=synchronize)
                if kind == "lazy":
                    incoming, handle = lazy_payload(clock)
                    incoming.wait_for_comm()  # Enqueues completion; does not finish the fake GPU tail.
                else:
                    incoming = IntermediateTensors(payload())
                before = clock.value
                self.run_final(runner, incoming)
                record = hook.records[0]
                self.assertEqual(record["start_monotonic_ns"], before + 47_000_000)
                self.assertEqual(record["duration_ms"], 27)
                self.assertNotIn("pp_receive_timing", record)
                self.assertTrue(record["transport_metadata"]["incoming"]["external_receive_synchronized"])

    def test_unknown_native_stream_contract_remains_raw_without_extra_pre_wait_sync(self):
        clock, runner, hook = self.setup_capture(synchronize_current_stream=lambda: self.fail("unproven stream"))
        hook._native_stream_proven = False
        incoming, handle = lazy_payload(clock)
        self.run_final(runner, incoming)
        record = hook.records[0]
        self.assertFalse(record["profile_eligible"])
        self.assertEqual(record["error_code"], "UNSUPPORTED_VLLM_FEATURE")
        self.assertEqual(handle.calls, 1)
        self.assertNotIn("pp_receive_timing", record)

    def test_loaded_persistent_capacity_and_descriptor_are_bound_independently_of_small_batch(self):
        clock, runner, hook = self.setup_capture()
        runner.max_num_tokens = 16
        runner.intermediate_tensors = IntermediateTensors(payload(rows=16))
        hook._bind_native_persistent()
        binding = hook.header["metadata"]["native_pp_persistent_binding"]
        self.assertEqual(binding["max_num_tokens"], 16)
        self.assertEqual(binding["descriptor"]["padded_tokens"], 16)
        self.assertEqual(binding["descriptor"]["resident_bytes_per_rank"], 8192)
        self.assertEqual(binding["scope"], "loaded_runner_intermediate_buffers")
        incoming, _ = lazy_payload(clock, rows=8)
        self.run_final(runner, incoming)
        self.assertEqual(hook.records[0]["query_tokens"], [4])
        self.assertTrue(hook.records[0]["profile_eligible"])
        self.assertEqual(hook.header["metadata"]["native_pp_persistent_binding"], binding)

    def test_large_or_inconsistent_loaded_buffers_are_rejected_before_binding(self):
        cases = ((32768, 16), (16, 8), (0, 16), (True, 16), (None, 16))
        for maximum, physical in cases:
            with self.subTest(maximum=maximum, physical=physical):
                _, runner, hook = self.setup_capture()
                runner.max_num_tokens = maximum
                runner.intermediate_tensors = IntermediateTensors(payload(rows=physical))
                with self.assertRaises(SimulationError):
                    hook._bind_native_persistent()
                self.assertNotIn("native_pp_persistent_binding", hook.header["metadata"])
        _, runner, hook = self.setup_capture()
        runner.max_num_tokens = 16
        tensors = payload(rows=16)
        tensors["residual"].dtype = "torch.float32"
        runner.intermediate_tensors = IntermediateTensors(tensors)
        with self.assertRaises(SimulationError):
            hook._bind_native_persistent()

    def test_loaded_persistent_binding_cannot_change_after_initial_observation(self):
        _, runner, hook = self.setup_capture()
        runner.max_num_tokens = 16
        runner.intermediate_tensors = IntermediateTensors(payload(rows=16))
        hook._bind_native_persistent()
        before = deepcopy(hook.header["metadata"]["native_pp_persistent_binding"])
        runner.max_num_tokens = 8
        runner.intermediate_tensors = IntermediateTensors(payload(rows=8))
        with self.assertRaises(SimulationError):
            hook._bind_native_persistent()
        self.assertEqual(hook.header["metadata"]["native_pp_persistent_binding"], before)

    def test_current_incoming_and_outgoing_payload_must_fit_actual_loaded_receive_buffer(self):
        for direction in ("incoming", "outgoing"):
            with self.subTest(direction=direction):
                clock, runner, hook = self.setup_capture(stage=1, pp=3 if direction == "outgoing" else 2)
                runner.max_num_tokens = 4
                runner.intermediate_tensors = IntermediateTensors(payload(rows=4))
                hook._bind_native_persistent()
                incoming, _ = lazy_payload(clock, rows=8 if direction == "incoming" else 4)
                if direction == "incoming":
                    self.run_final(runner, incoming)
                else:
                    self.assertIs(runner.execute_model(schedule({"r": 4}), incoming), runner.intermediate)
                self.assertFalse(hook.records[0]["profile_eligible"])
                self.assertEqual(hook.records[0]["error_code"], "UNSUPPORTED_VLLM_BOUNDARY")

    def test_first_stage_does_not_invent_a_loaded_receive_buffer_binding(self):
        _, _, hook = self.setup_capture(stage=0)
        hook._bind_native_persistent()
        self.assertNotIn("native_pp_persistent_binding", hook.header["metadata"])

    def test_missing_lazy_abi_proof_is_raw_and_does_not_install_a_wait(self):
        clock, runner, hook = self.setup_capture()
        incoming, handle = lazy_payload(clock)
        del incoming._comm_postprocess
        runner.consume = False
        self.run_final(runner, incoming)
        self.assertFalse(hook.records[0]["profile_eligible"])
        self.assertEqual(handle.calls, 0)
        self.assertNotIn("wait_for_comm", vars(incoming))
        self.assertEqual(hook._receive_methods, [])

    def test_actual_payload_metadata_mismatches_are_not_replaced_with_expected_values(self):
        for kind in ("dtype", "element-size", "hidden", "rows", "missing", "unknown"):
            with self.subTest(kind=kind):
                _, runner, hook = self.setup_capture(stage=0)
                tensors = payload()
                if kind == "dtype":
                    tensors["residual"].dtype = "torch.float32"
                elif kind == "element-size":
                    tensors["residual"].bytes = 4
                elif kind == "hidden":
                    tensors["residual"].shape = (8, 64)
                elif kind == "rows":
                    tensors = payload(rows=2)
                elif kind == "missing":
                    tensors.pop("residual")
                else:
                    tensors["extra"] = Tensor((8, 128))
                runner.intermediate = IntermediateTensors(tensors)
                self.assertIs(runner.execute_model(schedule({"r": 4})), runner.intermediate)
                self.assertFalse(hook.records[0]["profile_eligible"])

    def test_unconfigured_nondivisible_tensor_numel_is_not_accepted_as_expected_payload(self):
        _, runner, hook = self.setup_capture(stage=0)
        runner.intermediate = IntermediateTensors(payload(rows=1, hidden=127))
        runner.requests = {"r": state(1)}
        runner.execute_model(schedule({"r": 1}))
        record = hook.records[0]
        self.assertFalse(record["profile_eligible"])
        info = record["transport_metadata"]["outgoing"]
        self.assertNotIn("native_descriptor", info)

    def test_native_pp_without_tp_sends_full_tensors_without_receiver_allgather(self):
        _, runner, hook = self.setup_capture(stage=0, tp=1)
        runner.execute_model(schedule({"r": 4}))
        record = hook.records[0]
        self.assertTrue(record["profile_eligible"])
        info = record["transport_metadata"]["outgoing"]
        self.assertFalse(info["default_tp_sliced_p2p_receiver_allgather"])
        self.assertFalse(any(row["receiver_allgather"] for row in info["native_descriptor"]["tensors"]))
        self.assertEqual(info["native_descriptor"]["wire_bytes_per_rank"], 4096)

    def test_close_inside_original_wait_seals_capture_and_restores_temporary_method(self):
        clock, runner, hook = self.setup_capture()
        sealed = []
        def close_during_postprocess():
            hook.close()
            sealed.append(hook.export_capture())
        incoming, _ = lazy_payload(clock, after_wait=close_during_postprocess)
        self.assertIsNone(runner.execute_model(schedule({"r": 4}), incoming))
        self.assertEqual(hook.export_capture(), sealed[0])
        self.assertEqual(hook.records[0]["status"], "INCOMPLETE")
        self.assertNotIn("wait_for_comm", vars(incoming))
        self.assertEqual(hook._receive_methods, [])

    def test_close_inside_receive_or_final_synchronization_never_mutates_sealed_records(self):
        for close_call in (1, 2):
            with self.subTest(close_call=close_call):
                clock = ManualClock()
                holder, sealed, calls = {}, [], []
                def synchronize():
                    calls.append(True)
                    clock.advance(1)
                    if len(calls) == close_call:
                        holder["hook"].close()
                        sealed.append(holder["hook"].export_capture())
                _, runner, hook = self.setup_capture(clock=clock, synchronize=synchronize)
                holder["hook"] = hook
                incoming, _ = lazy_payload(clock)
                runner.execute_model(schedule({"r": 4}), incoming)
                runner.sample_tokens(None)
                self.assertEqual(hook.export_capture(), sealed[0])
                self.assertNotIn("wait_for_comm", vars(incoming))
                self.assertEqual(hook._receive_methods, [])

    def test_close_during_pre_wait_preparation_preserves_one_real_wait_and_its_exception(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                clock, holder, sealed = ManualClock(), {}, []
                def synchronize_current_stream():
                    holder["hook"].close()
                    sealed.append(holder["hook"].export_capture())
                _, runner, hook = self.setup_capture(clock=clock,
                    synchronize_current_stream=synchronize_current_stream)
                holder["hook"] = hook
                failure = RuntimeError("wait failed after close") if failed else None
                incoming, handle = lazy_payload(clock, wait_error=failure)
                if failed:
                    with self.assertRaises(RuntimeError) as raised:
                        runner.execute_model(schedule({"r": 4}), incoming)
                    self.assertIs(raised.exception, failure)
                else:
                    self.assertIsNone(runner.execute_model(schedule({"r": 4}), incoming))
                self.assertEqual(handle.calls, 1)
                self.assertEqual(incoming.calls, 1)
                self.assertEqual(hook.export_capture(), sealed[0])
                self.assertNotIn("wait_for_comm", vars(incoming))

    def test_temporary_wait_observer_does_not_retain_input_after_completion(self):
        clock, runner, hook = self.setup_capture()
        incoming, _ = lazy_payload(clock)
        reference = weakref.ref(incoming)
        self.run_final(runner, incoming)
        del incoming
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(hook._receive_methods, [])

    def test_native_async_output_is_still_raw_and_is_never_proactively_consumed(self):
        clock, runner, hook = self.setup_capture()
        runner.use_async_scheduling = True
        incoming, _ = lazy_payload(clock)
        future = self.run_final(runner, incoming)
        self.assertEqual(future.calls, 0)
        self.assertFalse(hook.records[0]["profile_eligible"])
        future.get_output()
        self.assertFalse(hook.records[0]["profile_eligible"])

    def test_record_limit_bounds_observers_without_affecting_real_consumption(self):
        clock, runner, hook = self.setup_capture(max_records=1)
        inputs = []
        for _ in range(4):
            incoming, handle = lazy_payload(clock)
            inputs.append((incoming, handle))
            self.run_final(runner, incoming)
        self.assertEqual(len(hook.records), 1)
        self.assertTrue(all(handle.calls == 1 for _, handle in inputs))
        self.assertEqual(hook._receive_methods, [])
        self.assertTrue(all("wait_for_comm" not in vars(incoming) for incoming, _ in inputs))

    def test_close_after_sampling_does_not_change_exported_completed_record(self):
        clock, runner, hook = self.setup_capture()
        incoming, _ = lazy_payload(clock)
        self.run_final(runner, incoming)
        before = deepcopy(hook.records)
        hook.close()
        self.assertEqual(hook.records, before)


if __name__ == "__main__":
    unittest.main()
