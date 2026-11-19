"""Source-shaped CPU fixtures only: no vLLM, Torch, GPU or fitted timings."""

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from mlsys_sim import Experiment, SimulationError
from mlsys_sim.profiles import profile_identity
from mlsys_sim.runtime_cost_capture import (
    RuntimeStepRecorder,
    PROTOCOL,
    apply_main_thread_affinity,
    preparation_policy,
    run_fixed_warmup,
    verify_main_thread_affinity,
)
from mlsys_sim.vllm_hooks import SUPPORTED_VERSION_PREFIX, _config_digest, _execution_config
from mlsys_sim.vllm_sla_worker import _VllmBackend, _progress_observer


class Clock:
    def __init__(self, increment=1000000):
        self.value, self.increment = 0, increment

    def __call__(self):
        self.value += self.increment
        return self.value


class Scheduler:
    def __init__(self):
        self.requests, self.selected = {}, {}
        self.running, self.waiting = [], []
        self.published = 0

    def schedule(self):
        for name, q in self.selected.items():
            self.requests[name].num_computed_tokens += q
        return NS(
            num_scheduled_tokens=dict(self.selected),
            total_num_scheduled_tokens=sum(self.selected.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
        )

    def update_from_output(self):
        self.published += 1


class Core:
    def __init__(self):
        self.scheduler = Scheduler()
        self.step_fn = self.step
        self.fail = False

    def step(self):
        self.scheduler.schedule()
        if self.fail:
            raise RuntimeError("native failure")
        self.scheduler.update_from_output()
        return {"native-result": True}


def fixture():
    exp = Experiment()
    exp = replace(
        exp,
        calibration=replace(
            exp.calibration,
            runtime_name="vllm",
            runtime_version=SUPPORTED_VERSION_PREFIX,
            allow_test_fixture=True,
        ),
    )
    policy = {"observer_protocol": PROTOCOL, "latency_adaptive": False, "main_thread_cpu": None}
    core = Core()
    recorder = RuntimeStepRecorder(
        core,
        exp,
        "fixture-session",
        policy=policy,
        runtime_version=SUPPORTED_VERSION_PREFIX,
        clock_ns=Clock(),
        thread_clock_ns=Clock(100000),
        allow_test_fixture=True,
    )
    snapshot = {"fixture_actual_execution": True}
    digest = _config_digest(snapshot)
    stage_exp = replace(exp, calibration=replace(exp.calibration, runtime_config_sha256=digest))
    binding = {
        "source": {"kind": "test_fixture"},
        "capture_errors": [],
        "capture_closed": True,
        "identity": profile_identity(stage_exp),
        "metadata": {"runtime_execution_config": snapshot, "runtime_config_sha256": digest},
    }
    return core, recorder, binding


def selected(core, members):
    core.scheduler.selected = {name: q for name, q, c, p in members}
    core.scheduler.requests = {
        name: NS(num_computed_tokens=c, num_prompt_tokens=p) for name, q, c, p in members
    }


class RuntimeCaptureTests(unittest.TestCase):
    def test_actual_scheduled_order_context_roles_and_core_wall_are_retained(self):
        core, recorder, binding = fixture()
        self.addCleanup(recorder.close)
        selected(core, [("warm", 4, 0, 4)])
        core.step_fn()
        recorder.bind_runtime(binding)
        recorder.begin_formal()
        selected(core, [("decode-first", 1, 10, 8), ("prefill-second", 4, 3, 8)])
        self.assertEqual(core.step_fn(), {"native-result": True})
        recorder.close()
        data = recorder.export_dataset()
        self.assertEqual(data["source"]["kind"], "test_fixture")
        self.assertEqual(data["domain"]["timing_boundary"], "engine_core_step_wall_v1")
        row = data["records"][0]
        self.assertEqual(row["source_order"], 2)
        self.assertEqual(row["query_tokens"], [1, 4])
        self.assertEqual(row["context_lengths"], [10, 3])
        self.assertEqual(row["request_phases"], ["decode", "prefill"])
        self.assertEqual(row["phase"], "mixed")
        self.assertGreater(row["duration_ms"], 0)
        self.assertEqual(len(recorder.export_capture()["iterations"]), 2)
        self.assertEqual(len(data["notes"]["warmup_record_ids"]), 1)
        self.assertEqual(core.scheduler.published, 2)

    def test_prefix_cached_context_is_post_schedule_cursor_minus_query(self):
        core, recorder, binding = fixture()
        self.addCleanup(recorder.close)
        recorder.bind_runtime(binding)
        recorder.begin_formal()
        selected(core, [("cached", 2, 8, 10)])
        core.step_fn()
        recorder.close()
        self.assertEqual(recorder.export_dataset()["records"][0]["context_lengths"], [8])

    def test_injection_requires_explicit_fixture_and_live_objects_cannot_be_spoofed(self):
        with self.assertRaisesRegex(SimulationError, "explicit test_fixture"):
            RuntimeStepRecorder(Core(), Experiment(), "invalid", clock_ns=Clock())
        with self.assertRaises(SimulationError):
            RuntimeStepRecorder(Core(), Experiment(), "invalid", runtime_version=SUPPORTED_VERSION_PREFIX)

    def test_warmup_only_is_not_formal_dataset_and_binding_is_required(self):
        core, recorder, binding = fixture()
        self.addCleanup(recorder.close)
        with self.assertRaises(SimulationError):
            recorder.begin_formal()
        selected(core, [("w", 4, 0, 4)])
        core.step_fn()
        recorder.bind_runtime(binding)
        recorder.close()
        with self.assertRaisesRegex(SimulationError, "No formal"):
            recorder.export_dataset()

    def test_native_errors_and_unusable_records_survive_without_becoming_costs(self):
        core, recorder, binding = fixture()
        self.addCleanup(recorder.close)
        recorder.bind_runtime(binding)
        recorder.begin_formal()
        selected(core, [("r", 4, 0, 4)])
        core.fail = True
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            core.step_fn()
        recorder.close()
        capture = recorder.export_capture()
        self.assertEqual(capture["iterations"][0]["status"], "FAILED")
        with self.assertRaises(SimulationError):
            recorder.export_dataset()

    def test_invalid_injected_clock_cannot_leave_an_active_hook_or_forge_costs(self):
        core, recorder, binding = fixture()
        recorder.bind_runtime(binding)
        recorder.begin_formal()
        recorder._clock = lambda: -1
        with self.assertRaises(SimulationError):
            core.step_fn()
        recorder.close()
        self.assertEqual(core.step_fn.__func__, Core.step)
        self.assertTrue(recorder.export_capture()["capture_errors"])
        with self.assertRaises(SimulationError):
            recorder.export_dataset()

    def test_outliers_remain_and_idle_steps_have_explicit_nontraining_lineage(self):
        core, recorder, binding = fixture()
        self.addCleanup(recorder.close)
        recorder.bind_runtime(binding)
        recorder.begin_formal()
        core.step_fn()
        for i in range(3):
            if i == 2:
                recorder._clock.increment = 100000000
            selected(core, [("r", 4, i, 8)])
            core.step_fn()
        recorder.close()
        data = recorder.export_dataset()
        self.assertEqual([r["source_order"] for r in data["records"]], [2, 3, 4])
        self.assertEqual(len(data["notes"]["idle_iterations"]), 1)
        self.assertGreater(data["records"][-1]["duration_ms"], data["records"][0]["duration_ms"] * 50)

    def test_lifecycle_restores_instance_methods_and_duplicate_attach_rejects(self):
        core, recorder, binding = fixture()
        with self.assertRaisesRegex(SimulationError, "already attached"):
            RuntimeStepRecorder(
                core,
                recorder.exp,
                "duplicate",
                allow_test_fixture=True,
                runtime_version=SUPPORTED_VERSION_PREFIX,
            )
        recorder.close()
        recorder.close()
        self.assertEqual(core.step_fn.__func__, Core.step)
        self.assertNotIn("schedule", vars(core.scheduler))
        self.assertFalse(hasattr(core, "_mlsys_runtime_cost_recorder"))

    def test_binding_identity_is_not_relabelled_as_the_core_cost_boundary(self):
        core, recorder, binding = fixture()
        self.addCleanup(recorder.close)
        bad = deepcopy(binding)
        bad["identity"]["runtime"]["timing_boundary"] = "engine_core_step_wall_v1"
        with self.assertRaisesRegex(SimulationError, "original stage boundary"):
            recorder.bind_runtime(bad)
        recorder.bind_runtime(binding)
        self.assertEqual(
            recorder.acquisition["runner_binding"]["identity"]["runtime"]["timing_boundary"],
            "step_includes_output_and_sampling_v1",
        )

    def test_fixed_preparation_is_independent_of_formal_concurrency(self):
        core, recorder, _ = fixture()
        self.addCleanup(recorder.close)

        class LLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompts, params, **kwargs):
                self.calls.append((len(prompts), len(prompts[0]["prompt_token_ids"])))
                return [NS(finished=True, outputs=[NS(token_ids=[1] * 16)]) for _ in prompts]

        llm = LLM()
        rows = run_fixed_warmup(llm, NS(max_tokens=16), prompt_token_id=1, recorder=recorder)
        self.assertEqual(len(rows), 20)
        self.assertEqual(llm.calls, [(b, n) for b in (1, 4, 8, 16) for n in (64, 128, 256, 512, 1024)])

    def test_affinity_is_applied_after_construction_and_verified_again(self):
        with (
            patch("os.sched_setaffinity", create=True) as setter,
            patch("os.sched_getaffinity", return_value={10}, create=True),
        ):
            self.assertEqual(apply_main_thread_affinity(10)["actual_main_thread_affinity"], [10])
            setter.assert_called_once_with(0, {10})
        with patch("os.sched_getaffinity", return_value={0, 12}, create=True):
            with self.assertRaises(SimulationError):
                verify_main_thread_affinity(10)
            self.assertEqual(verify_main_thread_affinity(None)["actual_main_thread_affinity"], [0, 12])

    def test_identity_extensions_are_opt_in_and_explicit_queue_settings_are_bound(self):
        config = NS(
            scheduler_config=NS(
                max_num_partial_prefills=2, max_long_partial_prefills=1, long_prefill_token_threshold=256
            )
        )
        with patch.dict("os.environ", {}, clear=True):
            original = _execution_config(config, Experiment(), False)
            self.assertNotIn("max_num_partial_prefills", original["scheduler_config"])
            self.assertNotIn("GPU_MAX_HW_QUEUES", original["environment"])
            with patch.dict(
                "os.environ", {"MLSYS_RUNTIME_COST_POLICY_JSON": json.dumps({"main_thread_cpu": None})}
            ):
                observed = _execution_config(config, Experiment(), False, runtime_cost_identity=True)
            self.assertEqual(observed["scheduler_config"]["max_num_partial_prefills"], 2)
            self.assertEqual(observed["runtime_cost_preparation"], {"main_thread_cpu": None})
            self.assertEqual(_execution_config(config, Experiment(), False), original)
            with patch.dict("os.environ", {"GPU_MAX_HW_QUEUES": "4"}):
                self.assertEqual(
                    _execution_config(config, Experiment(), False, runtime_cost_identity=True)["environment"]["GPU_MAX_HW_QUEUES"], "4"
                )

    def test_legacy_config_bytes_match_head_even_when_image_sets_hip_hsa_and_queue(self):
        config = NS(scheduler_config=NS(max_num_partial_prefills=2,
                    max_long_partial_prefills=1, long_prefill_token_threshold=256))
        def canonical(value):
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with patch.dict("os.environ", {}, clear=True):
            original = _execution_config(config, Experiment(), False)
            before = canonical(original)
            # SHA from the pre-runtime-cost HEAD implementation, evaluated on
            # exactly this namespace; that implementation ignores all3fields.
            self.assertEqual(_config_digest(original),
                             "1d3df1ede568f7b28ca9704a7953878a63cc86372df4a04fbc9a15fd407b0162")
            for environment in (
                    {"HSA_NO_SCRATCH_RECLAIM": "1", "HIP_FORCE_DEV_KERNARG": "1"},
                    {"GPU_MAX_HW_QUEUES": "4", "HSA_NO_SCRATCH_RECLAIM": "1", "HIP_FORCE_DEV_KERNARG": "1"}):
                with self.subTest(environment=environment), patch.dict("os.environ", environment):
                    self.assertEqual(canonical(_execution_config(config, Experiment(), False)), before)
                    runtime = _execution_config(config, Experiment(), False, runtime_cost_identity=True)
                    for name in ("GPU_MAX_HW_QUEUES", "HSA_NO_SCRATCH_RECLAIM", "HIP_FORCE_DEV_KERNARG"):
                        self.assertIn(name, runtime["environment"])
                        self.assertEqual(runtime["environment"][name], environment.get(name))

    def test_optional_api_observation_does_not_change_backend_output(self):
        core, recorder, _ = fixture()
        self.addCleanup(recorder.close)
        output = NS(request_id="r", finished=True, outputs=[NS(token_ids=[1, 2])])
        engine = NS(step=lambda: [output])
        expected = _VllmBackend(engine, None).step()
        self.assertEqual(_VllmBackend(engine, None, runtime_cost_recorder=recorder).step(), expected)
        self.assertEqual(len(recorder.api_returns), 1)

    def test_preparation_policy_does_not_claim_a_preconstruction_affinity(self):
        policy = preparation_policy(main_thread_cpu=None)
        self.assertIsNone(policy["main_thread_cpu"])
        self.assertNotIn("actual_main_thread_affinity", policy)
        self.assertNotIn("frequency", json.dumps(policy))

    def test_quiet_retains_the_same_progress_without_formal_stdout(self):
        state = {"completed": 3, "offered": 4}
        quiet, retained = _progress_observer({"cohort": "fixture"}, 4, 1, quiet=True, retain=True)
        printed, _ = _progress_observer({"cohort": "fixture"}, 4, 1, quiet=False, retain=False)
        stream = io.StringIO()
        with redirect_stdout(stream):
            quiet(state)
        self.assertEqual(stream.getvalue(), "")
        with redirect_stdout(stream):
            printed(state)
        self.assertEqual(json.loads(stream.getvalue()), retained[0])
        self.assertEqual(preparation_policy(quiet=True)["formal_progress"], "buffered_until_end")


if __name__ == "__main__":
    unittest.main()
