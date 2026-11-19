"""Opt-in source-pinned EngineCore wall capture, without GPU synchronization.

The observer's CPU bookkeeping is part of this instrumented Runtime domain.
It does not replace Scheduler token-ready or client API-feedback clocks.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, NoReturn

from .schema import Experiment, SimulationError
from .profiles import profile_identity
from .vllm_hooks import _config_digest, _execution_config, _runtime_version, _version


BOUNDARY = "engine_core_step_wall_v1"
PROTOCOL = "instrumented_engine_core_step_wall_v1"
_MISSING = object()


def _fail(message: str) -> NoReturn:
    raise SimulationError("INVALID_RUNTIME_COST_CAPTURE", message)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _fail(name + " must be an integer >= " + str(minimum))
    return value


def observe_cpu_host_facts() -> dict:
    """Read actual Linux CPU policy before construction; no affinity inference."""
    root = Path("/sys/devices/system/cpu")

    def read(path):
        try:
            return path.read_text().strip()
        except OSError:
            return None

    try:
        names = [
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ]
    except OSError:
        names = []
    return {
        "cpu_model": names[0] if names else None,
        "cpu_governors": {
            p.name: read(p / "scaling_governor") for p in sorted((root / "cpufreq").glob("policy*"))
        },
        "scaling_drivers": {
            p.name: read(p / "scaling_driver") for p in sorted((root / "cpufreq").glob("policy*"))
        },
        "cpu10_siblings": read(root / "cpu10/topology/thread_siblings_list"),
    }


def preparation_policy(*, main_thread_cpu: int | None = None, quiet: bool = False) -> dict:
    if main_thread_cpu is not None:
        _integer(main_thread_cpu, "main_thread_cpu")
    if type(quiet) is not bool:
        _fail("quiet must be boolean")
    return {
        "kind": "runtime_cost_preparation_v1",
        "observer_protocol": PROTOCOL,
        "observer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formal_progress": "buffered_until_end" if quiet else "synchronous_stdout",
        "main_thread_cpu": main_thread_cpu,
        "warmup_batches": [1, 4, 8, 16],
        "warmup_prompt_lengths": [64, 128, 256, 512, 1024],
        "warmup_output_tokens": 16,
        "latency_adaptive": False,
        "gpu_synchronization_added": False,
        "scheduler_capture_required": True,
        "cpu_host": observe_cpu_host_facts(),
    }


def apply_main_thread_affinity(cpu: int | None) -> dict:
    if cpu is not None:
        _integer(cpu, "main_thread_cpu")
        if not hasattr(os, "sched_setaffinity"):
            _fail("Explicit main-thread pinning requires Linux affinity support")
        getattr(os, "sched_setaffinity")(0, {cpu})
    return verify_main_thread_affinity(cpu)


def verify_main_thread_affinity(cpu: int | None) -> dict:
    actual = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpu is not None and actual != [cpu]:
        _fail("Main-thread affinity differs from the declared preparation policy")
    return {
        "requested_cpu": cpu,
        "actual_main_thread_affinity": actual,
        "verification_scope": "calling_thread_after_runtime_construction",
    }


def checkpoint_evidence(model: str | Path) -> dict:
    """Hash the actual local checkpoint bytes, outside every measured interval."""
    model = Path(model)
    if not model.is_dir():
        _fail("Runtime-cost capture requires the loaded local checkpoint directory")
    manifest: list[dict[str, Any]] = []
    for path in sorted(model.rglob("*")):
        if not path.is_file() or not (
            path.suffix in {".json", ".safetensors"} or path.name.endswith(".metadata")
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        manifest.append(
            {"file": str(path.relative_to(model)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    if not manifest or not any(row["file"].endswith(".safetensors") for row in manifest):
        _fail("A complete local safetensors checkpoint is required")
    return {"manifest": manifest, "manifest_sha256": _digest(manifest), "model_directory": str(model)}


def run_fixed_warmup(
    llm: Any, parameters: Any, *, prompt_token_id: int, recorder: "RuntimeStepRecorder"
) -> list[dict]:
    """Fixed source-declared preparation, independent of formal concurrency."""
    if getattr(parameters, "max_tokens", None) != 16:
        _fail("Fixed Runtime warmup requires exactly16 output tokens")
    if recorder.phase != "warmup":
        _fail("Preparation cannot be mixed into formal records")
    result = []
    for batch in (1, 4, 8, 16):
        for length in (64, 128, 256, 512, 1024):
            start = time.perf_counter_ns()
            answers = llm.generate(
                [{"prompt_token_ids": [prompt_token_id] * length} for _ in range(batch)],
                parameters,
                use_tqdm=False,
            )
            if len(answers) != batch or any(
                not row.finished or len(row.outputs) != 1 or len(row.outputs[0].token_ids) != 16
                for row in answers
            ):
                _fail("Fixed warmup did not produce every declared output")
            result.append(
                {
                    "batch_size": batch,
                    "input_tokens": length,
                    "output_tokens": 16,
                    "started_ns": start,
                    "finished_ns": time.perf_counter_ns(),
                }
            )
    recorder.acquisition["fixed_warmup_cohorts"] = deepcopy(result)
    return result


class RuntimeStepRecorder:
    """Reversible synchronous MRv1 core observer; injected objects are fixtures."""

    def __init__(
        self,
        core: Any,
        experiment: Experiment,
        session_id: str,
        *,
        policy: dict | None = None,
        runtime_version: str | None = None,
        clock_ns: Callable[[], int] | None = None,
        thread_clock_ns: Callable[[], int] | None = None,
        allow_test_fixture: bool = False,
        max_records: int = 100000,
    ):
        if type(allow_test_fixture) is not bool:
            _fail("allow_test_fixture must be boolean")
        if (clock_ns is not None or thread_clock_ns is not None) and not allow_test_fixture:
            _fail("Injected clocks require explicit test_fixture mode")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 160:
            _fail("A bounded process session_id is required")
        if getattr(core, "_mlsys_runtime_cost_recorder", None) is not None:
            _fail("A Runtime recorder is already attached")
        self.core, self.scheduler, self.exp = core, core.scheduler, experiment
        self.fixture, self.session_id = allow_test_fixture, session_id
        self.policy = deepcopy(policy if policy is not None else preparation_policy())
        if (
            self.policy.get("observer_protocol") != PROTOCOL
            or self.policy.get("latency_adaptive") is not False
        ):
            _fail("Explicit nonadaptive Runtime preparation protocol required")
        self._clock, self._thread_clock = (
            clock_ns or time.perf_counter_ns,
            thread_clock_ns or time.thread_time_ns,
        )
        self._max = _integer(max_records, "max_records", 1)
        self.phase = "warmup"
        self._closed, self._bound = False, False
        self._active: dict[str, Any] | None = None
        self._order, self._last_clock = 0, -1
        self.raw: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.api_returns: list[dict[str, Any]] = []
        self.acquisition: dict[str, Any] = {
            "protocol": PROTOCOL,
            "timing_boundary": BOUNDARY,
            "gpu_synchronization_added": False,
            "wall_includes": "schedule, execute/future consumption, sample, update/publish and active observers",
            "exclusive_runner_service": False,
            "api_return_boundary": "separately recorded when requested",
            "preparation_policy": deepcopy(self.policy),
            "source_kind": "test_fixture" if self.fixture else "measured",
        }
        self._runner: Any = None
        self._runner_capture: Any = None
        try:
            version = _version(runtime_version) if runtime_version is not None else _runtime_version()
            actual_version = _runtime_version() if not self.fixture else version
        except ImportError as error:
            raise SimulationError(
                "UNSUPPORTED_VLLM_ABI",
                "Pinned live vLLM is not installed; CPU injection requires explicit fixture mode",
            ) from error
        if not self.fixture:
            expected_policy = preparation_policy(
                main_thread_cpu=self.policy.get("main_thread_cpu"),
                quiet=self.policy.get("formal_progress") == "buffered_until_end",
            )
            if _digest(self.policy) != _digest(expected_policy):
                _fail("Actual host and fixed preparation/observer policy must match the configured domain")
            if version != actual_version:
                _fail("Declared and installed pinned vLLM versions differ")
            types = (
                (core, "vllm.v1.engine.core", "EngineCore"),
                (self.scheduler, "vllm.v1.core.sched.scheduler", "Scheduler"),
                (core.model_executor, "vllm.v1.executor.uniproc_executor", "UniProcExecutor"),
            )
            for obj, module, name in types:
                if type(obj) is not getattr(importlib.import_module(module), name):
                    _fail("Live capture requires the actual pinned " + name)
            runner = core.model_executor.driver_worker.worker.model_runner
            expected = importlib.import_module("vllm.v1.worker.gpu_model_runner").GPUModelRunner
            if type(runner) is not expected or getattr(runner, "use_async_scheduling", True):
                _fail("Live core capture requires synchronous MRv1")
            if (
                getattr(core.step_fn, "__func__", None) is not type(core).step
                or getattr(core, "batch_queue", None) is not None
                or getattr(core, "async_scheduling", False)
            ):
                _fail("Only the actual unqueued synchronous EngineCore.step boundary is supported")
            self._runner, self._runner_capture = runner, getattr(runner, "_mlsys_internal_capture", None)
            if (
                self._runner_capture is None
                or getattr(self.scheduler, "_mlsys_scheduler_capture", None) is None
            ):
                _fail("Live capture requires the same Runner binding and Scheduler observer protocol")
            actual_config = _execution_config(core.vllm_config, experiment, False, runtime_cost_identity=True)
            if actual_config.get("runtime_cost_preparation") != self.policy:
                _fail("Configured preparation policy must be bound before Runtime construction")
            self.acquisition["checkpoint"] = checkpoint_evidence(core.vllm_config.model_config.model)
            self.acquisition["core_execution_config"] = actual_config
        self.acquisition["runtime_version"] = version
        self._saved_step = vars(core).get("step_fn", _MISSING)
        self._saved_schedule = vars(self.scheduler).get("schedule", _MISSING)
        step, schedule = core.step_fn, self.scheduler.schedule

        @functools.wraps(schedule)
        def scheduled(*args, **kwargs):
            output = schedule(*args, **kwargs)
            try:
                if self._active is None or self._active.get("members") is not None:
                    _fail("Exactly one schedule call per core step is required")
                if getattr(output, "scheduled_spec_decode_tokens", {}) or getattr(
                    output, "scheduled_encoder_inputs", {}
                ):
                    _fail("Speculative or multimodal scheduling is unsupported")
                members: list[dict[str, Any]] = []
                for name, query in output.num_scheduled_tokens.items():
                    query = _integer(query, "scheduled query", 1)
                    request = self.scheduler.requests[name]
                    # This pinned Scheduler advances its cursor in _update_after_schedule.
                    context = _integer(request.num_computed_tokens, "advanced computed tokens") - query
                    if context < 0:
                        _fail("Scheduled cursor precedes its actual query")
                    prompt = _integer(request.num_prompt_tokens, "prompt tokens", 1)
                    role = "prefill" if context < prompt else "decode"
                    if role == "decode" and query != 1 or role == "prefill" and context + query > prompt:
                        _fail("Unit-query Decode and noncrossing Prefill required")
                    members.append(
                        {
                            "request_id": str(name),
                            "query_tokens": query,
                            "context_length": context,
                            "request_phase": role,
                            "prompt_tokens": prompt,
                        }
                    )
                if sum(m["query_tokens"] for m in members) != output.total_num_scheduled_tokens:
                    _fail("Scheduler token totals disagree")
                self._active["members"] = members
            except Exception as error:
                self._error(error)
            return output

        @functools.wraps(step)
        def observed(*args, **kwargs):
            self._order += 1
            row: dict[str, Any] = {
                "source_order": self._order,
                "capture_phase": self.phase,
                "members": None,
                "status": "FAILED",
                "record_id": f"{self.session_id}:core:{self._order}",
            }
            if self._active is not None:
                _fail("Concurrent/reentrant core steps are unsupported")
            self._active = row
            try:
                thread_start = _integer(self._thread_clock(), "thread clock")
                start = self._now()
            except Exception as error:
                self._active = None
                self._error(error)
                raise
            row["started_ns"] = start
            try:
                result = step(*args, **kwargs)
                row["status"] = "COMPLETED"
                return result
            except BaseException as error:
                row["error"] = {"code": getattr(error, "code", type(error).__name__), "message": str(error)}
                raise
            finally:
                try:
                    end = self._now()
                    thread_end = _integer(self._thread_clock(), "thread clock")
                    if thread_end < thread_start:
                        _fail("Thread clock moved backwards")
                    row.update(
                        finished_ns=end,
                        duration_ms=(end - start) / 1e6,
                        thread_cpu_ns=thread_end - thread_start,
                    )
                except Exception as error:
                    self._error(error)
                    row["clock_error"] = str(error)
                self._active = None
                if len(self.raw) < self._max:
                    self.raw.append(row)
                else:
                    self._error(
                        SimulationError(
                            "CAPTURE_LIMIT", "Runtime iteration limit reached; no trimming permitted"
                        )
                    )
                if row["status"] != "COMPLETED":
                    self._error(SimulationError("RUNTIME_STEP_FAILED", "A native core step failed"))

        try:
            core.step_fn, self.scheduler.schedule = observed, scheduled
            core._mlsys_runtime_cost_recorder = self
        except Exception:
            self.close()
            raise

    def _now(self):
        value = _integer(self._clock(), "wall clock")
        if value < self._last_clock:
            _fail("Core wall clock moved backwards")
        self._last_clock = value
        return value

    def _error(self, error):
        self.errors.append({"code": getattr(error, "code", type(error).__name__), "message": str(error)})

    def bind_runtime(self, binding: dict | None = None) -> None:
        """Use actual same-process Runner binding, retaining its original boundary."""
        actual = self._runner_capture.export_capture() if self._runner_capture is not None else None
        if not self.fixture:
            if binding is not None and binding != actual:
                _fail("Supplied binding differs from the actual same-process Runner capture")
            binding = actual
        if (
            not isinstance(binding, dict)
            or binding.get("capture_errors")
            or not binding.get("capture_closed")
        ):
            _fail("A completed separate Runner binding is required")
        expected_kind = "test_fixture" if self.fixture else "measured"
        if binding.get("source", {}).get("kind") != expected_kind:
            _fail("Runner binding source kind differs")
        metadata = binding["metadata"]
        snapshot, digest = metadata["runtime_execution_config"], metadata["runtime_config_sha256"]
        if _config_digest(snapshot) != digest:
            _fail("Runner execution fingerprint changed")
        if (
            self.exp.calibration.runtime_config_sha256
            and self.exp.calibration.runtime_config_sha256 != digest
        ):
            _fail("Actual Runtime execution config differs from experiment")
        stage_exp = replace(
            self.exp,
            calibration=replace(
                self.exp.calibration,
                backend="analytical",
                profile_path=None,
                analytical_costs=(),
                timing_boundary="step_includes_output_and_sampling_v1",
                runtime_config_sha256=digest,
            ),
        )
        if binding.get("identity") != profile_identity(stage_exp):
            _fail(
                "Actual loaded Runner model/device/runtime identity differs; binding retains its original stage boundary"
            )
        if not self.fixture and (
            not binding.get("records")
            or any(
                r.get("status") != "COMPLETED" or r.get("profile_eligible") is not True
                for r in binding["records"]
            )
        ):
            _fail("The initial real Runner binding is incomplete")
        if not self.fixture:
            base = {
                k: v
                for k, v in snapshot.items()
                if k not in {"actual_attention_backends", "torch_build", "sampling_policy"}
            }
            if base != self.acquisition["core_execution_config"]:
                _fail("Runner/Scheduler/core configured policies disagree")
            if snapshot["model_config"].get("enforce_eager") is not True:
                _fail("This initial live capture adapter supports the observed eager MRv1 domain only")
            for phase in ("prefill", "decode"):
                if getattr(self.exp.calibration, phase + "_execution_mode") != "eager" or snapshot.get(
                    "actual_attention_backends"
                ) != [getattr(self.exp.calibration, phase + "_attention_backend")]:
                    _fail("Declared phase/backend selectors differ from the actual eager Runner")
        self.exp = replace(self.exp, calibration=replace(self.exp.calibration, runtime_config_sha256=digest))
        self.acquisition["runner_binding"] = {
            "sha256": _digest(binding),
            "identity": binding["identity"],
            "metadata": deepcopy(metadata),
            "role": "loaded_model_runtime_binding_only_not_core_cost_samples",
        }
        self._bound = True

    def begin_formal(self) -> None:
        if not self._bound or self.phase != "warmup" or self._active is not None:
            _fail("Formal capture requires bound, drained preparation")
        if not self.fixture:
            cohorts = self.acquisition.get("fixed_warmup_cohorts", [])
            if [(r["batch_size"], r["input_tokens"], r["output_tokens"]) for r in cohorts] != [
                (b, n, 16) for b in (1, 4, 8, 16) for n in (64, 128, 256, 512, 1024)
            ]:
                _fail("The complete fixed warmup protocol must be observed before formal capture")
            self.acquisition["post_warmup_affinity"] = verify_main_thread_affinity(
                self.policy.get("main_thread_cpu")
            )
            if not self._runner_capture.export_capture().get("capture_closed"):
                _fail("Synchronized Runner warmup observer must close before formal capture")
            if self.scheduler.requests or self.scheduler.running or self.scheduler.waiting:
                _fail("Warmup requests must drain before formal capture")
        self.phase = "formal"
        self.acquisition["formal_started_ns"] = self._now()

    @contextmanager
    def api_step_boundary(self):
        start = self._now()
        try:
            yield
        finally:
            self.api_returns.append(
                {"started_ns": start, "finished_ns": self._now(), "source_order_through": self._order}
            )

    def close(self) -> None:
        if self._closed:
            return
        if self._active is not None:
            _fail("Cannot close during an active core step")
        if not self.fixture and self.phase == "formal":
            try:
                self.acquisition["post_formal_affinity"] = verify_main_thread_affinity(
                    self.policy.get("main_thread_cpu")
                )
            except Exception as error:
                self._error(error)
        for owner, name, saved in (
            (self.core, "step_fn", self._saved_step),
            (self.scheduler, "schedule", self._saved_schedule),
        ):
            if saved is _MISSING:
                vars(owner).pop(name, None)
            else:
                setattr(owner, name, saved)
        if getattr(self.core, "_mlsys_runtime_cost_recorder", None) is self:
            delattr(self.core, "_mlsys_runtime_cost_recorder")
        self._closed = True

    def export_capture(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "runtime_cost_raw_capture_v1",
            "session_id": self.session_id,
            "acquisition": deepcopy(self.acquisition),
            "iterations": deepcopy(self.raw),
            "api_returns": deepcopy(self.api_returns),
            "capture_errors": deepcopy(self.errors),
            "closed": self._closed,
        }

    def export_dataset(self) -> dict:
        from .runtime_cost import runtime_cost_domain

        if not self._closed or not self._bound or self.errors:
            _fail(
                "Only closed complete bound captures can become a Runtime-cost dataset; raw evidence remains available"
            )
        rows = []
        for raw in self.raw:
            if raw["capture_phase"] != "formal" or not raw["members"]:
                continue
            members = raw["members"]
            roles = [m["request_phase"] for m in members]
            phase = roles[0] if len(set(roles)) == 1 else "mixed"
            shape = {
                "phase": phase,
                "query_tokens": [m["query_tokens"] for m in members],
                "context_lengths": [m["context_length"] for m in members],
            }
            if phase == "mixed":
                shape["request_phases"] = roles
            rows.append(
                {
                    "record_id": raw["record_id"],
                    "session_id": self.session_id,
                    "group_id": "core-shape:" + _digest(shape),
                    "source_order": raw["source_order"],
                    **shape,
                    "stage_index": 0,
                    "duration_ms": raw["duration_ms"],
                    "workspace_peak_bytes": None,
                }
            )
        if not rows:
            _fail("No formal scheduled iterations were captured")
        core_exp = replace(
            self.exp,
            calibration=replace(
                self.exp.calibration,
                backend="runtime",
                timing_boundary=BOUNDARY,
                analytical_costs=(),
                profile_path=self.exp.calibration.profile_path or "runtime-cost-pending.json",
            ),
        )
        return {
            "schema_version": 1,
            "kind": "runtime_cost_dataset_v1",
            "dataset_id": self.session_id,
            "domain": runtime_cost_domain(core_exp),
            "source": {
                "kind": "test_fixture" if self.fixture else "measured",
                "reference": "Actual source-pinned instrumented EngineCore.step wall boundary",
            },
            "records": rows,
            "notes": {
                "acquisition": deepcopy(self.acquisition),
                "warmup_record_ids": [r["record_id"] for r in self.raw if r["capture_phase"] == "warmup"],
                "idle_iterations": [r["record_id"] for r in self.raw if not r["members"]],
                "raw_capture_sha256": _digest(self.export_capture()),
                "hardware_validated": False,
            },
        }
