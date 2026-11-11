"""Verify distribution contents and an installed wheel outside the checkout."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import venv
import zipfile


SMOKE_CODE = '''
from dataclasses import asdict, replace
from importlib.metadata import entry_points, metadata, version
import json
from pathlib import Path
import sys
import mlsys_sim
from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec,
                       MemoryConfig, RequestSpec, StrategySpec,
                       predict_experiment, simulate, profile_identity, comparison_contract)

assert Path(mlsys_sim.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert version("mlsys-simulator") == mlsys_sim.__version__
assert not metadata("mlsys-simulator").get_all("Requires-Dist")
assert "torch" not in sys.modules, "Optional GPU dependency was imported eagerly"
assert "vllm" not in sys.modules, "Optional serving dependency was imported eagerly"
from mlsys_sim.vllm_capacity_adapter import kv_pool_factory
assert callable(kv_pool_factory) and callable(mlsys_sim.validate_profile_repeats)
assert "torch" not in sys.modules and "vllm" not in sys.modules
assert all(callable(getattr(mlsys_sim, name)) for name in (
    "run_chapter1", "write_chapter1_report", "generate_workload", "run_load_sweep",
    "write_load_report", "simulate_dag", "run_dag_candidates", "write_dag_report",
    "validate_constraints", "evaluate_simulation_constraints", "evaluate_dag_constraints",
    "run_candidate_screening", "write_screening_report", "import_course_calibration",
    "write_course_calibration_report", "apply_course_calibration", "run_calibrated_chapter1",
    "validate_closed_loop_spec", "generate_request_stream", "closed_loop_policy_sha256",
    "predict_closed_loop", "validate_closed_loop", "collect_closed_loop"))
from mlsys_sim.load_validation_report import write_closed_loop_study_report
from mlsys_sim.vllm_sla_worker import formal_server_evidence
assert callable(write_closed_loop_study_report) and callable(formal_server_evidence)
assert "torch" not in sys.modules and "vllm" not in sys.modules
finite = {"schema_version": 1, "study_id": "installed-fixture", "cohort": "baseline",
          "concurrency_grid": [1, 2], "requests_per_run": 4, "replications": 1, "seed": 42,
          "think_ms": 0, "input_lengths": [2], "output_tokens": 2, "timeout_ms": 100, "max_run_ms": 1000}
finite_prediction = mlsys_sim.predict_closed_loop(Experiment(
    calibration=CalibrationSpec(backend="synthetic", prefill_ms=2, decode_ms=1)), finite)
assert all(row["result"]["counts"]["completed"] == 4 and row["resources_drained"]
           for row in finite_prediction["runs"])
assert mlsys_sim.generate_request_stream(finite)["request_stream_sha256"]
class FixtureClock:
    now = 1000000000
    def __call__(self): return self.now
class FeedbackBackend:
    source = {"kind": "test_fixture", "reference": "installed-wheel CPU feedback protocol"}
    def __init__(self, clock): self.clock, self.active = clock, {}
    def add_request(self, request):
        self.active[request["request_id"]] = 0
        return request["request_id"]
    def has_unfinished_requests(self): return bool(self.active)
    def abort_requests(self, names):
        for name in names: self.active.pop(name, None)
    def step(self):
        self.clock.now += 1000000
        rows = []
        for name in list(self.active):
            count = self.active[name] + 1
            self.active[name] = count
            rows.append({"native_request_id": name, "finished": count == 2, "output_tokens_actual": count})
            if count == 2: del self.active[name]
        return rows
feedback_clock = FixtureClock()
feedback = mlsys_sim.collect_closed_loop(FeedbackBackend(feedback_clock), finite, 2, clock_ns=feedback_clock)
assert feedback["completed_count"] == 4 and feedback["offered_count"] == 4
assert feedback["source"]["kind"] == "test_fixture" and feedback["timing_boundary"] == "runtime_API_feedback_v1"
plugins = [item for item in entry_points(group="vllm.general_plugins") if item.name == "mlsys_internal_capture"]
assert len(plugins) == 1
plugins[0].load()()
assert "torch" not in sys.modules and "vllm" not in sys.modules, "Dormant plugin activated without opt-in"
qwen_import = mlsys_sim.load_qwen3_model("qwen-inputs/qwen3-4b.hf.json", name="Qwen3-4B",
                                        revision="installed-qwen-fixture")
qwen = mlsys_sim.ModelSpec(**qwen_import["model"])
assert qwen.hidden_size == 2560 and qwen.query_projection_size == 4096
assert qwen.tie_word_embeddings and qwen.qk_norm and qwen.final_norm
assert qwen.architecture_config_sha256 == qwen_import["provenance"]["architecture_config_sha256"]
def fixture_transport(url, body, headers, timeout_s):
    if body is None:
        return [json.dumps({"version": "installed-fixture-vllm"}).encode()]
    packet = {"choices": [{"index": 0, "token_ids": [2, 3], "finish_reason": "length"}],
              "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    newline = chr(10)
    return [("data: " + json.dumps(packet) + newline * 2 + "data: [DONE]" + newline * 2).encode()]
client_fixture = mlsys_sim.collect_vllm(
    Experiment(workload=(RequestSpec("client-fixture", 0, 1, 2, prompt_token_ids=(1,)),)),
    {"schema_version": 1, "endpoint": "http://fixture.invalid", "model": "fixture-model",
     "source_reference": "Installed-wheel injected transport, no GPU or network"},
    transport=fixture_transport, allow_test_fixture=True)
assert client_fixture["source"]["kind"] == "test_fixture"
assert client_fixture["coalesced_delivery"] is True
assert client_fixture["observation"] is None
assert client_fixture["canonical_comparable"] is False
assert client_fixture["hardware_validated"] is False
mixed_model = mlsys_sim.AnalyticalModel(Experiment(
    calibration=CalibrationSpec(backend="synthetic", mixed_ms=3)))
assert mixed_model.estimate_step("mixed", [1, 4], [8, 0],
    request_phases=["decode", "prefill"]).duration_ms == 3
mixed_float = mlsys_sim.experiment_from_dict({"strategy": {"model": {
    "kv_bytes": 2.0, "activation_bytes": 4.0}}})
float_step = mlsys_sim.AnalyticalModel(mixed_float).estimate_step("decode", [1], [16])
assert float_step.breakdown["attention"]["flops_per_rank"] == 4 * 4 * 8 * 16 * 17
assert not any("dequant" in note.lower() for note in float_step.assumptions)
materialized = mlsys_sim.experiment_from_dict({"strategy": {"model": {
    "weight_bytes": 0.5, "weight_dequant": "materialized"}}})
fused = replace(materialized, strategy=replace(materialized.strategy,
    model=replace(materialized.strategy.model, weight_dequant="fused")))
materialized_memory = mlsys_sim.MemoryPlanner(materialized)
fused_memory = mlsys_sim.MemoryPlanner(fused)
assert (materialized_memory.check({"q": 4}, [4], "prefill").details["workspace_bytes"]
    == fused_memory.check({"q": 4}, [4], "prefill").details["workspace_bytes"] + 475136)
assert materialized_memory.check_kv({"q": 4}).details["workspace_bytes"] == 0
exp = Experiment(
    workload=tuple(RequestSpec(name, 0.0, 16, 24) for name in ("a", "b")),
    strategy=StrategySpec(max_batch_size=2, memory=MemoryConfig(kv_pool_bytes=24576)),
    calibration=CalibrationSpec(backend="synthetic", prefill_ms=5,
                                decode_ms=1, recompute_ms=5))
simulation = simulate(exp)
assert simulation.result["counts"]["completed"] == 2
assert simulation.result["preemption_count"] == 1
assert [row["output_tokens_actual"] for row in simulation.requests] == [24, 24]
Path("experiment.json").write_text(json.dumps(asdict(exp)))
pp = Experiment(workload=(RequestSpec("shape", 0.0, 16, 2),),
                infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
                strategy=StrategySpec(pipeline_parallel_size=2))
assert predict_experiment(pp)["online_metrics_reason"] == "SHAPE_ANALYSIS_HAS_NO_ARRIVALS_OR_QUEUEING"
Path("pp.json").write_text(json.dumps(asdict(pp)))
swap_exp = replace(exp, strategy=replace(exp.strategy, preemption="swap",
                    memory=replace(exp.strategy.memory, cpu_swap_bytes=65536)))
Path("swap.json").write_text(json.dumps(asdict(swap_exp)))
pp_swap = replace(swap_exp,
    infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
    strategy=replace(swap_exp.strategy, pipeline_parallel_size=2, max_batch_tokens=64,
                     pipeline_max_inflight_microbatches=2,
                     memory=replace(swap_exp.strategy.memory, kv_pool_bytes=12288)))
Path("pp-swap.json").write_text(json.dumps(asdict(pp_swap)))
cache_chunks = Experiment(
    workload=(RequestSpec("first", 0, 12, 1, prompt_token_ids=tuple(range(12))),
              RequestSpec("reuse", 20, 12, 1, prompt_token_ids=tuple(range(12)))),
    strategy=StrategySpec(prefix_cache=True, chunked_prefill=True, prefill_chunk_tokens=4,
                          max_batch_tokens=4, memory=MemoryConfig(kv_block_tokens=4)),
    calibration=CalibrationSpec(backend="synthetic", prefill_ms=5))
Path("prefix-chunk.json").write_text(json.dumps(asdict(cache_chunks)))
pp_cache_chunks = replace(cache_chunks,
    infra=InfraSpec(devices=(DeviceSpec("gpu0"), DeviceSpec("gpu1"))),
    strategy=replace(cache_chunks.strategy, pipeline_parallel_size=2, pipeline_p2p=False))
Path("pp-prefix-chunk.json").write_text(json.dumps(asdict(pp_cache_chunks)))
unsupported = asdict(pp)
unsupported["strategy"]["replica_routing"] = "unknown"
Path("unsupported.json").write_text(json.dumps(unsupported))
profile_base = Experiment(workload=(RequestSpec("profile-demo", 0, 16, 3),),
                          strategy=StrategySpec(scheduler="fifo"))
records = []
for name, phase, query, context, duration in [
    ("low", "prefill", 8, 0, 4), ("middle", "prefill", 16, 0, 6.2),
    ("high", "prefill", 24, 0, 8), ("decode-a", "decode", 1, 16, 1),
    ("decode-b", "decode", 1, 17, 1.2)]:
    records.append({"record_id": name, "group_id": name, "phase": phase,
                    "stage_index": 0, "query_tokens": [query], "context_lengths": [context],
                    "duration_ms": duration, "workspace_peak_bytes": None})
data = {"schema_version": 1, "dataset_id": "installed-wheel-fixture",
        "identity": profile_identity(profile_base),
        "source": {"kind": "test_fixture", "reference": "Hand calculations, not hardware."},
        "records": records}
data["notes"] = {"acquisition": {"session_id": "installed-training-fixture",
    "clock_domain": "cpu-fixture-monotonic", "capture_input_sha256": ["a" * 64]}}
Path("measurements.json").write_text(json.dumps(data))
repeated = json.loads(json.dumps(data))
repeated["dataset_id"] = "installed-independent-repeat-fixture"
repeated["notes"]["acquisition"].update(session_id="installed-repeat-fixture",
    capture_input_sha256=["b" * 64])
for row in repeated["records"]:
    row["record_id"] = "repeat-" + row["record_id"]
    row["group_id"] = "repeat-" + row["group_id"]
Path("repeats.json").write_text(json.dumps(repeated))
profile_config = asdict(profile_base)
profile_config["calibration"].update(backend="profile", profile_path="fit/profile.json",
                                       profile_version="smoke-v1", allow_test_fixture=True)
Path("profile-experiment.json").write_text(json.dumps(profile_config))
observation = {"schema_version": 1, "kind": "measured_run",
               "comparison_contract": comparison_contract(profile_base),
               "source": {"kind": "test_fixture", "reference": "Hand calculations, not hardware."},
               "metrics": {"ttft_p50_ms": 6.2, "tpot_p50_ms": 1.1,
                           "completion_rate": 1, "request_count": 1}}
Path("observation.json").write_text(json.dumps(observation))
communication = {"schema_version": 1, "kind": "communication_measurements",
                 "dataset_id": "installed-communication-fixture", "identity": profile_identity(pp),
                 "source": {"kind": "test_fixture", "reference": "Hand calculations, not RCCL measurements."},
                 "records": [{"record_id": str(size), "group_id": "p2p-two-ranks",
                              "primitive": "p2p", "ranks": 2, "message_bytes": size,
                              "duration_ms": .005 + size / 20e6} for size in [1000, 1000000]]}
Path("communication.json").write_text(json.dumps(communication))
cluster = Experiment(workload=(RequestSpec("p0", 0, 16, 1), RequestSpec("p1", 0, 16, 1)),
    infra=InfraSpec(devices=(DeviceSpec("replica0"), DeviceSpec("replica1"))),
    strategy=StrategySpec(replicas=2, replica_routing="round_robin"))
cluster_paths = []
for index in range(2):
    local = mlsys_sim.replica_experiment(cluster, index)
    local_records = []
    for name, phase, queries, contexts, duration in [
        ("single", "prefill", [16], [0], 2 + index),
        ("batch", "prefill", [16, 16], [0, 0], 3 + index),
        ("decode", "decode", [1, 1], [16, 16], 1 + index)]:
        local_records.append({"record_id": name, "group_id": name, "phase": phase,
            "stage_index": 0, "query_tokens": queries, "context_lengths": contexts,
            "duration_ms": duration, "workspace_peak_bytes": None})
    local_data = {"schema_version": 1, "dataset_id": f"installed-replica-{index}",
        "identity": profile_identity(local), "records": local_records,
        "source": {"kind": "test_fixture", "reference": "Local replica hand calculations only"}}
    local_profile = mlsys_sim.calibrate_dataset(local_data, profile_version="installed-replicas-v1",
                                               allow_test_fixture=True)
    path = f"replica-profile-{index}.json"
    Path(path).write_text(json.dumps(local_profile))
    cluster_paths.append(path)
cluster = replace(cluster, calibration=CalibrationSpec(backend="profile",
    replica_profile_paths=tuple(cluster_paths), profile_version="installed-replicas-v1", allow_test_fixture=True))
Path("cluster-profile.json").write_text(json.dumps(asdict(cluster)))
native = replace(pp, infra=InfraSpec(devices=tuple(DeviceSpec(f"native-{i}") for i in range(4))),
    strategy=replace(pp.strategy, tensor_parallel_size=2, pipeline_transport="vllm_native",
        max_batch_tokens=32, model=replace(pp.strategy.model, activation_dtype="float16")))
native_records = []
for stage in range(2):
    for phase, query, context in (("prefill", 16, 0), ("decode", 1, 16)):
        descriptor = mlsys_sim.native_pp_descriptor(native, query)
        assert descriptor["wire_bytes_per_rank"] * 2 == descriptor["resident_bytes_per_rank"]
        native_records.append({"record_id": f"{stage}-{phase}", "group_id": phase,
            "phase": phase, "stage_index": stage, "query_tokens": [query], "context_lengths": [context],
            "duration_ms": stage + 1, "workspace_peak_bytes": None,
            "pp_transport": {"outgoing" if stage == 0 else "incoming": descriptor}})
native_dataset = {"schema_version": 1, "dataset_id": "installed-native-pp-fixture",
    "identity": profile_identity(native), "records": native_records,
    "source": {"kind": "test_fixture", "reference": "Native PP hand geometry, not GPU/network timings"}}
native_profile = mlsys_sim.calibrate_dataset(native_dataset, profile_version="native-installed-v1",
                                            allow_test_fixture=True)
Path("native-profile.json").write_text(json.dumps(native_profile))
native = replace(native, calibration=replace(native.calibration, backend="profile",
    profile_path="native-profile.json", profile_version="native-installed-v1", allow_test_fixture=True))
Path("native-pp.json").write_text(json.dumps(asdict(native)))
print(json.dumps({"version": mlsys_sim.__version__, "wheel_api": "passed"}))
'''


def run(command: list[str], cwd: Path, env: dict[str, str], expected: int = 0) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if result.returncode != expected:
        raise RuntimeError(f"{command[0]} exited {result.returncode}, expected {expected}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def verify(dist: Path) -> dict:
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected one wheel and one source distribution in a clean dist directory.")
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "mlsys_sim/py.typed" in names, "Wheel is missing its typing marker"
        assert "mlsys_sim/__init__.py" in names
        assert "mlsys_sim/batching.py" in names
        assert "mlsys_sim/replicas.py" in names
        for module in ("hardware", "rocm_steps", "rccl_adapter", "runtime_adapter",
                       "hardware_validation", "release_adapter", "qwen3", "vllm_hooks", "vllm_capture", "pp_transport",
                       "pp_communication", "pp_benchmark", "pp_benchmark_worker", "workspace", "workspace_capture",
                       "capacity", "capacity_worker", "vllm_capacity_adapter"):
            assert f"mlsys_sim/{module}.py" in names
        for module in ("course_roofline", "workload", "load_sweep", "event_driver", "dag",
                       "constraints", "screening", "course_calibration", "load_validation",
                       "closed_loop_capture", "load_validation_report", "vllm_sla_worker"):
            assert f"mlsys_sim/{module}.py" in names
        assert not any(name.startswith(("tests/", "sample-runs/", "work/")) for name in names)
        assert archive.testzip() is None
    with tarfile.open(sdists[0]) as archive:
        source_names = {"/".join(member.name.split("/")[1:]) for member in archive.getmembers()}
        for required in ("pyproject.toml", "README.md", "README.zh-CN.md",
                         "examples/memory_pressure.json", "tests/test_public_api.py",
                         "examples/v1/step_measurements.fixture.json", "calibrate.py", "validate.py",
                         "examples/v1_2/prefix_cache.json", "docs/prefix-cache-and-chunking.md",
                         "examples/v1_3/mixed_batching.json", "examples/v1_3/pp_mixed_cache_swap.json",
                         "docs/mixed-batching.md",
                         "docs/multi-replica.md", "examples/v1_4/round_robin.json",
                         "examples/v1_4/replicas_pp_mixed_cache_swap.json",
                         "docs/rocm-collection.md", "docs/reference-data-audit.md",
                         "examples/v1_5/fixture-validation.plan.json", "examples/v1_5/runtime-trace.fixture.json",
                         "docs/qwen3-and-vllm-hooks.md", "examples/v1_6/qwen3-4b.hf.json",
                         "examples/v1_6/qwen3-r2-tp2-pp2.json", "examples/v1_6/hook-import.fixture.plan.json",
                         "docs/native-pp-capture.md", "tests/test_native_pp_capture.py",
                         "docs/native-pp-communication.md", "examples/v1_8/communication-train.fixture.json",
                         "examples/v1_8/primitive-capture.fixture.json",
                         "docs/workspace-and-capacity.md", "examples/v1_9/workspace-measurements.fixture.json",
                         "examples/v1_9/capacity-measurements.fixture.json",
                         "examples/v1_10/native-pp-collect.fixture.json",
                         "examples/v1_10/communication-train.fixture.json",
                         "examples/v1_10/communication-holdout.fixture.json",
                         "examples/v1_10/communication.profile.fixture.json",
                         "examples/v1_10/collector.plan.live-template.json",
                         "examples/v1_11/native-pp-collect.fixture.json",
                         "examples/v1_11/communication-train.fixture.json",
                         "examples/v1_11/communication-holdout.fixture.json",
                         "examples/v1_11/communication.profile.fixture.json",
                         "examples/v1_11/collector.plan.live-template.json",
                         "tests/test_stage_repeats.py", "tests/test_vllm_capacity_adapter.py",
                         "tests/test_rocm_workspace_domain.py",
                         "scripts/rocm_workspace_audit/README.md",
                         "scripts/rocm_workspace_audit/workspace_domain.py",
                         "scripts/rocm_workspace_audit/worker_workspace_capture.py",
                         "scripts/rocm_workspace_audit/workspace_stage_session.py",
                         "scripts/rocm_workspace_audit/workspace_stage_domain.py",
                         "scripts/rocm_workspace_audit/assemble_stage_workspace.py",
                         "scripts/rocm_workspace_audit/workspace_transport_audit.py",
                         "scripts/rocm_workspace_audit/synchronization_resource_audit.py",
                         "scripts/rocm_workspace_audit/hsa_signal_backing_capture.py",
                         "scripts/rocm_workspace_audit/hsa_signal_backing_probe.cpp",
                         "scripts/rocm_workspace_audit/hsa_signal_namespace_control.cpp",
                         "scripts/rocm_workspace_audit/kfd_event_probe.cpp",
                         "scripts/rocm_workspace_audit/kfd_event_audit.py",
                         "tests/test_workspace_repeats.py", "tests/test_workspace_repeat_cli.py",
                         "tests/test_quantized_workspace.py", "docs/quantization.md",
                         "tests/test_course_cli.py", "tests/test_course_roofline.py",
                         "tests/test_workload_generation.py", "tests/test_load_sweep.py",
                         "tests/test_event_driver.py", "tests/test_dag.py",
                         "examples/course/chapter1.plan.json", "examples/course/load-workload.spec.json",
                         "examples/course/load-poisson.plan.json", "examples/course/load-clients.plan.json",
                         "examples/course/load-queue.experiment.json", "examples/course/dag-fork-join.plan.json",
                         "examples/course/dag-candidates.plan.json", "docs/course-workflows.md",
                         "docs/course-chapter1.md", "docs/load-generation-and-sweep.md",
                         "docs/dag-and-model-pools.md",
                         "tests/test_constraints.py", "tests/test_screening.py",
                         "tests/test_course_calibration.py", "tests/test_course_constraints_cli.py",
                         "docs/joint-constraints.md", "docs/candidate-screening.md",
                         "docs/course-calibration.md", "docs/course-constraints-and-calibration.md",
                         "examples/course/course-calibration.fixture.json",
                         "tests/test_closed_loop_capture.py", "tests/test_load_validation.py",
                         "tests/test_load_validation_report.py", "tests/test_server_request_evidence.py",
                         "tests/test_vllm_sla_worker.py", "tests/test_course_sla_analysis.py",
                         "tests/test_course_sla_freeze.py", "docs/closed-loop-validation.md",
                         "scripts/analyze_course_sla_study.py", "scripts/freeze_course_sla_commitment.py",
                         "tests/test_course_sla_published.py",
                         "docs/course-sla-9070xt-20261111/README.md",
                         "docs/course-sla-9070xt-20261111/report.json",
                         "docs/course-sla-9070xt-20261111/summary.json",
                         "docs/course-sla-9070xt-20261111/cells.csv",
                         "docs/course-sla-9070xt-20261111/report.html",
                         "docs/course-sla-9070xt-20261111/baseline-latency.svg",
                         "docs/course-sla-9070xt-20261111/cohort-throughput.svg",
                         "docs/course-sla-9070xt-20261111/holdout-e2e.svg",
                         "examples/course/9070-sla/frozen-commitment/freeze-receipt.json",
                         "tests/test_native_pp_retention_budget.py",
                         "tests/test_rocm_workspace_worker.py",
                         "tests/test_rocm_workspace_stage_domain.py",
                         "tests/test_rocm_workspace_stage_assembly.py",
                         "tests/test_rocm_workspace_placement.py",
                         "tests/test_rocm_workspace_transport.py",
                         "tests/test_rocm_workspace_native_comm.py",
                         "tests/test_rocm_synchronization_resources.py",
                         "tests/test_rocm_hsa_signal_backing.py",
                         "tests/test_rocm_kfd_events.py",
                         "scripts/rocm_workspace_audit/native_probe.cpp",
                         "scripts/rocm_workspace_audit/torch_allocator_probe.cpp",
                         "scripts/rocm_workspace_audit/device_heap_controls.hip",
                         "docs/workspace-live-validation.md", "docs/capacity-live-validation.md",
                         "docs/multigpu-stage-validation.md",
                         "docs/online-policy-live-validation.md",
                         "docs/workspace-stage-validation.md",
                         "docs/pp2-synchronization-evidence.md",
                         "scripts/verify_distribution.py"):
            assert required in source_names, f"Source distribution missing {required}"
        assert not any(name.startswith(("sample-runs/", "work/")) for name in source_names)

    with tempfile.TemporaryDirectory(prefix="mlsys-installed-") as temporary:
        outside = Path(temporary)
        environment = outside / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        binaries = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        console = binaries / ("mlsys-sim.exe" if os.name == "nt" else "mlsys-sim")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("MLSYS_VLLM_CAPTURE_CONFIG", None)
        env.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        run([str(python), "-I", "-m", "pip", "install", "--no-deps", str(wheel)], outside, env)
        qwen_inputs = outside / "qwen-inputs"
        qwen_inputs.mkdir()
        for source in (Path(__file__).resolve().parents[1] / "examples/v1_6").glob("*.json"):
            (qwen_inputs / source.name).write_bytes(source.read_bytes())
        run([str(python), "-I", "-c", SMOKE_CODE], outside, env)
        run([str(python), "-I", "-m", "mlsys_sim.vllm_sla_worker", "--help"], outside, env)
        course_inputs = outside / "course-inputs"
        course_inputs.mkdir()
        for source in (Path(__file__).resolve().parents[1] / "examples/course").glob("*.json"):
            (course_inputs / source.name).write_bytes(source.read_bytes())
        run([str(console), "chapter1", "--plan", str(course_inputs / "chapter1.plan.json"),
             "--out", "course-chapter1"], outside, env)
        chapter = json.loads((outside / "course-chapter1/result.json").read_text())
        assert chapter["point_count"] == 18 and chapter["hardware_validated"] is False
        assert len(list((outside / "course-chapter1/artifacts").glob("*.svg"))) == 4
        run([str(console), "generate-workload", "--plan", str(course_inputs / "load-workload.spec.json"),
             "--out", "course-workload"], outside, env)
        generated = json.loads((outside / "course-workload/result.json").read_text())
        assert generated["requests"] and generated["provenance"]["seed"] == 42
        for axis in ("poisson", "clients"):
            path = course_inputs / f"load-{axis}.plan.json"
            plan = json.loads(path.read_text())
            plan["replications"], plan["min_completed_samples"] = 1, 1
            plan["workload"]["arrival"].update(duration_ms=1000, max_requests=256)
            plan["rate_grid_rps" if axis == "poisson" else "concurrency_grid"] = [25, 50] if axis == "poisson" else [1, 2]
            path.write_text(json.dumps(plan))
            target = f"course-{axis}"
            run([str(console), "sweep", "--experiment", str(course_inputs / "load-queue.experiment.json"),
                 "--plan", str(path), "--out", target], outside, env)
            sweep = json.loads((outside / target / "result.json").read_text())
            assert len(sweep["points"]) == 2 and sweep["hardware_validated"] is False
            assert (outside / target / "artifacts/report.html").is_file()
        run([str(console), "dag", "--plan", str(course_inputs / "dag-candidates.plan.json"),
             "--out", "course-dag"], outside, env)
        candidates = json.loads((outside / "course-dag/result.json").read_text())
        assert len(candidates["candidates"]) == len(candidates["ranking"]) == 3
        assert len({item["trace_sha256"] for item in candidates["candidates"]}) == 1
        assert all(len(item["nodes"]) == 52 and item["resources_drained"]
                   and item["status"] == "COMPLETED" for item in candidates["candidates"])
        assert (outside / "course-dag/artifacts/report.html").is_file()
        run([str(console), "import-course-calibration", "--input",
             str(course_inputs / "course-calibration.fixture.json"), "--source-kind", "test_fixture",
             "--chapter1-plan", str(course_inputs / "chapter1.plan.json"), "--out", "course-calibration"],
             outside, env)
        imported = json.loads((outside / "course-calibration/result.json").read_text())
        assert imported["calibration"]["coefficients"] == {"eta": 0.5, "beta": 0.8}
        assert imported["calibrated_chapter1"]["point_count"] == 18
        assert imported["hardware_validated"] is False
        assert (outside / "course-calibration/artifacts/calibration_patch.json").is_file()
        assert (outside / "course-calibration/artifacts/chapter1-calibrated/report.html").is_file()
        joint = {"schema_version": 1, "min_completed_samples": 1, "memory": {},
                 "latency": {"p99_e2e_ms": 100000},
                 "throughput": {"basis": "elapsed", "min_output_tokens_per_s": 0}}
        tiny = {"workload": [{"request_id": "r", "arrival_ms": 0, "input_tokens": 2, "output_tokens": 2}],
                "calibration": {"backend": "synthetic", "prefill_ms": 2, "decode_ms": 1}}
        dag_plan = json.loads((course_inputs / "dag-candidates.plan.json").read_text())
        dag_plan["pools"] = dag_plan.pop("candidates")[0]["pools"]
        load_plan = json.loads((course_inputs / "load-poisson.plan.json").read_text())
        load_plan["constraints"] = joint
        load_plan["rate_grid_rps"] = [10, 20]
        load_plan["workload"]["arrival"]["process"] = "periodic"
        load_plan["service_time"] = {"kind": "deterministic"}
        load_plan["slo"] = {"p99_e2e_ms": 100000}
        for kind, candidate, objective in (
            ("simulation", {"experiment": tiny}, "elapsed_ms"),
            ("sweep", {"experiment": json.loads((course_inputs / "load-queue.experiment.json").read_text()),
                       "load_plan": load_plan}, "worst_p99_e2e_ms"),
            ("dag", {"dag_plan": dag_plan}, "makespan_ms"),
        ):
            plan = {"schema_version": 1, "kind": kind, "source": "synthetic_teaching_trace",
                    "constraints": joint, "objective": {"metric": objective, "direction": "min"},
                    "candidates": [{"candidate_id": "installed", **candidate}]}
            path = course_inputs / f"installed-screen-{kind}.json"
            path.write_text(json.dumps(plan))
            target = f"course-screen-{kind}"
            run([str(console), "screen", "--plan", str(path), "--out", target], outside, env)
            selected = json.loads((outside / target / "result.json").read_text())
            assert [item["candidate_id"] for item in selected["ranking"]] == ["installed"]
            assert selected["hardware_validated"] is False
            assert (outside / target / "artifacts/replay.plan.json").is_file()
        primitive_inputs = outside / "primitive-inputs"
        primitive_inputs.mkdir()
        primitive_models = outside / "v1_6"
        primitive_models.mkdir()
        for source in qwen_inputs.glob("*.json"):
            (primitive_models / source.name).write_bytes(source.read_bytes())
        for source in (Path(__file__).resolve().parents[1] / "examples/v1_8").glob("*.json"):
            (primitive_inputs / source.name).write_bytes(source.read_bytes())
        run([str(console), "collect", "--kind", "pp-communication", "--experiment",
             str(primitive_inputs / "native-pp-collect.fixture.json"), "--plan",
             str(primitive_inputs / "collector.plan.fixture.json"), "--allow-test-fixture", "--out", "primitive-capture"],
             outside, env)
        run([str(console), "calibrate", "--native-pp-communication", "--dataset",
             str(primitive_inputs / "communication-train.fixture.json"), "--profile-version", "installed-comm-v1",
             "--allow-test-fixture", "--out", "primitive-fit"], outside, env)
        run([str(console), "validate", "--native-pp-communication", "--profile", "primitive-fit/profile.json",
             "--dataset", str(primitive_inputs / "communication-holdout.fixture.json"),
             "--allow-test-fixture", "--out", "primitive-holdout"], outside, env)
        for mode, filename in (("predict", "native-pp-predict.fixture.json"),
                               ("simulate", "native-pp-serving.fixture.json")):
            target = f"primitive-{mode}"
            run([str(console), mode, "--experiment", str(primitive_inputs / filename), "--out", target], outside, env)
            result = json.loads((outside / target / "result.json").read_text())
            manifest = json.loads((outside / target / "manifest.json").read_text())
            assert manifest["native_pp_communication_profiles"][0]["source_kind"] == "test_fixture"
            if mode == "simulate":
                assert result["counts"]["completed"] == 5
                assert result["swap"]["completed_transfers_by_direction"] == {"D2H": 1, "H2D": 1}
            else:
                assert result["communication"]["p2p_source"] == "test_fixture_native_pp_communication_profile"
        small_message_inputs = outside / "v1_10"
        small_message_inputs.mkdir()
        for source in (Path(__file__).resolve().parents[1] / "examples/v1_10").glob("*.json"):
            (small_message_inputs / source.name).write_bytes(source.read_bytes())
        run([str(console), "calibrate", "--native-pp-communication", "--native-pp-model", "size-lookup",
             "--dataset", str(small_message_inputs / "communication-train.fixture.json"),
             "--profile-version", "installed-small-message-v1", "--allow-test-fixture",
             "--out", "small-message-fit"], outside, env)
        lookup = json.loads((outside / "small-message-fit/profile.json").read_text())
        lookup_result = json.loads((outside / "small-message-fit/result.json").read_text())
        checked_lookup = json.loads((small_message_inputs / "communication.profile.fixture.json").read_text())
        assert lookup_result["profile_method"] == "piecewise_size_lookup_v1"
        assert lookup_result["hardware_validated"] is False
        assert lookup["source"]["kind"] == "test_fixture"
        assert lookup["benchmark"]["protocol"] == "native_pp_primitives_v2"
        assert lookup["groups"] == checked_lookup["groups"]
        assert all(knot["sample_count"] == 5 for group in lookup["groups"] for knot in group["knots"])
        assert [interval["supported"] for interval in lookup["groups"][0]["intervals"]] == [True, False, True]
        run([str(console), "validate", "--native-pp-communication", "--profile", "small-message-fit/profile.json",
             "--dataset", str(small_message_inputs / "communication-holdout.fixture.json"),
             "--allow-test-fixture", "--out", "small-message-holdout"], outside, env)
        lookup_holdout = json.loads((outside / "small-message-holdout/result.json").read_text())
        assert lookup_holdout["hardware_validated"] is False
        assert lookup_holdout["coverage"] == {"total_records": 3, "evaluated_records": 2,
                                              "rejected_records": 1, "fraction": 2 / 3}
        assert lookup_holdout["metrics"]["mae_ms"] < 1e-12
        assert [row["status"] for row in lookup_holdout["records"]] == ["EVALUATED", "OUT_OF_DOMAIN", "EVALUATED"]
        assert all(row["training_size_relation"] == "unseen" for row in lookup_holdout["records"])
        policy_inputs = outside / "v1_11"
        policy_inputs.mkdir()
        for source in (Path(__file__).resolve().parents[1] / "examples/v1_11").glob("*.json"):
            (policy_inputs / source.name).write_bytes(source.read_bytes())
        run([str(console), "calibrate", "--native-pp-communication", "--native-pp-model", "size-lookup",
             "--dataset", str(policy_inputs / "communication-train.fixture.json"),
             "--profile-version", "installed-simple-policy-v1", "--allow-test-fixture",
             "--out", "simple-policy-fit"], outside, env)
        policy_profile = json.loads((outside / "simple-policy-fit/profile.json").read_text())
        policy_environment = {"NCCL_P2P_LL_THRESHOLD": "0", "NCCL_P2P_LL128_THRESHOLD": "0"}
        assert policy_profile["benchmark"]["protocol"] == "native_pp_primitives_v3"
        assert policy_profile["benchmark"]["p2p_protocol_policy"] == "simple_only_thresholds_v1"
        assert policy_profile["benchmark"]["p2p_protocol_environment"] == policy_environment
        assert policy_profile["benchmark"]["stack"]["p2p_protocol_environment"] == policy_environment
        assert policy_profile["source"]["kind"] == "test_fixture"
        checked_policy = json.loads((policy_inputs / "communication.profile.fixture.json").read_text())
        assert policy_profile["groups"] == checked_policy["groups"]
        run([str(console), "validate", "--native-pp-communication", "--profile", "simple-policy-fit/profile.json",
             "--dataset", str(policy_inputs / "communication-holdout.fixture.json"),
             "--allow-test-fixture", "--out", "simple-policy-holdout"], outside, env)
        policy_holdout = json.loads((outside / "simple-policy-holdout/result.json").read_text())
        assert policy_holdout["hardware_validated"] is False
        assert policy_holdout["coverage"] == {"total_records": 2, "evaluated_records": 2,
                                             "rejected_records": 0, "fraction": 1}
        assert policy_holdout["metrics"]["mae_ms"] < 1e-12
        assert all(row["training_size_relation"] == "unseen" for row in policy_holdout["records"])
        workspace_inputs = outside / "v1_9"
        workspace_inputs.mkdir()
        communication_inputs = outside / "v1_8"
        communication_inputs.mkdir()
        for source in primitive_inputs.glob("*.json"):
            (communication_inputs / source.name).write_bytes(source.read_bytes())
        for source in (Path(__file__).resolve().parents[1] / "examples/v1_9").glob("*.json"):
            (workspace_inputs / source.name).write_bytes(source.read_bytes())
        run([str(console), "calibrate", "--workspace", "--dataset",
             str(workspace_inputs / "workspace-measurements.fixture.json"), "--profile-version", "installed-ws-v1",
             "--allow-test-fixture", "--out", "workspace-fit"], outside, env)
        training_workspace = json.loads((workspace_inputs / "workspace-measurements.fixture.json").read_text())
        training_workspace.setdefault("notes", {})["acquisition"] = {
            "session_id": "installed-workspace-training-fixture", "clock_domain": "cpu-fixture-monotonic",
            "capture_input_sha256": ["c" * 64]}
        (outside / "workspace-training.json").write_text(json.dumps(training_workspace))
        repeated_workspace = json.loads(json.dumps(training_workspace))
        repeated_workspace["dataset_id"] = "installed-workspace-repeat-fixture"
        repeated_workspace["notes"]["acquisition"].update(
            session_id="installed-workspace-independent-fixture", capture_input_sha256=["d" * 64])
        for row in repeated_workspace["records"]:
            row["record_id"] = "repeat-" + row["record_id"]
            row["group_id"] = "repeat-" + row["group_id"]
        (outside / "workspace-repeat.json").write_text(json.dumps(repeated_workspace))
        run([str(console), "calibrate", "--workspace", "--dataset", "workspace-training.json",
             "--profile-version", "installed-workspace-repeats-v1", "--allow-test-fixture",
             "--out", "workspace-repeat-fit"], outside, env)
        run([str(console), "validate", "--workspace-repeats", "--profile", "workspace-repeat-fit/profile.json",
             "--training-dataset", "workspace-repeat-fit/dataset.json", "--dataset", "workspace-repeat.json",
             "--allow-test-fixture", "--out", "workspace-repeat-validation"], outside, env)
        workspace_repeat_report = json.loads((outside / "workspace-repeat-validation/result.json").read_text())
        assert workspace_repeat_report["summary"]["coverage"] == 1
        assert workspace_repeat_report["summary"]["rank_coverage"] == 1
        assert workspace_repeat_report["summary"]["retained_shape_coverage"] == 1
        assert workspace_repeat_report["summary"]["underprediction_count"] == 0
        assert workspace_repeat_report["hardware_validated"] is False
        for mode in ("predict", "simulate"):
            target = f"workspace-{mode}"
            run([str(console), mode, "--experiment", str(workspace_inputs / "native-pp-workspace.fixture.json"),
                 "--out", target], outside, env)
            result = json.loads((outside / target / "result.json").read_text())
            manifest = json.loads((outside / target / "manifest.json").read_text())
            assert manifest["workspace_profiles"][0]["source_kind"] == "test_fixture"
            if mode == "simulate":
                assert result["counts"]["completed"] == 1
                assert all(row["workspace_source"] == "workspace_profile" for row in result["memory"])
        run([str(console), "collect", "--kind", "memory-capacity", "--experiment",
             str(workspace_inputs / "capacity-base.fixture.json"), "--plan",
             str(workspace_inputs / "capacity.plan.fixture.json"), "--allow-test-fixture", "--out", "capacity-capture"],
             outside, env)
        run([str(console), "validate", "--memory-capacity", "--dataset", "capacity-capture/dataset.json",
             "--allow-test-fixture", "--out", "capacity-validation"], outside, env)
        capacity_result = json.loads((outside / "capacity-validation/result.json").read_text())
        assert capacity_result["hardware_validated"] is False
        assert capacity_result["threshold_status"] == "BRACKETED"
        assert capacity_result["observed_brackets"] == [{"success_value": 4, "oom_value": 8,
                                                         "resolution": 4, "exact_threshold": None}]
        for mode in ("predict", "simulate"):
            target = f"native-pp-{mode}"
            run([str(console), mode, "--experiment", "native-pp.json", "--out", target], outside, env)
            native_result = json.loads((outside / target / "result.json").read_text())
            if mode == "simulate":
                assert native_result["counts"]["completed"] == 1
            else:
                assert native_result["communication"]["prefill_p2p_ms"] > 0
        run([str(console), "import-model", "--kind", "qwen3", "--config", str(qwen_inputs / "qwen3-4b.hf.json"),
             "--name", "Qwen3-4B", "--revision", "installed-qwen-fixture", "--out", "qwen-import"], outside, env)
        imported = json.loads((outside / "qwen-import/result.json").read_text())
        assert imported["dimensions"] == {"hidden_size": 2560, "query_projection_size": 4096,
                                            "kv_projection_size": 1024}
        run([str(console), "predict", "--experiment", str(qwen_inputs / "qwen3-4b-predict.json"),
             "--out", "qwen-predict"], outside, env)
        shape = json.loads((outside / "qwen-predict/result.json").read_text())
        assert shape["memory"][0]["weights_bytes"] == 8044936192
        run([str(console), "simulate", "--experiment", str(qwen_inputs / "qwen3-r2-tp2-pp2.json"),
             "--out", "qwen-replicas"], outside, env)
        served = json.loads((outside / "qwen-replicas/result.json").read_text())
        assert served["counts"]["completed"] == 10
        assert served["batching"]["mixed_batches_completed"] == 4
        assert served["prefix_cache"]["hit_tokens"] == 24
        assert served["swap"]["completed_transfers_by_direction"] == {"D2H": 2, "H2D": 2}
        assert served["swap"]["host_reserved_bytes"] == 0
        run([str(console), "collect", "--kind", "vllm-hook",
             "--experiment", str(qwen_inputs / "hook-experiment.fixture.json"),
             "--plan", str(qwen_inputs / "hook-import.fixture.plan.json"),
             "--out", "hook-fixture", "--allow-test-fixture"], outside, env)
        captured = json.loads((outside / "hook-fixture/result.json").read_text())
        assert captured["canonical_stage_ready"] is True
        assert captured["canonical_comparable"] is False
        assert captured["hardware_validated"] is False
        run([str(console), "calibrate", "--dataset", "hook-fixture/datasets/replica-0.json",
             "--profile-version", "installed-hook-v1", "--out", "hook-fit",
             "--allow-test-fixture"], outside, env)
        run([str(console), "collect", "--kind", "probe", "--out", "rocm-probe"], outside, env)
        probe = json.loads((outside / "rocm-probe/result.json").read_text())
        assert probe["hardware_available"] is False
        assert probe["hardware_validated"] is False
        hardware_inputs = outside / "hardware-inputs"
        hardware_inputs.mkdir()
        for source in (Path(__file__).resolve().parents[1] / "examples/v1_5").glob("*.json"):
            (hardware_inputs / source.name).write_bytes(source.read_bytes())
        run([str(console), "validate-hardware", "--plan", str(hardware_inputs / "fixture-validation.plan.json"),
             "--out", "hardware-fixture", "--allow-test-fixture"], outside, env)
        acceptance = json.loads((outside / "hardware-fixture/result.json").read_text())
        assert acceptance["verdict"] == "TEST_FIXTURE_ONLY"
        assert acceptance["criteria_passed"] is True
        assert acceptance["hardware_validated"] is False
        run([str(console), "validate-hardware", "--plan", str(hardware_inputs / "fixture-validation.plan.json"),
             "--out", "hardware-required-pass", "--allow-test-fixture", "--require-pass"], outside, env, expected=3)
        run([str(console), "collect", "--kind", "runtime-trace",
             "--experiment", str(hardware_inputs / "experiment.fixture.json"),
             "--raw", str(hardware_inputs / "runtime-trace.fixture.json"),
             "--out", "runtime-fixture", "--allow-test-fixture"], outside, env)
        trace = json.loads((outside / "runtime-fixture/result.json").read_text())
        observation = json.loads((outside / "runtime-fixture/observed.json").read_text())
        assert trace["canonical_comparable"] is True
        assert observation["source"]["kind"] == "test_fixture"
        assert trace["hardware_validated"] is False
        for example, completed in (("round_robin", 4), ("prefix_aware", 3),
                                   ("replicas_pp_mixed_cache_swap", 10)):
            source = Path(__file__).resolve().parents[1] / "examples" / "v1_4" / f"{example}.json"
            config = outside / f"{example}.json"
            config.write_bytes(source.read_bytes())
            run([str(console), "simulate", "--experiment", str(config), "--out", example], outside, env)
            replicated = json.loads((outside / example / "result.json").read_text())
            assert replicated["counts"]["completed"] == completed
            assert replicated["routing"]["assigned_requests"] == completed
            assert len(replicated["replicas"]) == 2
            if completed == 10:
                assert replicated["batching"]["mixed_batches_completed"] == 4
                assert replicated["prefix_cache"]["hit_tokens"] == 24
                assert replicated["prefix_cache"]["attached_block_references"] == 0
                assert replicated["swap"]["completed_transfers_by_direction"] == {"D2H": 2, "H2D": 2}
                assert replicated["swap"]["host_reserved_bytes"] == 0
        for mode, output in (("simulate", "cluster-profile-run"), ("predict", "cluster-profile-prediction")):
            run([str(console), mode, "--experiment", "cluster-profile.json", "--out", output], outside, env)
            report = json.loads((outside / output / "result.json").read_text())
            manifest = json.loads((outside / output / "manifest.json").read_text())
            assert report["calibration_source"] == "test_fixture_profile"
            assert len(report["profile_provenance"]["replicas"]) == 2
            assert len(manifest["replica_profiles"]) == 2
            assert all((outside / output / item["snapshot_file"]).is_file()
                       for item in manifest["replica_profiles"])
            if mode == "simulate":
                assert report["counts"]["completed"] == 2
                assert report["latency_ms"]["ttft"]["p50"] == 2.5
            else:
                assert len(report["replica_predictions"]) == 2
                assert report["replica_routing_applied"] is False
        for example, completed in (("mixed_batching", 2), ("pp_mixed_cache_swap", 5)):
            source = Path(__file__).resolve().parents[1] / "examples" / "v1_3" / f"{example}.json"
            config = outside / f"{example}.json"
            config.write_bytes(source.read_bytes())
            run([str(console), "simulate", "--experiment", str(config), "--out", example], outside, env)
            mixed = json.loads((outside / example / "result.json").read_text())
            assert mixed["counts"]["completed"] == completed
            assert mixed["batching"]["mixed_batches_completed"] > 0
            if completed == 5:
                assert mixed["prefix_cache"]["hit_tokens"] == 12
                assert mixed["prefix_cache"]["attached_block_references"] == 0
                assert mixed["swap"]["completed_transfers_by_direction"] == {"D2H": 1, "H2D": 1}
                assert mixed["swap"]["host_reserved_bytes"] == 0
                assert mixed["recomputed_tokens"] == 0
        simulated = json.loads(run([str(console), "simulate", "--experiment", "experiment.json",
                                   "--out", "simulation"], outside, env))
        assert simulated["status"] == "COMPLETED"
        prediction = json.loads(run([str(console), "predict", "--experiment", "pp.json",
                                    "--out", "prediction"], outside, env))
        assert prediction["status"] == "COMPLETED"
        online = json.loads(run([str(console), "simulate", "--experiment", "pp.json",
                                "--out", "pp-online"], outside, env))
        assert online["status"] == "COMPLETED"
        for config_name, output in [("swap.json", "cpu-swap"), ("pp-swap.json", "pp-swap")]:
            run([str(console), "simulate", "--experiment", config_name, "--out", output], outside, env)
            swapped = json.loads((outside / output / "result.json").read_text())
            assert swapped["counts"]["completed"] == 2
            assert swapped["recomputed_tokens"] == 0
            assert swapped["swap"]["completed_transfers_by_direction"]["D2H"] > 0
            assert swapped["swap"]["host_reserved_bytes"] == 0
        for config_name, output in [("prefix-chunk.json", "prefix-chunk"),
                               ("pp-prefix-chunk.json", "pp-prefix-chunk")]:
            run([str(console), "simulate", "--experiment", config_name, "--out", output], outside, env)
            cached = json.loads((outside / output / "result.json").read_text())
            assert cached["counts"]["completed"] == 2
            assert cached["prefill"]["completed_chunks"] == 4
            assert cached["prefill"]["reused_prefix_tokens"] == 8
            assert cached["prefill"]["computed_tokens"] == 16
            assert cached["prefix_cache"]["pinned_blocks"] == 0
            assert cached["prefix_cache"]["cached_blocks"] == 3
        failed = json.loads(run([str(console), "simulate", "--experiment", "unsupported.json",
                                "--out", "unsupported"], outside, env, expected=2))
        assert failed["error_code"] == "UNSUPPORTED_MECHANISM"
        run([str(console), "calibrate", "--dataset", "measurements.json", "--profile-version",
             "smoke-v1", "--out", "fit", "--allow-test-fixture"], outside, env)
        run([str(console), "validate", "--exact-repeats", "--training-dataset", "fit/dataset.json",
             "--profile", "fit/profile.json", "--dataset", "repeats.json", "--out", "exact-repeats",
             "--allow-test-fixture"], outside, env)
        repeated_report = json.loads((outside / "exact-repeats/result.json").read_text())
        assert repeated_report["summary"]["coverage"] == 1
        assert repeated_report["summary"]["mae_ms"] == 0
        assert repeated_report["unseen_shape_generalization"] is False
        assert repeated_report["hardware_validated"] is False
        run([str(console), "simulate", "--experiment", "profile-experiment.json",
             "--out", "profile-run"], outside, env)
        comparison = json.loads(run([str(console), "compare", "--predicted", "profile-run",
                                     "--observed", "observation.json", "--out", "comparison",
                                     "--allow-test-fixture"], outside, env))
        assert comparison["status"] == "COMPLETED"
        run([str(console), "validate", "--dataset", "measurements.json", "--profile-version",
             "heldout-v1", "--holdout-group", "middle", "--out", "heldout",
             "--allow-test-fixture"], outside, env)
        report = json.loads((outside / "heldout/result.json").read_text())
        assert report["summary"]["coverage"] == 1
        assert report["test_fixture_only"] is True
        assert report["hardware_validation"]["status"] == "not_asserted"
        run([str(console), "calibrate", "--communication", "--dataset", "communication.json",
             "--profile-version", "communication-v1", "--out", "communication-fit",
             "--allow-test-fixture"], outside, env)
        fit = json.loads((outside / "communication-fit/profile.json").read_text())
        assert fit["application"]["automatic_application"] is False
        assert 19.9 < fit["groups"][0]["infra_patch"]["link_bandwidth_gbps"] < 20.1
        run([str(python), "-I", "-m", "mlsys_sim", "--help"], outside, env)
    return {"wheel": wheel.name, "sdist": sdists[0].name,
            "distribution_contents": "passed", "installed_api_and_cli": "passed",
            "installed_pipeline_and_swap": "passed",
            "installed_prefix_cache_and_chunking": "passed",
            "installed_mixed_batches_pipeline_cache_and_swap": "passed",
            "installed_replica_routing_resources_and_profiles": "passed",
            "installed_rocm_collection_and_validation_fixture": "passed",
            "installed_qwen_geometry_dormant_plugin_and_capture_fixture": "passed",
            "installed_native_pp_exact_profile_and_transport": "passed",
            "installed_pp_primitives_collection_calibration_holdout_and_serving": "passed",
            "installed_native_pp_size_lookup_guarded_holdout": "passed",
            "installed_native_pp_simple_policy_lookup_holdout": "passed",
            "installed_workspace_calibration_and_capacity_brackets": "passed",
            "installed_workspace_exact_rank_repeats": "passed",
            "installed_exact_stage_repeats_and_lazy_capacity_adapter": "passed",
            "installed_course_chapter1_workload_load_and_dag": "passed",
            "installed_joint_constraints_screening_and_course_calibration": "passed",
            "installed_finite_closed_loop_and_lazy_sla_worker": "passed",
            "installed_calibration_comparison_holdout": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    args = parser.parse_args()
    print(json.dumps(verify(args.dist), sort_keys=True))


if __name__ == "__main__":
    main()
