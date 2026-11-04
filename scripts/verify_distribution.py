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
Path("measurements.json").write_text(json.dumps(data))
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
                       "hardware_validation", "release_adapter", "qwen3", "vllm_hooks", "vllm_capture"):
            assert f"mlsys_sim/{module}.py" in names
        assert not any(name.startswith(("tests/", "sample-runs/", "work/")) for name in names)
        assert archive.testzip() is None
    with tarfile.open(sdists[0]) as archive:
        names = {"/".join(member.name.split("/")[1:]) for member in archive.getmembers()}
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
                         "scripts/verify_distribution.py"):
            assert required in names, f"Source distribution missing {required}"
        assert not any(name.startswith(("sample-runs/", "work/")) for name in names)

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
        for config, output in [("swap.json", "cpu-swap"), ("pp-swap.json", "pp-swap")]:
            run([str(console), "simulate", "--experiment", config, "--out", output], outside, env)
            swapped = json.loads((outside / output / "result.json").read_text())
            assert swapped["counts"]["completed"] == 2
            assert swapped["recomputed_tokens"] == 0
            assert swapped["swap"]["completed_transfers_by_direction"]["D2H"] > 0
            assert swapped["swap"]["host_reserved_bytes"] == 0
        for config, output in [("prefix-chunk.json", "prefix-chunk"),
                               ("pp-prefix-chunk.json", "pp-prefix-chunk")]:
            run([str(console), "simulate", "--experiment", config, "--out", output], outside, env)
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
            "installed_calibration_comparison_holdout": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    args = parser.parse_args()
    print(json.dumps(verify(args.dist), sort_keys=True))


if __name__ == "__main__":
    main()
