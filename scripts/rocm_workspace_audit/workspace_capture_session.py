"""Actual pinned TP1 prefill acquisition; audit decides completeness separately."""

from pathlib import Path
from dataclasses import replace, asdict
import argparse
import ctypes
import gzip
import hashlib
import json
import os
import uuid
from typing import Any

from workspace_domain import (
    declared_domain_issues,
    observed_domain_issues,
    plain,
    require_domain,
    bound_config_sha256,
)


def dump(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def gz(path, value):
    with gzip.open(path, "xt") as f:
        json.dump(value, f, allow_nan=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--cohort", choices=["train", "repeat"], required=True)
    p.add_argument("--iterations", type=int, default=3)
    a = p.parse_args()
    assert 1 <= a.iterations <= 5
    root = Path(a.out)
    root.mkdir()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ.pop("MLSYS_VLLM_CAPTURE_CONFIG", None)
    from mlsys_sim import load_experiment

    e = load_experiment(a.experiment)
    require_domain(declared_domain_issues(asdict(e)))
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from mlsys_sim.vllm_hooks import _execution_config, _shape_model_matches, _loaded_rccl
    from mlsys_sim.hardware import probe_rocm

    properties = torch.cuda.get_device_properties(0)
    observed_domain: dict[str, Any] = {
        "schema_version": 1,
        "runtime_name": vllm.__name__,
        "runtime_version": str(vllm.__version__),
        "torch_version": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "hip_build_version": str(torch.version.hip),
        "visible_device_count": torch.cuda.device_count(),
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "device": {
            "index": 0,
            "name": str(properties.name),
            "memory_bytes": int(properties.total_memory),
            "architecture": getattr(properties, "gcnArchName", None),
        },
    }
    require_domain(observed_domain_issues(asdict(e), observed_domain, require_runner=False))
    probe = ctypes.CDLL("/app/work/native_probe.so")
    probe.mlsys_probe_begin.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    probe.mlsys_probe_end.argtypes = [ctypes.c_uint64]
    probe.mlsys_probe_write.argtypes = [ctypes.c_char_p]
    torch.cuda.memory._record_memory_history(
        enabled="all", context="all", stacks="all", max_entries=100000, clear_history=True
    )
    torch.cuda.synchronize(0)
    assert probe.mlsys_probe_begin(5001, b"explicit_ATen_LT_workspace_initialization") == 0
    native = ctypes.CDLL(str(Path(torch.__file__).parent / "lib/libtorch_hip.so"))
    getptr = getattr(native, "_ZN2at4cuda22getCUDABlasLtWorkspaceEv")
    getptr.argtypes = []
    getptr.restype = ctypes.c_void_p
    getsize = getattr(native, "_ZN2at4cuda26getCUDABlasLtWorkspaceSizeEv")
    getsize.argtypes = []
    getsize.restype = ctypes.c_size_t
    constructor = {"pointer": getptr(), "bytes": getsize(), "thread": os.getpid()}
    torch.cuda.synchronize(0)
    assert probe.mlsys_probe_end(5001) == 0
    llm = None
    try:
        llm = LLM(
            "/app/models/Qwen3-1.7B",
            dtype="bfloat16",
            enforce_eager=True,
            distributed_executor_backend="uni",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            async_scheduling=False,
            kv_cache_memory_bytes=268435456,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=True,
            generation_config="vllm",
            seed=0,
        )
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        for n in [8, 16, 32, 64, 64, 16, 8, 32]:
            llm.generate(
                [{"prompt_token_ids": [100] * n}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False),
                use_tqdm=False,
            )
        torch.cuda.synchronize(0)
        actual_parallel = runner.parallel_config
        config_parallel = runner.vllm_config.parallel_config
        observed_domain["runner"] = {
            "class_name": type(runner).__module__ + "." + type(runner).__name__,
            "tensor_parallel_size": getattr(actual_parallel, "tensor_parallel_size", None),
            "pipeline_parallel_size": getattr(actual_parallel, "pipeline_parallel_size", None),
            "config_tensor_parallel_size": getattr(config_parallel, "tensor_parallel_size", None),
            "config_pipeline_parallel_size": getattr(config_parallel, "pipeline_parallel_size", None),
            "data_parallel_size": getattr(actual_parallel, "data_parallel_size", None),
            "use_async_scheduling": getattr(runner, "use_async_scheduling", None),
        }
        rccl = _loaded_rccl()
        inventory = probe_rocm(rccl_library=rccl["path"])
        observed_domain["hardware"] = {
            "source_kind": inventory.get("source", {}).get("kind"),
            "status": inventory.get("status"),
            **inventory.get("versions", {}),
            "rccl_library_sha256": rccl["library_sha256"],
        }
        config = _execution_config(runner.vllm_config, e, observed_domain["runner"]["use_async_scheduling"])
        # These fields were historically derived from the declared Experiment.
        # New observations take them from the real Runner and verify both sources.
        config["tensor_parallel_size"] = observed_domain["runner"]["tensor_parallel_size"]
        config["pipeline_parallel_size"] = observed_domain["runner"]["pipeline_parallel_size"]
        config["async_scheduling"] = observed_domain["runner"]["use_async_scheduling"]
        config = plain(config)
        observed_domain = plain(observed_domain)
        require_domain(observed_domain_issues(asdict(e), observed_domain, config))
        digest = bound_config_sha256(asdict(e), config)
        e = replace(e, calibration=replace(e.calibration, runtime_config_sha256=digest))
        model_binding = _shape_model_matches(runner, e, _phase="prepared")
        bridge = ctypes.CDLL("/app/work/torch_allocator_probe.so")
        bridge.mlsys_torch_probe_install.argtypes = [ctypes.c_uint64]
        bridge.mlsys_torch_probe_begin.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        bridge.mlsys_torch_probe_end.argtypes = [ctypes.c_uint64]
        bridge.mlsys_torch_probe_write.argtypes = [ctypes.c_char_p]
        assert bridge.mlsys_torch_probe_install(65536) == 0

        def census():
            owners = []
            seen = set()

            def tensor(t, path, cat):
                if t.device.type != "cuda":
                    return
                s = t.untyped_storage()
                owners.append(
                    {
                        "path": path,
                        "category": cat,
                        "pointer": s.data_ptr(),
                        "bytes": s.nbytes(),
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "element_size": t.element_size(),
                        "tensor_offset": t.storage_offset(),
                        "stride": list(t.stride()),
                    }
                )

            def walk(x, path, cat, depth=0):
                if isinstance(x, torch.Tensor):
                    tensor(x, path, cat)
                    return
                if depth > 7 or id(x) in seen:
                    return
                seen.add(id(x))
                if isinstance(x, dict):
                    for k, v in x.items():
                        walk(v, path + "[" + str(k) + "]", cat, depth + 1)
                elif isinstance(x, (list, tuple)):
                    for i, v in enumerate(x):
                        walk(v, path + "[" + str(i) + "]", cat, depth + 1)
                elif (
                    not isinstance(x, torch.nn.Module)
                    and (x is runner or type(x).__module__.startswith("vllm."))
                    and hasattr(x, "__dict__")
                ):
                    for k, v in vars(x).items():
                        if k not in ["model", "kv_caches", "_mlsys_internal_capture", "vllm_config"]:
                            walk(v, path + "." + k, cat, depth + 1)

            for name, t in runner.model.named_parameters():
                tensor(t, "model.parameter:" + name, "weights")
            for name, t in runner.model.named_buffers():
                tensor(t, "model.buffer:" + name, "runtime")
            walk(runner.kv_caches, "runner.kv_caches", "kv_pool")
            walk(runner, "runner", "runtime")
            before = torch.cuda.memory._snapshot()
            pointer = getptr()
            size = getsize()
            after = torch.cuda.memory._snapshot()
            delta = after["device_traces"][0][len(before["device_traces"][0]) :]
            assert all(v["action"] == "snapshot" for v in delta)
            assert pointer == constructor["pointer"] and size == constructor["bytes"]
            owners.append(
                {
                    "path": "ATen.getCUDABlasLtWorkspace",
                    "category": "runtime",
                    "pointer": pointer,
                    "bytes": size,
                    "binding": "verified_exported_getter_no_allocation",
                    "element_size": 1,
                }
            )
            return owners

        def compact(snapshot):
            value = dict(snapshot)
            value["segments"] = [
                {**seg, "blocks": [{k: v for k, v in b.items() if k != "frames"} for b in seg["blocks"]]}
                for seg in snapshot["segments"]
            ]
            value["device_traces"] = [
                [{k: v for k, v in ev.items() if k != "frames"} for ev in events]
                for events in snapshot["device_traces"]
            ]
            return value

        def count_gpu(x, seen=None):
            seen = set() if seen is None else seen
            if id(x) in seen:
                return 0
            seen.add(id(x))
            if isinstance(x, torch.Tensor):
                return int(x.device.type == "cuda")
            if isinstance(x, dict):
                return sum(count_gpu(v, seen) for v in x.values())
            if isinstance(x, (list, tuple)):
                return sum(count_gpu(v, seen) for v in x)
            if type(x).__module__.startswith("vllm.") and hasattr(x, "__dict__"):
                return sum(count_gpu(v, seen) for v in vars(x).values())
            return 0

        current = [0]
        witness = []
        original = runner.sample_tokens

        def sample(*args, **kwargs):
            value = original(*args, **kwargs)
            if current[0]:
                witness.append(
                    {
                        "epoch_id": current[0],
                        "type": type(value).__module__ + "." + type(value).__name__,
                        "gpu_tensor_count": count_gpu(value),
                        "execute_state_is_none": runner.execute_model_state is None,
                        "prev_sampled_is_none": runner.input_batch.prev_sampled_token_ids is None,
                    }
                )
            return value

        runner.sample_tokens = sample
        session = uuid.uuid4().hex
        records = []
        for i, n in enumerate([16, 32] * a.iterations, 1):
            eid = 10000 + i
            owners_before = census()
            before = torch.cuda.memory._snapshot()
            current[0] = eid
            assert bridge.mlsys_torch_probe_begin(eid, ("runtime_step_transient:" + str(n)).encode()) == 0
            assert probe.mlsys_probe_begin(eid, ("prefill-" + str(n)).encode()) == 0
            outputs = llm.generate(
                [{"prompt_token_ids": [100] * n}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False),
                use_tqdm=False,
            )
            torch.cuda.synchronize(0)
            assert bridge.mlsys_torch_probe_end(eid) == 0
            assert probe.mlsys_probe_end(eid) == 0
            current[0] = 0
            after = torch.cuda.memory._snapshot()
            owners_after = census()
            assert len(outputs) == 1 and outputs[0].finished and len(outputs[0].outputs[0].token_ids) == 1
            row = {
                "record_id": session + "/" + a.cohort + "/q" + str(n) + "/i" + str(i),
                "group_id": session + "/" + a.cohort + "/q" + str(n),
                "epoch_id": eid,
                "phase": "prefill",
                "stage_index": 0,
                "query_tokens": [n],
                "context_lengths": [0],
                "before": compact(before),
                "after": compact(after),
                "owners_before": owners_before,
                "owners_after": owners_after,
            }
            gz(root / ("epoch-" + str(eid) + ".json.gz"), row)
            records.append(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ["before", "after", "owners_before", "owners_after"]
                }
            )
        assert bridge.mlsys_torch_probe_write(str(root / "torch-native-records.json").encode()) == 0
        assert probe.mlsys_probe_write(str(root / "native-checkpoint.json").encode()) == 0
        assert probe.mlsys_probe_stop() == 0
        dump(
            root / "session.json",
            {
                "status": "ACQUISITION_COMPLETED_AUDIT_REQUIRED",
                "complete_workspace": False,
                "session_id": session,
                "cohort": a.cohort,
                "fresh_pid": os.getpid(),
                "clock_domain": "linux-boot:" + Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "observed_runtime_version": str(vllm.__version__),
                "observed_runtime_domain": observed_domain,
                "torch_version": torch.__version__,
                "torch_git": torch.version.git_version,
                "experiment": asdict(e),
                "runtime_config": config,
                "model_binding": model_binding,
                "constructor": constructor,
                "records": records,
                "return_witness": witness,
                "history_max_entries": 100000,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
