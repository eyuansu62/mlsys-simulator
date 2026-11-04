"""CPU fake-backend evidence for synchronized ROCm callable stage collection."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlsys_sim import (DeviceSpec, Experiment, InfraSpec, ModelSpec, SimulationError,
                       StrategySpec, calibrate_dataset)
from mlsys_sim.rocm_steps import collect_steps


class FakeCUDA:
    def __init__(self, log, count):
        self.log = log
        self.count = count
        self.after = False
        self.available = True
        self.bad_peak = False

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def get_device_properties(self, index):
        return SimpleNamespace(name=f"AMD Test GPU {index}", total_memory=128 * 1024 * 1024)

    def synchronize(self, index):
        self.log.append(("sync", index))

    def reset_peak_memory_stats(self, index):
        self.log.append(("reset_peak", index))

    def memory_allocated(self, index):
        self.log.append(("allocated", index))
        return (4000 if self.after else 1000) + index

    def memory_reserved(self, index):
        self.log.append(("reserved", index))
        return (9000 if self.after else 2000) + index

    def max_memory_allocated(self, index):
        self.log.append(("peak_allocated", index))
        return float("inf") if self.bad_peak else 12000 + index

    def max_memory_reserved(self, index):
        self.log.append(("peak_reserved", index))
        return 16000 + index


class FakeAdapter:
    def __init__(self, cuda, log, *, attribution="none", workspace=None, fail_run=None, fail_close=False):
        self.cuda, self.log = cuda, log
        self.attribution, self.workspace = attribution, workspace
        self.fail_run, self.fail_close = fail_run, fail_close
        self.run_count = 0
        self.closed = 0
        self.shapes = []

    def metadata(self):
        self.log.append(("metadata",))
        return {"runtime_name": "fixture-runtime", "runtime_version": "fixture-runtime-v1",
                "timing_boundary": "step_includes_output_and_sampling_v1",
                "includes_local_collectives": True, "workspace_attribution": self.attribution}

    def prepare(self, sample):
        self.log.append(("prepare", sample["sample_id"]))
        self.shapes.append(deepcopy(sample))
        self.cuda.after = False

    def run_step(self, sample):
        self.run_count += 1
        self.log.append(("run", sample["sample_id"]))
        self.cuda.after = True
        if self.run_count == self.fail_run:
            raise RuntimeError("HIP out of memory (fixture)")

    def workspace_peak_bytes(self, sample):
        self.log.append(("workspace", sample["sample_id"]))
        return self.workspace

    def close(self):
        self.closed += 1
        self.log.append(("close",))
        if self.fail_close:
            raise RuntimeError("Runtime close failed (fixture)")


class FakeClock:
    def __init__(self, values, log):
        self.values, self.log = iter(values), log

    def __call__(self):
        value = next(self.values)
        self.log.append(("clock", value))
        return value


class RocmStepTests(unittest.TestCase):
    def environment(self, *, tp=1, pp=1):
        count = tp * pp
        log = []
        cuda = FakeCUDA(log, count)
        torch = SimpleNamespace(__version__="fixture-torch-v1", version=SimpleNamespace(hip="fixture-hip-v1"), cuda=cuda)
        inventory = {"schema_version": 1, "kind": "rocm_inventory", "status": "READY",
                     "source": {"kind": "test_fixture", "reference": "Injected hardware fixture"},
                     "versions": {"rocm_version": "fixture-rocm-v1", "driver_version": "fixture-driver-v1",
                                  "rccl_version": "fixture-rccl-v1"},
                     "torch": {"version": torch.__version__, "hip_version": torch.version.hip,
                               "cuda_available": True},
                     "devices": [{"index": i, "name": f"AMD Test GPU {i}",
                                  "memory_bytes": 128 * 1024 * 1024, "architecture": "fixture-gfx"}
                                 for i in range(count)], "issues": []}
        exp = Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(count))),
                         strategy=StrategySpec(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                             model=ModelSpec(name="dense-fixture", revision="fixture-model-v1",
                                             weight_dtype="float16", kv_dtype="float16", activation_dtype="float16")))
        return exp, inventory, torch, cuda, log

    def plan(self, *, stage=0, iterations=1, warmup=0, mixed=False):
        shape = {"sample_id": "sample", "group_id": "configuration", "phase": "prefill",
                 "stage_index": stage, "query_tokens": [4], "context_lengths": [0]}
        if mixed:
            shape.update(phase="mixed", query_tokens=[1, 4], context_lengths=[8, 4],
                         request_phases=["decode", "prefill"])
        return {"schema_version": 1, "dataset_id": "collection-fixture", "source_reference": "CPU timing fixture",
                "samples": [shape], "iterations": iterations, "warmup_iterations": warmup}

    def collect(self, exp, inventory, torch, adapter, plan, values, log, **kwargs):
        return collect_steps(exp, plan, adapter, inventory, torch_module=torch,
                             clock_ns=FakeClock(values, log), allow_test_fixture=True, **kwargs)

    def test_host_timing_brackets_all_participating_devices_and_excludes_prepare(self):
        exp, inventory, torch, cuda, log = self.environment(tp=2, pp=2)
        adapter = FakeAdapter(cuda, log)
        result = self.collect(exp, inventory, torch, adapter, self.plan(stage=1), [1_000_000, 4_000_000], log)
        record = result["dataset"]["records"][0]
        self.assertEqual(record["duration_ms"], 3)
        first_clock = log.index(("clock", 1_000_000))
        second_clock = log.index(("clock", 4_000_000))
        run = log.index(("run", "sample"))
        self.assertLess(log.index(("prepare", "sample")), first_clock)
        self.assertEqual(log[first_clock:second_clock + 1],
                         [("clock", 1_000_000), ("run", "sample"), ("sync", 2), ("sync", 3), ("clock", 4_000_000)])
        self.assertLess(log.index(("reset_peak", 2)), first_clock)
        self.assertLess(log.index(("reset_peak", 3)), first_clock)
        self.assertTrue(all(index in (2, 3) for operation, *rest in log
                            if operation == "sync" for index in rest))
        self.assertLess(first_clock, run)
        self.assertEqual(result["raw_records"][0]["device_indices"], [2, 3])
        self.assertEqual(adapter.closed, 1)

    def test_allocator_peaks_remain_raw_evidence_without_workspace_attribution(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = FakeAdapter(cuda, log, workspace=999999)
        result = self.collect(exp, inventory, torch, adapter, self.plan(), [0, 2_000_000], log)
        self.assertIsNone(result["dataset"]["records"][0]["workspace_peak_bytes"])
        raw = result["raw_records"][0]
        self.assertEqual(raw["allocator_baseline"][0]["allocated_bytes"], 1000)
        self.assertEqual(raw["allocator_after"][0]["peak_allocated_bytes"], 12000)
        self.assertEqual(raw["allocator_after"][0]["peak_reserved_bytes"], 16000)
        self.assertEqual(raw["workspace_source"], "not_attributed")
        self.assertNotIn(("workspace", "sample"), log)

    def test_runtime_workspace_mapping_uses_the_maximum_participant_bound(self):
        exp, inventory, torch, cuda, log = self.environment(tp=2, pp=2)
        adapter = FakeAdapter(cuda, log, attribution="runtime", workspace={2: 10, 3: 30})
        result = self.collect(exp, inventory, torch, adapter, self.plan(stage=1), [0, 1_000_000], log)
        self.assertEqual(result["dataset"]["records"][0]["workspace_peak_bytes"], 30)
        self.assertEqual(result["raw_records"][0]["workspace_source"], "runtime_reported_rank_max")

    def test_unknown_runtime_workspace_is_null_and_invalid_mapping_is_rejected(self):
        for workspace in (None, {0: 0, 1: 10}, -1, True, float("inf")):
            exp, inventory, torch, cuda, log = self.environment()
            adapter = FakeAdapter(cuda, log, attribution="runtime", workspace=workspace)
            with self.subTest(workspace=workspace):
                if workspace is None:
                    result = self.collect(exp, inventory, torch, adapter, self.plan(), [0, 1_000_000], log)
                    self.assertIsNone(result["dataset"]["records"][0]["workspace_peak_bytes"])
                else:
                    with self.assertRaises(SimulationError) as raised:
                        self.collect(exp, inventory, torch, adapter, self.plan(), [0, 1_000_000], log)
                    self.assertEqual(raised.exception.code, "INVALID_ALLOCATOR_SAMPLE")
                    self.assertNotIn("dataset", raised.exception.artifacts)

    def test_injected_dependencies_cannot_upgrade_inventory_to_measured(self):
        exp, inventory, torch, cuda, log = self.environment()
        inventory["source"]["kind"] = "measured"
        before = deepcopy(inventory)
        adapter = FakeAdapter(cuda, log)
        result = self.collect(exp, inventory, torch, adapter, self.plan(), [0, 1_000_000], log)
        self.assertEqual(inventory, before)
        self.assertEqual(result["source"]["kind"], "test_fixture")
        self.assertEqual(result["inventory"]["source"]["kind"], "test_fixture")
        self.assertFalse(result["hardware_validated"])
        adapter = FakeAdapter(cuda, log)
        with self.assertRaises(SimulationError) as raised:
            collect_steps(exp, self.plan(), adapter, inventory, torch_module=torch,
                          clock_ns=FakeClock([0, 1_000_000], log))
        self.assertEqual(raised.exception.code, "TEST_FIXTURE_NOT_ALLOWED")
        self.assertEqual(adapter.run_count, 0)

    def test_warmup_is_not_a_training_record_and_repeats_preserve_group_and_shape(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = FakeAdapter(cuda, log)
        result = self.collect(exp, inventory, torch, adapter, self.plan(iterations=2, warmup=2),
                              [1_000_000, 4_000_000, 10_000_000, 13_000_000], log)
        self.assertEqual(adapter.run_count, 4)
        self.assertEqual(len(result["raw_records"]), 2)
        self.assertEqual([record["group_id"] for record in result["dataset"]["records"]],
                         ["configuration", "configuration"])
        profile = calibrate_dataset(result["dataset"], profile_version="collected-fixture", allow_test_fixture=True)
        self.assertEqual(profile["samples"][0]["sample_count"], 2)
        self.assertEqual(profile["samples"][0]["duration_ms"], 3)
        self.assertEqual(profile["source"]["kind"], "test_fixture")
        self.assertEqual(result["dataset"]["notes"]["collection_provenance"], result["provenance"])
        self.assertTrue(all(len(digest) == 64 for digest in result["provenance"].values()))
        self.assertEqual(len([item for item in log if item[0] == "reset_peak"]), 2)

    def test_mixed_contexts_and_explicit_member_phases_survive_collection(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = FakeAdapter(cuda, log)
        result = self.collect(exp, inventory, torch, adapter, self.plan(mixed=True), [0, 1_000_000], log)
        record = result["dataset"]["records"][0]
        self.assertEqual(record["request_phases"], ["decode", "prefill"])
        self.assertEqual(record["query_tokens"], [1, 4])
        self.assertEqual(record["context_lengths"], [8, 4])
        self.assertEqual(adapter.shapes[0]["context_lengths"], [8, 4])

    def test_callback_oom_preserves_partial_evidence_and_does_not_publish_dataset(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = FakeAdapter(cuda, log, fail_run=2)
        with self.assertRaises(SimulationError) as raised:
            self.collect(exp, inventory, torch, adapter, self.plan(iterations=2), [0, 1_000_000, 2_000_000], log)
        self.assertEqual(raised.exception.code, "COLLECTION_FAILED")
        evidence = raised.exception.artifacts
        self.assertNotIn("dataset", evidence)
        self.assertEqual([row["status"] for row in evidence["raw_records"]], ["COMPLETED", "FAILED"])
        self.assertEqual(evidence["raw_records"][0]["duration_ms"], 1)
        self.assertIn("HIP out of memory", evidence["raw_records"][1]["error"])
        self.assertEqual(adapter.closed, 1)

    def test_bad_clocks_and_allocator_counters_fail_with_raw_diagnostics(self):
        for values in ([0, 0], [2, 1], [0, float("nan")], [0, True], [0, float("inf")],
                       [2, 3, 1, 2]):
            exp, inventory, torch, cuda, log = self.environment()
            adapter = FakeAdapter(cuda, log)
            with self.subTest(values=values), self.assertRaises(SimulationError) as raised:
                self.collect(exp, inventory, torch, adapter, self.plan(iterations=2 if len(values) == 4 else 1),
                              values, log)
            self.assertEqual(raised.exception.code, "INVALID_TIMING_SAMPLE")
            self.assertEqual(adapter.closed, 1)
        exp, inventory, torch, cuda, log = self.environment()
        cuda.bad_peak = True
        with self.assertRaises(SimulationError) as raised:
            self.collect(exp, inventory, torch, FakeAdapter(cuda, log), self.plan(), [0, 1_000_000], log)
        self.assertEqual(raised.exception.code, "INVALID_ALLOCATOR_SAMPLE")

    def test_current_hip_devices_must_match_inventory_and_non_hip_is_rejected(self):
        for variation in ("name", "memory", "torch", "no-hip", "unavailable"):
            exp, inventory, torch, cuda, log = self.environment()
            if variation == "name":
                inventory["devices"][0]["name"] = "Different GPU"
            elif variation == "memory":
                inventory["devices"][0]["memory_bytes"] += 1
            elif variation == "torch":
                inventory["torch"]["version"] = "Different torch"
            elif variation == "no-hip":
                torch.version.hip = None
            else:
                cuda.available = False
            adapter = FakeAdapter(cuda, log)
            with self.subTest(variation=variation), self.assertRaises(SimulationError) as raised:
                self.collect(exp, inventory, torch, adapter, self.plan(), [0, 1_000_000], log)
            expected = "HARDWARE_UNAVAILABLE" if variation in ("no-hip", "unavailable") else "HARDWARE_IDENTITY_MISMATCH"
            self.assertEqual(raised.exception.code, expected)
            self.assertEqual(adapter.run_count, 0)

    def test_missing_pytorch_is_a_clear_hardware_error_without_cpu_fallback(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = FakeAdapter(cuda, log)
        with patch("mlsys_sim.rocm_steps.importlib.import_module", side_effect=ImportError("no torch")):
            with self.assertRaises(SimulationError) as raised:
                collect_steps(exp, self.plan(), adapter, inventory, allow_test_fixture=True)
        self.assertEqual(raised.exception.code, "HARDWARE_UNAVAILABLE")
        self.assertEqual(adapter.run_count, 0)
        self.assertEqual(adapter.closed, 1)

    def test_invalid_plans_and_runtime_contracts_fail_before_execution(self):
        invalid = []
        for field, value in (("schema_version", True), ("iterations", 0), ("iterations", True),
                             ("warmup_iterations", -1), ("samples", []), ("unknown", 1)):
            plan = self.plan()
            plan[field] = value
            invalid.append(plan)
        plan = self.plan()
        plan["samples"][0]["stage_index"] = 1
        invalid.append(plan)
        plan = self.plan(mixed=True)
        plan["samples"][0]["request_phases"] = ["decode", "decode"]
        invalid.append(plan)
        plan = self.plan()
        plan["samples"] *= 2
        invalid.append(plan)
        for plan in invalid:
            exp, inventory, torch, cuda, log = self.environment()
            adapter = FakeAdapter(cuda, log)
            with self.subTest(plan=plan), self.assertRaises(SimulationError) as raised:
                self.collect(exp, inventory, torch, adapter, plan, [0, 1_000_000], log)
            self.assertEqual(raised.exception.code, "INVALID_COLLECTION_PLAN")
            self.assertEqual(adapter.run_count, 0)
        for field, value in (("timing_boundary", "kernel_only"), ("includes_local_collectives", False),
                             ("workspace_attribution", "peak-minus-baseline")):
            exp, inventory, torch, cuda, log = self.environment()
            adapter = FakeAdapter(cuda, log)
            metadata = adapter.metadata()
            metadata[field] = value
            adapter.metadata = lambda: metadata
            with self.subTest(field=field), self.assertRaises(SimulationError) as raised:
                self.collect(exp, inventory, torch, adapter, self.plan(), [0, 1_000_000], log)
            self.assertEqual(raised.exception.code, "INVALID_COLLECTION_PLAN")

    def test_close_failure_keeps_the_capture_as_failed_and_close_runs_once(self):
        exp, inventory, torch, cuda, log = self.environment()
        adapter = FakeAdapter(cuda, log, fail_close=True)
        with self.assertRaises(SimulationError) as raised:
            self.collect(exp, inventory, torch, adapter, self.plan(), [0, 1_000_000], log)
        self.assertEqual(raised.exception.code, "COLLECTION_FAILED")
        self.assertIn("close_error", raised.exception.artifacts)
        self.assertNotIn("dataset", raised.exception.artifacts)
        self.assertEqual(adapter.closed, 1)

    def test_cluster_collection_requires_explicit_local_replica_slice(self):
        exp, inventory, torch, cuda, log = self.environment()
        exp = replace(exp, infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
                      strategy=replace(exp.strategy, replicas=2))
        adapter = FakeAdapter(cuda, log)
        with self.assertRaises(SimulationError) as raised:
            self.collect(exp, inventory, torch, adapter, self.plan(), [0, 1_000_000], log)
        self.assertEqual(raised.exception.code, "INVALID_COLLECTION_PLAN")
        self.assertEqual(adapter.run_count, 0)

    def test_declared_measured_tp_requires_actual_runtime_rccl_binding(self):
        for evidence in ({}, {"rccl_runtime_version": "fixture-rccl-v1"},
                         {"rccl_library_sha256": "a" * 64},
                         {"rccl_runtime_version": "unknown", "rccl_library_sha256": "a" * 64}):
            exp, inventory, torch, cuda, log = self.environment(tp=2)
            inventory["source"]["kind"] = "measured"
            inventory["rccl_library"] = {"sha256": "a" * 64}
            adapter = FakeAdapter(cuda, log)
            metadata = {**adapter.metadata(), **evidence}
            adapter.metadata = lambda: metadata
            # Mock the import, not the public injection parameters, to exercise
            # the live-branch gate without creating a measured capture.
            with self.subTest(evidence=evidence), patch(
                    "mlsys_sim.rocm_steps.importlib.import_module", return_value=torch):
                with self.assertRaises(SimulationError) as raised:
                    collect_steps(exp, self.plan(), adapter, inventory)
            self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")
            self.assertEqual(adapter.run_count, 0)
            self.assertEqual(raised.exception.artifacts["runtime_metadata"], metadata)

    def test_runtime_rccl_version_or_library_mismatch_is_rejected_before_stage(self):
        for version, digest in (("different-runtime-rccl", "a" * 64), ("fixture-rccl-v1", "b" * 64)):
            exp, inventory, torch, cuda, log = self.environment(tp=2)
            inventory["source"]["kind"] = "measured"
            inventory["rccl_library"] = {"sha256": "a" * 64}
            adapter = FakeAdapter(cuda, log)
            metadata = {**adapter.metadata(), "rccl_runtime_version": version,
                        "rccl_library_sha256": digest, "rccl_binding_reference": "Loaded library test declaration"}
            adapter.metadata = lambda: metadata
            with self.subTest(version=version, digest=digest), patch(
                    "mlsys_sim.rocm_steps.importlib.import_module", return_value=torch):
                with self.assertRaises(SimulationError) as raised:
                    collect_steps(exp, self.plan(), adapter, inventory)
            self.assertEqual(raised.exception.code, "HARDWARE_IDENTITY_MISMATCH")
            self.assertEqual(adapter.run_count, 0)
            self.assertEqual(adapter.closed, 1)

    def test_matching_runtime_rccl_declaration_is_retained_in_provenance(self):
        exp, inventory, torch, cuda, log = self.environment(tp=2)
        inventory["source"]["kind"] = "measured"
        inventory["rccl_library"] = {"sha256": "a" * 64}
        adapter = FakeAdapter(cuda, log)
        metadata = {**adapter.metadata(), "rccl_runtime_version": "fixture-rccl-v1",
                    "rccl_library_sha256": "A" * 64, "rccl_binding_reference": "Loaded librccl path and symbol evidence"}
        adapter.metadata = lambda: metadata
        original = deepcopy(metadata)
        # This is a mocked live branch for contract testing, never a GPU claim.
        with patch("mlsys_sim.rocm_steps.importlib.import_module", return_value=torch):
            result = collect_steps(exp, self.plan(), adapter, inventory)
        self.assertEqual(metadata, original)
        expected = {**metadata, "rccl_library_sha256": "a" * 64}
        self.assertEqual(result["runtime_metadata"], expected)
        self.assertEqual(result["dataset"]["notes"]["runtime_metadata"], expected)
        self.assertIn("runtime_metadata_sha256", result["dataset"]["notes"]["collection_provenance"])
        self.assertEqual(result["dataset"]["identity"]["runtime"]["rccl_version"], "fixture-rccl-v1")
        self.assertFalse(result["hardware_validated"])

    def test_querying_a_library_without_a_hash_does_not_prove_runtime_binding(self):
        exp, inventory, torch, cuda, log = self.environment(tp=2)
        inventory["source"]["kind"] = "measured"
        adapter = FakeAdapter(cuda, log)
        metadata = {**adapter.metadata(), "rccl_runtime_version": "fixture-rccl-v1", "rccl_library_sha256": "a" * 64}
        adapter.metadata = lambda: metadata
        with patch("mlsys_sim.rocm_steps.importlib.import_module", return_value=torch):
            with self.assertRaises(SimulationError) as raised:
                collect_steps(exp, self.plan(), adapter, inventory)
        self.assertEqual(raised.exception.code, "HARDWARE_METADATA_INCOMPLETE")
        self.assertEqual(adapter.run_count, 0)


if __name__ == "__main__":
    unittest.main()
