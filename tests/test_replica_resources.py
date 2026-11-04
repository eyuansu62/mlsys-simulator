"""Replica placement, dynamic cache identity and independent host ownership."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (CalibrationSpec, DeviceSpec, Experiment, HostSwapManager, InfraSpec,
                       MemoryConfig, MemoryPlanner, RequestSpec, SimulationError, StrategySpec,
                       experiment_from_dict, load_experiment, replica_experiment)


class ReplicaResourceTests(unittest.TestCase):
    def test_replica_placement_and_local_profile_selection_are_explicit(self):
        exp = Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"g{i}") for i in range(8))),
            strategy=StrategySpec(replicas=2, tensor_parallel_size=2, pipeline_parallel_size=2),
            calibration=CalibrationSpec(backend="profile", replica_profile_paths=("a.json", "b.json")))
        local = replica_experiment(exp, 1)
        self.assertEqual([device.id for device in local.infra.devices], ["g4", "g5", "g6", "g7"])
        self.assertEqual(local.strategy.replicas, 1)
        self.assertEqual(local.strategy.tensor_parallel_size, 2)
        self.assertEqual(local.strategy.pipeline_parallel_size, 2)
        self.assertEqual(local.calibration.profile_path, "b.json")
        self.assertEqual(local.calibration.replica_profile_paths, ())
        for index in (-1, 2, True):
            with self.subTest(index=index), self.assertRaises(SimulationError):
                replica_experiment(exp, index)
        with self.assertRaises(SimulationError):
            MemoryPlanner(exp)

    def test_invalid_replica_controls_and_profile_counts_are_rejected(self):
        devices = {"devices": [{"id": "a"}, {"id": "b"}]}
        invalid = [{"strategy": {"replicas": 0}}, {"strategy": {"replicas": True}},
                   {"strategy": {"replicas": 2}},
                   {"strategy": {"replicas": 2}, "infra": {"devices": [*devices["devices"], {"id": "c"}]}},
                   {"strategy": {"replica_routing": 1}},
                   {"strategy": {"replicas": 2}, "infra": devices,
                    "calibration": {"backend": "profile", "profile_path": "a.json"}},
                   {"strategy": {"replicas": 2}, "infra": devices,
                    "calibration": {"backend": "profile", "replica_profile_paths": ["a.json"]}},
                   {"strategy": {"replicas": 2}, "infra": devices,
                    "calibration": {"backend": "profile", "replica_profile_paths": ["a.json", " "]}},
                   {"calibration": {"backend": "profile", "profile_path": "a.json",
                                     "replica_profile_paths": ["b.json"]}},
                   {"calibration": {"replica_profile_paths": ["a.json"]}}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(SimulationError):
                experiment_from_dict(config)
        invalid_parent = Experiment(strategy=StrategySpec(replicas=2))
        with self.assertRaises(SimulationError):
            replica_experiment(invalid_parent, 0)

    def test_profile_paths_resolve_from_the_referenced_calibration_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            section = root / "sections"
            section.mkdir()
            (section / "calibration.json").write_text(json.dumps({"backend": "profile",
                "replica_profile_paths": ["profiles/a.json", "../b.json"]}))
            config = root / "experiment.json"
            config.write_text(json.dumps({"strategy": {"replicas": 2},
                "infra": {"devices": [{"id": "a"}, {"id": "b"}]},
                "calibration": "sections/calibration.json"}))
            exp = load_experiment(config)
            self.assertEqual(exp.calibration.replica_profile_paths,
                             (str((section / "profiles/a.json").resolve()), str((root / "b.json").resolve())))

    def cached(self):
        seed = RequestSpec("seed", 0, 8, 1, prompt_token_ids=tuple(range(8)), cache_namespace="tenant")
        exp = Experiment(workload=(seed,), strategy=StrategySpec(prefix_cache=True,
                            memory=MemoryConfig(kv_block_tokens=4)))
        planner = MemoryPlanner(exp)
        self.assertTrue(planner.reserve({"seed": 8}, [8], "prefill").ok)
        self.assertEqual(planner.publish_prefix("seed", 8), 2)
        planner.release("seed")
        return planner

    def test_routing_probe_does_not_pin_register_count_or_touch_lru(self):
        planner = self.cached()
        future = RequestSpec("future", 10, 12, 1, prompt_token_ids=tuple(range(12)), cache_namespace="tenant")
        before = deepcopy((planner.snapshot(), planner.prefix_snapshot(), planner.allocations,
                           planner._prefix_specs, planner._prefix_request_keys))
        for _ in range(3):
            self.assertEqual(planner.peek_prefix_tokens(future), 8)
            self.assertEqual(planner.peek_prefix_tokens(replace(future, cache_namespace="other")), 0)
            self.assertEqual(planner.peek_prefix_tokens(replace(future, prompt_token_ids=())), 0)
        self.assertEqual((planner.snapshot(), planner.prefix_snapshot(), planner.allocations,
                          planner._prefix_specs, planner._prefix_request_keys), before)
        planner.register_request(future)
        self.assertEqual(planner.lookup_prefix("future"), 8)
        self.assertEqual(planner.prefix_snapshot()["hit_requests"], 1)
        self.assertEqual(planner.prefix_snapshot()["pinned_blocks"], 2)

    def test_dynamic_registration_is_idempotent_but_identity_cannot_change(self):
        planner = self.cached()
        request = RequestSpec("new", 10, 8, 1, prompt_token_ids=tuple(range(8)), cache_namespace="tenant")
        planner.register_request(request)
        planner.register_request(request)
        before = deepcopy((planner._prefix_specs, planner.prefix_snapshot()))
        for invalid in (replace(request, input_tokens=9), replace(request, cache_namespace="other"),
                        RequestSpec("bad", 0, 2, 1, prompt_token_ids=(0, True))):
            with self.subTest(invalid=invalid), self.assertRaises(SimulationError):
                planner.register_request(invalid)
            self.assertEqual((planner._prefix_specs, planner.prefix_snapshot()), before)

    def test_local_caches_and_simultaneous_dma_have_independent_capacity(self):
        first = RequestSpec("a", 0, 8, 2, prompt_token_ids=tuple(range(8)))
        second = replace(first, request_id="b")
        exp = Experiment(workload=(first, second), infra=InfraSpec(devices=(DeviceSpec("a"), DeviceSpec("b"))),
            strategy=StrategySpec(replicas=2, prefix_cache=True, preemption="swap",
                memory=MemoryConfig(kv_block_tokens=4, cpu_swap_bytes=4096,
                                    swap_d2h_bandwidth_gbps=.001)))
        local = [replica_experiment(exp, i) for i in range(2)]
        memories = [MemoryPlanner(item) for item in local]
        for memory, request in zip(memories, (first, second)):
            self.assertTrue(memory.reserve({request.request_id: 8}, [8], "prefill").ok)
        memories[0].publish_prefix("a", 8)
        self.assertEqual(memories[0].peek_prefix_tokens(replace(first, request_id="probe")), 4)
        self.assertEqual(memories[1].peek_prefix_tokens(replace(first, request_id="probe")), 0)
        swaps = [HostSwapManager(item, memory) for item, memory in zip(local, memories)]
        transfers = [manager.swap_out(request.request_id, 0) for manager, request in zip(swaps, (first, second))]
        self.assertEqual([transfer.start_ms for transfer in transfers], [0, 0])
        self.assertEqual(sum(manager.host_reserved_bytes for manager in swaps), 8192)
        swaps[0].complete(transfers[0], transfers[0].end_ms)
        self.assertNotIn("a", memories[0].allocations)
        self.assertEqual(memories[1].allocations, {"b": 8})
        self.assertEqual(swaps[1].snapshot(0)["completed_transfers_by_direction"]["D2H"], 0)
        swaps[1].complete(transfers[1], transfers[1].end_ms)
        self.assertEqual(memories[1].allocations, {})


if __name__ == "__main__":
    unittest.main()
