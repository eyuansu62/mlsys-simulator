"""Ownership, byte and virtual-time checks for the host KV swap resource."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from mlsys_sim.calibration import calibrate_dataset, write_json_exclusive
from mlsys_sim.layout import rank_kv_bytes_per_token
from mlsys_sim.memory import MemoryPlanner
from mlsys_sim.profiles import profile_identity
from mlsys_sim.schema import (CalibrationSpec, DeviceSpec, Experiment, InfraSpec,
                              MemoryConfig, SimulationError, StrategySpec)
from mlsys_sim.swap import HostSwapManager


def fixture(*, tp=1, pp=1, layers=(), blocks=8, host=65536, **memory_overrides):
    base = Experiment(infra=InfraSpec(devices=tuple(DeviceSpec(f"gpu{i}") for i in range(tp * pp))),
                      strategy=StrategySpec(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                                            stage_layers=layers, max_batch_tokens=64,
                                            pipeline_max_inflight_microbatches=4))
    values = dict(kv_block_tokens=16, kv_pool_bytes=rank_kv_bytes_per_token(base, 0) * 16 * blocks,
                  workspace_base_bytes=0, workspace_bytes_per_token=0,
                  runtime_bytes=0, safety_margin_bytes=0, cpu_swap_bytes=host,
                  swap_d2h_bandwidth_gbps=.001024, swap_h2d_bandwidth_gbps=.002048,
                  swap_latency_us=1000.0)
    values.update(memory_overrides)
    return replace(base, strategy=replace(base.strategy, memory=MemoryConfig(**values)))


class CopyLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.exp = fixture()
        self.memory = MemoryPlanner(self.exp)
        self.memory.reserve_kv({"a": 16})
        self.swap = HostSwapManager(self.exp, self.memory)

    def test_round_trip_pins_both_ends_and_uses_directional_bandwidth(self):
        physical = self.memory.snapshot()[0]["physical_reserved_bytes"]
        outgoing = self.swap.swap_out("a", 2.0)
        self.assertEqual((outgoing.start_ms, outgoing.end_ms, outgoing.bytes), (2, 11, 8192))
        self.assertEqual(self.memory.allocations, {"a": 16})
        self.assertEqual(self.swap.state("a"), "SWAPPING_OUT")
        self.assertEqual(self.swap.snapshot(2)["host_reserved_bytes"], 8192)
        with self.assertRaises(SimulationError):
            self.swap.complete(outgoing, 10.5)
        self.assertEqual(self.memory.allocations, {"a": 16})
        done = self.swap.complete(outgoing, 11)
        self.assertFalse(done["cancelled"])
        self.assertEqual(self.memory.allocations, {})
        self.assertEqual(self.swap.state("a"), "SWAPPED")
        self.assertEqual(self.swap.stored_tokens("a"), 16)
        incoming = self.swap.swap_in("a", 12)
        self.assertEqual((incoming.start_ms, incoming.end_ms), (12, 17))
        self.assertEqual(self.memory.allocations, {"a": 16})
        self.assertEqual(self.swap.snapshot(12)["host_reserved_bytes"], 8192)
        self.swap.complete(incoming, 17)
        result = self.swap.snapshot(17)
        self.assertEqual(result["host_reserved_bytes"], 0)
        self.assertEqual(result["host_peak_reserved_bytes"], 8192)
        self.assertEqual(result["completed_bytes_by_direction"], {"D2H": 8192, "H2D": 8192})
        self.assertEqual(result["dma_busy_ms"], 14)
        self.assertIsNone(self.swap.state("a"))
        self.assertEqual(self.memory.snapshot()[0]["physical_reserved_bytes"], physical)

    def test_mid_copy_snapshot_accounts_latency_and_data_without_mutation(self):
        transfer = self.swap.swap_out("a", 2)
        at_latency = self.swap.snapshot(3)
        middle = self.swap.snapshot(7)
        self.assertEqual(at_latency["dma_busy_ms"], 1)
        self.assertEqual(at_latency["pending_transferred_bytes_by_direction"]["D2H"], 0)
        self.assertEqual(middle["dma_busy_ms"], 5)
        self.assertEqual(middle["pending_transferred_bytes_by_direction"]["D2H"], 4096)
        self.assertEqual(middle["completed_bytes_by_direction"]["D2H"], 0)
        self.assertEqual(self.memory.allocations, {"a": 16})
        self.assertEqual(self.swap.pending_transfers, (transfer,))
        self.assertEqual(self.swap.snapshot(7), middle)

    def test_cancel_outgoing_waits_for_completion_then_frees_both_owners(self):
        transfer = self.swap.swap_out("a", 0)
        self.swap.cancel("a")
        self.assertEqual(self.memory.allocations, {"a": 16})
        self.assertEqual(self.swap.snapshot(2)["host_reserved_bytes"], 8192)
        done = self.swap.complete(transfer, transfer.end_ms)
        self.assertTrue(done["cancelled"])
        self.assertEqual(self.memory.allocations, {})
        self.assertEqual(self.swap.snapshot(transfer.end_ms)["host_reserved_bytes"], 0)
        self.assertIsNone(self.swap.state("a"))

    def test_cancel_restoring_waits_for_completion_then_frees_both_owners(self):
        outgoing = self.swap.swap_out("a", 0)
        self.swap.complete(outgoing, outgoing.end_ms)
        incoming = self.swap.swap_in("a", outgoing.end_ms)
        self.swap.cancel("a")
        self.assertEqual(self.memory.allocations, {"a": 16})
        self.assertEqual(self.swap.snapshot(incoming.start_ms)["host_reserved_bytes"], 8192)
        self.assertTrue(self.swap.complete(incoming, incoming.end_ms)["cancelled"])
        self.assertEqual(self.memory.allocations, {})
        self.assertEqual(self.swap.snapshot(incoming.end_ms)["host_reserved_bytes"], 0)

    def test_cancel_swapped_copy_is_immediate_but_unknown_id_does_not_touch_gpu(self):
        outgoing = self.swap.swap_out("a", 0)
        self.swap.complete(outgoing, outgoing.end_ms)
        self.memory.reserve_kv({"unrelated": 16})
        self.swap.cancel("a")
        self.swap.cancel("unrelated")
        self.assertEqual(self.memory.allocations, {"unrelated": 16})
        self.assertEqual(self.swap.snapshot(outgoing.end_ms)["host_reserved_bytes"], 0)

    def test_serialized_queue_holds_cancelled_buffers_until_its_own_boundary(self):
        self.memory.reserve_kv({"b": 16})
        first = self.swap.swap_out("a", 0)
        second = self.swap.swap_out("b", 0)
        self.assertEqual((first.end_ms, second.start_ms, second.end_ms), (9, 9, 18))
        self.swap.cancel("b")
        self.assertEqual(self.swap.snapshot(4)["dma_busy_ms"], 4)
        self.assertEqual(self.swap.snapshot(4)["host_reserved_bytes"], 16384)
        with self.assertRaises(SimulationError):
            self.swap.complete(second, 18)
        self.swap.complete(first, 9)
        self.assertEqual(self.memory.allocations, {"b": 16})
        self.assertEqual(self.swap.snapshot(9)["host_reserved_bytes"], 16384)
        self.swap.complete(second, 18)
        self.assertEqual(self.memory.allocations, {})
        self.assertEqual(self.swap.snapshot(18)["host_reserved_bytes"], 8192)
        self.assertEqual(self.swap.snapshot(18)["cancelled_transfer_bytes"], 8192)

    def test_duplicate_forged_and_wrong_lifecycle_actions_are_rejected(self):
        outgoing = self.swap.swap_out("a", 0)
        for action in [lambda: self.swap.swap_out("a", 0), lambda: self.swap.swap_in("a", 0),
                       lambda: self.swap.complete(replace(outgoing, bytes=1), outgoing.end_ms)]:
            with self.assertRaises(SimulationError) as raised:
                action()
            self.assertEqual(raised.exception.code, "INVALID_SWAP_STATE")
        self.swap.complete(outgoing, outgoing.end_ms)
        with self.assertRaises(SimulationError):
            self.swap.complete(outgoing, outgoing.end_ms)
        with self.assertRaises(SimulationError):
            self.swap.swap_out("missing", outgoing.end_ms)


class CapacityAndAccountingTests(unittest.TestCase):
    def test_host_capacity_rejection_is_atomic(self):
        exp = fixture(host=8191)
        memory = MemoryPlanner(exp)
        memory.reserve_kv({"a": 16})
        swap = HostSwapManager(exp, memory)
        before = swap.snapshot(0), memory.snapshot()
        with self.assertRaises(SimulationError) as raised:
            swap.swap_out("a", 0)
        self.assertEqual(raised.exception.code, "HOST_SWAP_CAPACITY_EXCEEDED")
        self.assertEqual((swap.snapshot(0), memory.snapshot()), before)

    def test_gpu_restore_rejection_is_atomic_and_cpu_copy_survives(self):
        exp = fixture(blocks=2)
        memory = MemoryPlanner(exp)
        memory.reserve_kv({"a": 16})
        swap = HostSwapManager(exp, memory)
        outgoing = swap.swap_out("a", 0)
        swap.complete(outgoing, outgoing.end_ms)
        memory.reserve_kv({"b": 32})
        before = swap.snapshot(9), memory.snapshot()
        with self.assertRaises(SimulationError) as raised:
            swap.swap_in("a", 9)
        self.assertEqual(raised.exception.code, "KV_CAPACITY_EXCEEDED")
        self.assertEqual((swap.snapshot(9), memory.snapshot()), before)
        self.assertEqual(swap.state("a"), "SWAPPED")

    def test_tp_pp_copy_traffic_includes_all_replicated_rank_blocks(self):
        exp = fixture(tp=4, pp=2, layers=(1, 3), blocks=8, host=32768)
        memory = MemoryPlanner(exp)
        self.assertEqual(memory.kv_bytes_for_tokens(0), 0)
        self.assertEqual(memory.kv_bytes_for_tokens(16), 16384)
        self.assertEqual(memory.kv_bytes_for_tokens(17), 32768)
        self.assertTrue(memory.reserve_kv({"a": 17}).ok)
        outgoing = HostSwapManager(exp, memory).swap_out("a", 0)
        self.assertEqual(outgoing.bytes, 32768)

    def test_kv_only_reservation_does_not_invent_workspace_queries_or_peaks(self):
        exp = fixture(blocks=4)
        data = {"schema_version": 1, "dataset_id": "fixture",
                "identity": profile_identity(exp),
                "source": {"kind": "test_fixture", "reference": "unit-test fixture"},
                "records": [{"record_id": "shape", "group_id": "shape", "phase": "prefill",
                             "stage_index": 0, "query_tokens": [4], "context_lengths": [0],
                             "duration_ms": 1, "workspace_peak_bytes": 123}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            write_json_exclusive(path, calibrate_dataset(data, profile_version="v1", allow_test_fixture=True))
            exp = replace(exp, calibration=CalibrationSpec(backend="profile", profile_path=str(path),
                                                           allow_test_fixture=True))
            memory = MemoryPlanner(exp)
            self.assertTrue(memory.reserve_kv({"a": 17}).ok)
            self.assertEqual(memory.snapshot()[0]["workspace_peak_bytes"], 0)
            self.assertEqual(memory.snapshot()[0]["workspace_source"], "not_executed")

    def test_block_only_failure_and_alone_checks_do_not_mutate(self):
        memory = MemoryPlanner(fixture(blocks=2))
        memory.reserve_kv({"a": 16})
        before = memory.snapshot()
        self.assertFalse(memory.reserve_kv({"b": 17}).ok)
        self.assertTrue(memory.check_kv({"b": 17}, alone=True).ok)
        self.assertEqual(memory.allocations, {"a": 16})
        self.assertEqual(memory.snapshot(), before)
        with self.assertRaises(SimulationError):
            memory.reserve_kv({"a": 15})

    def test_pipeline_buffer_budget_is_stage_specific_and_separate(self):
        exp = fixture(tp=2, pp=3, communication_buffer_bytes=37)
        ranks = MemoryPlanner(exp).snapshot()
        self.assertEqual([rank["pipeline_buffer_reserved_bytes"] for rank in ranks],
                         [65536, 65536, 131072, 131072, 65536, 65536])
        for rank in ranks:
            self.assertEqual(rank["physical_reserved_bytes"], rank["weights_bytes"]
                             + rank["kv_pool_reserved_bytes"] + 37
                             + rank["pipeline_buffer_reserved_bytes"])
        single = MemoryPlanner(fixture()).snapshot()[0]
        self.assertEqual(single["pipeline_buffer_reserved_bytes"], 0)

    def test_pipeline_buffers_can_make_a_deployment_statically_infeasible(self):
        exp = fixture(pp=2)
        budget = MemoryPlanner(exp).snapshot()[0]["fixed_budget_bytes"]
        devices = (replace(exp.infra.devices[0], memory_bytes=budget - 1), exp.infra.devices[1])
        with self.assertRaises(SimulationError) as raised:
            MemoryPlanner(replace(exp, infra=replace(exp.infra, devices=devices)))
        self.assertEqual(raised.exception.code, "STATIC_MEMORY_INFEASIBLE")

    def test_invalid_transfer_rates_and_nonadvancing_clock_have_no_reservations(self):
        for overrides, now in [({"swap_d2h_bandwidth_gbps": 1e308}, 0),
                               ({"swap_latency_us": 0}, 1e308)]:
            with self.subTest(overrides=overrides):
                exp = fixture(**overrides)
                memory = MemoryPlanner(exp)
                memory.reserve_kv({"a": 16})
                swap = HostSwapManager(exp, memory)
                with self.assertRaises(SimulationError) as raised:
                    swap.swap_out("a", now)
                self.assertEqual(raised.exception.code, "INVALID_SWAP_ESTIMATE")
                self.assertEqual(swap.host_reserved_bytes, 0)
                self.assertEqual(memory.allocations, {"a": 16})


if __name__ == "__main__":
    unittest.main()
