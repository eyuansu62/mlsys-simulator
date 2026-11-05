"""CPU admission checks for the pinned synchronous PP send-retention bound."""

from dataclasses import replace
from pathlib import Path
import unittest

from mlsys_sim import MemoryPlanner, SimulationError, load_experiment, simulate
from test_native_pp_transport import experiment


ROOT = Path(__file__).resolve().parents[1]


def full_pair_bytes(exp):
    strategy = exp.strategy
    tokens = strategy.pipeline_transport_max_tokens or strategy.max_batch_tokens
    return int(2 * tokens * strategy.model.hidden_size * strategy.model.activation_bytes)


class NativePPRetentionBudgetTests(unittest.TestCase):
    def test_one_wave_keeps_previous_and_current_outputs_on_every_sender(self):
        exp = experiment(tp=1, pp=3, pipeline_max_inflight_microbatches=1,
                         pipeline_transport_max_tokens=32)
        pair = full_pair_bytes(exp)
        ranks = MemoryPlanner(exp).snapshot()
        self.assertEqual([row["pipeline_buffer_reserved_bytes"] for row in ranks],
                         [2 * pair, 4 * pair, 2 * pair])
        for row in ranks[:2]:
            self.assertEqual(row["pipeline_buffer_breakdown"]["outgoing_inflight_bytes"], pair)
            self.assertEqual(row["pipeline_buffer_breakdown"]["outgoing_retention_slack_bytes"], pair)
        self.assertEqual(ranks[-1]["pipeline_buffer_breakdown"]["outgoing_retention_slack_bytes"], 0)

    def test_two_or_more_reserved_waves_do_not_add_a_redundant_pair(self):
        for inflight in (2, 8):
            with self.subTest(inflight=inflight):
                exp = experiment(pipeline_max_inflight_microbatches=inflight,
                                 pipeline_transport_max_tokens=32)
                pair = full_pair_bytes(exp)
                ranks = MemoryPlanner(exp).snapshot()
                for row in ranks[:2]:
                    self.assertEqual(row["pipeline_buffer_reserved_bytes"], inflight * pair)
                    self.assertEqual(row["pipeline_buffer_breakdown"]["outgoing_inflight_bytes"],
                                     inflight * pair)
                    self.assertEqual(row["pipeline_buffer_breakdown"]["outgoing_retention_slack_bytes"], 0)

    def test_receiver_slicing_and_legacy_transport_budgets_keep_their_contracts(self):
        native = experiment(pipeline_max_inflight_microbatches=1,
                            pipeline_transport_max_tokens=32)
        pair = full_pair_bytes(native)
        ranks = MemoryPlanner(native).snapshot()
        receiver = ranks[-1]["pipeline_buffer_breakdown"]
        self.assertEqual(receiver["incoming_inflight_bytes"], pair)
        self.assertEqual(receiver["incoming_slice_backing_bytes"], pair)
        self.assertEqual(receiver["persistent_input_copy_bytes"], pair)
        self.assertEqual(ranks[-1]["pipeline_buffer_reserved_bytes"], 3 * pair)
        legacy = replace(native, strategy=replace(native.strategy, pipeline_transport="full_hidden"))
        expected = int(legacy.strategy.max_batch_tokens * legacy.strategy.model.hidden_size
                       * legacy.strategy.model.activation_bytes)
        for row in MemoryPlanner(legacy).snapshot():
            self.assertEqual(row["pipeline_buffer_reserved_bytes"], expected)
            self.assertNotIn("pipeline_buffer_breakdown", row)

    def test_retained_pair_participates_in_exact_static_admission_boundary(self):
        exp = experiment(tp=1, pipeline_max_inflight_microbatches=1)
        sender = MemoryPlanner(exp).snapshot()[0]
        required = sender["fixed_budget_bytes"] + exp.strategy.memory.workspace_base_bytes
        devices = exp.infra.devices
        exact = replace(exp, infra=replace(exp.infra,
            devices=(replace(devices[0], memory_bytes=required), devices[1])))
        MemoryPlanner(exact)
        insufficient = replace(exact, infra=replace(exact.infra,
            devices=(replace(devices[0], memory_bytes=required - 1), devices[1])))
        with self.assertRaises(SimulationError) as raised:
            MemoryPlanner(insufficient)
        self.assertEqual(raised.exception.code, "STATIC_MEMORY_INFEASIBLE")
        self.assertGreaterEqual(full_pair_bytes(exp), 1)

    def test_retention_reserve_composes_with_replica_cache_chunk_swap_and_mixed(self):
        exp = load_experiment(ROOT / "examples/v1_7/qwen3-native-r2-tp2-pp2.json")
        self.assertEqual(exp.strategy.pipeline_max_inflight_microbatches, 1)
        run = simulate(exp)
        self.assertEqual(run.result["counts"]["completed"], 10)
        self.assertGreater(run.result["batching"]["mixed_batches_completed"], 0)
        self.assertGreater(run.result["prefix_cache"]["hit_tokens"], 0)
        self.assertEqual(run.result["swap"]["completed_transfers_by_direction"], {"D2H": 2, "H2D": 2})
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)
        for row in run.result["memory"]:
            self.assertEqual(row["used_blocks"], 0)
            breakdown = row["pipeline_buffer_breakdown"]
            self.assertEqual(breakdown["outgoing_retention_slack_bytes"],
                             breakdown["outgoing_inflight_bytes"])


if __name__ == "__main__":
    unittest.main()
