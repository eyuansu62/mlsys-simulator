"""Capacity boundaries for per-layer materialized weight dequantization."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import MemoryPlanner, SimulationError, load_experiment, simulate
from test_memory import fixture
from test_profiles import observation, profile
from test_workspace import configured, workspace_record


def materialized(*, tp=1, pp=1, layers=(), inflight=1, workspace_per_token=0):
    exp = fixture(tp=tp, pp=pp, layers=layers, blocks=16, workspace_per_token=workspace_per_token)
    return replace(exp, strategy=replace(exp.strategy,
        model=replace(exp.strategy.model, weight_bytes=0.5, weight_dequant="materialized"),
        pipeline_max_inflight_microbatches=inflight))


class MaterializedWorkspaceTests(unittest.TestCase):
    def test_invalid_parallel_degree_is_rejected_before_expansion_arithmetic(self):
        for field in ("tensor_parallel_size", "pipeline_parallel_size"):
            for value in (0, -1, False):
                exp = materialized()
                exp = replace(exp, strategy=replace(exp.strategy, **{field: value}))
                with self.subTest(field=field, value=value), self.assertRaises(SimulationError) as caught:
                    MemoryPlanner(exp)
                self.assertEqual(caught.exception.code, "INVALID_CONFIG")

    def test_one_layer_expansion_is_temporary_and_included_in_execution_peak(self):
        exp = materialized()
        fused = replace(exp, strategy=replace(exp.strategy,
            model=replace(exp.strategy.model, weight_dequant="fused")))
        planner, baseline = MemoryPlanner(exp), MemoryPlanner(fused)
        # Toy layer: Q/O 32,768 + K/V 8,192 + FFN 196,608 elements at 2 bytes.
        scratch = 475136
        self.assertEqual(planner.snapshot(), baseline.snapshot())
        for phase, target, query in (("prefill", 4, 4), ("decode", 5, 1), ("recompute", 4, 4)):
            result = planner.check({"r": target}, [query], phase)
            self.assertEqual(result.details["workspace_bytes"], 128 + scratch)
        self.assertTrue(planner.reserve({"r": 4}, [4], "prefill").ok)
        snapshot = planner.snapshot()[0]
        self.assertEqual(snapshot["workspace_peak_bytes"], 128 + scratch)
        self.assertEqual(snapshot["physical_peak_bytes"], snapshot["physical_reserved_bytes"] + 128 + scratch)
        self.assertEqual(snapshot["fixed_budget_bytes"], baseline.snapshot()[0]["fixed_budget_bytes"])

    def test_static_minimum_includes_expansion_at_exact_byte_boundary(self):
        exp = materialized()
        fixed = MemoryPlanner(exp).snapshot()[0]["fixed_budget_bytes"]
        for delta in (-1, 0):
            limited = replace(exp, infra=replace(exp.infra, devices=(
                replace(exp.infra.devices[0], memory_bytes=fixed + 128 + 475136 + delta),)))
            with self.subTest(delta=delta):
                if delta < 0:
                    with self.assertRaises(SimulationError) as caught:
                        MemoryPlanner(limited)
                    self.assertEqual(caught.exception.code, "STATIC_MEMORY_INFEASIBLE")
                else:
                    self.assertTrue(MemoryPlanner(limited).reserve({"r": 4}, [4], "prefill").ok)

    def test_weak_tp_rank_rejects_step_atomically_with_one_byte_shortfall(self):
        exp = materialized(tp=2, pp=2, layers=(1, 3), workspace_per_token=7)
        ranks = MemoryPlanner(exp).snapshot()
        devices = list(exp.infra.devices)
        devices[1] = replace(devices[1], memory_bytes=ranks[1]["fixed_budget_bytes"] + 128 + 237568 + 27)
        exp = replace(exp, infra=replace(exp.infra, devices=tuple(devices)))
        planner = MemoryPlanner(exp)
        before = planner.snapshot()
        decision = planner.reserve({"r": 4}, [4], "prefill")
        self.assertFalse(decision.ok)
        self.assertEqual(decision.code, "WORKSPACE_BUDGET_EXCEEDED")
        self.assertEqual([r["physical_ok"] for r in decision.details["ranks"]], [True, False, True, True])
        self.assertEqual(planner.allocations, {})
        self.assertEqual(planner.snapshot(), before)

    def test_pp_layer_counts_and_inflight_do_not_multiply_reused_scratch(self):
        for inflight in (1, 8):
            exp = materialized(tp=2, pp=2, layers=(1, 3), inflight=inflight)
            result = MemoryPlanner(exp).check({"r": 4}, [4], "prefill")
            self.assertEqual([r["workspace_bytes"] for r in result.details["ranks"]], [128 + 237568] * 4)

    def test_explicit_complete_workspace_profile_replaces_analytic_scratch(self):
        exp = materialized()
        with tempfile.TemporaryDirectory() as tmp:
            for peak in (0, 475136 + 17):
                rows = [workspace_record(exp, peaks=(peak,))]
                selected, _ = configured(exp, tmp, rows, version=f"materialized-{peak}")
                planner = MemoryPlanner(selected)
                self.assertTrue(planner.reserve({"r": 4}, [4], "prefill").ok)
                self.assertEqual(planner.snapshot()[0]["workspace_peak_bytes"], peak)
                self.assertEqual(planner.snapshot()[0]["workspace_source"], "workspace_profile")

    def test_stage_workspace_observation_replaces_scratch_but_missing_value_uses_fallback(self):
        exp = materialized()
        with tempfile.TemporaryDirectory() as tmp:
            for peak in (None, 0, 475136 + 17):
                path = Path(tmp) / f"stage-{peak}.json"
                path.write_text(json.dumps(profile([observation(workspace=peak)], experiment=exp)))
                selected = replace(exp, calibration=replace(exp.calibration, backend="profile",
                    profile_path=str(path), profile_version="v1", allow_test_fixture=True))
                decision = MemoryPlanner(selected).check({"r": 4}, [4], "prefill")
                self.assertEqual(decision.details["workspace_bytes"], 128 + 475136 if peak is None else peak)

    def test_kv_swap_destination_does_not_charge_compute_scratch(self):
        planner = MemoryPlanner(materialized())
        fixed = planner.snapshot()[0]["fixed_budget_bytes"]
        result = planner.reserve_kv({"restored": 4})
        self.assertTrue(result.ok)
        self.assertEqual(result.details["workspace_bytes"], 0)
        self.assertEqual(result.details["ranks"][0]["required_budget_bytes"], fixed)
        self.assertEqual(planner.snapshot()[0]["workspace_peak_bytes"], 0)

    def test_fractional_activation_storage_rounds_up_the_expanded_buffer(self):
        exp = materialized()
        exp = replace(exp, strategy=replace(exp.strategy,
            model=replace(exp.strategy.model, activation_bytes=2.000001)))
        self.assertEqual(MemoryPlanner(exp).check({"r": 4}, [4], "prefill").details["workspace_bytes"],
                         128 + 475137)

    def test_expansion_composes_with_replica_pp_cache_chunk_mixed_and_swap(self):
        path = Path(__file__).resolve().parents[1] / "examples/v1_4/replicas_pp_mixed_cache_swap.json"
        base = load_experiment(path)
        reference = simulate(base)
        exp = replace(base, strategy=replace(base.strategy,
            model=replace(base.strategy.model, weight_bytes=0.5, weight_dequant="materialized")))
        result = simulate(exp)
        self.assertEqual(result.requests, reference.requests)
        self.assertEqual(result.result["counts"]["completed"], 10)
        self.assertGreater(result.result["batching"]["mixed_batches_completed"], 0)
        for direction in ("D2H", "H2D"):
            self.assertGreater(result.result["swap"]["completed_transfers_by_direction"][direction], 0)
        self.assertGreater(result.result["prefix_cache"]["hit_tokens"], 0)
        for actual, original in zip(result.result["memory"], reference.result["memory"]):
            self.assertEqual(actual["workspace_peak_bytes"], original["workspace_peak_bytes"] + 475136)
            self.assertEqual(actual["used_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
