"""Public policy validation and compatibility with pre-pipeline V1 contracts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mlsys_sim import (Experiment, SimulationError, compare_runs,
                       comparison_contract, experiment_from_dict)


class PPSwapConfigurationTests(unittest.TestCase):
    def test_invalid_controls_are_rejected_before_scheduling(self):
        invalid = [
            {"preemption": "swap"},
            {"pipeline_microbatch_size": 0},
            {"max_batch_size": 2, "pipeline_microbatch_size": 3},
            {"pipeline_max_inflight_microbatches": 0},
            {"pipeline_feedback_latency_us": float("nan")},
            {"pipeline_feedback_latency_us": -1},
            {"pipeline_p2p": 1},
            {"swap_fallback": "recompute"},
            {"swap_fallback": "automatic"},
            {"memory": {"cpu_swap_bytes": -1}},
            {"memory": {"cpu_swap_bytes": True}},
            {"memory": {"swap_d2h_bandwidth_gbps": 0}},
            {"memory": {"swap_h2d_bandwidth_gbps": float("inf")}},
            {"memory": {"swap_latency_us": -1}},
        ]
        for strategy in invalid:
            with self.subTest(strategy=strategy), self.assertRaises(SimulationError):
                experiment_from_dict({"strategy": strategy})

    def test_explicit_swap_and_fallback_configuration_is_preserved(self):
        experiment = experiment_from_dict({"strategy": {"preemption": "swap",
            "swap_fallback": "recompute", "memory": {"cpu_swap_bytes": 32768,
            "swap_d2h_bandwidth_gbps": 4, "swap_h2d_bandwidth_gbps": 8}}})
        self.assertEqual(experiment.strategy.swap_fallback, "recompute")
        self.assertEqual(experiment.strategy.memory.swap_d2h_bandwidth_gbps, 4)
        self.assertEqual(experiment.strategy.memory.swap_h2d_bandwidth_gbps, 8)

    def test_original_v1_comparison_contract_uses_additive_policy_defaults(self):
        current = comparison_contract(Experiment())
        legacy = deepcopy(current)
        for field in ("pipeline_microbatch_size", "pipeline_max_inflight_microbatches",
                      "pipeline_feedback_latency_us", "pipeline_p2p", "swap_fallback"):
            legacy["strategy"].pop(field)
        for field in ("cpu_swap_bytes", "swap_d2h_bandwidth_gbps", "swap_h2d_bandwidth_gbps",
                      "swap_latency_us"):
            legacy["strategy"]["memory"].pop(field)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predicted = root / "predicted"
            predicted.mkdir()
            (predicted / "manifest.json").write_text(json.dumps({"backend": "simulated",
                                                                "comparison_contract": current}))
            (predicted / "result.json").write_text(json.dumps({"backend": "simulated", "status": "COMPLETED",
                                                               "counts": {"total": 0}}))
            observed = root / "observed.json"
            observed.write_text(json.dumps({"schema_version": 1, "kind": "measured_run",
                "comparison_contract": legacy,
                "source": {"kind": "test_fixture", "reference": "Legacy compatibility test only"},
                "metrics": {"request_count": 0}}))
            report = compare_runs(predicted, observed, allow_test_fixture=True)
            self.assertEqual(report["metrics"]["request_count"]["absolute_error"], 0)


if __name__ == "__main__":
    unittest.main()
