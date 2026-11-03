"""Public warmup dispatch isolates PP and host swap from measured statistics."""

from dataclasses import replace
from pathlib import Path
import unittest

from mlsys_sim import RequestSpec, load_experiment, simulate


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_swap_warmup_is_drained_before_a_fresh_measured_timeline(self):
        root = Path(__file__).resolve().parents[1]
        base = load_experiment(root / "examples/v1_1/pp_cpu_swap.json")
        experiment = replace(base, workload=(RequestSpec("measured", 0, 8, 1),),
                             execution=replace(base.execution, warmup=base.workload))
        run = simulate(experiment)
        self.assertEqual(run.result["status"], "COMPLETED")
        self.assertEqual([row["request_id"] for row in run.requests], ["measured"])
        self.assertLess(run.requests[0]["first_token_ms"], 10)
        warmup = run.result["warmup"]
        self.assertGreater(warmup["simulated_ms"], 50)
        self.assertGreater(warmup["swap"]["completed_transfers_by_direction"]["D2H"], 0)
        self.assertEqual(warmup["swap"]["host_reserved_bytes"], 0)
        self.assertEqual(run.result["swap"]["completed_transfers_by_direction"], {"D2H": 0, "H2D": 0})
        self.assertEqual(run.result["swap"]["host_reserved_bytes"], 0)
        self.assertTrue(all(rank["used_blocks"] == 0 for rank in run.result["memory"]))


if __name__ == "__main__":
    unittest.main()
