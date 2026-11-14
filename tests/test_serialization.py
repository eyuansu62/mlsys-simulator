"""Default additions must not invalidate historical replay or comparison data."""

from dataclasses import asdict, replace
import hashlib
import json
import unittest

from mlsys_sim import Experiment, RequestSpec
from mlsys_sim.capacity import config_sha256
from mlsys_sim.comparison import comparison_contract, _contract
from mlsys_sim.config import experiment_from_dict
from mlsys_sim.serialization import experiment_dict, strategy_dict


class SerializationTests(unittest.TestCase):
    def experiment(self):
        return Experiment(workload=(RequestSpec("a", 0, 16, 2), RequestSpec("b", 2, 32, 3)))

    def test_default_config_and_contract_match_committed_119_baseline(self):
        # Computed from an isolated clean fc6379f checkout, before adding controls.
        exp = self.experiment()
        self.assertEqual(config_sha256(exp), "fe5a125436e54fe6f3b83f659ad1963c20ca7183a457b7e8cf26fce22e8d7108")
        contract = comparison_contract(exp)
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(hashlib.sha256(encoded.encode()).hexdigest(),
                         "8f2466846f85847b7db10f2e8c2fad671a3d03f59fcb0849a42a3f5cee7cf1cd")
        self.assertEqual(_contract(contract, "test"), contract)

    def test_disabled_controls_are_equivalent_even_when_asdict_expands_nulls(self):
        exp = self.experiment()
        expanded = asdict(exp)
        self.assertIn("max_active_requests", expanded["strategy"])
        self.assertEqual(config_sha256(expanded), config_sha256(exp))
        self.assertEqual(experiment_from_dict(experiment_dict(exp)), exp)
        self.assertNotIn("max_active_requests", strategy_dict(exp.strategy))
        self.assertNotIn("gpu_memory_utilization", strategy_dict(exp.strategy)["memory"])

    def test_enabled_controls_remain_explicit_and_change_comparison_policy(self):
        base = self.experiment()
        changed = replace(base, strategy=replace(base.strategy, max_active_requests=1,
            memory=replace(base.strategy.memory, gpu_memory_utilization=0.75)))
        emitted = experiment_dict(changed)
        self.assertEqual(emitted["strategy"]["max_active_requests"], 1)
        self.assertEqual(emitted["strategy"]["memory"]["gpu_memory_utilization"], 0.75)
        self.assertEqual(experiment_from_dict(emitted), changed)
        self.assertNotEqual(config_sha256(changed), config_sha256(base))
        current = comparison_contract(changed)
        self.assertEqual(_contract(current, "test"), current)
        self.assertNotEqual(current, comparison_contract(base))

    def test_mapping_normalization_does_not_mutate_or_remove_other_defaults(self):
        original = asdict(self.experiment())
        before = json.dumps(original, sort_keys=True)
        normalized = experiment_dict(original)
        self.assertEqual(json.dumps(original, sort_keys=True), before)
        self.assertIn("profile_path", normalized["calibration"])
        self.assertIsNone(normalized["calibration"]["profile_path"])
        self.assertIn("ttft_sla_ms", normalized["evaluation"])
        self.assertEqual(experiment_dict({}), {})


if __name__ == "__main__":
    unittest.main()
