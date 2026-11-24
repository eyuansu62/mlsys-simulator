"""Declared exposure must apply on subsequent event-clock steps only."""

import unittest
from mlsys_sim.cpu_contention import SiblingExposure, validate_scenario
from mlsys_sim.schema import SimulationError, StepEstimate


class Toy:
    provenance = {"model_sha256": "a" * 64}

    def estimate_step(self, phase, q, c, stage_index, **kwargs):
        return StepEstimate(8.0, {}, "base", ())


def scenario():
    return {
        "kind": "declared_c2_sibling_exposure_v1",
        "base_model_sha256": "a" * 64,
        "increment_ms": {"prefill": 3.0, "decode": 2.0},
        "windows": [[[1, 3], [4, 5]]],
    }


class ContentionTests(unittest.TestCase):
    def spec(self):
        return {
            "concurrency_grid": [2],
            "input_lengths": [64, 128],
            "output_tokens": 16,
            "requests_per_run": 4,
            "replications": 1,
            "think_ms": 0,
        }

    def test_causal_half_open_steps(self):
        s = scenario()
        validate_scenario(s, self.spec())
        model = SiblingExposure(Toy(), s, 0)
        got = [
            model.estimate_step(
                "prefill" if i == 1 else "decode",
                [64, 128] if i == 1 else [1, 1],
                [0, 0] if i == 1 else [64, 128],
            ).duration_ms
            for i in range(6)
        ]
        self.assertEqual(got, [8.0, 11.0, 10.0, 8.0, 10.0, 8.0])
        self.assertEqual([row[-1] for row in model.steps], got)

    def test_reject_foreign_model_ranges_and_replication(self):
        s = scenario()
        s["base_model_sha256"] = "b" * 64
        with self.assertRaises(SimulationError):
            SiblingExposure(Toy(), s, 0)
        for intervals in ([[2, 5], [3, 7]], [[0, 33]], [[False, 3]], [[1, 1]]):
            s["windows"] = [intervals]
            with self.assertRaises(SimulationError):
                validate_scenario(s, self.spec())
        with self.assertRaises(SimulationError):
            validate_scenario(scenario(), {**self.spec(), "concurrency_grid": [2, 4]})
