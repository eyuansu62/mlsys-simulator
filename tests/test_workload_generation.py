"""Traffic generation is bounded, portable and independent of length RNGs."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from mlsys_sim.schema import SimulationError
from mlsys_sim.workload import canonical_sha256, generate_workload, request_specs


def specification(process="poisson", rate=100, duration=100, maximum=100):
    return {"schema_version": 1, "seed": 42,
            "arrival": {"process": process, "rate_rps": rate, "duration_ms": duration,
                        "max_requests": maximum},
            "tokens": {"input": {"min": 8, "max": 32}, "output": {"min": 1, "max": 4}}}


class WorkloadTests(unittest.TestCase):
    def test_periodic_half_open_and_canonical_contract(self):
        spec = specification("periodic")
        before = deepcopy(spec)
        run = generate_workload(spec)
        self.assertEqual(spec, before)
        self.assertEqual([row["arrival_ms"] for row in run["requests"]], list(range(0, 100, 10)))
        self.assertEqual(run["provenance"]["requests_sha256"], canonical_sha256(run["requests"]))
        self.assertEqual(len(request_specs(run)), 10)
        self.assertIsInstance(request_specs(run)[0].prompt_token_ids, tuple)

    def test_seeded_poisson_exact_replay_and_independent_substreams(self):
        first = specification()
        run = generate_workload(first, replication=1)
        self.assertEqual(run, generate_workload(first, replication=1))
        self.assertNotEqual(run, generate_workload(first, replication=2))
        second = deepcopy(first)
        second["tokens"]["input"] = 512
        second["tokens"]["output"] = 77
        changed = generate_workload(second, replication=1)
        self.assertEqual([row["arrival_ms"] for row in run["requests"]],
                         [row["arrival_ms"] for row in changed["requests"]])
        self.assertTrue(all(8 <= row["input_tokens"] <= 32 and 1 <= row["output_tokens"] <= 4
                            for row in run["requests"]))
        second["seed"] = 43
        self.assertNotEqual(changed["requests"], generate_workload(second, replication=1)["requests"])

    def test_reject_invalid_bounds_unknown_fields_and_bool(self):
        changes = [("seed", True), ("schema_version", True), ("seed", 2**63), ("seed", 1.0),
                   ("request_prefix", ""), ("timeout_ms", 0), ("priority", False)]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(SimulationError):
                spec = specification()
                spec[key] = value
                generate_workload(spec)
        for key, value in (("rate_rps", 0), ("rate_rps", math.inf), ("rate_rps", math.nan),
                           ("rate_rps", 10**1000), ("rate_rps", 5e-324),
                           ("rate_rps", 1e308), ("duration_ms", 0), ("max_requests", 0),
                           ("max_requests", True), ("max_requests", 100001), ("extra", 1)):
            with self.subTest(key=key, value=str(value)[:20]), self.assertRaises(SimulationError):
                spec = specification()
                spec["arrival"][key] = value
                generate_workload(spec)
        for lengths in (True, 0, -1, {"min": 5, "max": 4}, {"min": 1, "max": 5, "x": 2}):
            with self.subTest(lengths=lengths), self.assertRaises(SimulationError):
                spec = specification()
                spec["tokens"]["input"] = lengths
                generate_workload(spec)

    def test_bound_failure_cannot_silently_truncate(self):
        spec = specification(rate=100, maximum=10)
        with patch("mlsys_sim.workload.random.Random.random", return_value=.01):
            with self.assertRaisesRegex(SimulationError, "no silent truncation"):
                generate_workload(spec)
        with patch("mlsys_sim.workload.random.Random.random", return_value=0):
            with self.assertRaisesRegex(SimulationError, "not representable"):
                generate_workload(spec)

    def test_prompt_identity_requires_matching_fixed_input(self):
        spec = specification("periodic")
        spec["tokens"]["input"] = 8
        spec["prompt_token_ids"] = list(range(8))
        spec["cache_namespace"] = "same-model-tokenizer"
        self.assertEqual(request_specs(generate_workload(spec))[0].prompt_token_ids, tuple(range(8)))
        spec["prompt_token_ids"] = [True] * 8
        with self.assertRaises(SimulationError):
            generate_workload(spec)


if __name__ == "__main__":
    unittest.main()
