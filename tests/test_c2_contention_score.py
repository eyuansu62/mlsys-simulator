import unittest

from scripts.score_c2_contention import measures


class C2PositiveEvidenceTests(unittest.TestCase):
    def test_a_slow_prefill_alone_cannot_pass_the_joint_target(self):
        steps = []
        for k in range(4):
            steps.append(["prefill", [64, 64], [0, 0], 27.0])
            steps.extend([["decode", [1, 1], [64, 64], 17.0] for _ in range(4)])
        windows = [
            dict(start=i, stop=i + 2, center=i, condition=c)
            for i, c in [(5, "sham"), (10, "sibling"), (15, "separate")]
        ]
        steps[10][-1] = 41.0
        _, hits, _ = measures(steps, windows)
        self.assertEqual(hits["sibling"], 0)
        steps[11][-1] = 27.0
        _, hits, counts = measures(steps, windows)
        self.assertEqual((hits["sibling"], counts["sibling"]), (1, 1))
        steps[10][-1] = 37.8
        _, hits, _ = measures(steps, windows)
        self.assertEqual(hits["sibling"], 0)

    def test_unbound_window_is_rejected(self):
        with self.assertRaises(ValueError):
            measures(
                [["decode", [1, 1], [65, 65], 17.0]], [dict(start=0, stop=1, center=0, condition="sibling")]
            )
