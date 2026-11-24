import unittest
from scripts.analyze_tpot_windows import replay_windows


def step(start, end, phase, count=1):
    members = [{"request_phase": phase, "query_tokens": 1, "context_length": 9}] * count
    return dict(
        capture_phase="formal",
        status="COMPLETED",
        started_ns=start,
        finished_ns=end,
        members=members,
        duration_ms=(end - start) / 1e6,
    )


class WindowReplayTests(unittest.TestCase):
    def test_clip_both_token_edges_and_preserve_other_work(self):
        steps = [
            step(0, 4_000_000, "prefill"),
            step(5_000_000, 11_000_000, "decode"),
            step(12_000_000, 20_000_000, "decode"),
        ]
        times = [2_000_000 + i * 1_000_000 for i in range(16)]
        req = [dict(stream_index=0, token_ready_ns=times, tpot_ms=1.0)]
        raw = {"closed": True, "iterations": steps}
        result = replay_windows(req, raw, lambda k: 8 if k[0] == "prefill" else 6)
        _, a, b, h, f = result[0]
        self.assertEqual(a, 1.0)
        self.assertAlmostEqual(b, (2 + 6 + 5) / 15)
        self.assertAlmostEqual(h, (2 + 7 + 5 * 7 / 8) / 15)
        self.assertAlmostEqual(f, (4 + 6 + 5 * 6 / 8) / 15)
        self.assertAlmostEqual(a - b, 2 / 15)

    def test_overlaps_truncation_and_disordered_tokens_fail_closed(self):
        times = list(range(16))
        good = dict(stream_index=0, token_ready_ns=times, tpot_ms=1e-6)
        for capture, req in [
            ({"closed": False, "iterations": []}, good),
            ({"closed": True, "iterations": [step(0, 8, "decode"), step(7, 15, "decode")]}, good),
            ({"closed": True, "iterations": [step(0, 20, "decode")]}, {**good, "token_ready_ns": times[:-1]}),
            (
                {"closed": True, "iterations": [step(0, 20, "decode")]},
                {**good, "token_ready_ns": times[::-1]},
            ),
        ]:
            with self.assertRaises(ValueError):
                replay_windows([req], capture, lambda k: 1.0)
