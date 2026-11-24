import json
from pathlib import Path
from collections import deque
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.analyze_c2_tail import analysis, load_sensors
from scripts.c2_tail_probe import StageTimeline


def record(i, role, duration):
    start = i * 100_000_000
    members = (
        [
            {
                "request_phase": role,
                "query_tokens": 64 if role == "prefill" else 1,
                "context_length": 0 if role == "prefill" else 64,
            }
        ]
        * 2
        if role
        else []
    )
    return {
        "source_order": i,
        "started_ns": start,
        "finished_ns": start + int(duration * 1e6),
        "duration_ms": duration,
        "thread_cpu_ns": int(duration * 1e6),
        "members": members,
        "capture_phase": "formal",
        "status": "COMPLETED",
    }


class TailTimelineTests(unittest.TestCase):
    def fixture(self):
        pairs = [
            ("", 2),
            ("prefill", 40),
            ("decode", 17),
            ("prefill", 27),
            ("decode", 17),
            ("decode", 29),
            ("prefill", 41),
            ("decode", 28),
            ("decode", 17),
            ("prefill", 27),
            ("decode", 17),
        ]
        return {
            "session_id": "s",
            "closed": True,
            "iterations": [record(i + 1, *row) for i, row in enumerate(pairs)],
        }

    def test_paired_requires_immediate_matched_decode_and_excludes_first_prefill(self):
        core = self.fixture()
        report = analysis(core)
        self.assertEqual((report["prefill_ge38"], report["paired"]), (2, 1))
        self.assertTrue(report["events"][0]["first_online"])
        event = report["events"][1]
        self.assertEqual(event["source_order"], 7)
        self.assertEqual(event["neighbors"][2]["same_shape_median_ms"], 27)
        self.assertEqual(event["neighbors"][1]["same_shape_median_ms"], 17)
        # A second neighbor alone does not meet the definition.
        core["iterations"][5] = record(6, "decode", 17)
        core["iterations"][7] = record(8, "decode", 17)
        core["iterations"][8] = record(9, "decode", 30)
        self.assertEqual(analysis(core)["paired"], 0)

    def test_stage_clocks_must_enclose_same_native_step(self):
        core = self.fixture()
        rows = []
        for r in core["iterations"]:
            start = r["started_ns"]
            end = r["finished_ns"]
            rows.append([r["source_order"], "core", start - 10, 2, 0, 0, 0, 0, end + 10, 3, 0, 0, 0, 0])
        timeline = {"session_id": "s", "kind": "c2_stage_timeline_v1", "gc": [], "missing": [], "rows": rows}
        self.assertEqual(analysis(core, timeline)["paired"], 1)
        rows[0][2] += 11
        with self.assertRaises(ValueError):
            analysis(core, timeline)

    def test_sensor_intervals_are_clipped_to_episode_window(self):
        core = self.fixture()
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "sensor.jsonl"
            p.write_text(
                "\n".join(
                    json.dumps(x)
                    for x in [
                        {"keys": ["clk"], "hwmon_labels": {}},
                        [500000000, 500000100, 100],
                        [699000000, 700000000, 200],
                        [910000000, 990000000, 300],
                    ]
                )
            )
            report = analysis(core, sensors=load_sensors(p))
        self.assertEqual([r[-1] for r in report["events"][1]["sensors"]["window_rows"]], [100, 200])


class ProbeTests(unittest.TestCase):
    def test_scope_restored_on_exception(self):
        class Recorder:
            session_id = "s"
            _order = 3

        class Owner:
            def execute_model(self):
                raise LookupError("worker")

        with TemporaryDirectory() as tmp:
            probe = object.__new__(StageTimeline)
            probe.recorder = Recorder()
            probe.rows = {}
            probe.patches = []
            probe.missing = []
            probe.scope = None
            probe.collections = []
            probe.output = Path(tmp)
            owner = Owner()
            probe.bind(owner, "execute_model", "execute")
            with (
                patch("scripts.c2_tail_probe.stamp", side_effect=[(1, 1, 0, 0, 0, 0), (3, 2, 0, 0, 0, 0)]),
                self.assertRaises(LookupError),
            ):
                owner.execute_model()
            probe.close()
            self.assertNotIn("execute_model", vars(owner))
            raw = json.loads((Path(tmp) / "stage-timeline.json").read_bytes())
            self.assertEqual(raw["rows"][0][:2], [3, "execute"])


class StreamTimingTests(unittest.TestCase):
    def test_pending_event_is_retained_without_synchronization(self):
        # The deployed worker loads the probe as a sibling script.
        import importlib.util
        from scripts import c2_tail_probe

        with patch.dict(sys.modules, {"c2_tail_probe": c2_tail_probe}):
            spec = importlib.util.spec_from_file_location(
                "deep_probe", Path(c2_tail_probe.__file__).with_name("c2_forward_probe.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        class Event:
            ready = False

            def query(self):
                return self.ready

            def elapsed_time(self, other):
                self.assertion(other.ready)
                return 7.5

            @staticmethod
            def assertion(value):
                if not value:
                    raise AssertionError("pending device work read prematurely")

        probe = object.__new__(module.ForwardTimeline)
        start, end = Event(), Event()
        probe.pool, probe.gpu_rows, probe.inflight = deque(), [], deque([(12, start, end)])
        probe.collect_ready()
        self.assertEqual(len(probe.inflight), 1)
        end.ready = True
        probe.collect_ready()
        self.assertEqual(probe.gpu_rows, [[12, 7.5]])
        self.assertEqual(list(probe.pool), [(start, end)])
