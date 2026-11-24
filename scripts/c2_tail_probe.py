"""Run a diagnostic-only closed-loop worker with per-stage CPU and wall clocks."""

from array import array
import argparse
import functools
import gc
import json
import resource
import sys
import time
from pathlib import Path

_CLOCK = time.perf_counter_ns
_MISSING = object()


def stamp():
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    return (
        _CLOCK(),
        time.thread_time_ns(),
        usage.ru_nvcsw,
        usage.ru_nivcsw,
        usage.ru_minflt,
        usage.ru_majflt,
    )


class StageTimeline:
    def __init__(self, backend, output):
        self.core = backend.engine.engine_core.engine_core
        self.recorder = backend.runtime_cost_recorder
        self.output = output
        self.rows = {}
        self.collections = []
        self.patches = []
        self.missing = []
        self.scope = None
        self.gc_start = None

    def bind(self, owner, name, stage):
        original = getattr(owner, name, None)
        if not callable(original):
            self.missing.append(stage)
            return
        saved = vars(owner).get(name, _MISSING)

        records = self.rows[stage] = array("q")

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            order = self.recorder._order + int(stage == "core")
            before = stamp()
            prior = self.scope
            self.scope = (stage, order)
            try:
                return original(*args, **kwargs)
            finally:
                after = stamp()
                self.scope = prior
                records.extend((order, *before, *after))

        setattr(owner, name, wrapped)
        self.patches.append((owner, name, saved, wrapped))

    def gc_event(self, phase, info):
        if phase == "start":
            self.gc_start = (_CLOCK(), self.scope, info["generation"])
        elif self.gc_start is not None:
            self.collections.append([*self.gc_start, _CLOCK(), self.scope])
            self.gc_start = None

    def attach(self):
        if self.recorder is None or self.recorder.fixture or self.recorder.phase != "formal":
            raise ValueError("Timeline requires a warmed, measured native worker")
        self.bind(self.core, "step_fn", "core")
        self.bind(self.core.scheduler, "schedule", "schedule")
        self.bind(self.core.scheduler, "update_from_output", "publish")
        executor = self.core.model_executor
        self.bind(executor, "execute_model", "execute")
        self.bind(executor, "sample_tokens", "sample")
        driver = getattr(executor, "driver_worker", None)
        gpu_worker = getattr(driver, "worker", driver)
        runner = getattr(gpu_worker, "model_runner", None)
        self.bind(runner, "execute_model", "runner_execute")
        self.bind(runner, "sample_tokens", "runner_sample")
        gc.callbacks.append(self.gc_event)

    def close(self):
        if self.gc_event in gc.callbacks:
            gc.callbacks.remove(self.gc_event)
        for owner, name, saved, wrapper in reversed(self.patches):
            if vars(owner).get(name) is wrapper:
                if saved is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, saved)
        data = {
            "kind": "c2_stage_timeline_v1",
            "role": "diagnostic",
            "session_id": self.recorder.session_id,
            "missing": self.missing,
            "stamp_fields": [
                "wall_ns",
                "thread_ns",
                "voluntary_switches",
                "involuntary_switches",
                "minor_faults",
                "major_faults",
            ],
            "layout": ["source_order", "stage", "start_stamp", "end_stamp"],
            "rows": [
                [packed[i], stage, *packed[i + 1 : i + 13]]
                for stage, packed in self.rows.items()
                for i in range(0, len(packed), 13)
            ],
            "gc": self.collections,
        }
        with (self.output / "stage-timeline.json").open("x") as handle:
            json.dump(data, handle, separators=(",", ":"), allow_nan=False)


def main(argv=None):
    from mlsys_sim import closed_loop_capture, vllm_sla_worker

    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", required=True, choices=["pilot"])
    parser.add_argument("--out", required=True, type=Path)
    known, _ = parser.parse_known_args(argv)
    if "--runtime-cost-capture" not in argv:
        parser.error("Native EngineCore capture is required")
    base = vllm_sla_worker._VllmBackend
    collect = closed_loop_capture.collect_closed_loop
    probes = []

    class Backend(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            probe = StageTimeline(self, known.out)
            probes.append(probe)
            probe.attach()

    def recorded(*args, **kwargs):
        try:
            return collect(*args, **kwargs)
        finally:
            while probes:
                probes.pop().close()

    vllm_sla_worker._VllmBackend = Backend
    closed_loop_capture.collect_closed_loop = recorded
    try:
        return vllm_sla_worker.main(argv)
    finally:
        while probes:
            probes.pop().close()
        vllm_sla_worker._VllmBackend = base
        closed_loop_capture.collect_closed_loop = collect


if __name__ == "__main__":
    raise SystemExit(main())
