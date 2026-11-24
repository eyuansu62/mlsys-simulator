"""Inspect the runner forward path in a dedicated diagnostic worker on ROCm."""

from collections import deque
from pathlib import Path
import json
import sys

import c2_tail_probe as outer


class ForwardTimeline(outer.StageTimeline):
    def attach(self):
        super().attach()
        import torch

        executor = self.core.model_executor
        driver = executor.driver_worker
        runner = getattr(driver, "worker", driver).model_runner
        for method in (
            "_update_states",
            "_prepare_inputs",
            "_determine_batch_execution_and_padding",
            "_build_attention_metadata",
            "_preprocess",
        ):
            self.bind(runner, method, method[1:])
        self.bind(runner.model, "compute_logits", "logits")
        self.gpu_rows = []
        self.inflight = deque()
        self.pool = deque(
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(16)
        )
        self.unavailable = []
        self.bind(runner, "_model_forward", "forward")
        original = runner._model_forward
        saved = vars(runner).get("_model_forward", outer._MISSING)

        def timed(*args, **kwargs):
            if len(self.inflight) >= 8:
                self.collect_ready()
            order = self.recorder._order
            if not self.pool:
                self.unavailable.append(order)
                return original(*args, **kwargs)
            start, end = self.pool.popleft()
            stream = torch.cuda.current_stream()
            start.record(stream)
            try:
                return original(*args, **kwargs)
            finally:
                end.record(stream)
                self.inflight.append((order, start, end))

        runner._model_forward = timed
        self.patches.append((runner, "_model_forward", saved, timed))

    def collect_ready(self):
        while self.inflight and self.inflight[0][2].query():
            order, start, end = self.inflight.popleft()
            self.gpu_rows.append([order, start.elapsed_time(end)])
            self.pool.append((start, end))

    def close(self):
        if hasattr(self, "inflight"):
            self.collect_ready()
            with (Path(self.output) / "forward-stream.json").open("x") as handle:
                json.dump(
                    {
                        "session_id": self.recorder.session_id,
                        "role": "diagnostic",
                        "scope": "ROCm events on current stream around runner._model_forward; interval includes stream idle",
                        "synchronization_added": False,
                        "elapsed_ms": self.gpu_rows,
                        "unresolved_orders": [v[0] for v in self.inflight] + self.unavailable,
                    },
                    handle,
                )
        super().close()


if __name__ == "__main__":
    outer.StageTimeline = ForwardTimeline
    raise SystemExit(outer.main(sys.argv[1:]))
