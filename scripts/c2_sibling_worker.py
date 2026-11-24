"""Apply a preregistered CPU pulse schedule in an isolated native worker."""

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def background(pattern):
    if pattern == "copy":
        import ctypes

        size = 256 * 1024
        left = ctypes.create_string_buffer(b"x" * size, size)
        right = ctypes.create_string_buffer(b"y" * size, size)
        left_ptr, right_ptr = ctypes.addressof(left), ctypes.addressof(right)

    os.nice(15)
    print("READY", flush=True)
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd == "QUIT":
            return
        cpu, active = cmd
        os.sched_setaffinity(0, {cpu})
        start = time.perf_counter_ns()
        thread = time.thread_time_ns()
        print(start, flush=True)
        acc = 1
        while active and not select.select([sys.stdin], [], [], 0)[0]:
            if pattern == "copy":
                for _ in range(4):
                    ctypes.memmove(left_ptr, right_ptr, size)
                    ctypes.memmove(right_ptr, left_ptr, size)
            else:
                for i in range(256):
                    acc = (acc * 33 + i) & 65535
            if time.perf_counter_ns() - start > 2_000_000_000:
                raise TimeoutError("CPU pulse exceeded two seconds")
        if sys.stdin.readline().strip() != "STOP":
            raise RuntimeError("Missing pulse stop")
        print(json.dumps([time.perf_counter_ns(), time.thread_time_ns() - thread]), flush=True)


class PulseDriver:
    def __init__(self, schedule):
        self.windows = {row["start"]: row for row in schedule["windows"]}
        self.active = None
        self.records = []
        self.index = -1
        topology = Path("/sys/devices/system/cpu/cpu10/topology/thread_siblings_list").read_text().strip()
        if topology != "10,22" or os.sched_getaffinity(0) != {10}:
            raise ValueError("Worker must be pinned to CPU10 on the measured SMT topology")
        pattern = schedule.get("cpu_pattern", "scalar")
        if pattern not in ("scalar", "copy"):
            raise ValueError("Unknown background workload")
        self.child = subprocess.Popen(
            [sys.executable, "-B", __file__, "--cpu-child", pattern],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.child.stdout.readline().strip() != "READY":
            raise RuntimeError("Pulse worker unavailable")

    def send(self, text):
        self.child.stdin.write(text + "\n")
        self.child.stdin.flush()

    def stop(self):
        self.send("STOP")
        finish, cpu_time = json.loads(self.child.stdout.readline())
        self.active.update(end_ns=finish, child_cpu_ns=cpu_time)
        self.records.append(self.active)
        self.active = None

    def before_step(self):
        self.index += 1
        if self.active and self.index == self.active["stop"]:
            self.stop()
        if self.index in self.windows:
            window = self.windows[self.index]
            if self.active:
                raise ValueError("Overlapping pulse windows")
            self.send(
                json.dumps([23 if window["condition"] == "separate" else 22, window["condition"] != "sham"])
            )
            self.active = {**window, "start_ns": int(self.child.stdout.readline())}

    def close(self):
        try:
            if self.child.poll() is None:
                if self.active:
                    self.stop()
                self.send(json.dumps("QUIT"))
                self.child.wait(timeout=3)
        finally:
            if self.child.poll() is None:
                self.child.kill()
                self.child.wait()
            self.child.stdin.close()
            self.child.stdout.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--cpu-child" and argv[1] in ("scalar", "copy"):
        background(argv[1])
        return 0
    from mlsys_sim import closed_loop_capture, vllm_sla_worker

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pulse-plan", required=True, type=Path)
    extra, worker_args = parser.parse_known_args(argv)
    out = Path(worker_args[worker_args.index("--out") + 1])
    schedule = json.loads(extra.pulse_plan.read_bytes())
    base, collect = vllm_sla_worker._VllmBackend, closed_loop_capture.collect_closed_loop
    drivers = []

    class Backend(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.pulses = PulseDriver(schedule)
            drivers.append(self.pulses)

        def step(self):
            self.pulses.before_step()
            return super().step()

    def collect_with_pulses(*a, **kw):
        try:
            return collect(*a, **kw)
        finally:
            if drivers:
                driver = drivers.pop()
                driver.close()
                with (out / "cpu-pulses.json").open("x") as f:
                    json.dump(
                        {
                            "role": schedule["role"],
                            "windows": driver.records,
                            "api_steps": driver.index + 1,
                            "changed_threads": "one task-owned CPU child",
                        },
                        f,
                    )

    vllm_sla_worker._VllmBackend = Backend
    closed_loop_capture.collect_closed_loop = collect_with_pulses
    try:
        return vllm_sla_worker.main(worker_args)
    finally:
        while drivers:
            drivers.pop().close()
        vllm_sla_worker._VllmBackend = base
        closed_loop_capture.collect_closed_loop = collect


if __name__ == "__main__":
    raise SystemExit(main())
