"""Exact heterogeneous request streams released by real terminal feedback.

One EventDriver owns the deployment for the entire stream. Future requests are
not registered with its scheduler, and request-local failure releases the next
entry without resetting cache, model state, or in-flight resource ownership.
"""

from dataclasses import replace

from .config import validate_experiment
from .event_driver import EventDriver
from .schema import Experiment, RequestSpec, SimulationError
from .state import TERMINAL


def simulate_sequential(experiment: Experiment, requests: tuple[RequestSpec, ...] | list[RequestSpec]) -> dict:
    """Run an exact ordered list in one initially empty resource domain.

    ``arrival_ms`` in every input must be zero: measured ready times are set by
    preceding terminal feedback. ``run`` contains the original engine result and
    only actually admitted requests. ``not_started`` preserves every remaining
    supplied ID when the global execution stops. Static constructor errors are
    raised for the caller to classify; they are not fabricated observations.
    """
    if not isinstance(requests, (tuple, list)) or not requests:
        raise SimulationError("INVALID_CONFIG", "Sequential requests must be a nonempty exact list")
    if (experiment.workload or experiment.execution.warmup
            or experiment.evaluation.measurement_window_ms is not None):
        raise SimulationError("INVALID_CONFIG", "Sequential execution requires empty workload/warmup and no arrival window")
    ordered = tuple(requests)
    validate_experiment(replace(experiment, workload=ordered))
    if any(spec.arrival_ms != 0 for spec in ordered):
        raise SimulationError("INVALID_CONFIG", "Sequential input arrivals must be zero; terminal feedback sets ready time")
    driver = EventDriver(experiment)
    offered = 0
    active: str | None = None
    reason: str | None = None
    while True:
        if driver.stopped:
            reason = driver.stop_reason or "SEQUENTIAL_STOPPED"
            break
        if active is not None and driver.requests[active].status in TERMINAL:
            active = None
        if active is None and offered < len(ordered):
            if driver.now >= driver.cutoff_ms:
                reason = driver.cutoff_reason
                break
            spec = replace(ordered[offered], arrival_ms=driver.now)
            try:
                driver.accept_request(spec)
            except SimulationError as error:
                if error.code not in {"SIMULATION_EVENT_LIMIT", "NO_AVAILABLE_REPLICA"}:
                    raise
                reason = error.code
                break
            offered += 1
            active = spec.request_id
        driver.schedule_ready()
        if driver.stopped:
            reason = driver.stop_reason or "SEQUENTIAL_STOPPED"
            break
        # Admission may fail a request synchronously (for example an insufficient
        # KV pool). It remains a real terminal row, then the next entry is tried.
        if active is not None and driver.requests[active].status in TERMINAL:
            continue
        event = driver.next_event()
        if event is None:
            if driver.has_unfinished() or driver.has_inflight() or offered < len(ordered):
                reason = "SEQUENTIAL_NO_PROGRESS"
            break
        when = event[0]
        driver.set_time(when)
        # Completion and timeout events at a timestamp all commit before the
        # next request is released. The cutoff is inclusive for existing work.
        while (event := driver.next_event()) is not None and event[0] == when:
            driver.advance_one()
            if driver.stopped:
                break
    run = driver.finalize(reason=reason)
    return {
        "run": run,
        "first_compute_start_ms": dict(driver.first_compute_start_ms),
        "not_started": [spec.request_id for spec in ordered[offered:]],
        "stop_reason": reason or driver.stop_reason,
        "resources_drained": (not driver.has_inflight() and not driver.has_unfinished()
                              and all(row["status"] != "UNFINISHED" for row in run.requests)),
        "elapsed_ms": run.result["simulated_ms"],
    }
