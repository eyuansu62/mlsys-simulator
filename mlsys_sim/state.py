"""Shared request lifecycle and public simulation results."""

from dataclasses import dataclass, field

TERMINAL = {"COMPLETED", "FAILED", "TIMED_OUT", "UNFINISHED", "NOT_ADMITTED"}


@dataclass
class SimulationResult:
    result: dict
    requests: list[dict]
    events: list[dict]


@dataclass
class RequestState:
    request_id: str
    arrival_ms: float
    input_tokens: int
    priority: int
    status: str = "CREATED"
    generated: int = 0
    computed: int = 0
    first_token_ms: float | None = None
    completion_ms: float | None = None
    token_times: list[float] = field(default_factory=list)
    preemptions: int = 0
    recomputed_tokens: int = 0
    recompute_ms: float = 0.0
    memory_wait_ms: float = 0.0
    blocked_since: float | None = None
    error_code: str | None = None

    swap_out_count: int = 0
    swap_in_count: int = 0
    swap_out_bytes: int = 0
    swap_in_bytes: int = 0
    swap_ms: float = 0.0
