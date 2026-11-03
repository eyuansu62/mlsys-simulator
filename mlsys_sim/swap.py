"""Conservative host KV swap resources and a serialized bidirectional DMA lane.

Only byte counts and ownership are simulated. Source and destination storage
remain pinned through completion; cancellation never frees an in-flight copy.
The caller must establish a safe compute/pipeline boundary before swap_out.
"""

from dataclasses import asdict, dataclass
import math

from .memory import MemoryPlanner
from .schema import Experiment, SimulationError


@dataclass(frozen=True)
class SwapTransfer:
    transfer_id: int
    request_id: str
    direction: str
    tokens: int
    bytes: int
    start_ms: float
    end_ms: float


@dataclass
class _Entry:
    tokens: int
    bytes: int
    state: str
    transfer_id: int | None
    cancelled: bool = False


def _number(value: object, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationError("INVALID_CONFIG", f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise SimulationError("INVALID_CONFIG", f"{name} must be finite") from exc
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        raise SimulationError("INVALID_CONFIG", f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return result


class HostSwapManager:
    """Reserve a bounded host pool and schedule block-aligned KV copies.

    Both directions share one serial DMA resource. A queued/cancelled operation
    still runs to its planned boundary; engines may not reuse its buffers early.
    GPU reservations use the existing fixed pool and never shrink that pool.
    """

    def __init__(self, experiment: Experiment, memory: MemoryPlanner):
        self.memory = memory
        config = experiment.strategy.memory
        capacity = config.cpu_swap_bytes
        if type(capacity) is not int or not 0 <= capacity <= 2**63 - 1:
            raise SimulationError("INVALID_CONFIG", "cpu_swap_bytes must be a nonnegative signed 64-bit integer")
        self.capacity = capacity
        self.d2h_bandwidth_gbps = _number(config.swap_d2h_bandwidth_gbps, "swap_d2h_bandwidth_gbps")
        self.h2d_bandwidth_gbps = _number(config.swap_h2d_bandwidth_gbps, "swap_h2d_bandwidth_gbps")
        self.latency_ms = _number(config.swap_latency_us, "swap_latency_us", zero=True) / 1000
        self._entries: dict[str, _Entry] = {}
        self._pending: dict[int, SwapTransfer] = {}
        self._next_id = 1
        self._tail_ms = 0.0
        self._last_completed_ms = 0.0
        self._host_reserved_bytes = 0
        self._host_peak_bytes = 0
        self._completed_busy_ms = 0.0
        self._completed_bytes = {"D2H": 0, "H2D": 0}
        self._completed_counts = {"D2H": 0, "H2D": 0}
        self._cancelled_transfer_bytes = 0

    @property
    def busy_until_ms(self) -> float:
        return self._tail_ms

    @property
    def host_reserved_bytes(self) -> int:
        return self._host_reserved_bytes

    @property
    def pending_transfers(self) -> tuple[SwapTransfer, ...]:
        return tuple(self._pending[key] for key in sorted(self._pending))

    def _time(self, now_ms: float) -> float:
        now = _number(now_ms, "now_ms", zero=True)
        if now < self._last_completed_ms:
            raise SimulationError("INVALID_SWAP_STATE", "Swap time cannot precede a committed completion")
        return now

    @staticmethod
    def _id(request_id: str) -> str:
        if not isinstance(request_id, str) or not request_id:
            raise SimulationError("INVALID_SWAP_STATE", "request_id must be a nonempty string")
        return request_id

    def _plan(self, request_id: str, direction: str, tokens: int, size: int, now_ms: float) -> SwapTransfer:
        now = self._time(now_ms)
        bandwidth = self.d2h_bandwidth_gbps if direction == "D2H" else self.h2d_bandwidth_gbps
        rate = bandwidth * 1e6  # Decimal GB/s -> bytes/ms.
        try:
            duration = self.latency_ms + size / rate
        except (ZeroDivisionError, OverflowError) as exc:
            raise SimulationError("INVALID_SWAP_ESTIMATE", "Swap duration is not representable") from exc
        start = max(now, self._tail_ms)
        end = start + duration
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(duration) or duration <= 0:
            raise SimulationError("INVALID_SWAP_ESTIMATE", "Swap rate/duration must remain positive and finite")
        if not math.isfinite(end) or end <= start:
            raise SimulationError("INVALID_SWAP_ESTIMATE", "Swap duration cannot advance the virtual clock")
        return SwapTransfer(self._next_id, request_id, direction, tokens, size, start, end)

    def _enqueue(self, transfer: SwapTransfer) -> None:
        self._pending[transfer.transfer_id] = transfer
        self._next_id += 1
        self._tail_ms = transfer.end_ms

    def swap_out(self, request_id: str, now_ms: float) -> SwapTransfer:
        request_id = self._id(request_id)
        if request_id in self._entries:
            raise SimulationError("INVALID_SWAP_STATE", "Request already has host ownership or a pending copy")
        tokens = self.memory.allocations.get(request_id)
        if tokens is None:
            raise SimulationError("INVALID_SWAP_STATE", "Cannot swap a request without GPU KV")
        size = self.memory.kv_bytes_for_tokens(tokens)
        if self._host_reserved_bytes + size > self.capacity:
            raise SimulationError("HOST_SWAP_CAPACITY_EXCEEDED", "Host KV capacity cannot reserve this request")
        transfer = self._plan(request_id, "D2H", tokens, size, now_ms)
        self._entries[request_id] = _Entry(tokens, size, "SWAPPING_OUT", transfer.transfer_id)
        self._host_reserved_bytes += size
        self._host_peak_bytes = max(self._host_peak_bytes, self._host_reserved_bytes)
        self._enqueue(transfer)
        return transfer

    def swap_in(self, request_id: str, now_ms: float) -> SwapTransfer:
        request_id = self._id(request_id)
        entry = self._entries.get(request_id)
        if entry is None or entry.state != "SWAPPED" or entry.cancelled:
            raise SimulationError("INVALID_SWAP_STATE", "A complete, live host copy is required for restore")
        if request_id in self.memory.allocations:
            raise SimulationError("INVALID_SWAP_STATE", "Restore cannot replace an existing GPU allocation")
        transfer = self._plan(request_id, "H2D", entry.tokens, entry.bytes, now_ms)
        decision = self.memory.reserve_kv({request_id: entry.tokens})
        if not decision.ok:
            raise SimulationError(decision.code or "KV_CAPACITY_EXCEEDED", "GPU KV destination cannot be reserved")
        entry.state, entry.transfer_id = "SWAPPING_IN", transfer.transfer_id
        self._enqueue(transfer)
        return transfer

    def complete(self, transfer: SwapTransfer, now_ms: float) -> dict:
        if not isinstance(transfer, SwapTransfer) or self._pending.get(transfer.transfer_id) != transfer:
            raise SimulationError("INVALID_SWAP_STATE", "Unknown, forged or already completed swap transfer")
        now = self._time(now_ms)
        if now < transfer.end_ms:
            raise SimulationError("INVALID_SWAP_STATE", "Transfer has not reached its completion boundary")
        if transfer.transfer_id != min(self._pending):
            raise SimulationError("INVALID_SWAP_STATE", "Earlier serialized transfers must complete first")
        entry = self._entries.get(transfer.request_id)
        if entry is None or entry.transfer_id != transfer.transfer_id:
            raise SimulationError("INVALID_SWAP_STATE", "Transfer ownership is inconsistent")
        if self.memory.allocations.get(transfer.request_id) != transfer.tokens:
            raise SimulationError("INVALID_SWAP_STATE", "Pinned GPU KV changed before copy completion")

        self._pending.pop(transfer.transfer_id)
        self._last_completed_ms = now
        self._completed_busy_ms += transfer.end_ms - transfer.start_ms
        self._completed_bytes[transfer.direction] += transfer.bytes
        self._completed_counts[transfer.direction] += 1
        if entry.cancelled:
            self._cancelled_transfer_bytes += transfer.bytes
            self.memory.release(transfer.request_id)
            self._free_host(transfer.request_id)
        elif transfer.direction == "D2H":
            self.memory.release(transfer.request_id)
            entry.state, entry.transfer_id = "SWAPPED", None
        else:
            self._free_host(transfer.request_id)
        return {"request_id": transfer.request_id, "direction": transfer.direction,
                "tokens": transfer.tokens, "bytes": transfer.bytes, "cancelled": entry.cancelled,
                "start_ms": transfer.start_ms, "end_ms": transfer.end_ms,
                "duration_ms": transfer.end_ms - transfer.start_ms}

    def _free_host(self, request_id: str) -> None:
        entry = self._entries.pop(request_id)
        self._host_reserved_bytes -= entry.bytes

    def cancel(self, request_id: str) -> None:
        request_id = self._id(request_id)
        entry = self._entries.get(request_id)
        if entry is None:
            return
        if entry.state == "SWAPPED":
            self._free_host(request_id)
        else:
            entry.cancelled = True

    def state(self, request_id: str) -> str | None:
        request_id = self._id(request_id)
        entry = self._entries.get(request_id)
        return entry.state if entry is not None else None

    def stored_tokens(self, request_id: str) -> int:
        request_id = self._id(request_id)
        entry = self._entries.get(request_id)
        if entry is None:
            raise SimulationError("INVALID_SWAP_STATE", "Request has no host KV ownership")
        return entry.tokens

    def snapshot(self, now_ms: float) -> dict:
        """Observe resource state and executed copy fragments without mutation."""
        now = self._time(now_ms)
        pending = self.pending_transfers
        partial_busy = sum(max(0.0, min(now, transfer.end_ms) - transfer.start_ms)
                           for transfer in pending)
        progressing_bytes = {"D2H": 0.0, "H2D": 0.0}
        for transfer in pending:
            if now >= transfer.end_ms:
                copied = float(transfer.bytes)
            else:
                rate = (self.d2h_bandwidth_gbps if transfer.direction == "D2H"
                        else self.h2d_bandwidth_gbps) * 1e6
                copied = min(float(transfer.bytes), max(0.0, now - transfer.start_ms - self.latency_ms) * rate)
            progressing_bytes[transfer.direction] += copied
        return {
            "host_capacity_bytes": self.capacity,
            "host_reserved_bytes": self._host_reserved_bytes,
            "host_peak_reserved_bytes": self._host_peak_bytes,
            "host_free_bytes": self.capacity - self._host_reserved_bytes,
            "completed_bytes_by_direction": dict(self._completed_bytes),
            "completed_transfers_by_direction": dict(self._completed_counts),
            "pending_transferred_bytes_by_direction": progressing_bytes,
            "dma_busy_ms": self._completed_busy_ms + partial_busy,
            "completed_dma_busy_ms": self._completed_busy_ms,
            "cancelled_transfer_bytes": self._cancelled_transfer_bytes,
            "busy_until_ms": self._tail_ms,
            "pending_transfers": [asdict(transfer) for transfer in pending],
            "entries": [{"request_id": request_id, **asdict(entry)}
                        for request_id, entry in sorted(self._entries.items())],
            "model": "serialized_bidirectional_dma_no_early_buffer_reuse",
            "source": "analytical_assumption",
            "block_aligned_kv_bytes": True,
        }
