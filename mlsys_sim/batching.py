"""Per-request phase contracts for homogeneous and mixed stage batches."""

from collections.abc import Sequence
from typing import NoReturn

from .schema import SimulationError


def batch_members(phase: str, query_tokens: Sequence[int],
                  request_phases: Sequence[str] | None = None, *,
                  error_code: str = "INVALID_CONFIG") -> tuple[str, ...]:
    """Validate roles without conflating one-token prefill with decode.

    Mixed batches contain both serving phases. Recomputation keeps its own
    homogeneous recovery boundary. Legacy batches omit the additive metadata.
    """
    def invalid(message: str) -> NoReturn:
        raise SimulationError(error_code, message)

    if phase not in ("prefill", "decode", "recompute", "mixed"):
        invalid("phase must be prefill, decode, recompute or mixed")
    if phase == "mixed":
        if (not isinstance(request_phases, (tuple, list))
                or len(request_phases) != len(query_tokens)
                or any(member not in ("prefill", "decode") for member in request_phases)
                or set(request_phases) != {"prefill", "decode"}):
            invalid("mixed batches require both prefill and decode request_phases, one per query")
        members = tuple(request_phases)
    else:
        if request_phases is not None:
            invalid("request_phases is only valid for mixed batches")
        members = (phase,) * len(query_tokens)
    if any(member == "decode" and query != 1 for member, query in zip(members, query_tokens)):
        invalid("decode members must each process one query token")
    return members


def batch_phase(request_phases: Sequence[str]) -> str:
    """Name a nonempty selection; mixed metadata is required only for mixing."""
    if not request_phases:
        raise SimulationError("INVALID_CONFIG", "A batch must contain at least one request")
    return request_phases[0] if len(set(request_phases)) == 1 else "mixed"
