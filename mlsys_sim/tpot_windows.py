"""Population P99 for a fixed closed-loop workload from complete native token windows."""

from collections import defaultdict
from collections.abc import Sequence
import math
import re
from statistics import median
from typing import NoReturn

from .reporting import _percentiles
from .schema import SimulationError
from .workload import canonical_sha256


def _invalid(message: str) -> NoReturn:
    raise SimulationError("INVALID_TPOT_WINDOW_PROFILE", message)


def _sha(value, field):
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        _invalid(f"Expected a SHA-256 digest for {field}")
    return value


def _positive(value):
    if type(value) not in (int, float):
        _invalid("TPOT windows require finite positive durations")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        _invalid("TPOT windows require finite positive durations")
    return value


def _policy(spec, concurrency):
    from .load_validation import validate_closed_loop_spec

    spec = validate_closed_loop_spec(spec)
    if (
        type(concurrency) is not int
        or concurrency not in spec["concurrency_grid"]
        or spec["output_tokens"] < 2
    ):
        _invalid("TPOT requires a declared concurrency and at least two output tokens")
    # Seeds, study labels, grid order and number of replications do not change an individual client policy.
    return canonical_sha256(
        {
            **{
                k: spec[k]
                for k in (
                    "requests_per_run",
                    "input_lengths",
                    "prompt_token_id",
                    "output_tokens",
                    "think_ms",
                    "timeout_ms",
                    "max_run_ms",
                )
            },
            "concurrency": concurrency,
        }
    )


class TpotWindowProfile:
    """Transfer measured whole-request residuals at corresponding baseline TPOT ranks.

    A window is [stream index, TPOT replaying measured shapes, measured server TPOT].
    Requests stay intact across their first and last published tokens. Each fresh native
    process contributes one forecast quantile, irrespective of other processes' spread.
    """

    def __init__(self, evidence: dict, *, allow_test_fixture: bool = False):
        if (
            not isinstance(evidence, dict)
            or evidence.get("kind") != "tpot_window_calibration_v1"
            or evidence.get("role") != "calibration"
        ):
            _invalid("Only a calibration acquisition can supply a TPOT window profile")
        source = evidence.get("source")
        if type(allow_test_fixture) is not bool or (
            source != "measured" and not (source == "test_fixture" and allow_test_fixture)
        ):
            _invalid("TPOT calibration needs a measured source or explicit fixture opt-in")
        self._source = source
        self._base = _sha(evidence.get("base_file_sha256"), "base runtime artifact")
        self._plan = _sha(evidence.get("plan_sha256"), "calibration plan")
        rows = evidence.get("sessions")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
            _invalid("A TPOT profile requires 1..256 declared sessions")
        ids: set[str] = set()
        digests: set[str] = set()
        self._streams: set[str] = set()
        self._effects: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        total_windows = 0
        for row in rows:
            if (
                not isinstance(row, dict)
                or not {"spec", "concurrency", "session_id", "capture_sha256", "stream_sha256", "windows"}
                <= row.keys()
            ):
                _invalid("Incomplete calibration session")
            if row.get("role", "calibration") != "calibration":
                _invalid("A TPOT fit cannot contain a holdout acquisition")
            key = _policy(row["spec"], row["concurrency"])
            session = row["session_id"]
            capture = _sha(row["capture_sha256"], "raw Core trace")
            stream = _sha(row["stream_sha256"], "request stream")
            if (
                not isinstance(session, str)
                or not session
                or session in ids
                or capture in digests
                or (key, stream) in seen
            ):
                _invalid("Repeated session, capture or within-policy request stream")
            ids.add(session)
            digests.add(capture)
            seen.add((key, stream))
            self._streams.add(stream)
            windows = row["windows"]
            count = row["spec"]["requests_per_run"]
            total_windows += count
            if total_windows > 250_000:
                _invalid("TPOT profile exceeds the finite-study request budget")
            if not isinstance(windows, list) or len(windows) != count:
                _invalid("Incomplete request window sample")
            ranked = []
            for item in windows:
                if (
                    not isinstance(item, list)
                    or len(item) != 3
                    or type(item[0]) is not int
                    or not 0 <= item[0] < count
                ):
                    _invalid("Invalid request-window identity")
                index, replay, measured = item
                ranked.append((_positive(replay), index, _positive(measured)))
            if len({index for _, index, _ in ranked}) != count:
                _invalid("Repeated request window")
            self._effects[key].append(tuple(measured - replay for replay, _, measured in sorted(ranked)))
        if any(len(effects) < 2 for effects in self._effects.values()):
            _invalid("Every policy needs at least two independent calibration processes")
        self._checksum = canonical_sha256(evidence)

    @property
    def provenance(self) -> dict:
        return {
            "kind": "ranked_native_tpot_windows_v1",
            "profile_sha256": self._checksum,
            "base_file_sha256": self._base,
            "plan_sha256": self._plan,
            "source": self._source,
            "scope": "finite_cohort_server_token_ready",
            "unit": "ms_per_output_interval",
        }

    def forecast(
        self,
        simulated: Sequence[float],
        spec: dict,
        concurrency: int,
        base_file_sha256: str,
        stream_sha256: str,
    ) -> dict:
        key = _policy(spec, concurrency)
        if base_file_sha256 != self._base or key not in self._effects:
            _invalid("TPOT window profile does not cover this runtime/workload policy")
        if _sha(stream_sha256, "forecast stream") in self._streams:
            _invalid("An independent forecast cannot use a calibration stream")
        if len(simulated) != spec["requests_per_run"]:
            _invalid("Incomplete simulated cohort for TPOT forecast")
        ordered = sorted(_positive(x) for x in simulated)
        estimates = []
        for effects in self._effects[key]:
            corrected = [_positive(x + e) for x, e in zip(ordered, effects, strict=True)]
            value = _percentiles(corrected)["p99"]
            assert value is not None
            estimates.append(value)
        return {
            "p99_ms": median(estimates),
            "base_p99_ms": _percentiles(ordered)["p99"],
            "process_p99_ms": estimates,
            "sessions": len(estimates),
            "requests": len(ordered),
            "profile_sha256": self._checksum,
        }
