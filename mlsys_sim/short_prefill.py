"""Calibrated medians for ordered, zero-context short Prefill batches."""

from collections import defaultdict
from itertools import product
from statistics import median

from . import runtime_cost as runtime
from .profiles import shape_key

KIND = "runtime_cost_short_prefill_v2"
ALGORITHM = "ordered_short_prefill_median_v1"
LIMIT = 0.05
MIN_STEPS = 4
BATCHES = (1, 2, 4)
NOTE = "Matched short Prefill uses the median of process medians; other steps use the base affine model."


def _shape_table(base, evidence):
    runtime._object(evidence, {"domain_sha256", "plan_sha256", "role", "sessions"}, set(), "shape evidence")
    if evidence["role"] != "calibration" or evidence["domain_sha256"] != base["domain"]["domain_sha256"]:
        runtime._error("Short Prefill requires calibration observations from the same runtime domain")
    runtime._digest(evidence["plan_sha256"], "plan_sha256")
    rules = {row["batch_size"]: row for row in base["policy"]["cells"] if row["phase"] == "prefill"}
    if not isinstance(evidence["sessions"], list) or not evidence["sessions"]:
        runtime._error("Missing short Prefill calibration sessions")
    ids: set[str] = set()
    hashes: set[str] = set()
    grouped: dict[str, list[float]] = defaultdict(list)
    for session in evidence["sessions"]:
        runtime._object(
            session, {"session_id", "batch_size", "capture_sha256", "shapes"}, set(), "shape session"
        )
        sid = runtime._text(session["session_id"], "session_id")
        checksum = runtime._digest(session["capture_sha256"], "capture_sha256")
        if sid in ids or checksum in hashes:
            runtime._error("Each short Prefill calibration session needs an independent capture")
        ids.add(sid)
        hashes.add(checksum)
        batch = runtime._int(session["batch_size"], "batch_size", 1)
        if batch not in BATCHES or batch not in rules:
            runtime._error("Unsupported short Prefill batch size")
        expected = set(product((64, 128), repeat=batch))
        rows = session["shapes"]
        if not isinstance(rows, list) or len(rows) != len(expected):
            runtime._error("Incomplete ordered short Prefill shape inventory")
        for row in rows:
            runtime._object(row, {"query_tokens", "durations_ms"}, set(), "shape observations")
            queries = row["query_tokens"]
            if (
                not isinstance(queries, list)
                or any(type(q) is not int for q in queries)
                or tuple(queries) not in expected
            ):
                runtime._error("Unknown or duplicate ordered short Prefill shape")
            expected.remove(tuple(queries))
            contexts = [0] * batch
            if not runtime._in_bounds(rules[batch], queries, contexts) or not runtime._within_scheduler(
                base["domain"]["strategy"], "prefill", queries
            ):
                runtime._error("Short Prefill shape outside original runtime envelope")
            durations = row["durations_ms"]
            if not isinstance(durations, list) or len(durations) < MIN_STEPS:
                runtime._error("Insufficient observations for a short Prefill session/shape")
            values = [runtime._number(x, "duration_ms", positive=True) for x in durations]
            grouped[shape_key("prefill", queries, contexts)].append(median(values))
    table = {}
    for batch in BATCHES:
        for shape in product((64, 128), repeat=batch):
            key = shape_key("prefill", shape, [0] * batch)
            values = grouped[key]
            if len(values) < 2:
                runtime._error("Two independent calibration sessions required for every short Prefill shape")
            estimate = median(values)
            errors = [abs(estimate - actual) / actual for actual in values]
            if max(errors) > LIMIT:
                runtime._error("Short Prefill session median error exceeds 5% for " + key)
            table[key] = {"duration_ms": estimate, "session_medians_ms": values, "relative_errors": errors}
    return table


def fit_short_prefill(
    base: dict, evidence: dict, *, calibration_id: str, allow_test_fixture: bool = False
) -> dict:
    """Extend an immutable affine artifact using fresh short Prefill calibration sessions."""
    if not isinstance(base, dict) or base.get("kind") != runtime.RUNTIME_COST_KIND:
        runtime._error("Expected a base runtime_cost_model_v1 artifact")
    checked = runtime.validate_runtime_cost_model(base, allow_test_fixture=allow_test_fixture)
    runtime._text(calibration_id, "calibration_id")
    if calibration_id == checked["calibration_id"]:
        runtime._error("Short Prefill calibration requires a distinct calibration ID")
    source = runtime._clone(evidence)
    extension = {
        "base_model_sha256": checked["model_sha256"],
        "base_calibration_id": checked["calibration_id"],
        "evidence": source,
        "table": _shape_table(checked, source),
    }
    checked.update(
        schema_version=2,
        kind=KIND,
        algorithm=ALGORITHM,
        calibration_id=calibration_id,
        limitations=[*runtime._LIMITATIONS, NOTE],
        short_prefill=extension,
    )
    checked["model_sha256"] = runtime._sha({k: v for k, v in checked.items() if k != "model_sha256"})
    return checked


def validate_short_prefill(artifact, experiment, allow_test_fixture):
    if (
        type(artifact.get("schema_version")) is not int
        or artifact["schema_version"] != 2
        or artifact.get("algorithm") != ALGORITHM
        or artifact.get("limitations") != [*runtime._LIMITATIONS, NOTE]
    ):
        runtime._error("Unknown short Prefill schema or algorithm")
    cid = runtime._text(artifact.get("calibration_id"), "calibration_id")
    if runtime._sha({k: v for k, v in artifact.items() if k != "model_sha256"}) != artifact.get(
        "model_sha256"
    ):
        runtime._error("Short Prefill artifact checksum changed")
    extension = runtime._object(
        artifact.get("short_prefill"),
        {"base_model_sha256", "base_calibration_id", "evidence", "table"},
        set(),
        "short Prefill extension",
    )
    if cid == extension["base_calibration_id"]:
        runtime._error("Short Prefill calibration ID collides with base")
    base = {k: v for k, v in artifact.items() if k != "short_prefill"}
    base.update(
        schema_version=1,
        kind=runtime.RUNTIME_COST_KIND,
        algorithm=runtime.RUNTIME_COST_ALGORITHM,
        limitations=runtime._LIMITATIONS,
        calibration_id=extension["base_calibration_id"],
        model_sha256=extension["base_model_sha256"],
    )
    runtime.validate_runtime_cost_model(base, experiment=experiment, allow_test_fixture=allow_test_fixture)
    table = _shape_table(base, extension["evidence"])
    if runtime._canonical(table) != runtime._canonical(extension["table"]):
        runtime._error("Short Prefill table differs from the observed shape medians")
    return runtime._clone(artifact)
