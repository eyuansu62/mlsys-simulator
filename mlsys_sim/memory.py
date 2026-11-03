"""Fixed-pool memory planning with separate physical and logical budgets.

Freeing request blocks returns logical slots to the preallocated KV pool; it
does not free the live pool allocation from device memory. Reservations are
checked for every TP rank and pipeline stage before any state is committed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .layout import rank_kv_bytes_per_token, rank_weight_bytes, stage_layer_counts
from .schema import Experiment, SimulationError


@dataclass(frozen=True)
class MemoryDecision:
    ok: bool
    code: str | None
    details: dict


@dataclass(frozen=True)
class _RankBudget:
    device_id: str
    stage_index: int
    tp_rank: int
    device_memory_bytes: int
    weights_bytes: int
    physical_reserved_bytes: int
    fixed_budget_bytes: int
    kv_bytes_per_token: int
    kv_block_bytes: int
    pool_blocks: int


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SimulationError("INVALID_CONFIG", f"{name} must be an integer >= {minimum}")
    return value


class MemoryPlanner:
    """Plan safe fixed-pool reservations for the supported dense shard plan."""

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.config = experiment.strategy.memory
        self.allocations: dict[str, int] = {}
        self._ranks: list[_RankBudget] = []
        self._workspace_peaks: dict[str, int] = {}
        self._workspace_sources: dict[str, str] = {}
        self._workspace_last_sources: dict[str, str] = {}
        self._profile_model = None
        if experiment.calibration.backend == "profile":
            from .profiles import ProfileModel
            self._profile_model = ProfileModel(experiment)
        self._peak_used_blocks = 0
        self._validate_memory_config()

        strategy = experiment.strategy
        tp = _integer(strategy.tensor_parallel_size, "tensor_parallel_size", 1)
        pp = _integer(strategy.pipeline_parallel_size, "pipeline_parallel_size", 1)
        if len(experiment.infra.devices) < tp * pp:
            raise SimulationError("INVALID_CONFIG", "insufficient devices for TP × PP layout")
        layers = stage_layer_counts(experiment)
        if len(layers) != pp or any(count <= 0 for count in layers):
            raise SimulationError("INVALID_CONFIG", "each pipeline stage must own at least one layer")
        if sum(layers) != strategy.model.num_layers:
            raise SimulationError("INVALID_CONFIG", "stage layers must cover all model layers")

        for stage in range(pp):
            kv_per_token = rank_kv_bytes_per_token(experiment, stage)
            if kv_per_token <= 0:
                raise SimulationError("INVALID_CONFIG", "KV bytes per token must be positive")
            block_bytes = kv_per_token * self.config.kv_block_tokens
            weights = rank_weight_bytes(experiment, stage)
            # Safety margin and external users are constraints, not allocations
            # owned by this model. Preserve this distinction in the snapshots.
            physical = (weights + self.config.kv_pool_bytes + self.config.graph_bytes
                        + self.config.runtime_bytes + self.config.communication_buffer_bytes)
            fixed_budget = (physical + self.config.other_bytes
                            + self.config.safety_margin_bytes)
            for tp_rank in range(tp):
                device = experiment.infra.devices[stage * tp + tp_rank]
                capacity = _integer(device.memory_bytes, f"{device.id}.memory_bytes", 1)
                if fixed_budget + self.config.workspace_base_bytes > capacity:
                    raise SimulationError(
                        "STATIC_MEMORY_INFEASIBLE",
                        f"{device.id}: fixed budget and base workspace require "
                        f"{fixed_budget + self.config.workspace_base_bytes} bytes, "
                        f"capacity is {capacity}",
                    )
                self._ranks.append(_RankBudget(
                    device_id=device.id,
                    stage_index=stage,
                    tp_rank=tp_rank,
                    device_memory_bytes=capacity,
                    weights_bytes=weights,
                    physical_reserved_bytes=physical,
                    fixed_budget_bytes=fixed_budget,
                    kv_bytes_per_token=kv_per_token,
                    kv_block_bytes=block_bytes,
                    pool_blocks=self.config.kv_pool_bytes // block_bytes,
                ))

    def _validate_memory_config(self) -> None:
        _integer(self.config.kv_block_tokens, "kv_block_tokens", 1)
        _integer(self.config.kv_pool_bytes, "kv_pool_bytes", 1)
        for name in (
            "workspace_base_bytes", "workspace_bytes_per_token", "graph_bytes",
            "runtime_bytes", "communication_buffer_bytes", "other_bytes",
            "safety_margin_bytes", "guard_blocks",
        ):
            _integer(getattr(self.config, name), name)

    def _blocks(self, tokens: int) -> int:
        block = self.config.kv_block_tokens
        return (tokens + block - 1) // block

    def check(
        self,
        target_tokens: Mapping[str, int],
        query_tokens: Sequence[int],
        phase: str,
        alone: bool = False,
    ) -> MemoryDecision:
        """Check a batch without modifying reservations, allocations, or peaks.

        Targets replace the named requests' desired total KV lengths. Other
        active requests continue consuming blocks unless ``alone`` is true.
        A decrease cannot implicitly evict a live request: call ``release`` at
        a safe execution boundary before rebuilding its KV instead.
        """
        if phase not in {"prefill", "decode", "recompute"}:
            raise SimulationError("INVALID_CONFIG", f"unsupported memory phase: {phase}")
        if not isinstance(target_tokens, Mapping):
            raise SimulationError("INVALID_CONFIG", "target_tokens must be a mapping")
        if len(target_tokens) != len(query_tokens):
            raise SimulationError("INVALID_CONFIG", "one query length is required per target request")
        queries = [_integer(value, "query_tokens", 1) for value in query_tokens]
        targets: dict[str, int] = {}
        for request_id, tokens in target_tokens.items():
            if not isinstance(request_id, str) or not request_id:
                raise SimulationError("INVALID_CONFIG", "request_id must be a nonempty string")
            targets[request_id] = _integer(tokens, f"target_tokens[{request_id}]", 1)
            if not alone and tokens < self.allocations.get(request_id, 0):
                raise SimulationError("INVALID_CONFIG", "KV targets cannot shrink a live allocation")

        projected = {} if alone else dict(self.allocations)
        projected.update(targets)
        used_before = sum(self._blocks(tokens) for tokens in self.allocations.values())
        used_after = sum(self._blocks(tokens) for tokens in projected.values())
        workspace = (self.config.workspace_base_bytes
                     + self.config.workspace_bytes_per_token * sum(queries))
        ranks: list[dict] = []
        for rank in self._ranks:
            rank_workspace = workspace
            workspace_source = "analytical_assumption"
            if self._profile_model is not None and queries:
                contexts = ([target - query for target, query in zip(targets.values(), queries)]
                            if phase == "decode" else [0] * len(queries))
                observed_workspace = self._profile_model.workspace_bytes(
                    queries, phase, rank.stage_index, context_lengths=contexts)
                if observed_workspace is not None:
                    rank_workspace = observed_workspace
                    workspace_source = "profile"
            budget = rank.fixed_budget_bytes + rank_workspace
            physical_ok = budget <= rank.device_memory_bytes
            logical_ok = used_after + self.config.guard_blocks <= rank.pool_blocks
            ranks.append({
                "device_id": rank.device_id,
                "stage_index": rank.stage_index,
                "tp_rank": rank.tp_rank,
                "pool_blocks": rank.pool_blocks,
                "used_before_blocks": used_before,
                "proposed_used_blocks": used_after,
                "additional_blocks": used_after - (0 if alone else used_before),
                "guard_blocks": self.config.guard_blocks,
                "workspace_bytes": rank_workspace,
                "workspace_source": workspace_source,
                "required_budget_bytes": budget,
                "device_memory_bytes": rank.device_memory_bytes,
                "physical_ok": physical_ok,
                "logical_ok": logical_ok,
            })
        details = {
            "phase": phase,
            "alone": alone,
            "proposed_allocations": projected,
            "workspace_bytes": max((rank["workspace_bytes"] for rank in ranks), default=workspace),
            "ranks": ranks,
        }
        if any(not rank["physical_ok"] for rank in ranks):
            return MemoryDecision(False, "WORKSPACE_BUDGET_EXCEEDED", details)
        if any(not rank["logical_ok"] for rank in ranks):
            return MemoryDecision(False, "KV_CAPACITY_EXCEEDED", details)
        return MemoryDecision(True, None, details)

    def reserve(
        self,
        target_tokens: Mapping[str, int],
        query_tokens: Sequence[int],
        phase: str,
    ) -> MemoryDecision:
        """Commit a reservation only after every participating rank passes."""
        decision = self.check(target_tokens, query_tokens, phase)
        if decision.ok:
            self.allocations.update(target_tokens)
            for rank in decision.details["ranks"]:
                device_id = rank["device_id"]
                previous = self._workspace_peaks.get(device_id, -1)
                if rank["workspace_bytes"] > previous:
                    self._workspace_peaks[device_id] = rank["workspace_bytes"]
                    self._workspace_sources[device_id] = rank["workspace_source"]
                elif (rank["workspace_bytes"] == previous
                      and self._workspace_sources[device_id] != rank["workspace_source"]):
                    self._workspace_sources[device_id] = "mixed"
                self._workspace_last_sources[device_id] = rank["workspace_source"]
            used = sum(self._blocks(tokens) for tokens in self.allocations.values())
            self._peak_used_blocks = max(self._peak_used_blocks, used)
        return decision

    def release(self, request_id: str) -> None:
        """Return a request's logical blocks, retaining the physical KV pool.

        The event engine is responsible for waiting for in-flight compute and
        transfers before calling this method. Repeated release is harmless.
        """
        self.allocations.pop(request_id, None)

    def snapshot(self) -> list[dict]:
        """Return independent per-rank accounting records and high watermarks."""
        used = sum(self._blocks(tokens) for tokens in self.allocations.values())
        snapshots = []
        for rank in self._ranks:
            free = rank.pool_blocks - used
            workspace_peak = self._workspace_peaks.get(rank.device_id, 0)
            snapshots.append({
                "device_id": rank.device_id,
                "stage_index": rank.stage_index,
                "tp_rank": rank.tp_rank,
                "weights_bytes": rank.weights_bytes,
                "device_memory_bytes": rank.device_memory_bytes,
                "physical_reserved_bytes": rank.physical_reserved_bytes,
                "fixed_budget_bytes": rank.fixed_budget_bytes,
                "kv_pool_reserved_bytes": self.config.kv_pool_bytes,
                "kv_bytes_per_token": rank.kv_bytes_per_token,
                "kv_block_bytes": rank.kv_block_bytes,
                "pool_blocks": rank.pool_blocks,
                "guard_blocks": self.config.guard_blocks,
                "used_blocks": used,
                "free_blocks": free,
                "allocatable_free_blocks": max(0, free - self.config.guard_blocks),
                "kv_used_bytes": used * rank.kv_block_bytes,
                "pool_unusable_tail_bytes": self.config.kv_pool_bytes % rank.kv_block_bytes,
                "peak_used_blocks": self._peak_used_blocks,
                "workspace_peak_bytes": workspace_peak,
                "workspace_source": self._workspace_sources.get(rank.device_id, "not_executed"),
                "workspace_last_source": self._workspace_last_sources.get(rank.device_id, "not_executed"),
                "physical_peak_bytes": rank.physical_reserved_bytes + workspace_peak,
                "budget_peak_bytes": rank.fixed_budget_bytes + workspace_peak,
            })
        return snapshots
