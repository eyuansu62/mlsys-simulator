"""Fixed-pool memory planning with separate physical and logical budgets.

Freeing request blocks returns logical slots to the preallocated KV pool; it
does not free the live pool allocation from device memory. Reservations are
checked for every TP rank and pipeline stage before any state is committed.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from .batching import batch_members
import hashlib

from .layout import rank_kv_bytes_per_token, rank_weight_bytes, stage_layer_counts
from .schema import Experiment, RequestSpec, SimulationError


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
    pipeline_buffer_reserved_bytes: int
    pipeline_buffer_breakdown: dict | None = None


_PrefixKey = tuple[str, int, bytes]


@dataclass
class _PrefixBlock:
    references: set[str]
    last_used: int


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SimulationError("INVALID_CONFIG", f"{name} must be an integer >= {minimum}")
    return value


class MemoryPlanner:
    """Plan safe fixed-pool reservations for the supported dense shard plan."""

    def __init__(self, experiment: Experiment):
        if experiment.strategy.replicas != 1:
            raise SimulationError("UNSUPPORTED_MECHANISM", "MemoryPlanner requires one replica; slice the experiment first")
        self.experiment = experiment
        self.config = experiment.strategy.memory
        self.allocations: dict[str, int] = {}
        self._prefix_enabled = experiment.strategy.prefix_cache
        self._prefix_capacity = _integer(experiment.strategy.prefix_cache_max_blocks,
                                         "prefix_cache_max_blocks", 1)
        self._prefix_specs = {request.request_id: request for request in experiment.workload}
        self._prefix_entries: dict[_PrefixKey, _PrefixBlock] = {}
        self._prefix_attached: dict[str, list[_PrefixKey]] = {}
        self._prefix_checked: set[str] = set()
        self._prefix_lookup_hits: dict[str, int] = {}
        self._prefix_committed_requests: set[str] = set()
        self._prefix_request_keys: dict[str, tuple[_PrefixKey, ...]] = {}
        self._prefix_clock = 0
        self._prefix_counts = {name: 0 for name in (
            "lookup_requests", "hit_requests", "miss_requests", "bypassed_requests", "hit_tokens",
            "published_blocks", "deduplicated_blocks", "evicted_blocks", "peak_cached_blocks",
            "abandoned_lookups", "abandoned_hit_tokens",
        )}
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
        _integer(strategy.pipeline_max_inflight_microbatches, "pipeline_max_inflight_microbatches", 1)
        _integer(strategy.max_batch_tokens, "max_batch_tokens", 1)
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
            boundaries = int(stage > 0) + int(stage < pp - 1)
            native_buffers = None
            if strategy.pipeline_transport == "vllm_native":
                from .pp_transport import native_pp_descriptor, pipeline_transport_max_tokens
                descriptor = native_pp_descriptor(experiment, pipeline_transport_max_tokens(experiment))
                resident = descriptor["resident_bytes_per_rank"]
                inflight = strategy.pipeline_max_inflight_microbatches
                # irecv allocates full backing before slicing. The callback's
                # default argument keeps that backing alive after AllGather has
                # allocated another full tensor. Runner additionally owns a
                # persistent full input-copy target on every receiving stage.
                outgoing = resident * inflight if stage < pp - 1 else 0
                incoming = resident * inflight if stage > 0 else 0
                backing = (sum(tensor["full_bytes"] for tensor in descriptor["tensors"]
                           if tensor["tp_sliced"]) * inflight if stage > 0 else 0)
                persistent = resident if stage > 0 else 0
                native_buffers = {
                    "scope": "native_pp_full_resident_per_rank_upper_bound",
                    "outgoing_inflight_bytes": outgoing,
                    "incoming_inflight_bytes": incoming,
                    "incoming_slice_backing_bytes": backing,
                    "persistent_input_copy_bytes": persistent,
                    "max_padded_tokens": descriptor["padded_tokens"],
                }
                pipeline_buffers = outgoing + incoming + backing + persistent
            else:
                pipeline_buffers = int(
                    boundaries * strategy.pipeline_max_inflight_microbatches
                    * strategy.max_batch_tokens * strategy.model.hidden_size
                    * strategy.model.activation_bytes)
            # Safety margin and external users are constraints, not allocations
            # owned by this model. Preserve this distinction in the snapshots.
            physical = (weights + self.config.kv_pool_bytes + self.config.graph_bytes
                        + self.config.runtime_bytes + self.config.communication_buffer_bytes
                        + pipeline_buffers)
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
                    pipeline_buffer_reserved_bytes=pipeline_buffers,
                    pipeline_buffer_breakdown=native_buffers,
                ))

    def _validate_memory_config(self) -> None:
        _integer(self.config.kv_block_tokens, "kv_block_tokens", 1)
        _integer(self.config.kv_pool_bytes, "kv_pool_bytes", 1)
        for name in (
            "workspace_base_bytes", "workspace_bytes_per_token", "graph_bytes",
            "runtime_bytes", "communication_buffer_bytes", "other_bytes",
            "safety_margin_bytes", "guard_blocks",
            "cpu_swap_bytes",
        ):
            _integer(getattr(self.config, name), name)

    def _blocks(self, tokens: int) -> int:
        block = self.config.kv_block_tokens
        return (tokens + block - 1) // block

    def _touch_prefix(self, key: _PrefixKey) -> None:
        self._prefix_clock += 1
        self._prefix_entries[key].last_used = self._prefix_clock

    def register_request(self, request: RequestSpec) -> None:
        """Register the identity of a request dynamically assigned to this replica."""
        if not isinstance(request, RequestSpec) or not isinstance(request.request_id, str) or not request.request_id:
            raise SimulationError("INVALID_CONFIG", "A request with a nonempty string ID is required")
        _integer(request.input_tokens, "input_tokens", 1)
        _integer(request.output_tokens, "output_tokens", 1)
        previous = self._prefix_specs.get(request.request_id)
        if previous is not None and previous != request:
            raise SimulationError("INVALID_PREFIX_STATE", "A registered request identity cannot change")
        if request.prompt_token_ids:
            # Validate without retaining cache references or changing LRU state.
            self._validate_prefix_identity(request)
        self._prefix_specs[request.request_id] = request

    def _validate_prefix_identity(self, request: RequestSpec) -> None:
        tokens = request.prompt_token_ids
        if (not isinstance(tokens, (tuple, list)) or len(tokens) != request.input_tokens
                or not isinstance(request.cache_namespace, str)
                or any(type(token) is not int or not 0 <= token < min(
                    2**63, self.experiment.strategy.model.vocab_size) for token in tokens)):
            raise SimulationError("INVALID_CONFIG", "Prefix identity requires one valid vocabulary token per input token")

    def _iter_prefix_keys(self, request: RequestSpec) -> Iterator[_PrefixKey]:
        self._validate_prefix_identity(request)
        tokens = request.prompt_token_ids
        block = self.config.kv_block_tokens
        digest = b""
        for end in range(block, request.input_tokens + 1, block):
            payload = b"".join(token.to_bytes(8, "big") for token in tokens[end - block:end])
            digest = hashlib.sha256(digest + payload).digest()
            yield request.cache_namespace, end, digest

    def _request_prefix_keys(self, request_id: str) -> tuple[_PrefixKey, ...]:
        """Hash each block with its parent digest, using linear identity storage."""
        if request_id in self._prefix_request_keys:
            return self._prefix_request_keys[request_id]
        result = tuple(self._iter_prefix_keys(self._prefix_specs[request_id]))
        self._prefix_request_keys[request_id] = result
        return result

    def peek_prefix_tokens(self, request: RequestSpec) -> int:
        """Observe a local cache match without registration, pinning or LRU changes."""
        if not self._prefix_enabled or not request.prompt_token_ids:
            return 0
        tokens = 0
        for key in self._iter_prefix_keys(request):
            if key[1] >= request.input_tokens or key not in self._prefix_entries:
                break
            tokens = key[1]
        return tokens

    def _idle_prefix_keys(self) -> list[_PrefixKey]:
        return sorted((key for key, entry in self._prefix_entries.items() if not entry.references),
                      key=lambda key: self._prefix_entries[key].last_used)

    def _evict_prefix_blocks(self, count: int) -> None:
        keys = self._idle_prefix_keys()
        if count > len(keys):
            raise SimulationError("RESOURCE_STATE_CHANGED", "Planned cache blocks are no longer evictable")
        for key in keys[:count]:
            del self._prefix_entries[key]
        self._prefix_counts["evicted_blocks"] += count

    def _used_blocks(self, allocations: Mapping[str, int]) -> int:
        private = sum(self._blocks(tokens) - len(self._prefix_attached.get(request_id, ()))
                      for request_id, tokens in allocations.items())
        return len(self._prefix_entries) + private

    def _projected_usage(self, allocations: Mapping[str, int], alone: bool) -> tuple[int, int]:
        if alone:
            return sum(self._blocks(tokens) for tokens in allocations.values()), 0
        used = self._used_blocks(allocations)
        capacity = min(rank.pool_blocks for rank in self._ranks) - self.config.guard_blocks
        evictions = min(max(0, used - capacity), len(self._idle_prefix_keys()))
        return used - evictions, evictions

    def _record_block_peak(self) -> None:
        self._peak_used_blocks = max(self._peak_used_blocks, self._used_blocks(self.allocations))

    def lookup_prefix(self, request_id: str) -> int:
        """Attach immutable cached prompt blocks, leaving a token for logits.

        Only explicit token IDs in the same namespace can match. A lookup is
        performed once per request; repeated calls observe its current attached
        prefix without changing the counters or LRU order. Generated tokens and
        incomplete blocks are never cached.
        """
        if not self._prefix_enabled:
            return 0
        if request_id in self._prefix_checked:
            return min(self._prefix_lookup_hits.get(request_id, 0),
                       len(self._prefix_attached.get(request_id, ())) * self.config.kv_block_tokens)
        if request_id in self.allocations:
            raise SimulationError("INVALID_PREFIX_STATE", "Prefix lookup must precede request allocation")
        request = self._prefix_specs.get(request_id)
        request_keys = (self._request_prefix_keys(request_id)
                        if request is not None and request.prompt_token_ids else ())
        self._prefix_checked.add(request_id)
        self._prefix_counts["lookup_requests"] += 1
        if request is None or not request.prompt_token_ids:
            self._prefix_counts["bypassed_requests"] += 1
            return 0
        block = self.config.kv_block_tokens
        keys: list[_PrefixKey] = []
        for key in request_keys:
            if key[1] >= request.input_tokens:
                break
            if key not in self._prefix_entries:
                break
            keys.append(key)
        tokens = len(keys) * block
        self._prefix_lookup_hits[request_id] = tokens
        if keys:
            self.allocations[request_id] = tokens
            self._prefix_attached[request_id] = keys
            for key in keys:
                self._prefix_entries[key].references.add(request_id)
                self._touch_prefix(key)
            self._record_block_peak()
            self._prefix_counts["hit_requests"] += 1
            self._prefix_counts["hit_tokens"] += tokens
        else:
            self._prefix_counts["miss_requests"] += 1
        return tokens

    def abandon_prefix_lookup(self, request_id: str) -> None:
        """Roll back a hit whose request has not been admitted to computation.

        A failed trial admission must not pin idle cache blocks and block an
        incumbent decoder. Committed reservations and publications make this
        operation invalid; engines must never use it for an admitted request.
        """
        hit = self._prefix_lookup_hits.get(request_id, 0)
        if not hit:
            return
        if (request_id in self._prefix_committed_requests
                or self.allocations.get(request_id) != hit
                or len(self._prefix_attached.get(request_id, ())) * self.config.kv_block_tokens != hit):
            raise SimulationError("INVALID_PREFIX_STATE", "Only an unadmitted prefix lookup can be abandoned")
        self.release(request_id)
        self._prefix_checked.remove(request_id)
        self._prefix_lookup_hits.pop(request_id)
        self._prefix_counts["lookup_requests"] -= 1
        self._prefix_counts["hit_requests"] -= 1
        self._prefix_counts["hit_tokens"] -= hit
        self._prefix_counts["abandoned_lookups"] += 1
        self._prefix_counts["abandoned_hit_tokens"] += hit

    def publish_prefix(self, request_id: str, computed_tokens: int) -> int:
        """Replace complete private prompt blocks with shared cache references.

        Engines call this only at a safe compute/feedback boundary. Publishing
        changes ownership, without copying bytes or adding physical blocks. If
        every cache entry is pinned, additional prompt blocks remain private.
        The return value counts newly attached blocks, including deduplication.
        """
        computed = _integer(computed_tokens, "computed_tokens")
        if not self._prefix_enabled:
            return 0
        request = self._prefix_specs.get(request_id)
        if request is None or not request.prompt_token_ids:
            return 0
        if computed > self.allocations.get(request_id, 0):
            raise SimulationError("INVALID_PREFIX_STATE", "Cannot publish KV outside a live allocation")
        request_keys = self._request_prefix_keys(request_id)
        attached = self._prefix_attached.setdefault(request_id, [])
        published = 0
        self._prefix_committed_requests.add(request_id)
        for key in request_keys:
            if key[1] > min(computed, request.input_tokens):
                break
            if key in attached:
                continue
            if key not in self._prefix_entries:
                if len(self._prefix_entries) >= self._prefix_capacity:
                    if not self._idle_prefix_keys():
                        continue
                    self._evict_prefix_blocks(1)
                self._prefix_entries[key] = _PrefixBlock(set(), 0)
                self._prefix_counts["published_blocks"] += 1
                self._prefix_counts["peak_cached_blocks"] = max(
                    self._prefix_counts["peak_cached_blocks"], len(self._prefix_entries))
            else:
                self._prefix_counts["deduplicated_blocks"] += 1
            self._prefix_entries[key].references.add(request_id)
            attached.append(key)
            self._touch_prefix(key)
            published += 1
        return published

    def prefix_snapshot(self) -> dict:
        """Observe cache identity-independent counters and block ownership."""
        pinned = sum(bool(entry.references) for entry in self._prefix_entries.values())
        shared = sum(len(entry.references) > 1 for entry in self._prefix_entries.values())
        eligible = self._prefix_counts["hit_requests"] + self._prefix_counts["miss_requests"]
        return {
            "enabled": self._prefix_enabled, "capacity_blocks": self._prefix_capacity,
            **self._prefix_counts,
            "cached_blocks": len(self._prefix_entries), "pinned_blocks": pinned,
            "evictable_blocks": len(self._prefix_entries) - pinned, "shared_blocks": shared,
            "attached_block_references": sum(len(keys) for keys in self._prefix_attached.values()),
            "private_blocks": self._used_blocks(self.allocations) - len(self._prefix_entries),
            "hit_rate": self._prefix_counts["hit_requests"] / eligible if eligible else None,
            "block_tokens": self.config.kv_block_tokens,
            "policy": "full_prompt_blocks_sha256_ancestry_and_namespace_lru_unreferenced_only",
            "identity": "sha256_parent_digest_and_unsigned_8byte_big_endian_block_token_ids",
            "entries": [{"prefix_tokens": key[1], "reference_count": len(entry.references),
                         "request_ids": sorted(entry.references), "last_used": entry.last_used}
                        for key, entry in sorted(self._prefix_entries.items(),
                                                 key=lambda item: item[1].last_used)],
        }

    def kv_bytes_for_tokens(self, tokens: int) -> int:
        """Block-aligned KV copy bytes, summed over all stages and TP ranks.

        This includes replicated GQA KV heads and the unused tail of the last
        allocated block. It describes copy traffic, not an extra GPU allocation.
        """
        tokens = _integer(tokens, "tokens")
        return self._blocks(tokens) * sum(rank.kv_block_bytes for rank in self._ranks)

    def check_kv(self, target_tokens: Mapping[str, int], alone: bool = False) -> MemoryDecision:
        """Check only logical KV slots for a DMA destination, without compute.

        GPU physical capacity for the fixed pool is established at construction.
        No synthetic query or Profile lookup is performed for a byte copy.
        """
        if not isinstance(target_tokens, Mapping):
            raise SimulationError("INVALID_CONFIG", "target_tokens must be a mapping")
        if type(alone) is not bool:
            raise SimulationError("INVALID_CONFIG", "alone must be a boolean")
        targets: dict[str, int] = {}
        for request_id, tokens in target_tokens.items():
            if not isinstance(request_id, str) or not request_id:
                raise SimulationError("INVALID_CONFIG", "request_id must be a nonempty string")
            targets[request_id] = _integer(tokens, f"target_tokens[{request_id}]", 1)
            if not alone and tokens < self.allocations.get(request_id, 0):
                raise SimulationError("INVALID_CONFIG", "KV targets cannot shrink a live allocation")
        projected = {} if alone else dict(self.allocations)
        projected.update(targets)
        used_before = self._used_blocks(self.allocations)
        used_after, cache_evictions = self._projected_usage(projected, alone)
        ranks = [{"device_id": rank.device_id, "stage_index": rank.stage_index,
                  "tp_rank": rank.tp_rank, "pool_blocks": rank.pool_blocks,
                  "used_before_blocks": used_before, "proposed_used_blocks": used_after,
                  "additional_blocks": used_after - (0 if alone else used_before),
                  "guard_blocks": self.config.guard_blocks, "workspace_bytes": 0,
                  "workspace_source": "not_required_for_copy",
                  "required_budget_bytes": rank.fixed_budget_bytes,
                  "device_memory_bytes": rank.device_memory_bytes,
                  "physical_ok": rank.fixed_budget_bytes <= rank.device_memory_bytes,
                  "logical_ok": used_after + self.config.guard_blocks <= rank.pool_blocks}
                 for rank in self._ranks]
        details = {"phase": "kv_copy", "alone": alone, "proposed_allocations": projected,
                   "cache_eviction_blocks": cache_evictions,
                   "workspace_bytes": 0, "ranks": ranks}
        if any(not rank["physical_ok"] for rank in ranks):
            return MemoryDecision(False, "STATIC_MEMORY_INFEASIBLE", details)
        if any(not rank["logical_ok"] for rank in ranks):
            return MemoryDecision(False, "KV_CAPACITY_EXCEEDED", details)
        return MemoryDecision(True, None, details)

    def reserve_kv(self, target_tokens: Mapping[str, int]) -> MemoryDecision:
        """Atomically reserve a copy destination without changing Workspace peaks."""
        decision = self.check_kv(target_tokens)
        if decision.ok:
            self._evict_prefix_blocks(decision.details["cache_eviction_blocks"])
            self.allocations.update(target_tokens)
            self._prefix_committed_requests.update(target_tokens)
            self._record_block_peak()
        return decision

    def check(
        self,
        target_tokens: Mapping[str, int],
        query_tokens: Sequence[int],
        phase: str,
        alone: bool = False,
        *,
        request_phases: Sequence[str] | None = None,
    ) -> MemoryDecision:
        """Check a batch without modifying reservations, allocations, or peaks.

        Targets replace the named requests' desired total KV lengths. Other
        active requests continue consuming blocks unless ``alone`` is true.
        A decrease cannot implicitly evict a live request: call ``release`` at
        a safe execution boundary before rebuilding its KV instead.
        """
        if not isinstance(target_tokens, Mapping):
            raise SimulationError("INVALID_CONFIG", "target_tokens must be a mapping")
        if len(target_tokens) != len(query_tokens):
            raise SimulationError("INVALID_CONFIG", "one query length is required per target request")
        queries = [_integer(value, "query_tokens", 1) for value in query_tokens]
        batch_members(phase, queries, request_phases)
        targets: dict[str, int] = {}
        for request_id, tokens in target_tokens.items():
            if not isinstance(request_id, str) or not request_id:
                raise SimulationError("INVALID_CONFIG", "request_id must be a nonempty string")
            targets[request_id] = _integer(tokens, f"target_tokens[{request_id}]", 1)
            if not alone and tokens < self.allocations.get(request_id, 0):
                raise SimulationError("INVALID_CONFIG", "KV targets cannot shrink a live allocation")

        if any(query > target for query, target in zip(queries, targets.values())):
            raise SimulationError("INVALID_CONFIG", "A KV target must include its query tokens")
        projected = {} if alone else dict(self.allocations)
        projected.update(targets)
        used_before = self._used_blocks(self.allocations)
        used_after, cache_evictions = self._projected_usage(projected, alone)
        workspace = (self.config.workspace_base_bytes
                     + self.config.workspace_bytes_per_token * sum(queries))
        ranks: list[dict] = []
        for rank in self._ranks:
            rank_workspace = workspace
            workspace_source = "analytical_assumption"
            if self._profile_model is not None and queries:
                contexts = [target - query for target, query in zip(targets.values(), queries)]
                if phase == "mixed":
                    observed_workspace = self._profile_model.workspace_bytes(
                        queries, phase, rank.stage_index, context_lengths=contexts,
                        request_phases=request_phases)
                else:
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
            "cache_eviction_blocks": cache_evictions,
            "workspace_bytes": max((rank["workspace_bytes"] for rank in ranks), default=workspace),
            "ranks": ranks,
        }
        if phase == "mixed":
            details["request_phases"] = list(request_phases or ())
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
        *,
        request_phases: Sequence[str] | None = None,
    ) -> MemoryDecision:
        """Commit a reservation only after every participating rank passes."""
        decision = self.check(target_tokens, query_tokens, phase, request_phases=request_phases)
        if decision.ok:
            self._evict_prefix_blocks(decision.details["cache_eviction_blocks"])
            self.allocations.update(target_tokens)
            self._prefix_committed_requests.update(target_tokens)
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
            self._record_block_peak()
        return decision

    def release(self, request_id: str) -> None:
        """Return a request's logical blocks, retaining the physical KV pool.

        The event engine is responsible for waiting for in-flight compute and
        transfers before calling this method. Repeated release is harmless.
        """
        self.allocations.pop(request_id, None)
        self._prefix_committed_requests.discard(request_id)
        for key in self._prefix_attached.pop(request_id, ()):
            self._prefix_entries[key].references.discard(request_id)
            self._touch_prefix(key)

    def snapshot(self) -> list[dict]:
        """Return independent per-rank accounting records and high watermarks."""
        used = self._used_blocks(self.allocations)
        cache = self.prefix_snapshot()
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
                "pipeline_buffer_reserved_bytes": rank.pipeline_buffer_reserved_bytes,
                **({"pipeline_buffer_breakdown": dict(rank.pipeline_buffer_breakdown)}
                   if rank.pipeline_buffer_breakdown is not None else {}),
                "kv_bytes_per_token": rank.kv_bytes_per_token,
                "kv_block_bytes": rank.kv_block_bytes,
                "pool_blocks": rank.pool_blocks,
                "guard_blocks": self.config.guard_blocks,
                "used_blocks": used,
                "prefix_cached_blocks": cache["cached_blocks"],
                "prefix_pinned_blocks": cache["pinned_blocks"],
                "prefix_shared_blocks": cache["shared_blocks"],
                "private_blocks": cache["private_blocks"],
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
