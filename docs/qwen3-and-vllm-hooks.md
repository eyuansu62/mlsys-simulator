# Qwen3 geometry and vLLM internal capture

Version 1.6 describes dense Qwen3 models without conflating residual features
with attention-head features. The same model description feeds weight storage,
analytical service, TP/PP layouts, KV capacity and evidence identity. Simulation
continues to use CPU resources; optional hooks instrument an installed Runtime.

## Import a supported HF configuration

```bash
mlsys-sim import-model --kind qwen3 --config examples/v1_6/qwen3-4b.hf.json \
  --name Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --out runs/qwen3-import
mlsys-sim predict --experiment examples/v1_6/qwen3-4b-predict.json --out runs/qwen3-predict
mlsys-sim simulate --experiment examples/v1_6/qwen3-r2-tp2-pp2.json --out runs/qwen3-combined
```

Use new output directories. Import reads JSON and creates model/provenance
artifacts; it does not download weights, execute model-provided code or infer
a missing revision. The examples contain architecture metadata and synthetic
requests, with no user prompt or generated-output text.

The importer accepts explicitly declared dense `Qwen3ForCausalLM`, supported
float storage, gated SiLU MLP and full attention. Geometry and dtype/tied fields
must be explicit. Unsupported quantization, MoE/router options, active sliding
windows, attention bias and scaled RoPE fail rather than being approximated by
a different architecture. Known HF metadata is kept separate from modeled
behavior; unknown architecture options are not silently ignored.

```python
from mlsys_sim.qwen3 import (
    load_qwen3_model, model_from_qwen3_config,
    normalized_qwen3_architecture, qwen3_architecture_fingerprint,
)

bundle = load_qwen3_model(
    "examples/v1_6/qwen3-4b.hf.json", name="Qwen3-4B",
    revision="1cfa9a7208912126459214e8b04321603b3df60c",
)
model_data = bundle["model"]
```

`model_from_qwen3_config(config, *, name, revision, kv_dtype=None)` returns a
`ModelSpec`. KV storage defaults to the explicit model dtype; pass `kv_dtype`
when the actual Runtime uses a different supported unquantized KV format.
`load_qwen3_model` also records raw file SHA256, canonical full-config SHA256
and the normalized architecture mapping. These hashes describe different
provenance scopes and should not be substituted for each other.

## Residual, query and KV widths

`hidden_size` is the residual width. The read-only `query_projection_size`
equals `num_attention_heads × head_dim` and may differ from it. Query and output
projection parameters use both widths:

```text
Q/O parameters per TP rank = 2 × hidden_size × query_projection_size / TP
K/V parameters per TP rank = 2 × hidden_size × local_kv_heads × head_dim
KV bytes per token per rank = 2 × stage_layers × local_kv_heads × head_dim × kv_bytes
```

GQA head sharding/replication retains its existing integral constraints. The
PP activation payload uses residual width, not query width. Independent query
features do not multiply the KV head count or alter already computed context.

| Model metadata | Qwen3-1.7B | Qwen3-4B |
| --- | --- | --- |
| Layers | 28 | 36 |
| Residual hidden width | 2048 | 2560 |
| Query heads × head dimension | 16 × 128 | 32 × 128 |
| Query projection width | 2048 | 4096 |
| KV heads | 8 | 8 |
| Intermediate width | 6144 | 9728 |
| Vocabulary | 151936 | 151936 |
| Logical parameters with tying and Norm | 1,720,574,976 | 4,022,468,096 |
| BF16 logical weight bytes | 3,441,149,952 | 8,044,936,192 |
| Full-model BF16 KV bytes per token | 114688 | 147456 |

These are mathematical model counts under the supported shard plan. They are
not measured process VRAM, allocator overhead or a Runtime's vocabulary-padding
report. Physical copies and replicated tensors can make the sum over devices
different from the logical parameter count.

## Tying and Norm ownership

`tie_word_embeddings`, `qk_norm` and `final_norm` are optional ModelSpec booleans
with default `False`. Importing supported Qwen3 sets Q/K and final Norm true and
retains the configuration's explicit vocabulary-tying value.

Q/K Norm uses two shared `head_dim` vectors per layer and TP rank. Those weights
are not a separate vector for every head. The final RMSNorm uses one residual
width vector on the last stage of each TP rank. Their costs join the declared,
uncalibrated macro model; a calibrated whole-stage observation remains the
authority for a measured service time.

With PP=1, embedding and LM-head roles can alias the same vocabulary-sharded
allocation on a rank. With PP>1, first and last stages live on separate devices,
so each endpoint retains a physical copy even when its values are tied. Actual
LM-head computation and output communication still take place.

`architecture_config_sha256` binds a canonical execution-relevant mapping,
including geometry, dtype, Norm/RoPE/context choices and tying. Path strings and
HF/Transformers metadata versions do not change that fingerprint. A different
fingerprint cannot reuse an incompatible Profile merely because model name and
revision are the same. Default-false flags and an empty fingerprint are omitted
from canonical legacy identities; older complete Profile identities retain
their earlier structure.

## Serving combinations

The model description is shared by the existing schedulers. Prefix Cache still
needs full token ancestry and namespace; independent query width does not make
equal-length prompts identical. Chunked and mixed batches keep their member
contexts, and swap/recompute use KV head storage rather than residual width.
Replica ownership, TP atomic reservation and PP feedback rules remain in force.
CPU swap remains the simulator's declared recovery policy. Runtime capture
preserves observed native recovery; resumed-recompute boundaries outside the
supported service contract remain quarantined.

`qwen3-r2-tp2-pp2.json` is a deterministic synthetic demonstration on eight
declared 16GiB devices. Its small logical pools deliberately trigger recovery.
It completes ten requests, four mixed microbatches, 24 reused prefix Tokens and
two D2H/two H2D copies, with no recomputation and no final request/host ownership.
The observed makespan is 52.413216ms with its stated synthetic service and copy
parameters. P2P cost is explicitly disabled and each replica has one inflight
microbatch for inspectable accounting. These results verify software contracts,
not the throughput of eight real GPUs or a loaded 4B model.

## Reference package interpretation

The supplied RX9070XT archive's Qwen3 geometries can now be represented without
changing real dimensions. Its historical client-content SSE timestamps, C1
request-service rows and device peaks still lack whole-stage batch composition,
semantic server-token timing and attributed Workspace. Model import alone
therefore does not change the existing insufficient-evidence verdict or create
new TP/PP/cache/swap hardware coverage. See the
[reference audit](reference-data-audit.md).

## Internal capture boundaries

The integration targets the recorded vLLM build
`0.29.1.dev0+g98dff2a81.d20261002` and its MRv1
`vllm.v1.worker.gpu_model_runner` ABI. Engine V1 does not by itself identify
that runner: the same build can select the newer
`vllm.v1.worker.gpu.model_runner` implementation. Use the supported MRv1 path
with `VLLM_USE_V2_MODEL_RUNNER=0`; V2 is outside this hook ABI. Version
and method/field checks belong at installation. A different Runtime build is
not automatically supported because its public model name is the same.

The captured stage domain is dense unquantized Qwen3 generation with the
declared TP/PP layout. Native DP/EP/context parallelism, DBO, LoRA, speculative
decoding, pooling, multimodal/external KV transfer and unsupported resumed
history remain outside canonical service capture. Independent simulator
replicas and their CPU swap policy are separate from those native Runtime
features. Unsupported calls retain raw diagnostics and cannot provide a
fabricated stage cost.

Independent-stage eligibility also depends on the actual PP transport path.
In this pinned source, combined TP>1 and PP>1 normally uses tensor-sliced P2P
with a receiver-side TP AllGather. Lazy incoming-transfer postprocessing can
fall inside a Runner interval. Those intervals cannot be charged as isolated
stage service and then combined with the simulator's full-residual-per-rank
P2P model, so that transport remains raw diagnostics. Lazy incoming transport
or sequence parallelism (`pass_config.enable_sp`) outside the verified boundary
also remains ineligible. Successful synthetic TP/PP simulation is separate
from this measured-profile scope.

The runner capture retains the actual selected request IDs, scheduled query
lengths, computed-prefix contexts and original prompt lengths. A one-Token
query can finish Prefill, so member phase is not inferred from query length.
Cached prefixes and chunked Prefill keep their actual contexts. Mixed batches
retain ordered roles throughout capture and later Profile lookup.

A nonfinal PP stage can complete its forward service without sampling. A final
stage needs its supported execute/sampling completion boundary before it can
become a complete stage sample. An execute-only span does not silently include
future sampling. For async output, capture must wait for the Runtime's actual
`get_output()` call; it must not pre-consume a future. The original object's
return and repeated-call behavior remain the Runtime's responsibility.

Runner sampling candidates and Scheduler-published output serve different
purposes. Only actual published lifecycle events can establish the server
request trace: aborts, stale outputs and stopping rules can discard candidates.
Missing arrivals or terminal events do not become a complete canonical online
observation. Historical client SSE events are not substituted for that channel.

Per-rank records can form a canonical stage sample only when session, replica,
stage, shape, model/runtime identity and the complete TP group agree. Each
rank supplies its own elapsed service time; aggregation uses their maximum,
not subtraction of timestamps from unrelated hosts or clock domains. Missing,
failed or incomplete records remain explicit evidence gaps. Injected test
dependencies keep fixture identity across aggregation.

Raw allocated/reserved counters stay separate from attributed Workspace.
This hook does not attribute temporary allocations to Workspace, so its
canonical Workspace remains null; an unexpected nonnull claim is quarantined.
Synchronization and hook overhead are part of the capture protocol and need
to be considered before comparing it with a low-interference benchmark.
Successful CPU fixtures establish software consistency; live vLLM/HIP behavior
and hardware prediction accuracy still require measurements on the AMD host.

The pinned implementation contracts can be inspected in the primary
[MRv1 runner source](https://github.com/vllm-project/vllm/blob/98dff2a81/vllm/v1/worker/gpu_model_runner.py)
and [Qwen3 model source](https://github.com/vllm-project/vllm/blob/98dff2a81/vllm/model_executor/models/qwen3.py).
This library instruments those interfaces; it does not distribute a fork of
the Runtime implementation.

## Enable capture explicitly

Installing the library registers a dormant general-plugin entry point. vLLM
can discover all general plugins by default, so the entry point itself returns
without importing vLLM/Torch or patching anything when
`MLSYS_VLLM_CAPTURE_CONFIG` is unset. If using a `VLLM_PLUGINS` allowlist,
include `mlsys_internal_capture` alongside any other plugins the application needs.

An activation file has the following required fields:

```json
{
  "schema_version": 1,
  "experiment": "bound-experiment.json",
  "session_id": "capture-session-01",
  "output_dir": "captures",
  "capture_scheduler": true,
  "record_prompt_token_ids": false,
  "max_records": 1000,
  "max_events": 100000
}
```

The experiment and output path resolve relative to this configuration file.
The experiment must describe the actual device capacities, model revision,
precision, TP/PP layout and installed software identity. Build it from real
inventory and imported model metadata; synthetic device defaults cannot stand
in for an AMD host. Scheduler limits, chunk/prefix settings, block size and
dtype must also agree with the capture contract and its decode-first mapping.
Native priority/tie ordering must be compatible with the observed enqueue
order; unsupported ordering remains diagnostic evidence.
`capture_scheduler` defaults to true.
`record_prompt_token_ids` defaults to false; enable it deliberately when those
actual numeric IDs are required for a Prefix Cache replay. The capture does not
store prompt/output text. Optional `max_records` and `max_events` bound runner
and Scheduler records, and `replica_index` labels an independently launched
replica. Distinct replicas must receive their matching coordinate/configuration.
Use a finite diagnostic cohort and choose limits appropriate to it. Import is
bounded to 32MiB per complete JSON export and 256MiB across its input files;
larger sessions must be collected as smaller independent cohorts. A limit-hit
diagnostic does not establish a complete lifecycle.

Prompt-ID SHA256 is retained even when numeric IDs are omitted. Borrowing IDs
from a supplied workload requires a matching observed hash. Native nonempty
cache salts become opaque SHA256 namespaces; empty/absent salts share the
default domain. Original input namespaces are interpreted literally. When
reimporting a previously derived opaque replay, explicitly enable
`bind_runtime_workload` again so captured namespaces are rebound without guessing
an encoding from a string prefix.

On the supported installed Runtime, select MRv1 and synchronous scheduling for
canonical stage samples, then activate the file for that process:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
MLSYS_VLLM_CAPTURE_CONFIG=/absolute/path/capture.config.json \
vllm serve /path/to/model --no-async-scheduling
```

This is a live-host setup template, not a command executed by the CPU examples.
The plugin validates the pinned ABI before installing constructors. It attaches
instance hooks to the actual Runner and Scheduler; unsupported versions or
features cannot be made valid by changing a version label in a raw file.
Explicit export/close reports errors. Interpreter teardown flush is a best-effort
fallback and does not establish a complete run.

Live TP capture inspects the actually mapped Linux RCCL shared library,
queries its Runtime version and retains its content hash. A static or ambiguous
library mapping is insufficient evidence. PyTorch collective build-version
macros do not substitute for this binding. The actual stage layout, selected
device and declared model geometry must agree with the bound experiment.

For an application with existing Python objects, use the reversible installers:

```python
from mlsys_sim.vllm_hooks import install_runner_hook, install_scheduler_hook

runner_capture = install_runner_hook(runner, experiment, session_id="capture-session-01")
scheduler_capture = install_scheduler_hook(
    scheduler, experiment, session_id="capture-session-01",
    record_prompt_token_ids=True,
)
# Let the Runtime invoke its real methods normally.
runner_capture.close()
scheduler_capture.close()
runner_data = runner_capture.export_capture()
scheduler_data = scheduler_capture.export_capture()
```

Direct installation must use the correct replica/stage/TP-rank coordinates.
The `VllmCaptureWorkerExtension` supplies only new `mlsys_capture_install`,
`mlsys_capture_export` and `mlsys_capture_close` RPC methods, letting the worker
integration obtain actual PP/TP ranks without overriding its existing load or
initialization methods. `uninstall_vllm_hooks_plugin()` restores constructor
patches and closes captures in the current process. Passing test clocks,
synchronizers, Torch modules or allocator providers marks runner records as
fixtures; a test Scheduler clock does the same for lifecycle records.

The worker RPC path is an alternative installation route for embedded vLLM:

```python
from dataclasses import asdict
from vllm import LLM

# Select VLLM_USE_V2_MODEL_RUNNER=0 before Runtime initialization.
llm = LLM(
    model="/path/to/local/model",
    tensor_parallel_size=experiment.strategy.tensor_parallel_size,
    pipeline_parallel_size=experiment.strategy.pipeline_parallel_size,
    async_scheduling=False,
    worker_extension_cls="mlsys_sim.vllm_hooks.VllmCaptureWorkerExtension",
)
llm.collective_rpc("mlsys_capture_install", args=({
    "experiment": asdict(experiment), "session_id": "capture-session-01",
    "output_dir": "/absolute/path/captures",
},))
# Run the intended workload through the Runtime.
responses = llm.collective_rpc("mlsys_capture_close")
captures = [response["capture"] for response in responses]
```

The extension class uses a standard dotted import path; this differs from the
`module:factory` syntax of generic stage adapters. Install each Runner once.
The worker-only route supplies stage capture; attach the Scheduler hook as well
when an online lifecycle observation is needed.
Programmatic attachment cannot establish that a live Scheduler started with
empty state. Canonical fresh-state eligibility uses the opt-in plugin's
post-constructor observation; other live attachments retain diagnostic evidence
until the required initial-state proof exists.

## Normalize raw rank and lifecycle files

Runner exports are complete JSON objects with
`kind="vllm_internal_capture"`; Scheduler exports use
`kind="vllm_scheduler_capture"`. Files are rank/process-specific rather than a
shared multi-process append log. A loader does not accept JSONL as one export.
An import plan lists the files to combine:

```json
{
  "schema_version": 1,
  "captures": ["rank0.json", "rank1.json", "scheduler.json"]
}
```

```bash
mlsys-sim collect --kind vllm-hook --experiment bound-experiment.json \
  --plan hook-import.plan.json --out runs/internal-capture
```

A single export can use `--raw FILE` instead of a plan. Each plan capture path
resolves relative to that plan. The library functions are
`load_vllm_captures(paths)` and
`normalize_vllm_capture(experiment, captures, *, allow_test_fixture=False,
runtime_trace=None, bind_runtime_workload=False)`.

`CalibrationSpec.runtime_config_sha256` binds the observed execution snapshot,
including backend/graph and other captured Runtime choices, independently of
the model architecture fingerprint. If the declared hash is empty, consistent
captured facts can bind an effective experiment and the provenance records that
derivation. A declared nonempty mismatch fails; no different Runtime settings
are silently combined under one model/version label.

Complete compatible synchronous rank groups produce per-replica datasets.
R=1 also has `dataset.json`; all accepted replica datasets are retained under
`datasets/replica-N.json`. The output keeps `experiment.json`, raw captures,
input hashes, `capture_summary` and `quarantined` reasons. A successful import
can have `canonical_stage_ready=false` when required ranks or complete samples
are absent or some groups are quarantined. Partial accepted records can remain
available as `canonical_stage_records_available`; that field does not assert
full-layout readiness. Async consumption intervals, overlapping service, unsupported
features and resumed recomputation remain raw diagnostics. Set
`async_scheduling=False` for the supported exclusive stage cost boundary.

Scheduler hooks observe enqueue and the published `EngineCoreOutput` tokens
and terminal statuses. They do not manufacture arrivals from runner outputs.
To derive a replay from actual observed IDs/arrivals/output counts rather than
require an exact match to the supplied workload, explicitly set
`"bind_runtime_workload": true` in the import plan. Provenance marks the result
as conditional replay. Prefix-cache replay still needs captured real prompt IDs
or matching IDs from the input experiment. Length or prefix-group labels do
not supply token ancestry. An independent semantic Runtime trace can be supplied
through the plan's `runtime_trace` path when no Scheduler channel is present.

Canonical stage readiness and online readiness are separate. Runner-only
capture can fit a supported stage Profile while lacking a comparable online
observation. Published lifecycle events need complete ingress/termination and
matching identities/clocks; capture errors or missing prompt identity can leave
online readiness false. Scheduler-derived throughput and goodput rates remain
null without an explicit measurement window: idle time before export/flush is
not a measured cohort denominator. Latency, counts and declared SLA judgment can
still be comparable. Every collection keeps `hardware_validated=false`;
explicit held-out acceptance remains a separate step.

The small hand-calculated fixture can be imported on CPU:

```bash
mlsys-sim collect --kind vllm-hook --experiment examples/v1_6/hook-experiment.fixture.json \
  --plan examples/v1_6/hook-import.fixture.plan.json --out runs/hook-fixture --allow-test-fixture
```

It contains one complete runner channel and no Scheduler lifecycle. The observed
software result is one accepted stage group, `canonical_stage_ready=true`,
`canonical_comparable=false` and `hardware_validated=false`. It does not record
a live vLLM or GPU run.
