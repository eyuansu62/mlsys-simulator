// Diagnostic only: this tool never produces a complete Workspace ledger.
// Compile against the machine's installed ROCprofiler-SDK 1.4.1 headers.
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
constexpr size_t kDomains = 9, kOps = 8192, kEvents = 1048576, kEpochs = 64;
constexpr size_t kName = 192, kLabel = 256, kPath = 4096, kAgents = 64;
enum Flags : uint32_t {
    OtherApi = 0, Allocate = 1, Free = 2, Vmm = 4, IpcExternal = 8,
    PoolAsync = 16, HostManaged = 32, Lifecycle = 64,
    UnclassifiedMemory = 128, UnknownOperation = 256, MemoryQuery = 512, PrivateScratch = 1024
};
constexpr uint32_t kMutators = Allocate | Free | Vmm | IpcExternal | PoolAsync |
    HostManaged | Lifecycle | UnclassifiedMemory | UnknownOperation | PrivateScratch;
constexpr uint32_t kUnsupported = Vmm | IpcExternal | PoolAsync | HostManaged |
    Lifecycle | UnclassifiedMemory | UnknownOperation | PrivateScratch;
struct Op { char name[kName]; uint32_t flags; uint8_t status_type; bool present; uint64_t enters, exits, other, filtered; };
struct Domain {
    rocprofiler_callback_tracing_kind_t kind; const char* name;
    Op ops[kOps]; size_t op_count;
    int enumerate_status = -999, configure_status = -999;
};
Domain g_domains[kDomains] = {
    {ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, "HIP_RUNTIME_API", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API, "HIP_COMPILER_API", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API, "HSA_CORE_API", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API, "HSA_AMD_EXT_API", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API, "HSA_IMAGE_EXT_API", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API, "HSA_FINALIZE_EXT_API", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION, "MEMORY_ALLOCATION", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY, "SCRATCH_MEMORY", {}, 0},
    {ROCPROFILER_CALLBACK_TRACING_RCCL_API, "RCCL_API", {}, 0}
};
struct Event {
    uint64_t sequence, callback_sequence, timestamp_ns, thread_id, correlation, ancestor, external, epoch;
    uint32_t domain, operation, phase, flags;
    int64_t status;
    uint8_t status_type;  // 0=not decoded; 1=hipError_t; 2=hsa_status_t; 3=ncclResult_t
    uint64_t allocation_address, allocation_size, allocation_agent;
    bool allocation_payload;
    uint64_t scratch_agent, scratch_queue, scratch_flags, scratch_args_kind, scratch_size;
    bool scratch_payload, scratch_size_known;
};
struct Epoch {
    uint64_t id, begin_ns, end_ns, events, mutator_events, mutator_enters, unsupported_events, unsupported_enters, unknown_events;
    uint64_t dropped_at_begin, dropped_at_end, recursive_at_begin, recursive_at_end;
    uint64_t sync_exit_ns, last_mutator_ns;
    char label[kLabel]; bool ended;
};
struct AgentFact {
    uint64_t handle, gpu_id; uint32_t type, node_id;
    int32_t logical_node_id, logical_node_type_id;
    char name[kName], product_name[kName];
};
AgentFact g_agents[kAgents]; size_t g_agent_count = 0, g_agent_reported_count = 0;
uint64_t g_agent_callback_count = 0;
int g_agent_query_status = -999;
bool g_agent_inventory_invalid = false;
Event g_events[kEvents]; Epoch g_epochs[kEpochs];
pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t g_stop_lock = PTHREAD_MUTEX_INITIALIZER;
thread_local bool g_in_callback = false;
std::atomic<uint64_t> g_callbacks_inflight{0}, g_recursive{0};
size_t g_count = 0, g_limit = 262144, g_epoch_count = 0;
int g_active_epoch = -1;
uint64_t g_attempted = 0, g_observed = 0, g_filtered = 0, g_missing_user_data = 0, g_dropped = 0, g_unknown = 0;
bool g_metadata_truncated = false, g_bounds_valid = true, g_configured = false;
bool g_retain_all = false, g_retain_option_valid = true;
bool g_init_called = false, g_started = false, g_ready = false, g_finalized = false;
bool g_stop_invoked = false, g_stop_completed = false, g_observation_closed = false, g_fini_stop_skipped = false;
int g_stop_origin = 0, g_stop_result = -999, g_drain_status = -999;
uint64_t g_stop_attempts = 0, g_stop_rejected = 0, g_stop_begin_ns = 0, g_stop_end_ns = 0, g_callbacks_after_stop = 0;
char g_stop_status_name[192] = {}, g_stop_status_text[512] = {};
uint32_t g_sdk_version = 0, g_priority = 0;
char g_sdk_string[512] = {}, g_output[kPath] = {};
rocprofiler_context_id_t g_context{0};
int g_create_status = -999, g_valid_status = -999, g_context_valid = 0;
int g_start_status = -999, g_stop_status = -999;
uint64_t g_init_ns = 0, g_finalize_ns = 0;

uint64_t now_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t) != 0) return 0;
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}
bool copy_text(char* dst, size_t size, const char* src) {
    if (!src || size == 0) return false;
    size_t n = strnlen(src, size);
    size_t take = n < size ? n : size - 1;
    std::memcpy(dst, src, take); dst[take] = '\0';
    return n < size;
}
bool has(const char* s, const char* part) { return std::strstr(s, part) != nullptr; }
bool is(const char* s, const char* value) { return std::strcmp(s, value) == 0; }
uint32_t classify(const char* name, size_t domain, uint8_t& status_type) {
    char n[kName]{};
    for (size_t i = 0; name[i] && i + 1 < sizeof(n); ++i)
        n[i] = name[i] >= 'A' && name[i] <= 'Z' ? char(name[i] + ('a' - 'A')) : name[i];
    status_type = 0;
    if (domain == 8) {
        // These public RCCL entry points return ncclResult_t. Error-string
        // functions return const char*, so their union is deliberately not decoded.
        const char* known_result[] = {"ncclgetversion", "ncclgetuniqueid", "ncclcomminitrank",
            "ncclcomminitall", "ncclcomminitrankconfig", "ncclcommfinalize", "ncclcommdestroy",
            "ncclcommabort", "ncclcommsplit", "ncclmemalloc", "ncclmemfree", "ncclcommcount",
            "ncclcommcudevice", "ncclcommuserrank", "ncclcommgetasyncerror"};
        for (const char* candidate : known_result) if (is(n, candidate)) { status_type = 3; break; }
        if (is(n, "ncclgetversion") || is(n, "ncclgeterrorstring") || is(n, "ncclgetlasterror") ||
            is(n, "ncclcommcount") || is(n, "ncclcommcudevice") || is(n, "ncclcommuserrank") ||
            is(n, "ncclcommgetasyncerror")) return MemoryQuery;
        uint32_t flags = Lifecycle;
        if (is(n, "ncclmemalloc")) flags |= Allocate;
        if (is(n, "ncclmemfree")) flags |= Free;
        return flags;  // New/collective RCCL surfaces remain conservative lifecycle evidence.
    }
    if (domain == 7) return PrivateScratch;
    if (domain == 6) {
        uint32_t f = has(n, "vmem") ? Vmm : 0;
        if (has(n, "allocate")) f |= Allocate;
        else if (has(n, "free")) f |= Free;
        else f |= UnknownOperation;
        return f;
    }
    // Decode only families whose official APIs return HIP/HSA status. Other APIs
    // explicitly retain status=null; we do not reinterpret arbitrary retval unions.
    if (domain == 0 && (has(n, "hipmalloc") || has(n, "hipfree") ||
        has(n, "hipextmalloc") || has(n, "hipmem") || has(n, "hiphost") ||
        has(n, "hipipc") || has(n, "hipexternalmemory") ||
        is(n, "hipdevicesynchronize") || is(n, "hipstreamsynchronize"))) status_type = 1;
    if (domain >= 2 && (has(n, "hsa_memory_") || has(n, "hsa_amd_memory_") ||
        has(n, "hsa_amd_vmem_") || has(n, "hsa_amd_ipc_") ||
        has(n, "hsa_amd_svm_") || has(n, "hsa_ext_image_") ||
        is(n, "hsa_signal_create") || is(n, "hsa_signal_destroy"))) status_type = 2;
    uint32_t f = 0;
    const bool memory_surface = has(n, "mem") || has(n, "malloc") || has(n, "free") ||
        has(n, "alloc") || has(n, "ipc") || has(n, "svm");
    const bool query = has(n, "getinfo") || has(n, "get_info") ||
        has(n, "getattribute") || has(n, "getaccess") || has(n, "getallocationgranularity") ||
        has(n, "pointerinfo") || has(n, "pointer_info") || has(n, "pointerget") ||
        has(n, "getdefaultmempool") || has(n, "getmempool");
    if (has(n, "vmem") || has(n, "hipmemcreate") || has(n, "hipmemrelease") ||
        has(n, "hipmemmap") || has(n, "hipmemunmap") || has(n, "hipmemaddress") ||
        has(n, "hipmemretain") || has(n, "hipmemsetaccess")) f |= Vmm;
    if (has(n, "ipc") || has(n, "externalmemory") || has(n, "shareablehandle") ||
        has(n, "importfrom") || has(n, "exportto")) f |= IpcExternal;
    if (!query && (has(n, "mempool") || has(n, "memory_pool") ||
        ((has(n, "alloc") || has(n, "free")) && has(n, "async")))) f |= PoolAsync;
    if (!query && (has(n, "hostmalloc") || has(n, "hostalloc") || has(n, "hostfree") ||
        has(n, "allochost") || has(n, "mallocmanaged") || has(n, "hostregister") ||
        has(n, "hostunregister") || has(n, "memory_lock") || has(n, "memory_unlock") ||
        has(n, "svm"))) f |= HostManaged;
    if (!query && (has(n, "malloc") || has(n, "_allocate") || has(n, "memalloc") ||
        has(n, "arraycreate") || has(n, "array3dcreate"))) f |= Allocate;
    if (!query && (has(n, "free") || has(n, "memrelease") || has(n, "handle_release"))) f |= Free;
    if (!query && ((domain == 1 && (has(n, "register") || has(n, "unregister"))) ||
        has(n, "registerfatbinary") || has(n, "unregisterfatbinary") ||
        has(n, "queue_create") || has(n, "queue_destroy") ||
        has(n, "streamcreate") || has(n, "streamdestroy") ||
        has(n, "graphcreate") || has(n, "graphdestroy") ||
        has(n, "moduleload") || has(n, "moduleunload") ||
        has(n, "eventcreate") || has(n, "eventdestroy") ||
        has(n, "signal_create") || has(n, "signal_destroy") ||
        has(n, "code_object") || has(n, "executable_create") ||
        has(n, "executable_destroy") || has(n, "executable_load") ||
        has(n, "image_create") || has(n, "image_destroy") ||
        has(n, "sampler_create") || has(n, "sampler_destroy"))) f |= Lifecycle;
    if (query && f == 0) f = MemoryQuery;
    // Do not silently accept an unclassified new memory-management surface.
    if (memory_surface && f == 0 && !has(n, "memcpy") && !has(n, "memset") &&
        !has(n, "memory_copy") && !has(n, "memory_fill") &&
        !has(n, "async_copy") && !has(n, "allow_access")) f = UnclassifiedMemory;
    return f;
}
size_t domain_index(rocprofiler_callback_tracing_kind_t kind) {
    for (size_t i = 0; i < kDomains; ++i) if (g_domains[i].kind == kind) return i;
    return kDomains;
}
rocprofiler_status_t agent_inventory_callback(rocprofiler_agent_version_t version,
    const void** agents, size_t count, void*) {
    ++g_agent_callback_count; g_agent_reported_count = count;
    if (version != ROCPROFILER_AGENT_INFO_VERSION_0 || !agents || count > kAgents || g_agent_callback_count != 1) {
        g_agent_inventory_invalid = true; return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; ++i) {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
        if (!agent || agent->size < sizeof(*agent)) {
            g_agent_inventory_invalid = true; continue;
        }
        AgentFact& fact = g_agents[g_agent_count++];
        fact.handle = agent->id.handle; fact.type = uint32_t(agent->type);
        fact.node_id = agent->node_id; fact.gpu_id = agent->gpu_id;
        fact.logical_node_id = agent->logical_node_id; fact.logical_node_type_id = agent->logical_node_type_id;
        if (!copy_text(fact.name, sizeof(fact.name), agent->name ? agent->name : "") ||
            !copy_text(fact.product_name, sizeof(fact.product_name), agent->product_name ? agent->product_name : ""))
            g_agent_inventory_invalid = true;
        if (!fact.name[0] || (agent->type != ROCPROFILER_AGENT_TYPE_CPU && agent->type != ROCPROFILER_AGENT_TYPE_GPU))
            g_agent_inventory_invalid = true;
        for (size_t j = 0; j + 1 < g_agent_count; ++j) if (g_agents[j].handle == fact.handle) g_agent_inventory_invalid = true;
    }
    return g_agent_inventory_invalid ? ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT : ROCPROFILER_STATUS_SUCCESS;
}
int enumerate_operation(rocprofiler_callback_tracing_kind_t kind,
                        rocprofiler_tracing_operation_t operation, void* user) {
    auto* domain = static_cast<Domain*>(user);
    const char* name = nullptr;
    auto rc = rocprofiler_query_callback_tracing_kind_operation_name(kind, operation, &name, nullptr);
    if (rc != ROCPROFILER_STATUS_SUCCESS || !name || operation >= kOps) {
        g_metadata_truncated = true; return 0;
    }
    Op& op = domain->ops[operation];
    if (op.present || !copy_text(op.name, sizeof(op.name), name)) g_metadata_truncated = true;
    if (!op.present) ++domain->op_count;
    op.present = true;
    op.flags = classify(op.name, size_t(domain - g_domains), op.status_type);
    return 0;
}
void append_locked(Event event) {
    event.sequence = ++g_attempted;
    if (g_count < g_limit) g_events[g_count++] = event;
    else ++g_dropped;  // Saturate rather than overwrite: any loss invalidates the audit.
}
void callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void*) {
    g_callbacks_inflight.fetch_add(1, std::memory_order_acq_rel);
    if (g_in_callback) {
        g_recursive.fetch_add(1, std::memory_order_relaxed);
        g_callbacks_inflight.fetch_sub(1, std::memory_order_acq_rel); return;
    }
    g_in_callback = true;
    const uint64_t timestamp = now_ns();
    const size_t di = domain_index(record.kind);
    Op* op = di < kDomains && record.operation < kOps && g_domains[di].ops[record.operation].present
        ? &g_domains[di].ops[record.operation] : nullptr;
    Event event{};
    event.timestamp_ns = timestamp; event.thread_id = record.thread_id;
    event.correlation = record.correlation_id.internal; event.ancestor = record.correlation_id.ancestor;
    event.external = record.correlation_id.external.value;
    event.domain = uint32_t(di); event.operation = record.operation; event.phase = record.phase;
    event.flags = op ? op->flags : UnknownOperation;
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && op && op->status_type && record.payload) {
        if (op->status_type == 1) {
            const auto* p = static_cast<const rocprofiler_callback_tracing_hip_api_data_t*>(record.payload);
            if (p->size >= sizeof(*p)) { event.status = p->retval.hipError_t_retval; event.status_type = 1; }
        } else if (op->status_type == 2) {
            const auto* p = static_cast<const rocprofiler_callback_tracing_hsa_api_data_t*>(record.payload);
            if (p->size >= sizeof(*p)) { event.status = p->retval.hsa_status_t_retval; event.status_type = 2; }
        } else if (op->status_type == 3) {
            const auto* p = static_cast<const rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
            if (p->size >= sizeof(*p)) { event.status = p->retval.ncclResult_t_retval; event.status_type = 3; }
        }
    }
    if (di == 6 && record.payload) {
        const auto* p = static_cast<const rocprofiler_callback_tracing_memory_allocation_data_t*>(record.payload);
        if (p->size >= sizeof(*p)) {
            event.allocation_payload = true; event.allocation_address = p->address.handle;
            event.allocation_size = p->allocation_size; event.allocation_agent = p->agent_id.handle;
        }
    }
    if (di == 7 && record.payload) {
        const auto* p = static_cast<const rocprofiler_callback_tracing_scratch_memory_data_t*>(record.payload);
        if (p->size >= sizeof(*p)) {
            event.scratch_payload = true; event.scratch_agent = p->agent_id.handle;
            event.scratch_queue = p->queue_id.handle; event.scratch_flags = uint64_t(p->flags);
            event.scratch_args_kind = uint64_t(p->args_kind);
            if (record.operation == ROCPROFILER_SCRATCH_MEMORY_ALLOC && record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
                event.scratch_size = p->args.alloc_end.size; event.scratch_size_known = true;
            }
        }
    }
    pthread_mutex_lock(&g_lock);
    if (g_stop_completed && timestamp > g_stop_end_ns) ++g_callbacks_after_stop;
    event.callback_sequence = ++g_observed;
    if (!user_data) ++g_missing_user_data;
    if (op) {
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) ++op->enters;
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) ++op->exits;
        else ++op->other;
    }
    if (!op || event.flags & UnknownOperation) ++g_unknown;
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER && user_data)
        user_data->value = g_active_epoch >= 0 ? g_epochs[g_active_epoch].id : 0;
    const bool began_in_epoch = record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && user_data && user_data->value;
    if (g_active_epoch >= 0) {
        Epoch& e = g_epochs[g_active_epoch]; event.epoch = e.id; ++e.events;
        if (event.flags & UnknownOperation) ++e.unknown_events;
        if (event.flags & kMutators) ++e.mutator_events;
        if (event.flags & kUnsupported) ++e.unsupported_events;
        if (record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT && event.flags & kMutators) {
            ++e.mutator_enters; e.last_mutator_ns = timestamp;
            if (event.flags & kUnsupported) ++e.unsupported_enters;
        }
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && op &&
            is(op->name, "hipDeviceSynchronize") && event.status_type == 1 && event.status == 0)
            e.sync_exit_ns = timestamp;
    }
    const bool retain = g_retain_all || (event.flags & kMutators) || g_active_epoch >= 0 || began_in_epoch ||
        (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER && record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT);
    if (retain) append_locked(event);
    else { ++g_filtered; if (op) ++op->filtered; }
    pthread_mutex_unlock(&g_lock);
    g_in_callback = false;
    g_callbacks_inflight.fetch_sub(1, std::memory_order_acq_rel);
}
void json_text(FILE* f, const char* s) {
    std::fputc('"', f);
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p; ++p) {
        if (*p == '"' || *p == '\\') { std::fputc('\\', f); std::fputc(*p, f); }
        else if (*p < 32) std::fprintf(f, "\\u%04x", unsigned(*p));
        else std::fputc(*p, f);
    }
    std::fputc('"', f);
}
int write_report(const char* path) {
    if (!path || strnlen(path, kPath) >= kPath - 40) return EINVAL;
    char temporary[kPath]{};
    std::snprintf(temporary, sizeof(temporary), "%s.tmp.%ld", path, long(getpid()));
    int fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return errno;
    FILE* f = fdopen(fd, "w");
    if (!f) { int error = errno; close(fd); unlink(temporary); return error; }
    pthread_mutex_lock(&g_lock);
    std::fprintf(f, "{\n\"schema_version\":1,\"kind\":\"native_allocation_api_diagnostic\",\"probe_protocol_version\":2,\"probe_protocol\":\"native_allocation_api_probe_v2\",\"complete_workspace\":false,"
        "\"hardware_validated\":false,\"scope\":\"configured_process_hip_hsa_api_calls_only\",\"pid\":%ld,"
        "\"clock\":\"CLOCK_MONOTONIC_RAW_host_callback_boundary\",\"sdk_version\":%u,\"sdk_string\":", long(getpid()), g_sdk_version);
    json_text(f, g_sdk_string);
    std::fprintf(f, ",\"priority\":%u,\"configured\":%s,\"init_called\":%s,\"ready\":%s,"
        "\"init_ns\":%" PRIu64 ",\"create_status\":%d,\"valid_status\":%d,\"context_valid\":%d,"
        "\"start_status\":%d,\"stop_status\":%d,\"finalize_callback_called\":%s,\"finalize_ns\":%" PRIu64 ","
        "\"buffered_services_used\":false,\"buffer_flush_applicable\":false,\"sdk_buffer_drop_count\":null,"
        "\"callback_drain_method\":\"stop_context_then_no_inflight_callbacks\",\"callbacks_inflight\":%" PRIu64 ","
        "\"max_events\":%zu,\"observed_callbacks\":%" PRIu64 ",\"filtered_nonmutator_events\":%" PRIu64 ",\"missing_user_data\":%" PRIu64 ",\"attempted_events\":%" PRIu64 ",\"retained_events\":%zu,\"dropped_events\":%" PRIu64 ","
        "\"recursive_suppressed\":%" PRIu64 ",\"unknown_events\":%" PRIu64 ",\"metadata_truncated\":%s,\"bounds_valid\":%s,\n\"domains\":[",
        g_priority, g_configured ? "true" : "false", g_init_called ? "true" : "false", g_ready ? "true" : "false",
        g_init_ns, g_create_status, g_valid_status, g_context_valid, g_start_status, g_stop_status,
        g_finalized ? "true" : "false", g_finalize_ns, g_callbacks_inflight.load(), g_limit, g_observed, g_filtered, g_missing_user_data, g_attempted, g_count,
        g_dropped, g_recursive.load(), g_unknown, g_metadata_truncated ? "true" : "false", g_bounds_valid ? "true" : "false");
    for (size_t i = 0; i < kDomains; ++i) {
        Domain& d = g_domains[i];
        if (i) std::fputc(',', f);
        std::fprintf(f, "{\"index\":%zu,\"kind\":%d,\"name\":", i, int(d.kind)); json_text(f, d.name);
        std::fprintf(f, ",\"enumerate_status\":%d,\"configure_status\":%d,\"operation_count\":%zu,\"operations\":[", d.enumerate_status, d.configure_status, d.op_count);
        bool first = true;
        for (size_t j = 0; j < kOps; ++j) if (d.ops[j].present) {
            if (!first) std::fputc(',', f); first = false;
            std::fprintf(f, "{\"id\":%zu,\"name\":", j); json_text(f, d.ops[j].name);
            std::fprintf(f, ",\"flags\":%u,\"decoded_exit_status_type\":%u,\"enter_count\":%" PRIu64 ",\"exit_count\":%" PRIu64 ",\"other_phase_count\":%" PRIu64 ",\"filtered_nonmutator_events\":%" PRIu64 "}", d.ops[j].flags, unsigned(d.ops[j].status_type), d.ops[j].enters, d.ops[j].exits, d.ops[j].other, d.ops[j].filtered);
        }
        std::fputs("]}", f);
    }
    std::fprintf(f, "],\n\"agent_inventory\":{\"version\":0,\"query_status\":%d,\"callback_count\":%" PRIu64 ",\"reported_count\":%zu,\"retained_count\":%zu,\"invalid\":%s,\"max_agents\":%zu,\"agents\":[", g_agent_query_status, g_agent_callback_count, g_agent_reported_count, g_agent_count, g_agent_inventory_invalid ? "true" : "false", kAgents);
    for (size_t i = 0; i < g_agent_count; ++i) {
        const AgentFact& a = g_agents[i]; if (i) std::fputc(',', f);
        std::fprintf(f, "{\"handle\":%" PRIu64 ",\"type\":%u,\"type_name\":", a.handle, a.type);
        json_text(f, a.type == uint32_t(ROCPROFILER_AGENT_TYPE_CPU) ? "CPU" : a.type == uint32_t(ROCPROFILER_AGENT_TYPE_GPU) ? "GPU" : "UNKNOWN");
        std::fputs(",\"name\":", f); json_text(f, a.name); std::fputs(",\"product_name\":", f); json_text(f, a.product_name);
        std::fprintf(f, ",\"node_id\":%u,\"gpu_id\":%" PRIu64 ",\"logical_node_id\":%d,\"logical_node_type_id\":%d}", a.node_id, a.gpu_id, a.logical_node_id, a.logical_node_type_id);
    }
    std::fputs("]},\n\"stop_evidence\":{\"invoked\":", f);
    std::fputs(g_stop_invoked ? "true" : "false", f);
    std::fprintf(f, ",\"origin\":%d,\"actual_sdk_stop_calls\":%" PRIu64 ",\"rejected_requests\":%" PRIu64 ",\"begin_ns\":%" PRIu64 ",\"end_ns\":%" PRIu64 ",\"stop_result\":%d,\"drain_status\":%d,\"completed\":%s,\"observation_window_closed\":%s,\"callbacks_after_stop\":%" PRIu64 ",\"fini_skipped_second_stop\":%s,\"sdk_status_name\":",
        g_stop_origin, g_stop_attempts, g_stop_rejected, g_stop_begin_ns, g_stop_end_ns, g_stop_result, g_drain_status,
        g_stop_completed ? "true" : "false", g_observation_closed ? "true" : "false", g_callbacks_after_stop, g_fini_stop_skipped ? "true" : "false");
    json_text(f, g_stop_status_name); std::fputs(",\"sdk_status_text\":", f); json_text(f, g_stop_status_text); std::fputc('}', f);
    std::fputs(",\n\"retention_mode\":", f); json_text(f, g_retain_all ? "all_callbacks_v1" : "mutators_and_epoch_v1");
    std::fprintf(f, ",\"retain_all_option_valid\":%s,\n\"filter_policy\":", g_retain_option_valid ? "true" : "false");
    json_text(f, g_retain_all ? "all_APIs_global" : "all_mutators_global_and_all_APIs_in_or_entered_epoch");
    std::fputs(",\n\"flag_bits\":{\"allocate\":1,\"free\":2,\"vmm\":4,\"ipc_external\":8,\"pool_async\":16,\"host_managed\":32,\"runtime_compiler_lifecycle\":64,\"unclassified_memory\":128,\"unknown_operation\":256,\"memory_query\":512,\"kernel_private_scratch\":1024},\n\"epochs\":[", f);
    for (size_t i = 0; i < g_epoch_count; ++i) {
        const Epoch& e = g_epochs[i]; if (i) std::fputc(',', f);
        std::fprintf(f, "{\"id\":%" PRIu64 ",\"label\":", e.id); json_text(f, e.label);
        std::fprintf(f, ",\"begin_ns\":%" PRIu64 ",\"end_ns\":%" PRIu64 ",\"ended\":%s,\"api_events\":%" PRIu64 ","
            "\"mutator_events\":%" PRIu64 ",\"mutator_enters\":%" PRIu64 ",\"unsupported_events\":%" PRIu64 ",\"unsupported_enters\":%" PRIu64 ",\"unknown_events\":%" PRIu64 ","
            "\"dropped_at_begin\":%" PRIu64 ",\"dropped_at_end\":%" PRIu64 ",\"recursive_at_begin\":%" PRIu64 ",\"recursive_at_end\":%" PRIu64 ","
            "\"successful_device_sync_exit_ns\":%" PRIu64 ",\"last_mutator_enter_ns\":%" PRIu64 "}",
            e.begin_ns, e.end_ns, e.ended ? "true" : "false", e.events, e.mutator_events, e.mutator_enters, e.unsupported_events, e.unsupported_enters,
            e.unknown_events, e.dropped_at_begin, e.dropped_at_end, e.recursive_at_begin, e.recursive_at_end,
            e.sync_exit_ns, e.last_mutator_ns);
    }
    std::fputs("],\n\"events\":[\n", f);
    for (size_t i = 0; i < g_count; ++i) {
        const Event& e = g_events[i]; if (i) std::fputs(",\n", f);
        std::fprintf(f, "{\"seq\":%" PRIu64 ",\"callback_seq\":%" PRIu64 ",\"ns\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"cid\":%" PRIu64 ",\"ancestor\":%" PRIu64 ",\"external\":%" PRIu64 ",\"epoch\":%" PRIu64 ",\"domain\":%u,\"op\":%u,\"phase\":%u,\"flags\":%u,\"status_type\":%u,\"status\":",
            e.sequence, e.callback_sequence, e.timestamp_ns, e.thread_id, e.correlation, e.ancestor, e.external, e.epoch,
            e.domain, e.operation, e.phase, e.flags, unsigned(e.status_type));
        if (e.status_type) std::fprintf(f, "%" PRId64, e.status); else std::fputs("null", f);
        if (e.allocation_payload) std::fprintf(f, ",\"raw_allocation_payload\":{\"address_or_handle\":%" PRIu64 ",\"size_bytes\":%" PRIu64 ",\"agent_handle\":%" PRIu64 ",\"success_proven\":false}", e.allocation_address, e.allocation_size, e.allocation_agent);
        if (e.scratch_payload) {
            std::fprintf(f, ",\"raw_scratch_payload\":{\"agent_handle\":%" PRIu64 ",\"queue_handle\":%" PRIu64 ",\"flags\":%" PRIu64 ",\"args_kind\":%" PRIu64 ",\"allocation_size_known\":%s,\"allocation_size_bytes\":", e.scratch_agent, e.scratch_queue, e.scratch_flags, e.scratch_args_kind, e.scratch_size_known ? "true" : "false");
            if (e.scratch_size_known) std::fprintf(f, "%" PRIu64, e.scratch_size); else std::fputs("null", f);
            std::fputs(",\"workspace_attribution_supported\":false}", f);
        }
        std::fputc('}', f);
    }
    std::fputs("\n],\"write_complete\":true,\"limitations\":[\"Not an ownership ledger or Workspace completeness proof\",\"Only normalized allocation address/size/agent; no success or ownership inferred\",\"No driver bypass or device-side heap coverage\",\"Non-memory API return statuses are not generically decoded\",\"Successful device synchronize is required in each epoch\"]}\n", f);
    pthread_mutex_unlock(&g_lock);
    int error = std::ferror(f) ? EIO : 0;
    if (!error && std::fflush(f) != 0) error = errno ? errno : EIO;
    if (!error && fsync(fd) != 0) error = errno ? errno : EIO;
    if (std::fclose(f) != 0 && !error) error = errno ? errno : EIO;
    // Publish a completed file atomically, without replacing an earlier run.
    if (!error && link(temporary, path) != 0) error = errno;
    unlink(temporary);
    return error;
}
int tool_init(rocprofiler_client_finalize_t, void*) {
    g_init_called = true; g_init_ns = now_ns();
    const char* retain = std::getenv("MLSYS_NATIVE_PROBE_RETAIN_ALL");
    if (retain) {
        if (is(retain, "1")) g_retain_all = true;
        else if (!is(retain, "0")) { g_retain_option_valid = false; g_bounds_valid = false; }
    }
    const char* bound = std::getenv("MLSYS_NATIVE_PROBE_MAX_EVENTS");
    if (bound) {
        char* end = nullptr; errno = 0; unsigned long long value = std::strtoull(bound, &end, 10);
        if (errno || !end || *end || value == 0 || value > kEvents) g_bounds_valid = false;
        else g_limit = size_t(value);
    }
    g_agent_query_status = int(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
        agent_inventory_callback, sizeof(rocprofiler_agent_v0_t), nullptr));
    g_create_status = int(rocprofiler_create_context(&g_context));
    if (g_create_status == ROCPROFILER_STATUS_SUCCESS) {
        bool all = true;
        for (auto& d : g_domains) {
            d.enumerate_status = int(rocprofiler_iterate_callback_tracing_kind_operations(d.kind, enumerate_operation, &d));
            d.configure_status = int(rocprofiler_configure_callback_tracing_service(g_context, d.kind, nullptr, 0, callback, nullptr));
            all = all && d.enumerate_status == ROCPROFILER_STATUS_SUCCESS && d.configure_status == ROCPROFILER_STATUS_SUCCESS && d.op_count > 0;
        }
        g_valid_status = int(rocprofiler_context_is_valid(g_context, &g_context_valid));
        if (g_valid_status == ROCPROFILER_STATUS_SUCCESS && g_context_valid) {
            g_start_status = int(rocprofiler_start_context(g_context));
            g_started = g_start_status == ROCPROFILER_STATUS_SUCCESS;
        }
        g_ready = all && g_started && g_bounds_valid && !g_metadata_truncated &&
            g_agent_query_status == ROCPROFILER_STATUS_SUCCESS && g_agent_callback_count == 1 &&
            g_agent_count > 0 && !g_agent_inventory_invalid;
    }
    // Keep the tool registered even if setup is partial, so finalization emits evidence.
    return 0;
}
int stop_observation_window(bool explicit_request) {
    if (g_in_callback) return EDEADLK;
    pthread_mutex_lock(&g_stop_lock);
    pthread_mutex_lock(&g_lock);
    if (g_stop_invoked) {
        const int result = g_stop_result;
        pthread_mutex_unlock(&g_lock); pthread_mutex_unlock(&g_stop_lock); return result;
    }
    if (explicit_request && (g_finalized || g_active_epoch >= 0)) {
        ++g_stop_rejected;
        pthread_mutex_unlock(&g_lock); pthread_mutex_unlock(&g_stop_lock); return EBUSY;
    }
    g_stop_invoked = true; g_stop_origin = explicit_request ? 1 : 2;
    g_stop_begin_ns = now_ns();
    const bool started = g_started;
    if (started) ++g_stop_attempts;
    pthread_mutex_unlock(&g_lock);
    const auto rc = started ? rocprofiler_stop_context(g_context) : ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
    const char* status_name = rocprofiler_get_status_name(rc);
    const char* status_text = rocprofiler_get_status_string(rc);
    const uint64_t deadline = now_ns() + 1000000000ULL;
    while (g_callbacks_inflight.load(std::memory_order_acquire) && now_ns() < deadline) {
        timespec pause{0, 100000}; nanosleep(&pause, nullptr);
    }
    pthread_mutex_lock(&g_lock);
    g_stop_status = int(rc);
    g_drain_status = g_callbacks_inflight.load(std::memory_order_acquire) ? ETIMEDOUT : 0;
    g_stop_result = rc == ROCPROFILER_STATUS_SUCCESS ? g_drain_status : int(rc);
    g_stop_end_ns = now_ns(); g_stop_completed = true;
    g_observation_closed = rc == ROCPROFILER_STATUS_SUCCESS && g_drain_status == 0;
    copy_text(g_stop_status_name, sizeof(g_stop_status_name), status_name ? status_name : "UNKNOWN_STATUS");
    copy_text(g_stop_status_text, sizeof(g_stop_status_text), status_text ? status_text : "");
    const int result = g_stop_result;
    pthread_mutex_unlock(&g_lock); pthread_mutex_unlock(&g_stop_lock); return result;
}
void tool_fini(void*) {
    // A successful Python stop closes the observation window before SDK teardown
    // can unregister the context. Preserve that actual status; do not stop twice.
    if (!g_stop_invoked) (void) stop_observation_window(false);
    else g_fini_stop_skipped = true;
    pthread_mutex_lock(&g_lock);
    g_finalized = true; g_finalize_ns = now_ns();
    pthread_mutex_unlock(&g_lock);
    if (g_output[0]) (void) write_report(g_output);
}
}  // namespace

extern "C" __attribute__((visibility("default")))
rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t version, const char* text,
    uint32_t priority, rocprofiler_client_id_t* client) {
    g_configured = true; g_sdk_version = version; g_priority = priority;
    if (client) client->name = "mlsys-native-allocation-diagnostic";
    if (!copy_text(g_sdk_string, sizeof(g_sdk_string), text ? text : "")) g_metadata_truncated = true;
    const char* output = std::getenv("MLSYS_NATIVE_PROBE_OUT");
    if (output && !copy_text(g_output, sizeof(g_output), output)) g_metadata_truncated = true;
    static rocprofiler_tool_configure_result_t config{sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, nullptr};
    return &config;
}

extern "C" __attribute__((visibility("default")))
int mlsys_probe_begin(uint64_t epoch, const char* label) {
    pthread_mutex_lock(&g_lock);
    if (!g_ready || g_stop_invoked || g_finalized || !epoch || g_active_epoch >= 0 || g_epoch_count >= kEpochs ||
        !label || strnlen(label, kLabel) >= kLabel) { pthread_mutex_unlock(&g_lock); return EINVAL; }
    for (size_t i = 0; i < g_epoch_count; ++i) if (g_epochs[i].id == epoch) {
        pthread_mutex_unlock(&g_lock); return EEXIST;
    }
    Epoch& e = g_epochs[g_epoch_count];
    e.id = epoch; e.begin_ns = now_ns(); e.dropped_at_begin = g_dropped;
    e.recursive_at_begin = g_recursive.load(); copy_text(e.label, sizeof(e.label), label);
    g_active_epoch = int(g_epoch_count++);
    pthread_mutex_unlock(&g_lock); return 0;
}
extern "C" __attribute__((visibility("default")))
int mlsys_probe_end(uint64_t epoch) {
    pthread_mutex_lock(&g_lock);
    if (g_active_epoch < 0 || g_epochs[g_active_epoch].id != epoch || g_finalized) {
        pthread_mutex_unlock(&g_lock); return EINVAL;
    }
    Epoch& e = g_epochs[g_active_epoch];
    e.end_ns = now_ns(); e.ended = true; e.dropped_at_end = g_dropped;
    e.recursive_at_end = g_recursive.load(); g_active_epoch = -1;
    pthread_mutex_unlock(&g_lock); return 0;
}
extern "C" __attribute__((visibility("default")))
int mlsys_probe_write(const char* path) { return write_report(path); }

extern "C" __attribute__((visibility("default")))
int mlsys_probe_stop() { return stop_observation_window(true); }
