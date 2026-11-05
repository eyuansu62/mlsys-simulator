// Diagnostic only: this tool never produces a complete Workspace ledger.
// Compile against the machine's installed ROCprofiler-SDK 1.4.1 headers.
// Actual GCC13 std_thread.h uses this implementation guard solely to expose
// its declared _State type. No layout/dispatch offset is invented here.
#define _GLIBCXX_THREAD_IMPL 1
#include <thread>
#undef _GLIBCXX_THREAD_IMPL
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
#include <dlfcn.h>
#include <optional>
#include <map>
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

namespace {
constexpr size_t kDomains = 9, kOps = 8192, kEvents = 4194304, kEpochs = 64;
constexpr size_t kName = 192, kLabel = 256, kPath = 4096, kAgents = 64, kThreadOrigins = 256;
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
    uint64_t rccl_comm, rccl_stream, rccl_count, rccl_output_address;
    uint64_t rccl_send, rccl_receive;
    int64_t rccl_datatype, rccl_peer, rccl_root, rccl_world, rccl_rank;
    uint8_t rccl_payload_kind;  // 0=absent, 1=collective, 2=init output, 3=group
    bool rccl_comm_known, rccl_peer_known, rccl_root_known;
    bool rccl_send_known, rccl_receive_known;
    uint64_t thread_origin_id;
    uint64_t hip_object, hip_output_address, hip_stream, hip_pointer, hip_size, hip_flags, hip_value, hip_props_address;
    uint64_t hip_pool_max_size;
    int64_t hip_device_index;
    uint64_t hip_device_source_cid;
    int64_t hip_attribute, hip_alloc_type, hip_handle_types, hip_location_type, hip_location_id;
    uint8_t hip_payload_kind;  // 1=event, 2=pool create, 3=pool attribute, 4=host registration
    bool hip_object_known, hip_value_known, hip_properties_known, hip_reserved_zero;
    uint64_t hsa_signal_handle, hsa_signal_output, hsa_consumers[4];
    int64_t hsa_initial_value;
    uint32_t hsa_num_consumers;
    uint8_t hsa_signal_kind;
    bool hsa_signal_known, hsa_consumers_observed;
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
pthread_mutex_t g_finalize_lock = PTHREAD_MUTEX_INITIALIZER;
thread_local bool g_in_callback = false;
struct InitCallScope { uint64_t cid, epoch, tid, output_address; };
thread_local InitCallScope g_init_call_stack[8]{};
thread_local size_t g_init_call_depth = 0;
thread_local uint64_t g_thread_origin_id = 0;
thread_local int64_t g_current_device_observed = -1;
thread_local uint64_t g_current_device_source_cid = 0;
struct ThreadOrigin {
    uint64_t id, parent_init_cid, parent_epoch, parent_tid, init_output_address;
    uint64_t start_address, module_base, module_offset, argument_address;
    uint64_t create_begin_ns, create_end_ns, child_start_ns, child_end_ns, child_tid, start_invocations;
    int create_return = -999;
    bool module_known, module_path_canonical;
    char module_path[kPath];
    bool stdthread_trampoline_matched, state_dispatch_observed;
    uint64_t state_vtable, state_typeinfo, state_dispatch, run_member_offset;
    uint64_t vtable_module_base, typeinfo_module_base, dispatch_module_base;
    char vtable_module[kPath], typeinfo_module[kPath], dispatch_module[kPath];
};
ThreadOrigin g_thread_origins[kThreadOrigins]; size_t g_thread_origin_count = 0;
pthread_mutex_t g_thread_origin_lock = PTHREAD_MUTEX_INITIALIZER;
std::atomic<uint64_t> g_thread_origin_errors{0}, g_thread_origin_dropped{0}, g_init_stack_errors{0};
using PthreadCreate = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
std::atomic<PthreadCreate> g_real_pthread_create{nullptr};
struct ThreadStartContext { void* (*start)(void*); void* argument; size_t index; };
struct StdThreadControl {
    bool attempted, completed, trampoline_observed;
    int status = -999;
    uint64_t start_address, module_base, module_offset, run_member_offset, member_pointer_size, state_size;
    char module_path[kPath];
};
StdThreadControl g_stdthread_control;
pthread_mutex_t g_stdthread_control_lock = PTHREAD_MUTEX_INITIALIZER;
thread_local bool g_stdthread_control_active = false;
std::atomic<uint64_t> g_callbacks_inflight{0}, g_recursive{0};
std::atomic<uint64_t> g_open_api_calls{0}, g_open_api_underflows{0}, g_api_enters_after_sdk_stop{0};
std::atomic<uint64_t> g_stop_sdk_return_ns{0};
uint64_t g_open_apis_at_sdk_stop_return = 0, g_open_apis_at_window_close = 0, g_drain_wait_iterations = 0;
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
std::optional<rocprofiler_client_id_t> g_client_id;
rocprofiler_client_finalize_t g_client_finalize = nullptr;
bool g_explicit_finalize_requested = false;
uint64_t g_explicit_finalize_begin_ns = 0, g_fini_callback_count = 0;
int g_final_write_status = -999;
rocprofiler_context_id_t g_context{0};
int g_create_status = -999, g_valid_status = -999, g_context_valid = 0;
int g_start_status = -999, g_stop_status = -999;
uint64_t g_init_ns = 0, g_finalize_ns = 0;

uint64_t now_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t) != 0) return 0;
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}
bool read_pointer_word(uint64_t address, uint64_t& value) {
    if (!address) return false;
    iovec local{&value, sizeof(value)}, remote{reinterpret_cast<void*>(uintptr_t(address)), sizeof(value)};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == ssize_t(sizeof(value));
}
bool module_for_address(uint64_t address, char* path, uint64_t& base) {
    Dl_info module{};
    if (!address || !dladdr(reinterpret_cast<void*>(uintptr_t(address)), &module) || !module.dli_fname || !module.dli_fbase)
        return false;
    base = uint64_t(reinterpret_cast<uintptr_t>(module.dli_fbase));
    return realpath(module.dli_fname, path) != nullptr;
}
void observe_stdthread_dispatch(ThreadOrigin& origin, void* argument) {
    pthread_mutex_lock(&g_stdthread_control_lock);
    const bool matched = g_stdthread_control.completed &&
        origin.start_address == g_stdthread_control.start_address;
    const uint64_t run_offset = g_stdthread_control.run_member_offset;
    pthread_mutex_unlock(&g_stdthread_control_lock);
    if (!matched) return;
    origin.stdthread_trampoline_matched = true; origin.run_member_offset = run_offset;
    // Only after the observed CPU control identifies the exact runtime
    // trampoline, inspect its declared _State dispatch using the compiler's
    // actual virtual member-pointer encoding. process_vm_readv avoids blindly
    // dereferencing an opaque caller argument or inaccessible/absent RTTI.
    origin.state_dispatch_observed = read_pointer_word(uint64_t(reinterpret_cast<uintptr_t>(argument)), origin.state_vtable)
        && origin.state_vtable >= sizeof(uint64_t)
        && read_pointer_word(origin.state_vtable - sizeof(uint64_t), origin.state_typeinfo)
        && read_pointer_word(origin.state_vtable + run_offset, origin.state_dispatch)
        && module_for_address(origin.state_vtable, origin.vtable_module, origin.vtable_module_base)
        && module_for_address(origin.state_typeinfo, origin.typeinfo_module, origin.typeinfo_module_base)
        && module_for_address(origin.state_dispatch, origin.dispatch_module, origin.dispatch_module_base);
}
void* observed_thread_start(void* opaque) {
    const int entered_errno = errno;
    auto* context = static_cast<ThreadStartContext*>(opaque);
    const auto start = context->start;
    void* argument = context->argument;
    const size_t index = context->index;
    std::free(context);  // Do not retain an additional user/context reference.
    pthread_mutex_lock(&g_thread_origin_lock);
    auto& record = g_thread_origins[index];
    record.child_tid = uint64_t(syscall(SYS_gettid)); record.child_start_ns = now_ns();
    ++record.start_invocations; g_thread_origin_id = record.id;
    pthread_mutex_unlock(&g_thread_origin_lock);
    errno = entered_errno;
    void* result = start(argument);  // Exactly the original routine, argument and return.
    const int returned_errno = errno;
    pthread_mutex_lock(&g_thread_origin_lock); record.child_end_ns = now_ns(); pthread_mutex_unlock(&g_thread_origin_lock);
    errno = returned_errno;
    return result;
}
int observe_pthread_create(pthread_t* thread, const pthread_attr_t* attributes,
                           void* (*start)(void*), void* argument) {
    const int entered_errno = errno;
    auto real = g_real_pthread_create.load(std::memory_order_acquire);
    if (!real) {
        real = reinterpret_cast<PthreadCreate>(dlsym(RTLD_NEXT, "pthread_create"));
        if (!real) { g_thread_origin_errors.fetch_add(1); return EAGAIN; }
        g_real_pthread_create.store(real, std::memory_order_release);
    }
    if (g_stdthread_control_active) {
        auto& control = g_stdthread_control;
        control.start_address = uint64_t(reinterpret_cast<uintptr_t>(start));
        control.trampoline_observed = module_for_address(control.start_address, control.module_path, control.module_base);
        if (control.trampoline_observed && control.start_address >= control.module_base)
            control.module_offset = control.start_address - control.module_base;
        else control.trampoline_observed = false;
        errno = entered_errno; return real(thread, attributes, start, argument);
    }
    if (g_in_callback || !g_init_call_depth) { errno = entered_errno; return real(thread, attributes, start, argument); }
    const auto parent = g_init_call_stack[g_init_call_depth - 1];
    auto* context = static_cast<ThreadStartContext*>(std::malloc(sizeof(ThreadStartContext)));
    if (!context) {
        g_thread_origin_dropped.fetch_add(1);
        errno = entered_errno;
        return real(thread, attributes, start, argument);  // Runtime behavior unchanged; evidence becomes partial.
    }
    Dl_info module{};
    const uint64_t address = uint64_t(reinterpret_cast<uintptr_t>(start));
    const bool known = dladdr(reinterpret_cast<void*>(start), &module) != 0 && module.dli_fbase && module.dli_fname;
    pthread_mutex_lock(&g_thread_origin_lock);
    if (g_thread_origin_count >= kThreadOrigins) {
        pthread_mutex_unlock(&g_thread_origin_lock); std::free(context); g_thread_origin_dropped.fetch_add(1);
        errno = entered_errno; return real(thread, attributes, start, argument);
    }
    const size_t index = g_thread_origin_count++;
    auto& record = g_thread_origins[index];
    record.id = index + 1; record.parent_init_cid = parent.cid; record.parent_epoch = parent.epoch;
    record.parent_tid = parent.tid; record.init_output_address = parent.output_address;
    record.start_address = address; record.argument_address = uint64_t(reinterpret_cast<uintptr_t>(argument));
    record.module_known = known;
    if (known) {
        record.module_base = uint64_t(reinterpret_cast<uintptr_t>(module.dli_fbase));
        if (address >= record.module_base) record.module_offset = address - record.module_base;
        else record.module_known = false;
        if (realpath(module.dli_fname, record.module_path)) record.module_path_canonical = true;
        else if (strnlen(module.dli_fname, kPath) < kPath)
            std::memcpy(record.module_path, module.dli_fname, std::strlen(module.dli_fname) + 1);
        else { record.module_known = false; g_thread_origin_errors.fetch_add(1); }
    }
    observe_stdthread_dispatch(record, argument);
    record.create_begin_ns = now_ns();
    context->start = start; context->argument = argument; context->index = index;
    pthread_mutex_unlock(&g_thread_origin_lock);
    errno = entered_errno;
    const int status = real(thread, attributes, observed_thread_start, context);
    const int returned_errno = errno;
    pthread_mutex_lock(&g_thread_origin_lock); record.create_return = status; record.create_end_ns = now_ns(); pthread_mutex_unlock(&g_thread_origin_lock);
    if (status != 0) std::free(context);  // POSIX guarantees the child was not created on failure.
    errno = returned_errno;
    return status;
}
bool copy_text(char* dst, size_t size, const char* src) {
    if (!src || size == 0) return false;
    size_t n = strnlen(src, size);
    size_t take = n < size ? n : size - 1;
    std::memcpy(dst, src, take); dst[take] = '\0';
    return n < size;
}
// A worker may inherit one template, but every process must publish its own
// file. Only the literal %p token is recognized; this is not a printf format.
// Keep this small function self-contained for the CPU formatter regression.
// MLSYS_OUTPUT_PATH_FORMATTER_BEGIN
bool expand_output_path(char* dst, size_t capacity, const char* source) {
    if (!dst || !capacity) return false;
    dst[0] = '\0';
    if (!source || strnlen(source, capacity) >= capacity) return false;
    char pid[32]{};
    int length = std::snprintf(pid, sizeof(pid), "%ld", long(getpid()));
    if (length <= 0 || size_t(length) >= sizeof(pid)) return false;
    size_t used = 0;
    for (size_t i = 0; source[i]; ++i) {
        const char* piece = source + i;
        size_t count = 1;
        if (source[i] == '%') {
            if (source[i + 1] != 'p') { dst[0] = '\0'; return false; }
            ++i; piece = pid; count = size_t(length);
        }
        if (count >= capacity - used) { dst[0] = '\0'; return false; }
        std::memcpy(dst + used, piece, count); used += count;
    }
    dst[used] = '\0';
    return true;
}
// MLSYS_OUTPUT_PATH_FORMATTER_END
bool has(const char* s, const char* part) { return std::strstr(s, part) != nullptr; }
bool is(const char* s, const char* value) { return std::strcmp(s, value) == 0; }
uint32_t classify(const char* name, size_t domain, uint8_t& status_type) {
    char n[kName]{};
    for (size_t i = 0; name[i] && i + 1 < sizeof(n); ++i)
        n[i] = name[i] >= 'A' && name[i] <= 'Z' ? char(name[i] + ('a' - 'A')) : name[i];
    status_type = 0;
    if (domain == 0) {
        const char* queries[] = {"hipctxgetcurrent", "hipctxgetdevice", "hipctxgetapiversion",
                                "hipgetdevice", "hipgetdevicecount", "hipdevicegetattribute", "hipdeviceprimaryctxgetstate",
                                "hipmemgetaddressrange"};
        for (const char* query : queries) if (is(n, query)) { status_type = 1; return MemoryQuery; }
        // "hipCtx" contains the letters "ipc" but is not an IPC API. Context
        // changes still retain lifecycle classification; they are not queries.
        if (std::strncmp(n, "hipctx", 6) == 0 || std::strncmp(n, "hipdeviceprimaryctx", 19) == 0)
            return Lifecycle;
    }
    if (domain == 8) {
        // These public RCCL entry points return ncclResult_t. Error-string
        // functions return const char*, so their union is deliberately not decoded.
        const char* known_result[] = {"ncclgetversion", "ncclgetuniqueid", "ncclcomminitrank",
            "ncclcomminitall", "ncclcomminitrankconfig", "ncclcommfinalize", "ncclcommdestroy",
            "ncclcommabort", "ncclcommsplit", "ncclmemalloc", "ncclmemfree", "ncclcommcount",
            "ncclcommcudevice", "ncclcommuserrank", "ncclcommgetasyncerror",
            "ncclallreduce", "ncclallgather", "ncclbroadcast", "ncclreduce", "ncclreducescatter",
            "ncclsend", "ncclrecv", "ncclgroupstart", "ncclgroupend"};
        for (const char* candidate : known_result) if (is(n, candidate)) { status_type = 3; break; }
        if (is(n, "ncclgetversion") || is(n, "ncclgeterrorstring") || is(n, "ncclgetlasterror") ||
            is(n, "ncclcommcount") || is(n, "ncclcommcudevice") || is(n, "ncclcommuserrank") ||
            is(n, "ncclcommgetasyncerror")) return MemoryQuery;
        uint32_t flags = Lifecycle;
        if (is(n, "ncclmemalloc")) flags |= Allocate;
        if (is(n, "ncclmemfree")) flags |= Free;
        return flags;  // Typed collectives still retain conservative lifecycle flags.
    }
    if (domain == 7) return PrivateScratch;
    if (domain == 6) {
        uint32_t f = has(n, "vmem") ? uint32_t(Vmm) : uint32_t(0);
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
    if (domain == 0 && (is(n, "hipeventcreatewithflags") || is(n, "hipeventcreate") ||
        is(n, "hipeventdestroy") || is(n, "hipeventquery") || is(n, "hipeventrecord") || is(n, "hipeventrecord_spt") ||
        is(n, "hipeventrecordwithflags") || is(n, "hipeventsynchronize") ||
        is(n, "hipstreamwaitevent") || is(n, "hipstreamwaitevent_spt") ||
        is(n, "hipextmodulelaunchkernel") || is(n, "hipmodulelaunchkernel") || is(n, "hiplaunchkernel") ||
        is(n, "hipsetdevice"))) status_type = 1;
    if (domain >= 2 && (has(n, "hsa_memory_") || has(n, "hsa_amd_memory_") ||
        has(n, "hsa_amd_vmem_") || has(n, "hsa_amd_ipc_") ||
        has(n, "hsa_amd_svm_") || has(n, "hsa_ext_image_") ||
        is(n, "hsa_signal_create") || is(n, "hsa_signal_destroy"))) status_type = 2;
    uint32_t f = 0;
    const bool ipc_api = (domain == 0 && std::strncmp(n, "hipipc", 6) == 0) ||
                         (domain == 3 && std::strncmp(n, "hsa_amd_ipc_", 12) == 0);
    const bool memory_surface = has(n, "mem") || has(n, "malloc") || has(n, "free") ||
        has(n, "alloc") || ipc_api || has(n, "svm");
    const bool query = has(n, "getinfo") || has(n, "get_info") ||
        has(n, "getattribute") || has(n, "getaccess") || has(n, "getallocationgranularity") ||
        has(n, "pointerinfo") || has(n, "pointer_info") || has(n, "pointerget") ||
        has(n, "getdefaultmempool") || has(n, "getmempool");
    if (has(n, "vmem") || has(n, "hipmemcreate") || has(n, "hipmemrelease") ||
        has(n, "hipmemmap") || has(n, "hipmemunmap") || has(n, "hipmemaddress") ||
        has(n, "hipmemretain") || has(n, "hipmemsetaccess")) f |= Vmm;
    if (ipc_api || has(n, "externalmemory") || has(n, "shareablehandle") ||
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
    if (rc != ROCPROFILER_STATUS_SUCCESS || !name || operation < 0 || size_t(operation) >= kOps) {
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
void capture_rccl_payload(Event& event, const rocprofiler_callback_tracing_rccl_api_data_t& data,
                          const char* name) {
    // Field names and types are from the installed SDK api_args.h (SHA256
    // 3e4fa86c4d9e7aee00f5c83d7c7d430eba9fa3bafa2e6fb1d877b21e94834842).
    // Never read an init output pointer on ENTER or after a failed return.
    auto collective = [&](auto comm, auto stream, size_t count, auto datatype) {
        event.rccl_payload_kind = 1;
        event.rccl_comm = uint64_t(reinterpret_cast<uintptr_t>(comm));
        event.rccl_comm_known = true;
        event.rccl_stream = uint64_t(reinterpret_cast<uintptr_t>(stream));
        event.rccl_count = uint64_t(count); event.rccl_datatype = int64_t(datatype);
    };
    if (is(name, "ncclAllReduce")) {
        const auto& a = data.args.ncclAllReduce; collective(a.comm, a.stream, a.count, a.datatype);
        event.rccl_send = uint64_t(reinterpret_cast<uintptr_t>(a.sendbuff)); event.rccl_receive = uint64_t(reinterpret_cast<uintptr_t>(a.recvbuff));
        event.rccl_send_known = event.rccl_receive_known = true;
    } else if (is(name, "ncclAllGather")) {
        const auto& a = data.args.ncclAllGather; collective(a.comm, a.stream, a.sendcount, a.datatype);
        event.rccl_send = uint64_t(reinterpret_cast<uintptr_t>(a.sendbuff)); event.rccl_receive = uint64_t(reinterpret_cast<uintptr_t>(a.recvbuff));
        event.rccl_send_known = event.rccl_receive_known = true;
    } else if (is(name, "ncclBroadcast")) {
        const auto& a = data.args.ncclBroadcast; collective(a.comm, a.stream, a.count, a.datatype);
        event.rccl_send = uint64_t(reinterpret_cast<uintptr_t>(a.sendbuff)); event.rccl_receive = uint64_t(reinterpret_cast<uintptr_t>(a.recvbuff));
        event.rccl_send_known = event.rccl_receive_known = true;
        event.rccl_root = a.root; event.rccl_root_known = true;
    } else if (is(name, "ncclReduce")) {
        const auto& a = data.args.ncclReduce; collective(a.comm, a.stream, a.count, a.datatype);
        event.rccl_send = uint64_t(reinterpret_cast<uintptr_t>(a.sendbuff)); event.rccl_receive = uint64_t(reinterpret_cast<uintptr_t>(a.recvbuff));
        event.rccl_send_known = event.rccl_receive_known = true;
        event.rccl_root = a.root; event.rccl_root_known = true;
    } else if (is(name, "ncclReduceScatter")) {
        const auto& a = data.args.ncclReduceScatter; collective(a.comm, a.stream, a.recvcount, a.datatype);
        event.rccl_send = uint64_t(reinterpret_cast<uintptr_t>(a.sendbuff)); event.rccl_receive = uint64_t(reinterpret_cast<uintptr_t>(a.recvbuff));
        event.rccl_send_known = event.rccl_receive_known = true;
    } else if (is(name, "ncclSend")) {
        const auto& a = data.args.ncclSend; collective(a.comm, a.stream, a.count, a.datatype);
        event.rccl_send = uint64_t(reinterpret_cast<uintptr_t>(a.sendbuff)); event.rccl_send_known = true;
        event.rccl_peer = a.peer; event.rccl_peer_known = true;
    } else if (is(name, "ncclRecv")) {
        const auto& a = data.args.ncclRecv; collective(a.comm, a.stream, a.count, a.datatype);
        event.rccl_receive = uint64_t(reinterpret_cast<uintptr_t>(a.recvbuff)); event.rccl_receive_known = true;
        event.rccl_peer = a.peer; event.rccl_peer_known = true;
    } else if (is(name, "ncclCommInitRank") || is(name, "ncclCommInitRankConfig")) {
        ncclComm_t* output = nullptr;
        event.rccl_payload_kind = 2;
        if (is(name, "ncclCommInitRank")) {
            const auto& a = data.args.ncclCommInitRank;
            output = a.newcomm; event.rccl_world = a.nranks; event.rccl_rank = a.myrank;
        } else {
            const auto& a = data.args.ncclCommInitRankConfig;
            output = a.comm; event.rccl_world = a.nranks; event.rccl_rank = a.myrank;
        }
        event.rccl_output_address = uint64_t(reinterpret_cast<uintptr_t>(output));
        if (event.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && event.status_type == 3 && event.status == 0 && output) {
            event.rccl_comm = uint64_t(reinterpret_cast<uintptr_t>(*output));
            event.rccl_comm_known = true;
        }
    } else if (is(name, "ncclGroupStart") || is(name, "ncclGroupEnd")) event.rccl_payload_kind = 3;
}
void capture_hip_control_payload(Event& event, const rocprofiler_callback_tracing_hip_api_data_t& data,
                                 const char* name) {
    // Installed SDK hip/api_args.h SHA f3d5cb278d2d13233ac8848e5eb14a68d5b175af48a3233bf706c237bb60cf26.
    const bool success = event.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && event.status_type == 1 && event.status == 0;
    auto event_create = [&](hipEvent_t* output, unsigned flags) {
        event.hip_payload_kind = 1; event.hip_output_address = uint64_t(reinterpret_cast<uintptr_t>(output));
        event.hip_flags = flags;
        if (success && output) { event.hip_object = uint64_t(reinterpret_cast<uintptr_t>(*output)); event.hip_object_known = true; }
    };
    auto event_use = [&](hipEvent_t handle, hipStream_t stream) {
        event.hip_payload_kind = 1; event.hip_object = uint64_t(reinterpret_cast<uintptr_t>(handle));
        event.hip_object_known = true; event.hip_stream = uint64_t(reinterpret_cast<uintptr_t>(stream));
    };
    if (is(name, "hipEventCreateWithFlags")) {
        event_create(data.args.hipEventCreateWithFlags.event, data.args.hipEventCreateWithFlags.flags);
    } else if (is(name, "hipEventCreate")) event_create(data.args.hipEventCreate.event, 0);
    else if (is(name, "hipEventDestroy")) event_use(data.args.hipEventDestroy.event, nullptr);
    else if (is(name, "hipEventQuery")) event_use(data.args.hipEventQuery.event, nullptr);
    else if (is(name, "hipEventRecord")) event_use(data.args.hipEventRecord.event, data.args.hipEventRecord.stream);
    else if (is(name, "hipEventRecord_spt")) event_use(data.args.hipEventRecord_spt.event, data.args.hipEventRecord_spt.stream);
    else if (is(name, "hipEventSynchronize")) event_use(data.args.hipEventSynchronize.event, nullptr);
    else if (is(name, "hipStreamWaitEvent")) {
        event_use(data.args.hipStreamWaitEvent.event, data.args.hipStreamWaitEvent.stream);
        event.hip_flags = data.args.hipStreamWaitEvent.flags;
    } else if (is(name, "hipStreamWaitEvent_spt")) {
        event_use(data.args.hipStreamWaitEvent_spt.event, data.args.hipStreamWaitEvent_spt.stream);
        event.hip_flags = data.args.hipStreamWaitEvent_spt.flags;
    }
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
    else if (is(name, "hipEventRecordWithFlags")) {
        event_use(data.args.hipEventRecordWithFlags.event, data.args.hipEventRecordWithFlags.stream);
        event.hip_flags = data.args.hipEventRecordWithFlags.flags;
    }
#endif
    else if (is(name, "hipGetDevice") || is(name, "hipCtxGetDevice")) {
        event.hip_payload_kind = 5;
        event.hip_device_index = -1; event.hip_device_source_cid = 0;
        const int* output = is(name, "hipGetDevice") ? data.args.hipGetDevice.deviceId : data.args.hipCtxGetDevice.device;
        event.hip_output_address = uint64_t(reinterpret_cast<uintptr_t>(output));
        if (success && output) {
            event.hip_device_index = *output;
            event.hip_device_source_cid = event.correlation;
            g_current_device_observed = *output; g_current_device_source_cid = event.correlation;
        }
    } else if (is(name, "hipSetDevice")) {
        event.hip_payload_kind = 6; event.hip_device_index = data.args.hipSetDevice.deviceId; event.hip_device_source_cid = 0;
        if (success) {
            g_current_device_observed = event.hip_device_index; g_current_device_source_cid = event.correlation;
            event.hip_device_source_cid = event.correlation;
        }
    } else if (is(name, "hipDeviceSynchronize")) event.hip_payload_kind = 7;
    else if (is(name, "hipStreamSynchronize")) {
        event.hip_payload_kind = 8;
        event.hip_stream = uint64_t(reinterpret_cast<uintptr_t>(data.args.hipStreamSynchronize.stream));
    } else if (std::strncmp(name, "hipCtx", 6) == 0 && !is(name, "hipCtxGetCurrent") && !is(name, "hipCtxGetApiVersion")) {
        g_current_device_observed = -1; g_current_device_source_cid = 0;
    } else if (is(name, "hipMemPoolCreate")) {
        const auto& a = data.args.hipMemPoolCreate; event.hip_payload_kind = 2;
        event.hip_output_address = uint64_t(reinterpret_cast<uintptr_t>(a.mem_pool));
        event.hip_props_address = uint64_t(reinterpret_cast<uintptr_t>(a.pool_props));
        if (success && a.mem_pool && a.pool_props) {
            event.hip_object = uint64_t(reinterpret_cast<uintptr_t>(*a.mem_pool)); event.hip_object_known = true;
            event.hip_alloc_type = int64_t(a.pool_props->allocType); event.hip_handle_types = int64_t(a.pool_props->handleTypes);
            event.hip_location_type = int64_t(a.pool_props->location.type); event.hip_location_id = a.pool_props->location.id;
            event.hip_pool_max_size = a.pool_props->maxSize; event.hip_properties_known = true;
            event.hip_reserved_zero = a.pool_props->win32SecurityAttributes == nullptr;
            for (unsigned char value : a.pool_props->reserved) if (value) event.hip_reserved_zero = false;
        }
    } else if (is(name, "hipMemPoolSetAttribute")) {
        const auto& a = data.args.hipMemPoolSetAttribute; event.hip_payload_kind = 3;
        event.hip_object = uint64_t(reinterpret_cast<uintptr_t>(a.mem_pool)); event.hip_object_known = true;
        event.hip_attribute = int64_t(a.attr); event.hip_pointer = uint64_t(reinterpret_cast<uintptr_t>(a.value));
        if (success && a.value && a.attr == hipMemPoolAttrReleaseThreshold) {
            event.hip_value = *static_cast<const uint64_t*>(a.value); event.hip_value_known = true;
        }
    } else if (is(name, "hipMemPoolDestroy")) {
        event.hip_payload_kind = 9;
        event.hip_object = uint64_t(reinterpret_cast<uintptr_t>(data.args.hipMemPoolDestroy.mem_pool));
        event.hip_object_known = true;
    } else if (is(name, "hipHostRegister")) {
        const auto& a = data.args.hipHostRegister; event.hip_payload_kind = 4;
        event.hip_pointer = uint64_t(reinterpret_cast<uintptr_t>(a.hostPtr)); event.hip_size = a.sizeBytes; event.hip_flags = a.flags;
    } else if (is(name, "hipHostUnregister")) {
        event.hip_payload_kind = 10;
        event.hip_pointer = uint64_t(reinterpret_cast<uintptr_t>(data.args.hipHostUnregister.hostPtr));
    }
}
void capture_hsa_signal_payload(Event& event, const rocprofiler_callback_tracing_hsa_api_data_t& data,
                                const char* name) {
    // Installed hsa/api_args.h SHA 6554bb62453be665ed86995fe315245b9d1125f308a7ae668e71109e7ca9c4e0.
    if (is(name, "hsa_signal_create")) {
        const auto& a = data.args.hsa_signal_create;
        event.hsa_signal_kind = 1; event.hsa_initial_value = a.initial_value;
        event.hsa_num_consumers = a.num_consumers;
        event.hsa_signal_output = uint64_t(reinterpret_cast<uintptr_t>(a.signal));
        if (event.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && event.status_type == 2 && event.status == 0) {
            if (a.signal) { event.hsa_signal_handle = a.signal->handle; event.hsa_signal_known = true; }
            if (a.num_consumers <= 4 && (!a.num_consumers || a.consumers)) {
                event.hsa_consumers_observed = true;
                for (size_t i = 0; i < a.num_consumers; ++i) event.hsa_consumers[i] = a.consumers[i].handle;
            }
        }
    } else if (is(name, "hsa_signal_destroy")) {
        event.hsa_signal_kind = 2; event.hsa_signal_handle = data.args.hsa_signal_destroy.signal.handle;
        event.hsa_signal_known = true;
    }
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
    const bool ordinary_api = di < 6 || di == 8;
    if (ordinary_api && record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        g_open_api_calls.fetch_add(1, std::memory_order_acq_rel);
        const uint64_t stopped = g_stop_sdk_return_ns.load(std::memory_order_acquire);
        if (stopped && timestamp > stopped) g_api_enters_after_sdk_stop.fetch_add(1);
    } else if (ordinary_api && record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
        uint64_t count = g_open_api_calls.load(std::memory_order_acquire);
        while (count && !g_open_api_calls.compare_exchange_weak(count, count - 1, std::memory_order_acq_rel)) {}
        if (!count) g_open_api_underflows.fetch_add(1);
    }
    Op* op = di < kDomains && record.operation >= 0 && size_t(record.operation) < kOps && g_domains[di].ops[record.operation].present
        ? &g_domains[di].ops[record.operation] : nullptr;
    Event event{};
    event.hip_device_index = g_current_device_observed;
    event.hip_device_source_cid = g_current_device_source_cid;
    event.thread_origin_id = g_thread_origin_id;
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
    if (di == 8 && op && record.payload) {
        const auto* p = static_cast<const rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
        if (p->size >= sizeof(*p)) capture_rccl_payload(event, *p, op->name);
    }
    if (di == 0 && op && record.payload) {
        const auto* p = static_cast<const rocprofiler_callback_tracing_hip_api_data_t*>(record.payload);
        if (p->size >= sizeof(*p)) capture_hip_control_payload(event, *p, op->name);
        if (event.hip_payload_kind && event.hip_payload_kind != 5 && event.hip_payload_kind != 6) {
            event.hip_device_index = g_current_device_observed; event.hip_device_source_cid = g_current_device_source_cid;
        }
    }
    if (di == 2 && op && record.payload) {
        const auto* p = static_cast<const rocprofiler_callback_tracing_hsa_api_data_t*>(record.payload);
        if (p->size >= sizeof(*p)) capture_hsa_signal_payload(event, *p, op->name);
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
    if (di == 8 && op && (is(op->name, "ncclCommInitRank") || is(op->name, "ncclCommInitRankConfig"))) {
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
            if (g_init_call_depth < 8)
                g_init_call_stack[g_init_call_depth++] = {event.correlation, event.epoch, event.thread_id, event.rccl_output_address};
            else g_init_stack_errors.fetch_add(1);
        } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
            if (g_init_call_depth && g_init_call_stack[g_init_call_depth - 1].cid == event.correlation) --g_init_call_depth;
            else g_init_stack_errors.fetch_add(1);
        }
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
        "\"max_events\":%zu,\"event_size_bytes\":%zu,\"event_storage_capacity_bytes\":%zu,\"active_event_byte_bound\":%zu,"
        "\"observed_callbacks\":%" PRIu64 ",\"filtered_nonmutator_events\":%" PRIu64 ",\"missing_user_data\":%" PRIu64 ",\"attempted_events\":%" PRIu64 ",\"retained_events\":%zu,\"dropped_events\":%" PRIu64 ","
        "\"recursive_suppressed\":%" PRIu64 ",\"unknown_events\":%" PRIu64 ",\"metadata_truncated\":%s,\"bounds_valid\":%s,\n\"domains\":[",
        g_priority, g_configured ? "true" : "false", g_init_called ? "true" : "false", g_ready ? "true" : "false",
        g_init_ns, g_create_status, g_valid_status, g_context_valid, g_start_status, g_stop_status,
        g_finalized ? "true" : "false", g_finalize_ns, g_callbacks_inflight.load(), g_limit, sizeof(Event), sizeof(g_events), g_limit * sizeof(Event), g_observed, g_filtered, g_missing_user_data, g_attempted, g_count,
        g_dropped, g_recursive.load(), g_unknown, g_metadata_truncated ? "true" : "false", g_bounds_valid ? "true" : "false");
    for (size_t i = 0; i < kDomains; ++i) {
        Domain& d = g_domains[i];
        if (i) std::fputc(',', f);
        std::fprintf(f, "{\"index\":%zu,\"kind\":%d,\"name\":", i, int(d.kind)); json_text(f, d.name);
        std::fprintf(f, ",\"enumerate_status\":%d,\"configure_status\":%d,\"operation_count\":%zu,\"operations\":[", d.enumerate_status, d.configure_status, d.op_count);
        bool first = true;
        for (size_t j = 0; j < kOps; ++j) if (d.ops[j].present) {
            if (!first) std::fputc(',', f);
            first = false;
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
    std::fprintf(f, ",\"api_call_drain_evidence\":{\"method\":\"ordinary_api_enter_exit_counter_v1\","
        "\"ordinary_domains\":[0,1,2,3,4,5,8],\"normalized_and_scratch_counted\":false,"
        "\"sdk_stop_return_ns\":%" PRIu64 ",\"window_closed_ns\":%" PRIu64 ","
        "\"open_calls_at_sdk_stop_return\":%" PRIu64 ",\"open_calls_at_window_close\":%" PRIu64 ","
        "\"counter_underflows\":%" PRIu64 ",\"api_enters_after_sdk_stop_return\":%" PRIu64 ","
        "\"wait_iterations\":%" PRIu64 ",\"timeout_ns\":1000000000}",
        g_stop_sdk_return_ns.load(), g_stop_end_ns, g_open_apis_at_sdk_stop_return, g_open_apis_at_window_close,
        g_open_api_underflows.load(), g_api_enters_after_sdk_stop.load(), g_drain_wait_iterations);
    std::fputs(",\n\"retention_mode\":", f); json_text(f, g_retain_all ? "all_callbacks_v1" : "mutators_and_epoch_v1");
    std::fprintf(f, ",\"retain_all_option_valid\":%s,\n\"filter_policy\":", g_retain_option_valid ? "true" : "false");
    json_text(f, g_retain_all ? "all_APIs_global" : "all_mutators_global_and_all_APIs_in_or_entered_epoch");
    pthread_mutex_lock(&g_thread_origin_lock);
    std::fprintf(f, ",\"rccl_thread_origins\":{\"schema_version\":1,\"method\":\"pthread_create_from_observed_rccl_init_v1\","
        "\"capacity\":%zu,\"record_count\":%zu,\"dropped_records\":%" PRIu64 ",\"observation_errors\":%" PRIu64 ","
        "\"init_stack_errors\":%" PRIu64 ",\"records\":[",
        kThreadOrigins, g_thread_origin_count, g_thread_origin_dropped.load(), g_thread_origin_errors.load(), g_init_stack_errors.load());
    for (size_t i = 0; i < g_thread_origin_count; ++i) {
        const auto& origin = g_thread_origins[i]; if (i) std::fputc(',', f);
        std::fprintf(f, "{\"id\":%" PRIu64 ",\"parent_init_cid\":%" PRIu64 ",\"parent_epoch\":%" PRIu64 ","
            "\"parent_tid\":%" PRIu64 ",\"init_output_address\":%" PRIu64 ",\"start_address\":%" PRIu64 ","
            "\"module_base\":%" PRIu64 ",\"module_offset\":%" PRIu64 ",\"argument_address\":%" PRIu64 ","
            "\"create_begin_ns\":%" PRIu64 ",\"create_end_ns\":%" PRIu64 ",\"create_returncode\":%d,"
            "\"child_tid\":%" PRIu64 ",\"child_start_ns\":%" PRIu64 ",\"child_end_ns\":%" PRIu64 ","
            "\"start_routine_invocations\":%" PRIu64 ",\"module_known\":%s,\"module_path_canonical\":%s,\"module_path\":",
            origin.id, origin.parent_init_cid, origin.parent_epoch, origin.parent_tid, origin.init_output_address,
            origin.start_address, origin.module_base, origin.module_offset, origin.argument_address,
            origin.create_begin_ns, origin.create_end_ns, origin.create_return, origin.child_tid, origin.child_start_ns,
            origin.child_end_ns, origin.start_invocations, origin.module_known ? "true" : "false",
            origin.module_path_canonical ? "true" : "false");
        json_text(f, origin.module_path);
        std::fprintf(f, ",\"stdthread_trampoline_matched\":%s,\"state_dispatch_observed\":%s,"
            "\"state_vtable\":%" PRIu64 ",\"state_typeinfo\":%" PRIu64 ",\"state_dispatch\":%" PRIu64 ","
            "\"run_member_offset\":%" PRIu64 ",\"vtable_module_base\":%" PRIu64 ","
            "\"typeinfo_module_base\":%" PRIu64 ",\"dispatch_module_base\":%" PRIu64 ",\"vtable_module\":",
            origin.stdthread_trampoline_matched ? "true" : "false", origin.state_dispatch_observed ? "true" : "false",
            origin.state_vtable, origin.state_typeinfo, origin.state_dispatch, origin.run_member_offset,
            origin.vtable_module_base, origin.typeinfo_module_base, origin.dispatch_module_base);
        json_text(f, origin.vtable_module); std::fputs(",\"typeinfo_module\":", f); json_text(f, origin.typeinfo_module);
        std::fputs(",\"dispatch_module\":", f); json_text(f, origin.dispatch_module); std::fputc('}', f);
    }
    std::fputs("]}", f); pthread_mutex_unlock(&g_thread_origin_lock);
    std::fprintf(f, ",\"explicit_client_finalize\":{\"schema_version\":1,\"requested\":%s,"
        "\"official_callback_available\":%s,\"actual_client_id_saved\":%s,\"request_begin_ns\":%" PRIu64 ","
        "\"actual_tool_fini_callback_count\":%" PRIu64 ",\"api\":\"rocprofiler_client_finalize_t\","
        "\"sdk_return_type\":\"void\",\"sdk_return_status\":null}",
        g_explicit_finalize_requested ? "true" : "false", g_client_finalize ? "true" : "false",
        g_client_id.has_value() ? "true" : "false", g_explicit_finalize_begin_ns, g_fini_callback_count);
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
        std::fprintf(f, "{\"seq\":%" PRIu64 ",\"callback_seq\":%" PRIu64 ",\"ns\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"cid\":%" PRIu64 ",\"ancestor\":%" PRIu64 ",\"external\":%" PRIu64 ",\"epoch\":%" PRIu64 ",\"thread_origin_id\":%" PRIu64 ",\"domain\":%u,\"op\":%u,\"phase\":%u,\"flags\":%u,\"status_type\":%u,\"status\":",
            e.sequence, e.callback_sequence, e.timestamp_ns, e.thread_id, e.correlation, e.ancestor, e.external, e.epoch,
            e.thread_origin_id, e.domain, e.operation, e.phase, e.flags, unsigned(e.status_type));
        if (e.status_type) std::fprintf(f, "%" PRId64, e.status); else std::fputs("null", f);
        if (e.allocation_payload) std::fprintf(f, ",\"raw_allocation_payload\":{\"address_or_handle\":%" PRIu64 ",\"size_bytes\":%" PRIu64 ",\"agent_handle\":%" PRIu64 ",\"success_proven\":false}", e.allocation_address, e.allocation_size, e.allocation_agent);
        if (e.scratch_payload) {
            std::fprintf(f, ",\"raw_scratch_payload\":{\"agent_handle\":%" PRIu64 ",\"queue_handle\":%" PRIu64 ",\"flags\":%" PRIu64 ",\"args_kind\":%" PRIu64 ",\"allocation_size_known\":%s,\"allocation_size_bytes\":", e.scratch_agent, e.scratch_queue, e.scratch_flags, e.scratch_args_kind, e.scratch_size_known ? "true" : "false");
            if (e.scratch_size_known) std::fprintf(f, "%" PRIu64, e.scratch_size); else std::fputs("null", f);
            std::fputs(",\"workspace_attribution_supported\":false}", f);
        }
        if (e.rccl_payload_kind) {
            const char* kind = e.rccl_payload_kind == 1 ? "collective" : e.rccl_payload_kind == 2 ? "init" : "group";
            std::fputs(",\"raw_rccl_payload\":{\"kind\":", f); json_text(f, kind);
            std::fputs(",\"comm_handle\":", f);
            if (e.rccl_comm_known) std::fprintf(f, "%" PRIu64, e.rccl_comm); else std::fputs("null", f);
            if (e.rccl_payload_kind == 1) {
                std::fprintf(f, ",\"stream_handle\":%" PRIu64 ",\"count\":%" PRIu64 ",\"datatype\":%" PRId64,
                             e.rccl_stream, e.rccl_count, e.rccl_datatype);
                if (e.rccl_send_known) std::fprintf(f, ",\"send_buffer_address\":%" PRIu64, e.rccl_send);
                if (e.rccl_receive_known) std::fprintf(f, ",\"receive_buffer_address\":%" PRIu64, e.rccl_receive);
                if (e.rccl_peer_known) std::fprintf(f, ",\"peer\":%" PRId64, e.rccl_peer);
                if (e.rccl_root_known) std::fprintf(f, ",\"root\":%" PRId64, e.rccl_root);
            } else if (e.rccl_payload_kind == 2) {
                std::fprintf(f, ",\"comm_output_address\":%" PRIu64 ",\"world_size\":%" PRId64 ",\"rank\":%" PRId64,
                             e.rccl_output_address, e.rccl_world, e.rccl_rank);
            }
            std::fputc('}', f);
        }
        if (e.hip_payload_kind) {
            const char* kind = e.hip_payload_kind == 1 ? "event" : e.hip_payload_kind == 2 ? "pool_create" :
                               e.hip_payload_kind == 3 ? "pool_attribute" : e.hip_payload_kind == 4 ? "host_registration" :
                               e.hip_payload_kind == 5 ? "device_query" : e.hip_payload_kind == 6 ? "device_set" :
                               e.hip_payload_kind == 7 ? "device_sync" : e.hip_payload_kind == 8 ? "stream_sync" :
                               e.hip_payload_kind == 9 ? "pool_destroy" : "host_unregistration";
            std::fputs(",\"raw_hip_control_payload\":{\"kind\":", f); json_text(f, kind);
            std::fputs(",\"object_handle\":", f);
            if (e.hip_object_known) std::fprintf(f, "%" PRIu64, e.hip_object); else std::fputs("null", f);
            std::fprintf(f, ",\"output_address\":%" PRIu64 ",\"stream_handle\":%" PRIu64 ",\"pointer\":%" PRIu64 ","
                           "\"size_bytes\":%" PRIu64 ",\"flags\":%" PRIu64,
                         e.hip_output_address, e.hip_stream, e.hip_pointer, e.hip_size, e.hip_flags);
            std::fputs(",\"device_index\":", f);
            if (e.hip_device_index >= 0) std::fprintf(f, "%" PRId64, e.hip_device_index); else std::fputs("null", f);
            std::fprintf(f, ",\"device_source_cid\":%" PRIu64, e.hip_device_source_cid);
            if (e.hip_payload_kind == 2) {
                std::fprintf(f, ",\"properties_address\":%" PRIu64 ",\"properties_observed\":%s,\"alloc_type\":%" PRId64 ","
                    "\"handle_types\":%" PRId64 ",\"location_type\":%" PRId64 ",\"location_id\":%" PRId64 ","
                    "\"max_size_bytes\":%" PRIu64 ",\"reserved_zero\":%s",
                    e.hip_props_address, e.hip_properties_known ? "true" : "false", e.hip_alloc_type, e.hip_handle_types,
                    e.hip_location_type, e.hip_location_id, e.hip_pool_max_size, e.hip_reserved_zero ? "true" : "false");
            } else if (e.hip_payload_kind == 3) {
                std::fprintf(f, ",\"attribute\":%" PRId64 ",\"value\":", e.hip_attribute);
                if (e.hip_value_known) std::fprintf(f, "%" PRIu64, e.hip_value); else std::fputs("null", f);
            }
            std::fputc('}', f);
        }
        if (e.hsa_signal_kind) {
            std::fputs(",\"raw_hsa_signal_payload\":{\"operation\":", f);
            json_text(f, e.hsa_signal_kind == 1 ? "create" : "destroy");
            std::fputs(",\"signal_handle\":", f);
            if (e.hsa_signal_known) std::fprintf(f, "%" PRIu64, e.hsa_signal_handle); else std::fputs("null", f);
            std::fprintf(f, ",\"output_address\":%" PRIu64 ",\"initial_value\":%" PRId64 ",\"num_consumers\":%u,"
                           "\"consumers_observed\":%s,\"consumer_agents\":[",
                         e.hsa_signal_output, e.hsa_initial_value, e.hsa_num_consumers,
                         e.hsa_consumers_observed ? "true" : "false");
            if (e.hsa_consumers_observed) for (size_t j = 0; j < e.hsa_num_consumers; ++j) {
                if (j) std::fputc(',', f);
                std::fprintf(f, "%" PRIu64, e.hsa_consumers[j]);
            }
            std::fputs("]}", f);
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
int tool_init(rocprofiler_client_finalize_t finalize_func, void*) {
    g_client_finalize = finalize_func;
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
    g_stop_sdk_return_ns.store(now_ns(), std::memory_order_release);
    g_open_apis_at_sdk_stop_return = g_open_api_calls.load(std::memory_order_acquire);
    const char* status_name = rocprofiler_get_status_name(rc);
    const char* status_text = rocprofiler_get_status_string(rc);
    const uint64_t deadline = now_ns() + 1000000000ULL;
    while ((g_callbacks_inflight.load(std::memory_order_acquire) || g_open_api_calls.load(std::memory_order_acquire))
           && now_ns() < deadline) {
        ++g_drain_wait_iterations;
        timespec pause{0, 100000}; nanosleep(&pause, nullptr);
    }
    pthread_mutex_lock(&g_lock);
    g_stop_status = int(rc);
    g_open_apis_at_window_close = g_open_api_calls.load(std::memory_order_acquire);
    g_drain_status = (g_callbacks_inflight.load(std::memory_order_acquire) || g_open_apis_at_window_close)
        ? ETIMEDOUT : (g_open_api_underflows.load() || g_api_enters_after_sdk_stop.load() ? EPROTO : 0);
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
    ++g_fini_callback_count; g_finalized = true; g_finalize_ns = now_ns();
    pthread_mutex_unlock(&g_lock);
    const int written = g_output[0] ? write_report(g_output) : EINVAL;
    pthread_mutex_lock(&g_lock); g_final_write_status = written; pthread_mutex_unlock(&g_lock);
}
}  // namespace

extern "C" __attribute__((visibility("default")))
int pthread_create(pthread_t* thread, const pthread_attr_t* attributes,
                   void* (*start)(void*), void* argument) noexcept {
    return observe_pthread_create(thread, attributes, start, argument);
}

extern "C" __attribute__((visibility("default")))
rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t version, const char* text,
    uint32_t priority, rocprofiler_client_id_t* client) {
    g_configured = true; g_sdk_version = version; g_priority = priority;
    if (client && client->size >= sizeof(*client)) {
        client->name = "mlsys-native-allocation-diagnostic";
        g_client_id.emplace(*client);  // Copy construction preserves the const SDK handle.
    }
    if (!copy_text(g_sdk_string, sizeof(g_sdk_string), text ? text : "")) g_metadata_truncated = true;
    const char* output = std::getenv("MLSYS_NATIVE_PROBE_OUT");
    if (output && !expand_output_path(g_output, sizeof(g_output), output)) {
        g_metadata_truncated = true; g_bounds_valid = false;
    }
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

extern "C" __attribute__((visibility("default")))
int mlsys_probe_stdthread_control() {
    if (g_in_callback || g_init_call_depth) return EDEADLK;
    pthread_mutex_lock(&g_stdthread_control_lock);
    auto& control = g_stdthread_control;
    if (control.attempted) { const int result = control.status; pthread_mutex_unlock(&g_stdthread_control_lock); return result; }
    control.attempted = true;
    // This implementation is deliberately bound to the actual installed
    // GCC13 x86-64 Itanium ABI and std_thread.h, not an arbitrary object layout.
    using Run = void (std::thread::_State::*)();
    struct Encoding { ptrdiff_t pointer, adjustment; } encoding{};
    const Run member = &std::thread::_State::_M_run;
    control.member_pointer_size = sizeof(member); control.state_size = sizeof(std::thread::_State);
    if (sizeof(void*) != 8 || sizeof(member) != sizeof(encoding) || _GLIBCXX_RELEASE != 13) {
        control.status = ENOTSUP; pthread_mutex_unlock(&g_stdthread_control_lock); return control.status;
    }
    std::memcpy(&encoding, &member, sizeof(member));
    if (encoding.pointer <= 0 || !(encoding.pointer & 1) || encoding.adjustment != 0 ||
        uint64_t(encoding.pointer - 1) > 256 || (encoding.pointer - 1) % sizeof(void*) != 0) {
        control.status = ENOTSUP; pthread_mutex_unlock(&g_stdthread_control_lock); return control.status;
    }
    control.run_member_offset = uint64_t(encoding.pointer - 1);
    g_stdthread_control_active = true;
    try {
        std::thread cpu_only([] {}); cpu_only.join();
        control.status = control.trampoline_observed ? 0 : EIO;
    } catch (...) { control.status = EIO; }
    g_stdthread_control_active = false; control.completed = control.status == 0;
    const int result = control.status; pthread_mutex_unlock(&g_stdthread_control_lock);
    return result;
}

extern "C" __attribute__((visibility("default")))
int mlsys_probe_write_stdthread_control(const char* path) {
    if (!path || !path[0] || strnlen(path, kPath) >= kPath) return EINVAL;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return errno;
    FILE* file = fdopen(fd, "w");
    if (!file) { const int error = errno; close(fd); return error; }
    pthread_mutex_lock(&g_stdthread_control_lock);
    const auto& control = g_stdthread_control;
    std::fprintf(file, "{\"schema_version\":1,\"method\":\"observed_stdthread_state_dispatch_v1\",\"pid\":%ld,"
        "\"attempted\":%s,\"completed\":%s,\"control_returncode\":%d,\"trampoline_observed\":%s,"
        "\"start_address\":%" PRIu64 ",\"module_base\":%" PRIu64 ",\"module_offset\":%" PRIu64 ","
        "\"run_member_offset\":%" PRIu64 ",\"member_pointer_size\":%" PRIu64 ",\"state_size\":%" PRIu64 ","
        "\"gxx_abi_version\":%d,\"glibcxx_release\":%d,\"glibcxx_date\":%d,\"header_reference\":",
        long(getpid()), control.attempted ? "true" : "false", control.completed ? "true" : "false", control.status,
        control.trampoline_observed ? "true" : "false", control.start_address, control.module_base, control.module_offset,
        control.run_member_offset, control.member_pointer_size, control.state_size,
        __GXX_ABI_VERSION, _GLIBCXX_RELEASE, __GLIBCXX__);
    json_text(file, "/usr/include/c++/13/bits/std_thread.h");
    std::fputs(",\"header_sha256\":\"bfaeaa41bf061e826c96cbee7ee3c453250918b7c3e7cc96b01693ee31751233\",\"module_path\":", file);
    json_text(file, control.module_path); std::fputs("}\n", file);
    pthread_mutex_unlock(&g_stdthread_control_lock);
    int error = std::ferror(file) ? EIO : 0;
    if (std::fclose(file) != 0 && !error) error = errno ? errno : EIO;
    return error;
}

extern "C" __attribute__((visibility("default")))
int mlsys_probe_write_live_signals(const char* path) {
    // This is a read-only replay of observed API generations, not a query of
    // opaque runtime memory and not a claim that a signal has been destroyed.
    if (!path || !path[0] || strnlen(path, kPath) >= kPath) return EINVAL;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return errno;
    FILE* file = fdopen(fd, "w");
    if (!file) { const int error = errno; close(fd); return error; }
    struct Live { const Event* event; uint64_t generation; };
    std::map<uint64_t, Live> live;
    std::map<uint64_t, uint64_t> generations;
    uint64_t reused_live = 0, unknown_destroy = 0;
    pthread_mutex_lock(&g_lock);
    for (size_t i = 0; i < g_count; ++i) {
        const auto& event = g_events[i];
        if (event.phase != ROCPROFILER_CALLBACK_PHASE_EXIT || event.status_type != 2 || event.status != 0 ||
            !event.hsa_signal_known || !event.hsa_signal_handle) continue;
        const uint64_t handle = event.hsa_signal_handle;
        if (event.hsa_signal_kind == 1) {
            if (live.count(handle)) ++reused_live;
            live[handle] = {&event, ++generations[handle]};
        } else if (event.hsa_signal_kind == 2) {
            if (!live.erase(handle)) ++unknown_destroy;
        }
    }
    std::fprintf(file, "{\"schema_version\":1,\"kind\":\"observed_live_hsa_signal_generations\","
        "\"complete_workspace\":false,\"hardware_validated\":false,\"pid\":%ld,"
        "\"snapshot_monotonic_raw_ns\":%" PRIu64 ",\"event_cursor\":%" PRIu64 ",\"dropped_events\":%" PRIu64 ","
        "\"reused_live_handle_count\":%" PRIu64 ",\"unknown_destroy_count\":%" PRIu64 ",\"generations\":[",
        long(getpid()), now_ns(), g_attempted, g_dropped, reused_live, unknown_destroy);
    bool first = true;
    for (const auto& pair : live) {
        const auto& event = *pair.second.event;
        if (!first) std::fputc(',', file);
        first = false;
        std::fprintf(file, "{\"signal_handle\":%" PRIu64 ",\"generation\":%" PRIu64 ","
            "\"create_cid\":%" PRIu64 ",\"create_seq\":%" PRIu64 ",\"create_epoch\":%" PRIu64 ","
            "\"create_tid\":%" PRIu64 ",\"initial_value\":%" PRId64 ",\"num_consumers\":%u,"
            "\"consumers_observed\":%s,\"consumer_agents\":[",
            pair.first, pair.second.generation, event.correlation, event.sequence, event.epoch, event.thread_id,
            event.hsa_initial_value, event.hsa_num_consumers, event.hsa_consumers_observed ? "true" : "false");
        if (event.hsa_consumers_observed) for (size_t j = 0; j < event.hsa_num_consumers; ++j) {
            if (j) std::fputc(',', file);
            std::fprintf(file, "%" PRIu64, event.hsa_consumers[j]);
        }
        std::fputs("]}", file);
    }
    std::fputs("]}\n", file);
    const uint64_t dropped = g_dropped;
    pthread_mutex_unlock(&g_lock);
    int error = std::ferror(file) ? EIO : 0;
    if (std::fclose(file) != 0 && !error) error = errno ? errno : EIO;
    return error ? error : reused_live || dropped ? EPROTO : 0;
}

extern "C" __attribute__((visibility("default")))
int mlsys_probe_finalize() {
    // Installed registration.h gives a void callback which asks the SDK to
    // invoke tool_fini and suppress its later atexit invocation. Never invoke
    // tool_fini directly or invent a success status for the void SDK API.
    if (g_in_callback) return EDEADLK;
    pthread_mutex_lock(&g_finalize_lock);
    pthread_mutex_lock(&g_lock);
    if (g_explicit_finalize_requested || g_finalized) {
        pthread_mutex_unlock(&g_lock); pthread_mutex_unlock(&g_finalize_lock); return EALREADY;
    }
    if (!g_stop_completed || !g_observation_closed || g_stop_result != 0 || g_active_epoch >= 0 ||
        !g_client_finalize || !g_client_id.has_value()) {
        pthread_mutex_unlock(&g_lock); pthread_mutex_unlock(&g_finalize_lock); return EBUSY;
    }
    g_explicit_finalize_requested = true; g_explicit_finalize_begin_ns = now_ns();
    auto finalize = g_client_finalize;
    auto client = *g_client_id;  // This struct has a const member; do not assign it.
    pthread_mutex_unlock(&g_lock);
    finalize(client);  // Only the SDK itself may deliver the real final callback.
    pthread_mutex_lock(&g_lock);
    const int result = g_finalized && g_fini_callback_count == 1 && g_final_write_status != -999
        ? g_final_write_status : EIO;
    pthread_mutex_unlock(&g_lock); pthread_mutex_unlock(&g_finalize_lock);
    return result;
}
