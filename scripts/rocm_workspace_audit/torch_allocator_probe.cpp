// Task-local observation only. Never establishes complete Workspace ownership.
// Uses the ACTUAL installed hipified header, whose namespace remains c10::cuda.
// Keep this shared object loaded until the process exits: the hook cannot detach.
#include <c10/hip/HIPCachingAllocator.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != 1
#error "Compile with the actual Torch ABI: _GLIBCXX_USE_CXX11_ABI=1"
#endif

namespace {
namespace Allocator = c10::cuda::CUDACachingAllocator;
using TraceEntry = c10::CachingDeviceAllocator::TraceEntry;
constexpr uint64_t kEvents = 262144, kEpochs = 64;
constexpr size_t kMetadata = 256, kLabel = 256, kPath = 4096;
constexpr uint64_t kBoundary = UINT64_MAX;
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "The allocator callback must not acquire libatomic locks");
static_assert(static_cast<int>(TraceEntry::ALLOC) == 0 &&
              static_cast<int>(TraceEntry::FREE_REQUESTED) == 1 &&
              static_cast<int>(TraceEntry::FREE_COMPLETED) == 2 &&
              static_cast<int>(TraceEntry::SEGMENT_ALLOC) == 3 &&
              static_cast<int>(TraceEntry::SEGMENT_FREE) == 4 &&
              static_cast<int>(TraceEntry::SEGMENT_MAP) == 5 &&
              static_cast<int>(TraceEntry::SEGMENT_UNMAP) == 6 &&
              static_cast<int>(TraceEntry::SNAPSHOT) == 7 &&
              static_cast<int>(TraceEntry::OOM) == 8, "Actual allocator action ABI changed");

enum Flags : uint64_t {
    UnknownAction = 1, CrossThread = 2, OutsideEpoch = 4,
    DuringBoundary = 8, MetadataTruncated = 16, ClockFailure = 32, ThreadIdFailure = 64
};
enum Issues : uint64_t {
    DroppedEvents = 1, UnknownActions = 2, UnownedAllocations = 4,
    CrossThreadAllocations = 8, TruncatedMetadata = 16,
    BoundaryOverlap = 32, FailedClock = 64, AllocatorOOM = 128,
    MappingRequiresAudit = 256, OpenEpoch = 512, UnknownThreadId = 1024
};
struct Event {
    uint64_t sequence, timestamp_ns, thread_id, epoch_id, owner_token;
    uint64_t address, size_bytes, stream, mempool_first, mempool_second, flags;
    uint64_t user_metadata_size, retained_metadata_size;
    int32_t action, device;
    char user_metadata[kMetadata];
};
struct Epoch {
    uint64_t id, owner_thread_id, begin_ns, end_ns, begin_sequence, end_sequence;
    uint64_t boundary_before, boundary_after;
    char label[kLabel];
    bool ended;
    std::atomic<uint64_t> observed{0}, dropped{0}, unknown{0}, unowned_allocations{0};
    std::atomic<uint64_t> cross_thread_allocations{0}, metadata_truncated{0};
    std::atomic<uint64_t> boundary_overlap{0}, clock_failure{0}, oom{0}, mapping{0};
    std::atomic<uint64_t> thread_failure{0};
};

// BSS, saturating, never overwritten. Callback performs no heap allocation.
Event g_events[kEvents];
std::atomic<uint64_t> g_ready[kEvents]{};
Epoch g_epochs[kEpochs];
std::atomic<uint64_t> g_install_state{0}, g_attempted{0}, g_inflight{0};
std::atomic<uint64_t> g_active{0}, g_boundary_callbacks{0};
std::atomic<uint64_t> g_dropped{0}, g_unknown{0}, g_metadata_truncated{0};
std::atomic<uint64_t> g_clock_failure{0}, g_outside_epoch_allocations{0}, g_api_errors{0};
std::atomic<uint64_t> g_thread_failure{0};
std::atomic<uint64_t> g_control_busy{0};
uint64_t g_limit = 0, g_epoch_count = 0, g_install_thread = 0;

uint64_t thread_id() noexcept {
    const long value = syscall(SYS_gettid);
    return value > 0 ? static_cast<uint64_t>(value) : 0;
}
uint64_t now_ns() noexcept {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) return 0;
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(value.tv_nsec);
}
bool allocation_action(int action) noexcept {
    return action == TraceEntry::ALLOC || action == TraceEntry::SEGMENT_ALLOC ||
           action == TraceEntry::SEGMENT_MAP;
}

void callback(const TraceEntry& source) noexcept {
    // Called under Torch's allocator lock. No GIL, mutex, Torch/HIP API,
    // std::string construction, formatting, I/O, malloc or callback dispatch.
    g_inflight.fetch_add(1, std::memory_order_acq_rel);
    const uint64_t ticket = g_attempted.fetch_add(1, std::memory_order_relaxed);
    const uint64_t active = g_active.load(std::memory_order_acquire);
    const uint64_t tid = thread_id(), stamp = now_ns();
    const int action = static_cast<int>(source.action_);
    const bool known = action >= TraceEntry::ALLOC && action <= TraceEntry::OOM;
    Epoch* epoch = active > 0 && active <= kEpochs ? &g_epochs[active - 1] : nullptr;
    const bool owned = epoch && tid && tid == epoch->owner_thread_id;
    uint64_t flags = 0;
    if (!known) { flags |= UnknownAction; g_unknown.fetch_add(1, std::memory_order_relaxed); }
    if (!stamp) { flags |= ClockFailure; g_clock_failure.fetch_add(1, std::memory_order_relaxed); }
    if (!tid) { flags |= ThreadIdFailure; g_thread_failure.fetch_add(1, std::memory_order_relaxed); }
    if (active == kBoundary) {
        flags |= DuringBoundary;
        g_boundary_callbacks.fetch_add(1, std::memory_order_relaxed);
    } else if (!epoch) {
        flags |= OutsideEpoch;
        if (allocation_action(action)) g_outside_epoch_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    if (epoch && !owned) flags |= CrossThread;
    const size_t metadata_size = source.user_metadata_.size();
    const size_t retained = metadata_size < kMetadata ? metadata_size : kMetadata;
    if (metadata_size > kMetadata) {
        flags |= MetadataTruncated;
        g_metadata_truncated.fetch_add(1, std::memory_order_relaxed);
    }
    if (epoch) {
        epoch->observed.fetch_add(1, std::memory_order_relaxed);
        if (!known) epoch->unknown.fetch_add(1, std::memory_order_relaxed);
        if (!stamp) epoch->clock_failure.fetch_add(1, std::memory_order_relaxed);
        if (!tid) epoch->thread_failure.fetch_add(1, std::memory_order_relaxed);
        if (metadata_size > kMetadata) epoch->metadata_truncated.fetch_add(1, std::memory_order_relaxed);
        if (allocation_action(action) && !owned) {
            epoch->unowned_allocations.fetch_add(1, std::memory_order_relaxed);
            epoch->cross_thread_allocations.fetch_add(1, std::memory_order_relaxed);
        }
        if (action == TraceEntry::OOM) epoch->oom.fetch_add(1, std::memory_order_relaxed);
        if (action == TraceEntry::SEGMENT_MAP || action == TraceEntry::SEGMENT_UNMAP)
            epoch->mapping.fetch_add(1, std::memory_order_relaxed);
    }
    if (ticket < g_limit) {
        Event& event = g_events[ticket];
        event.sequence = ticket + 1; event.timestamp_ns = stamp; event.thread_id = tid;
        event.epoch_id = epoch ? epoch->id : 0;
        event.owner_token = owned ? epoch->id : 0;
        event.action = action; event.device = static_cast<int32_t>(source.device_);
        event.address = source.addr_; event.size_bytes = source.size_;
        event.stream = reinterpret_cast<uintptr_t>(source.stream_);
        event.mempool_first = source.mempool_.first;
        event.mempool_second = source.mempool_.second;
        event.flags = flags; event.user_metadata_size = metadata_size;
        event.retained_metadata_size = retained;
        if (retained) std::memcpy(event.user_metadata, source.user_metadata_.data(), retained);
        g_ready[ticket].store(1, std::memory_order_release);
    } else {
        g_dropped.fetch_add(1, std::memory_order_relaxed);
        if (epoch) epoch->dropped.fetch_add(1, std::memory_order_relaxed);
    }
    g_inflight.fetch_sub(1, std::memory_order_acq_rel);
}

struct Control {
    bool acquired;
    Control() : acquired(g_control_busy.exchange(1, std::memory_order_acquire) == 0) {}
    ~Control() { if (acquired) g_control_busy.store(0, std::memory_order_release); }
};
int reject(int code) {
    g_api_errors.fetch_add(1, std::memory_order_relaxed);
    return code;
}
bool valid_control() {
    return g_install_state.load(std::memory_order_acquire) == 2 && thread_id() == g_install_thread;
}
uint64_t issues(const Epoch& epoch) {
    uint64_t value = 0;
    if (epoch.dropped.load()) value |= DroppedEvents;
    if (epoch.unknown.load()) value |= UnknownActions;
    if (epoch.unowned_allocations.load()) value |= UnownedAllocations;
    if (epoch.cross_thread_allocations.load()) value |= CrossThreadAllocations;
    if (epoch.metadata_truncated.load()) value |= TruncatedMetadata;
    if (epoch.boundary_overlap.load() || epoch.boundary_after != epoch.boundary_before) value |= BoundaryOverlap;
    if (epoch.clock_failure.load()) value |= FailedClock;
    if (epoch.oom.load()) value |= AllocatorOOM;
    if (epoch.mapping.load()) value |= MappingRequiresAudit;
    if (epoch.thread_failure.load()) value |= UnknownThreadId;
    if (!epoch.ended) value |= OpenEpoch;
    return value;
}
const char* action_name(int action) {
    static const char* const names[] = {"alloc", "free_requested", "free_completed", "segment_alloc",
        "segment_free", "segment_map", "segment_unmap", "snapshot", "oom"};
    return action >= 0 && action < 9 ? names[action] : "UNKNOWN";
}
void json_bytes(FILE* stream, const char* bytes, size_t size) {
    std::fputc('"', stream);
    for (size_t i = 0; i < size; ++i) {
        const auto ch = static_cast<unsigned char>(bytes[i]);
        if (ch == '"' || ch == '\\') { std::fputc('\\', stream); std::fputc(ch, stream); }
        else if (ch < 32 || ch >= 127) std::fprintf(stream, "\\u%04x", unsigned(ch));
        else std::fputc(ch, stream);
    }
    std::fputc('"', stream);
}
} // namespace

extern "C" int mlsys_torch_probe_install(uint64_t max_events) noexcept {
    Control guard;
    if (!guard.acquired) return reject(EBUSY);
    uint64_t expected = 0;
    if (!g_install_state.compare_exchange_strong(expected, 1)) return reject(EALREADY);
    if (!max_events || max_events > kEvents) { g_install_state.store(3); return reject(EINVAL); }
    g_limit = max_events; g_install_thread = thread_id();
    if (!g_install_thread) { g_install_state.store(3); return reject(EIO); }
    // Caller must already have imported Torch and initialized the allocator.
    // Do not lazy-init a device or enable/reset memory history from this helper.
    if (!Allocator::get()) { g_install_state.store(3); return reject(ENODEV); }
    try {
        Allocator::attachAllocatorTraceTracker(&callback);
    } catch (...) {
        // No retry: an exception must not cause duplicate callback registration.
        g_install_state.store(3, std::memory_order_release);
        return reject(EIO);
    }
    g_install_state.store(2, std::memory_order_release);
    return 0;
}

extern "C" int mlsys_torch_probe_begin(uint64_t id, const char* label) noexcept {
    Control guard;
    if (!guard.acquired) return reject(EBUSY);
    if (!valid_control() || !id || !label || g_epoch_count == kEpochs) return reject(EINVAL);
    const size_t size = strnlen(label, kLabel);
    if (!size || size == kLabel) return reject(EINVAL);
    for (uint64_t i = 0; i < g_epoch_count; ++i) if (g_epochs[i].id == id) return reject(EEXIST);
    const uint64_t boundary_before = g_boundary_callbacks.load(std::memory_order_acquire);
    uint64_t expected = 0;
    if (!g_active.compare_exchange_strong(expected, kBoundary, std::memory_order_acq_rel)) return reject(EBUSY);
    Epoch& epoch = g_epochs[g_epoch_count];
    epoch.id = id; epoch.owner_thread_id = g_install_thread;
    epoch.boundary_before = boundary_before;
    epoch.begin_sequence = g_attempted.load(std::memory_order_acquire);
    epoch.begin_ns = now_ns();
    std::memcpy(epoch.label, label, size); epoch.label[size] = '\0';
    if (g_inflight.load(std::memory_order_acquire)) epoch.boundary_overlap.fetch_add(1);
    if (!epoch.begin_ns) epoch.clock_failure.fetch_add(1);
    ++g_epoch_count;
    g_active.store(g_epoch_count, std::memory_order_release);
    return 0;
}

extern "C" int mlsys_torch_probe_end(uint64_t id) noexcept {
    Control guard;
    if (!guard.acquired) return reject(EBUSY);
    if (!valid_control()) return reject(EINVAL);
    uint64_t active = g_active.load(std::memory_order_acquire);
    if (!active || active > kEpochs || g_epochs[active - 1].id != id) return reject(EINVAL);
    g_active.store(kBoundary, std::memory_order_release);
    Epoch& epoch = g_epochs[active - 1];
    if (g_inflight.load(std::memory_order_acquire)) epoch.boundary_overlap.fetch_add(1);
    epoch.end_ns = now_ns();
    epoch.end_sequence = g_attempted.load(std::memory_order_acquire);
    if (!epoch.end_ns || epoch.end_ns < epoch.begin_ns) epoch.clock_failure.fetch_add(1);
    epoch.ended = true;
    g_active.store(0, std::memory_order_release);
    epoch.boundary_after = g_boundary_callbacks.load(std::memory_order_acquire);
    if (g_inflight.load(std::memory_order_acquire)) epoch.boundary_overlap.fetch_add(1);
    return 0;
}

extern "C" int mlsys_torch_probe_write(const char* path) noexcept {
    Control guard;
    if (!guard.acquired) return reject(EBUSY);
    if (!valid_control() || g_active.load(std::memory_order_acquire) || !path) return reject(EINVAL);
    const size_t path_size = strnlen(path, kPath);
    if (!path_size || path_size > kPath - 64) return reject(EINVAL);
    if (g_inflight.load(std::memory_order_acquire)) return reject(EBUSY);
    const uint64_t cutoff = g_attempted.load(std::memory_order_acquire);
    const uint64_t retained = cutoff < g_limit ? cutoff : g_limit;
    for (uint64_t i = 0; i < retained; ++i)
        if (!g_ready[i].load(std::memory_order_acquire)) return reject(EBUSY);

    char temporary[kPath]{};
    std::snprintf(temporary, sizeof(temporary), "%s.tmp.%ld", path, long(getpid()));
    int fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return reject(errno);
    FILE* output = fdopen(fd, "w");
    if (!output) { const int error = errno; close(fd); unlink(temporary); return reject(error); }
    std::fprintf(output,
        "{\"schema_version\":1,\"kind\":\"torch_allocator_trace_probe\","
        "\"complete_workspace\":false,\"allocation_owner_classification_performed\":false,"
        "\"callback_scope\":\"all_allocator_tracker_callbacks_since_explicit_install\","
        "\"owner_policy\":\"active_epoch_install_thread_token_only\","
        "\"clock\":\"CLOCK_MONOTONIC_RAW\",\"metadata_encoding\":\"byte_preserving_latin1\","
        "\"hook_detachable\":false,\"callback_gpu_calls\":false,\"callback_gil\":false,"
        "\"callback_extra_locks\":false,\"abi_cxx11\":1,"
        "\"install_thread_id\":%" PRIu64 ",\"event_capacity\":%" PRIu64 ",\"event_limit\":%" PRIu64 ","
        "\"attempted_events_at_cutoff\":%" PRIu64 ",\"retained_events\":%" PRIu64 ","
        "\"dropped_events\":%" PRIu64 ",\"unknown_actions\":%" PRIu64 ","
        "\"truncated_metadata\":%" PRIu64 ",\"clock_failures\":%" PRIu64 ",\"thread_id_failures\":%" PRIu64 ","
        "\"outside_epoch_allocations\":%" PRIu64 ",\"api_errors\":%" PRIu64 ",\"epochs\":[",
        g_install_thread, kEvents, g_limit, cutoff, retained, g_dropped.load(), g_unknown.load(),
        g_metadata_truncated.load(), g_clock_failure.load(), g_thread_failure.load(),
        g_outside_epoch_allocations.load(), g_api_errors.load());
    for (uint64_t i = 0; i < g_epoch_count; ++i) {
        const Epoch& e = g_epochs[i];
        if (i) std::fputc(',', output);
        std::fprintf(output, "{\"id\":%" PRIu64 ",\"label\":", e.id);
        json_bytes(output, e.label, strnlen(e.label, kLabel));
        std::fprintf(output,
            ",\"owner_thread_id\":%" PRIu64 ",\"begin_ns\":%" PRIu64 ",\"end_ns\":%" PRIu64
            ",\"begin_sequence\":%" PRIu64 ",\"end_sequence\":%" PRIu64
            ",\"ended\":%s,\"observed_events\":%" PRIu64 ",\"dropped_events\":%" PRIu64
            ",\"unknown_actions\":%" PRIu64 ",\"unowned_allocations\":%" PRIu64
            ",\"cross_thread_allocations\":%" PRIu64 ",\"truncated_metadata\":%" PRIu64
            ",\"boundary_overlap\":%" PRIu64 ",\"boundary_callbacks\":%" PRIu64
            ",\"clock_failures\":%" PRIu64 ",\"thread_id_failures\":%" PRIu64
            ",\"oom_events\":%" PRIu64 ",\"mapping_events\":%" PRIu64
            ",\"observation_issue_mask\":%" PRIu64 ",\"observation_issues_present\":%s}",
            e.owner_thread_id, e.begin_ns, e.end_ns, e.begin_sequence, e.end_sequence, e.ended ? "true" : "false",
            e.observed.load(), e.dropped.load(), e.unknown.load(), e.unowned_allocations.load(),
            e.cross_thread_allocations.load(), e.metadata_truncated.load(), e.boundary_overlap.load(),
            e.boundary_after - e.boundary_before, e.clock_failure.load(), e.thread_failure.load(), e.oom.load(), e.mapping.load(),
            issues(e), issues(e) ? "true" : "false");
    }
    std::fputs("],\"events\":[", output);
    for (uint64_t i = 0; i < retained; ++i) {
        const Event& e = g_events[i];
        if (i) std::fputc(',', output);
        std::fprintf(output,
            "{\"sequence\":%" PRIu64 ",\"timestamp_ns\":%" PRIu64 ",\"thread_id\":%" PRIu64
            ",\"epoch_id\":%" PRIu64 ",\"owner_token\":%" PRIu64 ",\"action\":%d,\"action_name\":\"%s\""
            ",\"device\":%d,\"address\":%" PRIu64 ",\"size_bytes\":%" PRIu64 ",\"stream\":%" PRIu64
            ",\"mempool\":[%" PRIu64 ",%" PRIu64 "],\"flags\":%" PRIu64
            ",\"user_metadata_size\":%" PRIu64 ",\"retained_metadata_size\":%" PRIu64 ",\"user_metadata\":",
            e.sequence, e.timestamp_ns, e.thread_id, e.epoch_id, e.owner_token,
            e.action, action_name(e.action), e.device, e.address, e.size_bytes, e.stream,
            e.mempool_first, e.mempool_second, e.flags, e.user_metadata_size, e.retained_metadata_size);
        json_bytes(output, e.user_metadata, e.retained_metadata_size);
        std::fputc('}', output);
    }
    const bool stable = g_inflight.load(std::memory_order_acquire) == 0 &&
        g_attempted.load(std::memory_order_acquire) == cutoff;
    bool global_issues = !stable || g_dropped.load() || g_unknown.load() ||
        g_metadata_truncated.load() || g_clock_failure.load() || g_thread_failure.load() ||
        g_boundary_callbacks.load() || g_api_errors.load();
    for (uint64_t i = 0; i < g_epoch_count; ++i) if (issues(g_epochs[i])) global_issues = true;
    std::fprintf(output, "],\"checkpoint_stable\":%s,\"checkpoint_is_prefix_only\":true,"
                 "\"global_observation_issues_present\":%s,\"boundary_callbacks\":%" PRIu64
                 ",\"callbacks_inflight_at_write_end\":%" PRIu64
                 ",\"attempted_events_at_write_end\":%" PRIu64 "}\n",
                 stable ? "true" : "false", global_issues ? "true" : "false", g_boundary_callbacks.load(),
                 g_inflight.load(), g_attempted.load());
    int error = 0;
    if (std::ferror(output) || std::fflush(output) != 0 || fsync(fd) != 0) error = errno ? errno : EIO;
    if (std::fclose(output) != 0 && !error) error = errno;
    if (!error && link(temporary, path) != 0) error = errno;
    unlink(temporary);
    return error ? reject(error) : 0;
}
