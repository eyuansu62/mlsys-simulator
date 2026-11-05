// Passive Linux x86-64 libc/UAPI observation. Compile beside native_probe.cpp.
// No HSA/KFD operation is initiated by this translation unit.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#if !defined(__linux__) || !defined(__x86_64__)
#error "The ioctl register-preserving trampoline requires Linux x86-64 SysV ABI"
#endif
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <dlfcn.h>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <ctime>

namespace {
constexpr size_t kRows = 131072, kPath = 256;
enum Api : uint32_t { Ioctl = 1, Mmap = 2, Mmap64 = 3, Munmap = 4, Close = 5, Dup2 = 6, Dup3 = 7,
    Mremap = 8, CloseRange = 9 };
enum Op : uint32_t { Other = 0, CreateEvent = 1, DestroyEvent = 2, Allocate = 3, Free = 4,
    MapGpu = 5, UnmapGpu = 6, Wait = 7, GetVersion = 8, QueueControl = 9, OtherKnown = 10, AcquireVm = 11,
    SetEvent = 12, ResetEvent = 13, Xnack = 14, RuntimeEnable = 15, ScratchVa = 16, AvailableMemory = 17 };
struct Identity {
    uint64_t device, rdevice, inode;
    uint32_t mode, kind;  // 1=/dev/kfd, 2=render node, 0=other.
    int fd;
    bool observed, path_complete;
    char path[kPath];
};
struct Row {
    uint64_t sequence, call, ns, tid, pid;
    uint32_t api, op, phase;
    int64_t result;
    int saved_errno;
    Identity identity;
    Identity associated_identity;
    uint64_t request, argument, address, length, offset;
    int64_t protection, flags, target_fd;
    uint64_t values[8];
    bool payload_observed, output_valid;
};
Row g_rows[kRows];
size_t g_count = 0;
uint64_t g_attempted = 0, g_dropped = 0, g_read_errors = 0;
std::atomic<uint64_t> g_calls{0}, g_metadata_inflight{0}, g_mutators{0}, g_forward_errors{0}, g_reentrant{0};
std::atomic<uint64_t> g_process_pid{0};
bool g_related = false, g_inherited_unsupported = false, g_atfork_registered = false;
uint64_t g_fork_parent_pid = 0, g_discarded_parent_rows = 0;
pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
thread_local unsigned g_depth = 0;
thread_local bool g_writing = false;
using IoctlFunction = int (*)(int, unsigned long, ...);
using MmapFunction = void* (*)(void*, size_t, int, int, int, off_t);
using Mmap64Function = void* (*)(void*, size_t, int, int, int, off64_t);
using MunmapFunction = int (*)(void*, size_t);
using CloseFunction = int (*)(int);
using Dup2Function = int (*)(int, int);
using Dup3Function = int (*)(int, int, int);
using MremapFunction = void* (*)(void*, size_t, size_t, int, ...);
using CloseRangeFunction = int (*)(unsigned, unsigned, int);
std::atomic<void*> g_ioctl{nullptr}, g_mmap{nullptr}, g_mmap64{nullptr}, g_munmap{nullptr};
std::atomic<void*> g_close{nullptr}, g_dup2{nullptr}, g_dup3{nullptr};
std::atomic<void*> g_mremap{nullptr}, g_close_range{nullptr};
bool process_valid() {
    uint64_t process = g_process_pid.load(std::memory_order_acquire);
    const uint64_t actual = uint64_t(getpid());
    if (!process) {
        g_process_pid.compare_exchange_strong(process, actual);
        process = g_process_pid.load(std::memory_order_acquire);
    }
    return process == actual;
}

uint64_t now() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) return 0;
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}
template <typename T> bool read_output(const T* address, T& value) {
    if (!address) return false;
    const int previous = errno;
    iovec local{&value, sizeof(value)}, remote{const_cast<T*>(address), sizeof(value)};
    const bool okay = syscall(SYS_process_vm_readv, getpid(), &local, 1, &remote, 1, 0) == ssize_t(sizeof(value));
    errno = previous;
    return okay;
}
Identity identify(int fd) {
    Identity result{}; result.fd = fd;
    struct stat stat{};
    char link[64]{};
    if (fd < 0 || fstat(fd, &stat)) return result;
    result.observed = true; result.device = uint64_t(stat.st_dev);
    result.rdevice = uint64_t(stat.st_rdev); result.inode = uint64_t(stat.st_ino); result.mode = uint32_t(stat.st_mode);
    std::snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    const ssize_t count = readlink(link, result.path, kPath);
    if (count < 0 || count >= ssize_t(kPath)) return result;
    result.path[count] = '\0'; result.path_complete = true;
    if (S_ISCHR(stat.st_mode) && !std::strcmp(result.path, "/dev/kfd")) result.kind = 1;
    const char prefix[] = "/dev/dri/renderD";
    if (S_ISCHR(stat.st_mode) && !std::strncmp(result.path, prefix, sizeof(prefix) - 1)) {
        const char* digits = result.path + sizeof(prefix) - 1;
        if (*digits) {
            bool numeric = true;
            for (const char* p = digits; *p; ++p) numeric = numeric && *p >= '0' && *p <= '9';
            if (numeric) result.kind = 2;
        }
    }
    return result;
}
void* resolve(std::atomic<void*>& slot, const char* name) {
    void* value = slot.load(std::memory_order_acquire);
    if (value) return value;
    // A recursive dynamic-loader call is forwarded by the raw Linux bootstrap
    // path and makes the trace fail closed. Normal calls always use libc.
    if (g_depth > 1) { ++g_forward_errors; return nullptr; }
    value = dlsym(RTLD_NEXT, name);
    if (value) slot.store(value, std::memory_order_release);
    else ++g_forward_errors;
    return value;
}
Op operation(unsigned long request) {
    switch (request) {
        case AMDKFD_IOC_CREATE_EVENT: return CreateEvent;
        case AMDKFD_IOC_DESTROY_EVENT: return DestroyEvent;
        case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: return Allocate;
        case AMDKFD_IOC_FREE_MEMORY_OF_GPU: return Free;
        case AMDKFD_IOC_MAP_MEMORY_TO_GPU: return MapGpu;
        case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: return UnmapGpu;
        case AMDKFD_IOC_WAIT_EVENTS: return Wait;
        case AMDKFD_IOC_GET_VERSION: return GetVersion;
        case AMDKFD_IOC_ACQUIRE_VM: return AcquireVm;
        case AMDKFD_IOC_SET_EVENT: return SetEvent;
        case AMDKFD_IOC_RESET_EVENT: return ResetEvent;
        case AMDKFD_IOC_SET_XNACK_MODE: return Xnack;
        case AMDKFD_IOC_RUNTIME_ENABLE: return RuntimeEnable;
        case AMDKFD_IOC_SET_SCRATCH_BACKING_VA: return ScratchVa;
        case AMDKFD_IOC_AVAILABLE_MEMORY: return AvailableMemory;
        case AMDKFD_IOC_CREATE_QUEUE: case AMDKFD_IOC_DESTROY_QUEUE:
        case AMDKFD_IOC_UPDATE_QUEUE: case AMDKFD_IOC_SET_CU_MASK: return QueueControl;
        case AMDKFD_IOC_GET_PROCESS_APERTURES:
        case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: case AMDKFD_IOC_GET_CLOCK_COUNTERS:
        case AMDKFD_IOC_SET_MEMORY_POLICY: case AMDKFD_IOC_SET_TRAP_HANDLER:
        case AMDKFD_IOC_GET_TILE_CONFIG: return OtherKnown;
        default: return Other;
    }
}
bool mutator(Op op) { return op != Wait && op != GetVersion && op != QueueControl && op != OtherKnown && op != AvailableMemory; }
void payload(Row& row) {
    const auto* address = reinterpret_cast<const void*>(uintptr_t(row.argument));
    if (row.phase == 2 && row.result != 0) return;
    row.output_valid = row.phase == 2;
    switch (row.op) {
        case CreateEvent: {
            kfd_ioctl_create_event_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.event_page_offset; row.values[1] = a.event_trigger_data;
            row.values[2] = a.event_type; row.values[3] = a.auto_reset; row.values[4] = a.node_id;
            row.values[5] = a.event_id; row.values[6] = a.event_slot_index; break;
        }
        case DestroyEvent: {
            kfd_ioctl_destroy_event_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.event_id; break;
        }
        case Allocate: {
            kfd_ioctl_alloc_memory_of_gpu_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.va_addr; row.values[1] = a.size; row.values[2] = a.handle;
            row.values[3] = a.mmap_offset; row.values[4] = a.gpu_id; row.values[5] = a.flags; break;
        }
        case Free: {
            kfd_ioctl_free_memory_of_gpu_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.handle; break;
        }
        case MapGpu: case UnmapGpu: {
            kfd_ioctl_map_memory_to_gpu_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.handle; row.values[1] = a.device_ids_array_ptr;
            row.values[2] = a.n_devices; row.values[3] = a.n_success; break;
        }
        case AcquireVm: {
            kfd_ioctl_acquire_vm_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.drm_fd; row.values[1] = a.gpu_id;
            if (row.payload_observed) row.associated_identity = identify(int(a.drm_fd));
            break;
        }
        case SetEvent: {
            kfd_ioctl_set_event_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.event_id; row.values[1] = a.pad; break;
        }
        case ResetEvent: {
            kfd_ioctl_reset_event_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.event_id; row.values[1] = a.pad; break;
        }
        case Xnack: {
            kfd_ioctl_set_xnack_mode_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = uint64_t(int64_t(a.xnack_enabled)); break;
        }
        case RuntimeEnable: {
            kfd_ioctl_runtime_enable_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.r_debug; row.values[1] = a.mode_mask; row.values[2] = a.capabilities_mask; break;
        }
        case ScratchVa: {
            kfd_ioctl_set_scratch_backing_va_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.va_addr; row.values[1] = a.gpu_id; row.values[2] = a.pad; break;
        }
        case AvailableMemory: {
            kfd_ioctl_get_available_memory_args a{}; row.payload_observed = read_output(static_cast<const decltype(a)*>(address), a);
            row.values[0] = a.gpu_id; row.values[1] = a.available; row.values[2] = a.pad; break;
        }
        default: row.payload_observed = true; break;
    }
}
void append(Row row) {
    ++g_metadata_inflight;
    pthread_mutex_lock(&g_lock);
    row.sequence = ++g_attempted;
    if (row.identity.kind || row.associated_identity.kind) g_related = true;
    if (!row.ns || (row.op && !row.payload_observed && (row.phase == 1 || row.output_valid))) ++g_read_errors;
    if (g_count < kRows) g_rows[g_count++] = row;
    else ++g_dropped;
    pthread_mutex_unlock(&g_lock);
    --g_metadata_inflight;
}
Row begin(Api api, int fd = -1) {
    Row row{}; row.api = api; row.phase = 1; row.call = ++g_calls;
    row.tid = uint64_t(syscall(SYS_gettid)); row.pid = uint64_t(getpid());
    row.ns = now(); row.identity = identify(fd);
    return row;
}
void end(Row row, int64_t result, int error) {
    row.phase = 2; row.ns = now(); row.result = result; row.saved_errno = error;
    row.identity = identify(row.identity.fd);
    if (row.api == Ioctl) payload(row);
    append(row);
}
void json_text(FILE* file, const char* value) {
    std::fputc('"', file);
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
        if (*p == '"' || *p == '\\') { std::fputc('\\', file); std::fputc(*p, file); }
        else if (*p < 32) std::fprintf(file, "\\u%04x", unsigned(*p));
        else std::fputc(*p, file);
    }
    std::fputc('"', file);
}
void identity_json(FILE* file, const Identity& id) {
    std::fprintf(file, "{\"fd\":%d,\"observed\":%s,\"path_complete\":%s,\"kind\":%u,\"dev\":%" PRIu64
        ",\"rdev\":%" PRIu64 ",\"inode\":%" PRIu64 ",\"mode\":%u,\"rdev_major\":%u,\"rdev_minor\":%u,\"path\":",
        id.fd, id.observed ? "true" : "false", id.path_complete ? "true" : "false", id.kind,
        id.device, id.rdevice, id.inode, id.mode, unsigned(major(id.rdevice)), unsigned(minor(id.rdevice)));
    json_text(file, id.path); std::fputc('}', file);
}
void fork_prepare() { pthread_mutex_lock(&g_lock); }
void fork_parent() { pthread_mutex_unlock(&g_lock); }
void fork_child() {
    g_fork_parent_pid = g_process_pid.load(); g_discarded_parent_rows = g_count;
    g_inherited_unsupported = g_inherited_unsupported || g_related;
    g_process_pid.store(uint64_t(getpid())); g_count = 0; g_attempted = 0; g_dropped = 0; g_read_errors = 0;
    g_calls.store(0); g_metadata_inflight.store(0); g_mutators.store(0);
    g_depth = 0; g_writing = false; g_related = false;
    pthread_mutex_unlock(&g_lock);
}
__attribute__((constructor)) void initialize_fork_observation() {
    g_process_pid.store(uint64_t(getpid()));
    g_atfork_registered = pthread_atfork(fork_prepare, fork_parent, fork_child) == 0;
    if (!g_atfork_registered) ++g_forward_errors;
}
}  // namespace

extern "C" __attribute__((visibility("hidden")))
int mlsys_kfd_dispatch_ioctl(int fd, unsigned long request, uintptr_t argument) {
    const int entered_errno = errno; ++g_depth;
    auto real = reinterpret_cast<IoctlFunction>(resolve(g_ioctl, "ioctl"));
    const Identity id = identify(fd);
    const bool record = !g_writing && g_depth == 1 && process_valid() && id.kind == 1;
    if (!g_writing && g_depth > 1 && id.kind == 1) ++g_reentrant;
    Row row{}; bool changing = false;
    if (record) {
        row = begin(Ioctl, fd); row.request = request; row.argument = argument; row.op = operation(request);
        payload(row); changing = mutator(Op(row.op)); if (changing) ++g_mutators;
        append(row);
    }
    errno = entered_errno;
    const int result = real ? real(fd, request, argument) : int(syscall(SYS_ioctl, fd, request, argument));
    const int returned_errno = errno;
    if (record) { end(row, result, returned_errno); if (changing) --g_mutators; }
    --g_depth; errno = returned_errno; return result;
}
// Do not va_arg an ioctl caller that supplied no third argument. Preserve the
// incoming SysV RDX bits and tail-transfer to a fixed-arity ABI dispatcher.
extern "C" __attribute__((naked, visibility("default")))
int ioctl(int, unsigned long, ...) noexcept { __asm__ volatile("jmp mlsys_kfd_dispatch_ioctl"); }

extern "C" __attribute__((visibility("default")))
void* mmap(void* address, size_t length, int protection, int flags, int fd, off_t offset) noexcept {
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<MmapFunction>(resolve(g_mmap, "mmap"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) {
        row = begin(Mmap, fd); row.address = uintptr_t(address); row.length = length;
        row.offset = uint64_t(offset); row.protection = protection; row.flags = flags;
        ++g_mutators; append(row);
    }
    errno = previous;
    void* result = real ? real(address, length, protection, flags, fd, offset)
        : reinterpret_cast<void*>(syscall(SYS_mmap, address, length, protection, flags, fd, offset));
    const int error = errno;
    if (record) { end(row, int64_t(reinterpret_cast<uintptr_t>(result)), error); --g_mutators; }
    --g_depth; errno = error; return result;
}
extern "C" __attribute__((visibility("default")))
void* mmap64(void* address, size_t length, int protection, int flags, int fd, off64_t offset) noexcept {
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<Mmap64Function>(resolve(g_mmap64, "mmap64"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) {
        row = begin(Mmap64, fd); row.address = uintptr_t(address); row.length = length;
        row.offset = uint64_t(offset); row.protection = protection; row.flags = flags;
        ++g_mutators; append(row);
    }
    errno = previous;
    void* result = real ? real(address, length, protection, flags, fd, offset)
        : reinterpret_cast<void*>(syscall(SYS_mmap, address, length, protection, flags, fd, offset));
    const int error = errno;
    if (record) { end(row, int64_t(reinterpret_cast<uintptr_t>(result)), error); --g_mutators; }
    --g_depth; errno = error; return result;
}
extern "C" __attribute__((visibility("default")))
int munmap(void* address, size_t length) noexcept {
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<MunmapFunction>(resolve(g_munmap, "munmap"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) { row = begin(Munmap); row.address = uintptr_t(address); row.length = length; ++g_mutators; append(row); }
    errno = previous;
    const int result = real ? real(address, length) : int(syscall(SYS_munmap, address, length));
    const int error = errno;
    if (record) { end(row, result, error); --g_mutators; }
    --g_depth; errno = error; return result;
}

// close and dup target replacement are the bounded FD-lifetime surface.
// fcntl duplication and open only choose unused FDs and do not replace a live FD.
extern "C" __attribute__((visibility("default")))
int close(int fd) {
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<CloseFunction>(resolve(g_close, "close"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) { row = begin(Close, fd); ++g_mutators; append(row); }
    errno = previous;
    const int result = real ? real(fd) : int(syscall(SYS_close, fd));
    const int error = errno;
    if (record) { end(row, result, error); --g_mutators; }
    --g_depth; errno = error; return result;
}
extern "C" __attribute__((visibility("default")))
int dup2(int source, int target) noexcept {
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<Dup2Function>(resolve(g_dup2, "dup2"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) { row = begin(Dup2, source); row.target_fd = target; ++g_mutators; append(row); }
    errno = previous;
    const int result = real ? real(source, target) : int(syscall(SYS_dup2, source, target));
    const int error = errno;
    if (record) { end(row, result, error); --g_mutators; }
    --g_depth; errno = error; return result;
}
extern "C" __attribute__((visibility("default")))
int dup3(int source, int target, int flags) noexcept {
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<Dup3Function>(resolve(g_dup3, "dup3"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) { row = begin(Dup3, source); row.target_fd = target; row.flags = flags; ++g_mutators; append(row); }
    errno = previous;
    const int result = real ? real(source, target, flags) : int(syscall(SYS_dup3, source, target, flags));
    const int error = errno;
    if (record) { end(row, result, error); --g_mutators; }
    --g_depth; errno = error; return result;
}
extern "C" __attribute__((visibility("default")))
void* mremap(void* address, size_t old_length, size_t new_length, int flags, ...) noexcept {
    void* target = nullptr;
    if (flags & MREMAP_FIXED) {
        va_list arguments; va_start(arguments, flags); target = va_arg(arguments, void*); va_end(arguments);
    }
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<MremapFunction>(resolve(g_mremap, "mremap"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) {
        row = begin(Mremap); row.address = uintptr_t(address); row.length = old_length;
        row.values[0] = new_length; row.values[1] = uintptr_t(target); row.flags = flags;
        ++g_mutators; append(row);
    }
    errno = previous;
    void* result = real ? real(address, old_length, new_length, flags, target)
        : reinterpret_cast<void*>(syscall(SYS_mremap, address, old_length, new_length, flags, target));
    const int error = errno;
    if (record) { end(row, int64_t(reinterpret_cast<uintptr_t>(result)), error); --g_mutators; }
    --g_depth; errno = error; return result;
}
extern "C" __attribute__((visibility("default")))
int close_range(unsigned first, unsigned last, int flags) noexcept {
    const int previous = errno; ++g_depth;
    auto real = reinterpret_cast<CloseRangeFunction>(resolve(g_close_range, "close_range"));
    const bool record = !g_writing && g_depth == 1 && process_valid();
    if (!g_writing && g_depth > 1) ++g_reentrant;
    Row row{};
    if (record) {
        row = begin(CloseRange); row.address = first; row.length = last; row.flags = flags;
        ++g_mutators; append(row);
    }
    errno = previous;
    const int result = real ? real(first, last, flags) : int(syscall(SYS_close_range, first, last, flags));
    const int error = errno;
    if (record) { end(row, result, error); --g_mutators; }
    --g_depth; errno = error; return result;
}

extern "C" __attribute__((visibility("default")))
uint64_t mlsys_kfd_probe_cursor() {
    if (!process_valid()) return UINT64_MAX;  // A raw, unobserved fork may have copied a locked mutex.
    pthread_mutex_lock(&g_lock);
    const uint64_t value = !g_inherited_unsupported && !g_dropped && !g_read_errors && !g_forward_errors.load() && !g_reentrant.load() &&
        !g_metadata_inflight.load() && !g_mutators.load() && g_count == g_attempted ? uint64_t(g_count) : UINT64_MAX;
    pthread_mutex_unlock(&g_lock); return value;
}
extern "C" __attribute__((visibility("default")))
int mlsys_kfd_probe_write(const char* path) {
    if (!path || !path[0] || strnlen(path, 4096) >= 4096 || g_writing) return EINVAL;
    if (!process_valid()) return ECHILD;
    const int previous = errno; g_writing = true;
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { const int error = errno; g_writing = false; errno = previous; return error; }
    FILE* file = fdopen(fd, "w");
    if (!file) { const int error = errno; syscall(SYS_close, fd); g_writing = false; errno = previous; return error; }
    pthread_mutex_lock(&g_lock);
    std::fprintf(file, "{\"schema_version\":1,\"kind\":\"passive_kfd_event_api_tape\",\"complete_workspace\":false,"
        "\"hardware_validated\":false,\"scope\":\"observed_libc_kfd_and_mapping_prefix\",\"clock\":\"CLOCK_MONOTONIC_RAW\","
        "\"pid\":%ld,\"snapshot_ns\":%" PRIu64 ",\"capacity\":%zu,\"record_size_bytes\":%zu,\"storage_capacity_bytes\":%zu,"
        "\"attempted_rows\":%" PRIu64 ",\"retained_rows\":%zu,\"dropped_rows\":%" PRIu64 ",\"read_errors\":%" PRIu64 ","
        "\"forward_errors\":%" PRIu64 ",\"reentrant_calls\":%" PRIu64 ",\"active_mutators\":%" PRIu64 ",\"last_sequence\":%zu,"
        "\"fork_observation\":{\"atfork_registered\":%s,\"inherited_kfd_resources_unsupported\":%s,\"parent_pid\":%" PRIu64 ",\"discarded_parent_rows\":%" PRIu64 "},"
        "\"abi\":{\"machine\":\"linux_x86_64_sysv\",\"pointer_bytes\":%zu,\"off_t_bytes\":%zu,\"off64_t_bytes\":%zu,"
        "\"kfd_ioctl_major\":%u,\"kfd_ioctl_minor\":%u,\"create_event_bytes\":%zu,\"allocate_bytes\":%zu,\"available_memory_structure_bytes\":%zu,\"gtt_flag\":%u,\"vram_flag\":%u,\"userptr_flag\":%u,"
        "\"doorbell_flag\":%u,\"mmio_flag\":%u,\"map_shared\":%d,\"map_fixed\":%d,\"prot_read\":%d,\"prot_write\":%d,"
        "\"signal_event_type\":%u,\"runtime_enable_mask\":%u,\"runtime_ttmp_mask\":%u,\"requests\":{",
        long(getpid()), now(), kRows, sizeof(Row), sizeof(g_rows), g_attempted, g_count, g_dropped, g_read_errors,
        g_forward_errors.load(), g_reentrant.load(), g_mutators.load(), g_count,
        g_atfork_registered ? "true" : "false", g_inherited_unsupported ? "true" : "false", g_fork_parent_pid, g_discarded_parent_rows,
        sizeof(void*), sizeof(off_t), sizeof(off64_t),
        unsigned(KFD_IOCTL_MAJOR_VERSION), unsigned(KFD_IOCTL_MINOR_VERSION), sizeof(kfd_ioctl_create_event_args), sizeof(kfd_ioctl_alloc_memory_of_gpu_args),
        sizeof(kfd_ioctl_get_available_memory_args),
        unsigned(KFD_IOC_ALLOC_MEM_FLAGS_GTT), unsigned(KFD_IOC_ALLOC_MEM_FLAGS_VRAM), unsigned(KFD_IOC_ALLOC_MEM_FLAGS_USERPTR),
        unsigned(KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL), unsigned(KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP),
        MAP_SHARED, MAP_FIXED, PROT_READ, PROT_WRITE, unsigned(KFD_IOC_EVENT_SIGNAL),
        unsigned(KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK), unsigned(KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK));
    const unsigned long requests[] = {AMDKFD_IOC_CREATE_EVENT, AMDKFD_IOC_DESTROY_EVENT, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
        AMDKFD_IOC_FREE_MEMORY_OF_GPU, AMDKFD_IOC_MAP_MEMORY_TO_GPU, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU};
    for (size_t i = 0; i < sizeof(requests) / sizeof(requests[0]); ++i)
        std::fprintf(file, "%s\"%zu\":%lu", i ? "," : "", i + 1, requests[i]);
    std::fprintf(file, "},\"acquire_vm_request\":%lu,\"request_operations\":{", static_cast<unsigned long>(AMDKFD_IOC_ACQUIRE_VM));
    const unsigned long classified[] = {AMDKFD_IOC_CREATE_EVENT, AMDKFD_IOC_DESTROY_EVENT,
        AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, AMDKFD_IOC_FREE_MEMORY_OF_GPU, AMDKFD_IOC_MAP_MEMORY_TO_GPU,
        AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, AMDKFD_IOC_WAIT_EVENTS, AMDKFD_IOC_GET_VERSION, AMDKFD_IOC_ACQUIRE_VM,
        AMDKFD_IOC_CREATE_QUEUE, AMDKFD_IOC_DESTROY_QUEUE, AMDKFD_IOC_UPDATE_QUEUE, AMDKFD_IOC_SET_CU_MASK,
        AMDKFD_IOC_GET_PROCESS_APERTURES, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, AMDKFD_IOC_GET_CLOCK_COUNTERS,
        AMDKFD_IOC_SET_MEMORY_POLICY, AMDKFD_IOC_SET_TRAP_HANDLER, AMDKFD_IOC_GET_TILE_CONFIG,
        AMDKFD_IOC_SET_EVENT, AMDKFD_IOC_RESET_EVENT, AMDKFD_IOC_SET_XNACK_MODE,
        AMDKFD_IOC_RUNTIME_ENABLE, AMDKFD_IOC_SET_SCRATCH_BACKING_VA, AMDKFD_IOC_AVAILABLE_MEMORY};
    for (size_t i = 0; i < sizeof(classified) / sizeof(classified[0]); ++i)
        std::fprintf(file, "%s\"%lu\":%u", i ? "," : "", classified[i], unsigned(operation(classified[i])));
    std::fputs("},\"offsets\":{", file);
#define KFD_FIELD(type, member, key, comma) std::fprintf(file, comma "\"" key "\":[%zu,%zu]", offsetof(type, member), sizeof(((type*) nullptr)->member))
    KFD_FIELD(kfd_ioctl_create_event_args, event_page_offset, "create_page", "");
    KFD_FIELD(kfd_ioctl_create_event_args, event_id, "create_id", ",");
    KFD_FIELD(kfd_ioctl_create_event_args, event_slot_index, "create_slot", ",");
    KFD_FIELD(kfd_ioctl_alloc_memory_of_gpu_args, va_addr, "allocate_va", ",");
    KFD_FIELD(kfd_ioctl_alloc_memory_of_gpu_args, handle, "allocate_handle", ",");
    KFD_FIELD(kfd_ioctl_alloc_memory_of_gpu_args, mmap_offset, "allocate_mmap_offset", ",");
    KFD_FIELD(kfd_ioctl_alloc_memory_of_gpu_args, gpu_id, "allocate_gpu_id", ",");
    KFD_FIELD(kfd_ioctl_alloc_memory_of_gpu_args, flags, "allocate_flags", ",");
    KFD_FIELD(kfd_ioctl_acquire_vm_args, drm_fd, "acquire_drm_fd", ",");
    KFD_FIELD(kfd_ioctl_acquire_vm_args, gpu_id, "acquire_gpu_id", ",");
    KFD_FIELD(kfd_ioctl_get_available_memory_args, available, "available_memory_bytes", ",");
    KFD_FIELD(kfd_ioctl_get_available_memory_args, gpu_id, "available_memory_gpu_id", ",");
#undef KFD_FIELD
    std::fprintf(file, "}},\"cplusplus\":%ld,\"compiler_version\":", long(__cplusplus));
    json_text(file, __VERSION__); std::fputs(",\"compiled_source_path\":", file); json_text(file, __FILE__);
    std::fputs(",\"libc_bindings\":{", file);
    const char* symbols[] = {"ioctl", "mmap", "mmap64", "munmap", "close", "dup2", "dup3", "mremap", "close_range"};
    std::atomic<void*>* slots[] = {&g_ioctl, &g_mmap, &g_mmap64, &g_munmap, &g_close,
        &g_dup2, &g_dup3, &g_mremap, &g_close_range};
    for (size_t i = 0; i < sizeof(symbols) / sizeof(symbols[0]); ++i) {
        if (i) std::fputc(',', file);
        json_text(file, symbols[i]); std::fputs(":{\"address\":", file);
        void* address = slots[i]->load(std::memory_order_acquire);
        if (address) std::fprintf(file, "%" PRIu64, uint64_t(reinterpret_cast<uintptr_t>(address)));
        else std::fputs("null", file);
        Dl_info info{};
        const bool known = address && dladdr(address, &info) && info.dli_fname && info.dli_fbase;
        std::fprintf(file, ",\"dladdr_success\":%s,\"module_base\":%" PRIu64 ",\"module_path\":",
            known ? "true" : "false", known ? uint64_t(reinterpret_cast<uintptr_t>(info.dli_fbase)) : 0);
        json_text(file, known ? info.dli_fname : ""); std::fputs(",\"symbol_name\":", file);
        if (known && info.dli_sname) json_text(file, info.dli_sname); else std::fputs("null", file);
        std::fputc('}', file);
    }
    std::fputs("},\"events\":[", file);
    for (size_t i = 0; i < g_count; ++i) {
        const auto& row = g_rows[i];
        std::fprintf(file, "%s{\"seq\":%" PRIu64 ",\"call_id\":%" PRIu64 ",\"ns\":%" PRIu64 ",\"tid\":%" PRIu64
            ",\"pid\":%" PRIu64 ",\"api\":%u,\"operation\":%u,\"phase\":%u,\"retval\":%" PRId64 ",\"errno\":%d,\"fd_identity\":",
            i ? "," : "", row.sequence, row.call, row.ns, row.tid, row.pid, row.api, row.op, row.phase, row.result, row.saved_errno);
        identity_json(file, row.identity);
        std::fputs(",\"associated_fd_identity\":", file); identity_json(file, row.associated_identity);
        std::fprintf(file, ",\"request\":%" PRIu64 ",\"argument_address\":%" PRIu64 ",\"address\":%" PRIu64 ",\"length\":%" PRIu64
            ",\"offset\":%" PRIu64 ",\"protection\":%" PRId64 ",\"flags\":%" PRId64 ",\"target_fd\":%" PRId64
            ",\"payload_observed\":%s,\"output_valid\":%s,\"values\":[",
            row.request, row.argument, row.address, row.length, row.offset, row.protection, row.flags, row.target_fd,
            row.payload_observed ? "true" : "false", row.output_valid ? "true" : "false");
        for (size_t j = 0; j < 8; ++j) std::fprintf(file, "%s%" PRIu64, j ? "," : "", row.values[j]);
        std::fputs("]}", file);
    }
    std::fputs("],\"write_complete\":true}\n", file);
    pthread_mutex_unlock(&g_lock);
    int error = std::ferror(file) ? EIO : 0;
    if (!error && std::fflush(file)) error = errno ? errno : EIO;
    if (!error && fsync(fd)) error = errno ? errno : EIO;
    if (std::fclose(file) && !error) error = errno ? errno : EIO;
    g_writing = false; errno = previous; return error;
}
