// Task-local positive controls, never a Workspace-completeness verdict.
// Compile twice with hipcc -DMLSYS_CONTROL_KIND=1 (heap) or =2 (scratch).
#include <hip/hip_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#ifndef MLSYS_CONTROL_KIND
#error "Set MLSYS_CONTROL_KIND=1 for device heap or =2 for private scratch"
#endif
#if MLSYS_CONTROL_KIND != 1 && MLSYS_CONTROL_KIND != 2
#error "Unknown control kind"
#endif
#ifndef MLSYS_HEAP_BYTES
#define MLSYS_HEAP_BYTES 262144
#endif
#ifndef MLSYS_SCRATCH_WORDS
#define MLSYS_SCRATCH_WORDS 256
#endif

namespace {
constexpr unsigned kTrials = 3; // One real warmup, then two measured repetitions.
constexpr unsigned kThreads = 32, kRounds = 256;
constexpr size_t kHeapBytes = MLSYS_HEAP_BYTES;
constexpr unsigned kScratchWords = MLSYS_SCRATCH_WORDS;
constexpr uint64_t kScratchAdmissionBytes = 512ULL << 20;
static_assert(kHeapBytes >= 4096 && kHeapBytes <= (1U << 20) && kHeapBytes % 4 == 0,
              "Heap request must be 4KiB..1MiB, divisible by four");
static_assert(kScratchWords >= 64 && kScratchWords <= 1024 &&
              (kScratchWords & (kScratchWords - 1)) == 0,
              "Scratch array must have a bounded power-of-two number of words");

struct ApiCall { const char* operation; int status; };
struct Trial {
    uint64_t epoch = 0, pointer = 0, checksum = 0, expected_checksum = 0;
    uint64_t free_before = 0, free_held = 0, free_after = 0;
    unsigned seed = 0, mismatches = 0;
    bool warmup = false, began = false, ended = false, ok = false;
    bool allocation_nonnull = false, pointer_retained = false, freed = false;
};
ApiCall g_calls[256]; size_t g_call_count = 0;
Trial g_trials[kTrials];
bool g_failed = false, g_call_log_overflow = false, g_resource_skip = false;
const char* g_failure = "";
hipDeviceProp_t g_properties{};
hipFuncAttributes g_attributes{};
uint64_t g_private_admission_estimate = 0;

bool checked(const char* operation, hipError_t status) {
    if (g_call_count < 256) g_calls[g_call_count++] = {operation, static_cast<int>(status)};
    else { g_call_log_overflow = true; g_failed = true; }
    if (status != hipSuccess) {
        g_failed = true;
        if (!*g_failure) g_failure = operation;
        std::fprintf(stderr, "%s: HIP status=%d name=%s message=%s\n", operation,
                     int(status), hipGetErrorName(status), hipGetErrorString(status));
        return false;
    }
    return true;
}
bool semantic(bool condition, const char* message) {
    if (!condition) { g_failed = true; if (!*g_failure) g_failure = message; }
    return condition;
}
void json_text(FILE* file, const char* value) {
    std::fputc('"', file);
    if (value) for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
        if (*p == '\\' || *p == '"') { std::fputc('\\', file); std::fputc(*p, file); }
        else if (*p < 32 || *p >= 127) std::fprintf(file, "\\u%04x", unsigned(*p));
        else std::fputc(*p, file);
    }
    std::fputc('"', file);
}

struct Probe {
    void* library = nullptr;
    int (*begin)(uint64_t, const char*) = nullptr;
    int (*end)(uint64_t) = nullptr;
    int (*write)(const char*) = nullptr;
    int (*stop)() = nullptr;
    int load_status = -1, begin_status[kTrials] = {-1, -1, -1};
    int end_status[kTrials] = {-1, -1, -1}, stop_status = -1, write_status = -1;
    char error[512]{};

    bool load(const char* path) {
        // It must be the SAME tool already loaded through ROCP_TOOL_LIBRARIES.
        library = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!library) {
            const char* text = dlerror();
            std::snprintf(error, sizeof(error), "%s", text ? text : "tool was not preloaded");
            load_status = ENOENT; return semantic(false, "native probe is not preloaded");
        }
        begin = reinterpret_cast<decltype(begin)>(dlsym(library, "mlsys_probe_begin"));
        end = reinterpret_cast<decltype(end)>(dlsym(library, "mlsys_probe_end"));
        write = reinterpret_cast<decltype(write)>(dlsym(library, "mlsys_probe_write"));
        stop = reinterpret_cast<decltype(stop)>(dlsym(library, "mlsys_probe_stop"));
        load_status = begin && end && write && stop ? 0 : ENOSYS;
        return semantic(load_status == 0, "native probe exported API is missing");
    }
    bool start(unsigned i) {
        char label[128]{};
        std::snprintf(label, sizeof(label), "%s/%s/repetition-%u",
                      MLSYS_CONTROL_KIND == 1 ? "device-heap" : "private-scratch",
                      i == 0 ? "warmup" : "measured", i);
        begin_status[i] = begin(g_trials[i].epoch, label);
        g_trials[i].began = begin_status[i] == 0;
        return semantic(g_trials[i].began, "native probe begin rejected");
    }
    void finish(unsigned i) {
        if (!g_trials[i].began) return;
        end_status[i] = end(g_trials[i].epoch);
        g_trials[i].ended = end_status[i] == 0;
        semantic(g_trials[i].ended, "native probe end rejected");
    }
    void close(const char* checkpoint) {
        if (!stop || !write) return;
        stop_status = stop();
        semantic(stop_status == 0, "native probe stop rejected");
        write_status = write(checkpoint);
        semantic(write_status == 0, "native probe checkpoint write rejected");
        // No dlclose: the SDK owns a process-lifetime callback into this tool.
    }
};

__host__ __device__ uint32_t pattern(uint32_t index, uint32_t seed) {
    return (index * UINT32_C(2654435761)) ^ seed ^ UINT32_C(0xa5a55a5a);
}

#if MLSYS_CONTROL_KIND == 1
struct HeapState {
    uint64_t pointer, original_pointer, checksum, sentinel;
    uint32_t nonnull, mismatches, freed;
};
__global__ void heap_allocate_and_write(HeapState* state, unsigned seed) {
    if (blockIdx.x || threadIdx.x) return;
    auto* memory = static_cast<uint32_t*>(malloc(kHeapBytes));
    state->pointer = reinterpret_cast<uintptr_t>(memory);
    state->original_pointer = state->pointer;
    state->nonnull = memory != nullptr;
    state->freed = 0; state->mismatches = 0; state->checksum = 0;
    state->sentinel = UINT64_C(0x4d4c535953484541) ^ seed;
    if (!memory) return;
    volatile uint32_t* retained = memory;
    for (uint32_t i = 0; i < kHeapBytes / sizeof(uint32_t); ++i) retained[i] = pattern(i, seed);
}
__global__ void heap_read_and_verify(HeapState* state, unsigned seed) {
    if (blockIdx.x || threadIdx.x) return;
    const volatile uint32_t* memory = reinterpret_cast<const volatile uint32_t*>(state->pointer);
    if (!memory) { state->mismatches = 1; return; }
    uint64_t checksum = 0; uint32_t mismatches = 0;
    for (uint32_t i = 0; i < kHeapBytes / sizeof(uint32_t); ++i) {
        const uint32_t value = memory[i];
        checksum += value;
        if (value != pattern(i, seed)) ++mismatches;
    }
    state->checksum = checksum; state->mismatches = mismatches;
}
__global__ void heap_release(HeapState* state) {
    if (blockIdx.x || threadIdx.x) return;
    void* memory = reinterpret_cast<void*>(state->pointer);
    if (memory) { free(memory); state->freed = 1; }
    state->pointer = 0;
}

bool run_trials(Probe& probe) {
    HeapState* device = nullptr;
    if (!checked("hipMalloc(heap control state)", hipMalloc(reinterpret_cast<void**>(&device), sizeof(*device)))) return false;
    bool all_ok = true;
    for (unsigned i = 0; i < kTrials && all_ok; ++i) {
        Trial& trial = g_trials[i]; trial.epoch = 1001 + i; trial.warmup = i == 0; trial.seed = 10007 + i;
        size_t free_bytes = 0, total = 0;
        all_ok = checked("hipMemset(heap control state)", hipMemset(device, 0, sizeof(*device))) &&
                 checked("hipDeviceSynchronize(before heap epoch)", hipDeviceSynchronize()) &&
                 checked("hipMemGetInfo(before heap epoch)", hipMemGetInfo(&free_bytes, &total));
        trial.free_before = free_bytes;
        if (!all_ok || !probe.start(i)) { all_ok = false; break; }
        HeapState held{}, verified{}, released{};
        hipLaunchKernelGGL(heap_allocate_and_write, dim3(1), dim3(1), 0, 0, device, trial.seed);
        all_ok = checked("heap_allocate_and_write launch", hipGetLastError()) &&
                 checked("hipDeviceSynchronize(after device malloc/write)", hipDeviceSynchronize()) &&
                 checked("hipMemcpy(held heap state)", hipMemcpy(&held, device, sizeof(held), hipMemcpyDeviceToHost));
        trial.pointer = held.pointer; trial.allocation_nonnull = held.nonnull != 0 && held.pointer != 0;
        all_ok = semantic(trial.allocation_nonnull, "device malloc returned null") && all_ok;
        if (all_ok) {
            hipLaunchKernelGGL(heap_read_and_verify, dim3(1), dim3(1), 0, 0, device, trial.seed);
            all_ok = checked("heap_read_and_verify launch", hipGetLastError()) &&
                     checked("hipDeviceSynchronize(after held pointer read)", hipDeviceSynchronize()) &&
                     checked("hipMemcpy(verified heap state)", hipMemcpy(&verified, device, sizeof(verified), hipMemcpyDeviceToHost));
            trial.pointer_retained = verified.pointer == held.pointer && verified.original_pointer == held.pointer &&
                                     verified.sentinel == (UINT64_C(0x4d4c535953484541) ^ trial.seed);
            trial.checksum = verified.checksum; trial.mismatches = verified.mismatches;
            for (uint32_t j = 0; j < kHeapBytes / sizeof(uint32_t); ++j) trial.expected_checksum += pattern(j, trial.seed);
            all_ok = semantic(trial.pointer_retained && !trial.mismatches &&
                              trial.checksum == trial.expected_checksum, "device heap readback/sentinel verification failed") && all_ok;
        }
        if (all_ok) {
            all_ok = checked("hipMemGetInfo(held heap)", hipMemGetInfo(&free_bytes, &total));
            trial.free_held = free_bytes;
        }
        // A distinct kernel frees the retained pointer; not the allocating kernel.
        // Attempt cleanup after a semantic mismatch too. Original errors remain.
        hipLaunchKernelGGL(heap_release, dim3(1), dim3(1), 0, 0, device);
        bool release_ok = checked("heap_release launch", hipGetLastError()) &&
            checked("hipDeviceSynchronize(after device free)", hipDeviceSynchronize()) &&
            checked("hipMemcpy(released heap state)", hipMemcpy(&released, device, sizeof(released), hipMemcpyDeviceToHost));
        trial.freed = released.freed == 1 && released.pointer == 0 && released.original_pointer == held.pointer;
        release_ok = semantic(trial.freed || !trial.allocation_nonnull, "device heap free witness failed") && release_ok;
        bool final_sync = checked("hipDeviceSynchronize(final heap epoch boundary)", hipDeviceSynchronize());
        probe.finish(i);
        bool queried = checked("hipMemGetInfo(after heap epoch)", hipMemGetInfo(&free_bytes, &total));
        trial.free_after = free_bytes;
        trial.ok = all_ok && release_ok && final_sync && queried && trial.ended;
        all_ok = trial.ok;
        std::fprintf(stdout, "heap trial=%u warmup=%u ptr=0x%" PRIx64 " bytes=%zu checksum=%" PRIu64
                     " expected=%" PRIu64 " mismatches=%u retained=%u freed=%u ok=%u\n", i,
                     unsigned(trial.warmup), trial.pointer, kHeapBytes, trial.checksum, trial.expected_checksum,
                     trial.mismatches, unsigned(trial.pointer_retained), unsigned(trial.freed), unsigned(trial.ok));
        std::fflush(stdout);
    }
    const bool freed_state = checked("hipFree(heap control state)", hipFree(device));
    const bool drained = checked("hipDeviceSynchronize(after heap control cleanup)", hipDeviceSynchronize());
    return all_ok && freed_state && drained;
}
#else
// A volatile, dynamically indexed private array is observable local storage.
// localSizeBytes must independently prove that this build actually retained it.
__global__ __launch_bounds__(kThreads) void private_scratch(uint64_t* output, unsigned seed) {
    const unsigned tid = threadIdx.x;
    volatile uint32_t local[kScratchWords];
    for (unsigned i = 0; i < kScratchWords; ++i) local[i] = pattern(i, seed ^ tid);
    uint32_t state = seed ^ (tid * UINT32_C(747796405));
    for (unsigned i = 0; i < kRounds; ++i) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        const unsigned index = (state ^ i) & (kScratchWords - 1);
        const uint32_t previous = local[index];
        local[index] = previous ^ state;
        state += previous;
    }
    uint64_t checksum = state;
    for (unsigned i = 0; i < kScratchWords; ++i) checksum += local[i];
    output[tid] = checksum;
}
uint64_t scratch_reference(unsigned tid, unsigned seed) {
    uint32_t local[kScratchWords];
    for (unsigned i = 0; i < kScratchWords; ++i) local[i] = pattern(i, seed ^ tid);
    uint32_t state = seed ^ (tid * UINT32_C(747796405));
    for (unsigned i = 0; i < kRounds; ++i) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        const unsigned index = (state ^ i) & (kScratchWords - 1);
        const uint32_t previous = local[index]; local[index] = previous ^ state; state += previous;
    }
    uint64_t checksum = state;
    for (unsigned i = 0; i < kScratchWords; ++i) checksum += local[i];
    return checksum;
}
bool run_trials(Probe& probe) {
    if (!checked("hipFuncGetAttributes(private_scratch)", hipFuncGetAttributes(&g_attributes,
                 reinterpret_cast<const void*>(private_scratch)))) return false;
    if (!semantic(g_attributes.localSizeBytes >= kScratchWords * sizeof(uint32_t),
                  "compiled kernel did not retain the required private array")) return false;
    const uint64_t multiprocessors = g_properties.multiProcessorCount > 0 ? g_properties.multiProcessorCount : 0;
    const uint64_t resident_threads = g_properties.maxThreadsPerMultiProcessor > 0 ? g_properties.maxThreadsPerMultiProcessor : 0;
    const uint64_t local_bytes = g_attributes.localSizeBytes;
    if (!multiprocessors || !resident_threads || local_bytes > kScratchAdmissionBytes / 2 /
        multiprocessors / resident_threads) {
        g_resource_skip = true; return semantic(false, "private scratch admission estimate exceeds 512MiB");
    }
    // This admits bounded work, not a hard limit on opaque driver allocations.
    g_private_admission_estimate = local_bytes * resident_threads * multiprocessors * 2;
    size_t free_bytes = 0, total = 0;
    if (!checked("hipMemGetInfo(scratch admission)", hipMemGetInfo(&free_bytes, &total))) return false;
    if (free_bytes < g_private_admission_estimate + (128ULL << 20)) {
        g_resource_skip = true; return semantic(false, "insufficient free memory for scratch diagnostic");
    }
    uint64_t* device = nullptr;
    if (!checked("hipMalloc(scratch output)", hipMalloc(reinterpret_cast<void**>(&device), kThreads * sizeof(uint64_t)))) return false;
    bool all_ok = true;
    for (unsigned i = 0; i < kTrials && all_ok; ++i) {
        Trial& trial = g_trials[i]; trial.epoch = 2001 + i; trial.warmup = i == 0; trial.seed = 20011 + i;
        uint64_t output[kThreads]{};
        all_ok = checked("hipDeviceSynchronize(before scratch epoch)", hipDeviceSynchronize()) &&
                 checked("hipMemGetInfo(before scratch epoch)", hipMemGetInfo(&free_bytes, &total));
        trial.free_before = free_bytes;
        if (!all_ok || !probe.start(i)) { all_ok = false; break; }
        hipLaunchKernelGGL(private_scratch, dim3(1), dim3(kThreads), 0, 0, device, trial.seed);
        all_ok = checked("private_scratch launch", hipGetLastError()) &&
                 checked("hipDeviceSynchronize(after private scratch)", hipDeviceSynchronize()) &&
                 checked("hipMemcpy(scratch output)", hipMemcpy(output, device, sizeof(output), hipMemcpyDeviceToHost));
        if (all_ok) for (unsigned t = 0; t < kThreads; ++t) {
            const uint64_t expected = scratch_reference(t, trial.seed);
            trial.checksum += output[t]; trial.expected_checksum += expected;
            if (output[t] != expected) ++trial.mismatches;
        }
        all_ok = semantic(!trial.mismatches && trial.checksum == trial.expected_checksum,
                          "private scratch reference verification failed") && all_ok;
        const bool final_sync = checked("hipDeviceSynchronize(final scratch epoch boundary)", hipDeviceSynchronize());
        probe.finish(i);
        const bool queried = checked("hipMemGetInfo(after scratch epoch)", hipMemGetInfo(&free_bytes, &total));
        trial.free_after = free_bytes;
        trial.ok = all_ok && final_sync && queried && trial.ended;
        all_ok = trial.ok;
        std::fprintf(stdout, "scratch trial=%u warmup=%u localSizeBytes=%zu words=%u threads=%u rounds=%u"
                     " checksum=%" PRIu64 " expected=%" PRIu64 " mismatches=%u ok=%u\n", i,
                     unsigned(trial.warmup), g_attributes.localSizeBytes, kScratchWords, kThreads, kRounds,
                     trial.checksum, trial.expected_checksum, trial.mismatches, unsigned(trial.ok));
        std::fflush(stdout);
    }
    const bool freed_output = checked("hipFree(scratch output)", hipFree(device));
    const bool drained = checked("hipDeviceSynchronize(after scratch control cleanup)", hipDeviceSynchronize());
    return all_ok && freed_output && drained;
}
#endif

int write_result(const char* path, int device, const Probe& probe) {
    if (std::strlen(path) > 3900) return ENAMETOOLONG;
    char temporary[4096]{};
    std::snprintf(temporary, sizeof(temporary), "%s.tmp.%ld", path, long(getpid()));
    int fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return errno;
    FILE* file = fdopen(fd, "w");
    if (!file) { const int error = errno; close(fd); unlink(temporary); return error; }
    std::fprintf(file, "{\"schema_version\":1,\"kind\":\"device_allocation_positive_control\","
        "\"control\":\"%s\",\"status\":\"%s\",\"complete_workspace\":false,\"hardware_validated\":false,"
        "\"native_observation_verdict\":\"NOT_EVALUATED_BY_THIS_PROGRAM\",\"device_index\":%d,\"device_name\":",
        MLSYS_CONTROL_KIND == 1 ? "device_heap" : "private_scratch",
        g_resource_skip ? "SKIPPED_RESOURCE_GATE" : g_failed ? "FAILED" : "DIAGNOSTIC_CONTROL_SUCCEEDED", device);
    json_text(file, g_properties.name);
    std::fprintf(file, ",\"failure\":"); json_text(file, g_failure);
    std::fprintf(file, ",\"heap_request_bytes\":%zu,\"scratch_words\":%u,\"threads\":%u,\"rounds\":%u,"
        "\"heap_witness_applicable\":%s,\"scratch_attributes_applicable\":%s,"
        "\"scratch_attributes\":{\"localSizeBytes\":%zu,\"sharedSizeBytes\":%zu,\"numRegs\":%d,\"maxThreadsPerBlock\":%d},"
        "\"scratch_admission_estimate_bytes\":%" PRIu64 ",\"scratch_admission_limit_bytes\":%" PRIu64 ","
        "\"admission_is_not_driver_memory_limit\":true,\"trial_results\":[",
        kHeapBytes, kScratchWords, MLSYS_CONTROL_KIND == 1 ? 1U : kThreads,
        MLSYS_CONTROL_KIND == 2 ? kRounds : 0U, MLSYS_CONTROL_KIND == 1 ? "true" : "false",
        MLSYS_CONTROL_KIND == 2 ? "true" : "false", g_attributes.localSizeBytes, g_attributes.sharedSizeBytes,
        g_attributes.numRegs, g_attributes.maxThreadsPerBlock, g_private_admission_estimate, kScratchAdmissionBytes);
    for (unsigned i = 0; i < kTrials; ++i) {
        const Trial& t = g_trials[i]; if (i) std::fputc(',', file);
        std::fprintf(file, "{\"repetition\":%u,\"warmup\":%s,\"epoch_id\":%" PRIu64 ",\"seed\":%u,"
            "\"pointer\":%" PRIu64 ",\"allocation_nonnull\":%s,\"pointer_retained_across_kernel\":%s,\"freed_in_later_kernel\":%s,"
            "\"checksum\":%" PRIu64 ",\"expected_checksum\":%" PRIu64 ",\"verification_mismatches\":%u,"
            "\"free_bytes_before\":%" PRIu64 ",\"free_bytes_while_held\":%" PRIu64 ",\"free_bytes_after\":%" PRIu64 ","
            "\"probe_begin_status\":%d,\"probe_end_status\":%d,\"epoch_ended\":%s,\"passed_semantic_control\":%s}",
            i, t.warmup ? "true" : "false", t.epoch, t.seed, t.pointer, t.allocation_nonnull ? "true" : "false",
            t.pointer_retained ? "true" : "false", t.freed ? "true" : "false", t.checksum, t.expected_checksum,
            t.mismatches, t.free_before, t.free_held, t.free_after, probe.begin_status[i], probe.end_status[i],
            t.ended ? "true" : "false", t.ok ? "true" : "false");
    }
    std::fprintf(file, "],\"probe\":{\"load_status\":%d,\"stop_status\":%d,\"checkpoint_write_status\":%d,\"load_error\":",
                 probe.load_status, probe.stop_status, probe.write_status); json_text(file, probe.error);
    std::fprintf(file, "},\"hip_call_log_overflow\":%s,\"hip_calls\":[", g_call_log_overflow ? "true" : "false");
    for (size_t i = 0; i < g_call_count; ++i) {
        if (i) std::fputc(',', file);
        std::fputs("{\"operation\":", file); json_text(file, g_calls[i].operation);
        std::fprintf(file, ",\"status\":%d,\"error_name\":", g_calls[i].status);
        json_text(file, hipGetErrorName(static_cast<hipError_t>(g_calls[i].status))); std::fputc('}', file);
    }
    std::fputs("],\"limitations\":[\"Semantic kernel success is not proof of profiler visibility\","
        "\"Zero allocation callbacks after warmup can mean hidden heap or scratch-pool reuse\","
        "\"Free-memory queries are diagnostic only, never Workspace attribution\","
        "\"The scratch admission estimate is not a hard bound on opaque driver reserves\"]}\n", file);
    int error = 0;
    if (std::ferror(file) || std::fflush(file) != 0 || fsync(fd) != 0) error = errno ? errno : EIO;
    if (std::fclose(file) != 0 && !error) error = errno;
    if (!error && link(temporary, path) != 0) error = errno;
    unlink(temporary);
    return error;
}
} // namespace

int main(int argc, char** argv) {
    const char *probe_path = nullptr, *output = nullptr; int device = 0;
    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "--probe") || !std::strcmp(argv[i], "--out") ||
             !std::strcmp(argv[i], "--device")) && i + 1 < argc) {
            const char* name = argv[i++];
            if (!std::strcmp(name, "--probe")) probe_path = argv[i];
            else if (!std::strcmp(name, "--out")) output = argv[i];
            else {
                char* end = nullptr; errno = 0; const long parsed = std::strtol(argv[i], &end, 10);
                if (errno || !end || end == argv[i] || *end || parsed < 0 || parsed > 1023) return 2;
                device = static_cast<int>(parsed);
            }
        } else { std::fprintf(stderr, "Usage: %s --probe ABS_NATIVE_PROBE_SO --out UNIQUE_JSON [--device N]\n", argv[0]); return 2; }
    }
    if (!probe_path || !output || probe_path[0] != '/' || output[0] != '/') return 2;
    Probe probe;
    bool ready = checked("hipSetDevice(control)", hipSetDevice(device)) &&
                 checked("hipGetDeviceProperties(control)", hipGetDeviceProperties(&g_properties, device)) &&
                 probe.load(probe_path);
    if (ready) run_trials(probe);
    char checkpoint[4096]{};
    if (std::strlen(output) <= 3900) {
        std::snprintf(checkpoint, sizeof(checkpoint), "%s.native-checkpoint.json", output);
        probe.close(checkpoint);
    } else semantic(false, "output path is too long");
    const int write_status = write_result(output, device, probe);
    if (write_status) { std::fprintf(stderr, "write_result: errno=%d\n", write_status); return 3; }
    return g_resource_skip ? 4 : g_failed ? 1 : 0;
}
