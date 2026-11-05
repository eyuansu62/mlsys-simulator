// Independent ordinary-signal control for the passive Native/KFD observers.
// The only destroyed signals are the ones successfully created by this process.
#include <hsa/hsa.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
using Begin = int (*)(uint64_t, const char*);
using End = int (*)(uint64_t);
using Action = int (*)();
using Write = int (*)(const char*);
using Init = int (*)(const char*, uint64_t, char*, size_t);
using Sample = int (*)(uint64_t, char*, size_t);
template <typename T> T symbol(void* library, const char* name) {
    return reinterpret_cast<T>(dlsym(library, name));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fputs("Usage: hsa_signal_namespace_control OUT_DIR PROVIDER_SO ROCR_SO COUNT\n", stderr);
        return 2;
    }
    char* count_end = nullptr;
    errno = 0;
    const unsigned long count = std::strtoul(argv[4], &count_end, 10);
    if (errno || !count_end || *count_end || count < 1 || count > 1024) return 2;
    const std::string root = argv[1];
    const auto begin = symbol<Begin>(RTLD_DEFAULT, "mlsys_probe_begin");
    const auto end = symbol<End>(RTLD_DEFAULT, "mlsys_probe_end");
    const auto stop = symbol<Action>(RTLD_DEFAULT, "mlsys_probe_stop");
    const auto finalize = symbol<Action>(RTLD_DEFAULT, "mlsys_probe_finalize");
    const auto kfd_write = symbol<Write>(RTLD_DEFAULT, "mlsys_kfd_probe_write");
    void* cursor = dlsym(RTLD_DEFAULT, "mlsys_probe_cursor");
    void* provider = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!begin || !end || !stop || !finalize || !kfd_write || !cursor || !provider) return 3;
    const auto initialize = symbol<Init>(provider, "mlsys_hsa_signal_backing_init");
    const auto sample = symbol<Sample>(provider, "mlsys_hsa_signal_backing_sample");
    if (!initialize || !sample) return 3;
    const int fd = open((root + "/control.jsonl").c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return 4;
    FILE* file = fdopen(fd, "w");
    if (!file) { close(fd); return 4; }

    const auto initialized = hsa_init();
    const int begun = begin(8001, "cold_ordinary_hsa_signal_namespace_control");
    char output[65536]{};
    const int configured = initialize(argv[3], uint64_t(reinterpret_cast<uintptr_t>(cursor)),
                                     output, sizeof(output));
    std::fprintf(file, "{\"kind\":\"cold_ordinary_signal_control\",\"pid\":%ld,"
        "\"init_status\":%u,\"native_begin\":%d,\"provider_init_status\":%d,"
        "\"complete_workspace\":false,\"hardware_validated\":false}\n",
        long(getpid()), unsigned(initialized), begun, configured);
    if (configured == 0) std::fprintf(file, "%s\n", output);

    std::vector<hsa_signal_t> own;
    own.reserve(count);
    int failures = 0;
    if (initialized == HSA_STATUS_SUCCESS) {
        for (unsigned long i = 0; i < count; ++i) {
            hsa_signal_t signal{};
            const auto status = hsa_signal_create(0, 0, nullptr, &signal);
            std::fprintf(file, "{\"operation\":\"create_own_control\",\"index\":%lu,"
                "\"status\":%u,\"handle\":%" PRIu64 "}\n", i, unsigned(status), signal.handle);
            if (status != HSA_STATUS_SUCCESS) { ++failures; break; }
            own.push_back(signal);
        }
    }
    if (configured == 0) {
        for (const auto signal : own) {
            for (unsigned n = 0; n < 2; ++n) {
                const int status = sample(signal.handle, output, sizeof(output));
                std::fprintf(file, "{\"operation\":\"sample_own_control\",\"status\":%d,"
                    "\"handle\":%" PRIu64 ",\"index\":%u}\n", status, signal.handle, n);
                if (status == 0) std::fprintf(file, "%s\n", output);
                else ++failures;
            }
        }
    }
    for (const auto signal : own) {
        const auto status = hsa_signal_destroy(signal);
        std::fprintf(file, "{\"operation\":\"destroy_own_control\",\"status\":%u,"
            "\"handle\":%" PRIu64 "}\n", unsigned(status), signal.handle);
        if (status != HSA_STATUS_SUCCESS) ++failures;
    }
    const int ended = end(8001);
    const int stopped = stop();
    const int finalized = finalize();
    const int written = kfd_write((root + "/kfd.json").c_str());
    std::fprintf(file, "{\"native_end\":%d,\"native_stop\":%d,\"native_finalize\":%d,"
        "\"kfd_write\":%d,\"failures\":%d}\n", ended, stopped, finalized, written, failures);
    const bool write_error = std::ferror(file) != 0;
    const int closed = std::fclose(file);
    return initialized == HSA_STATUS_SUCCESS && begun == 0 && configured == 0 &&
        ended == 0 && stopped == 0 && finalized == 0 && written == 0 && !failures &&
        !write_error && closed == 0 ? 0 : 1;
}
