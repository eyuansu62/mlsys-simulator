// Callback-external, public-API HSA signal allocation observations.
// No signal object is dereferenced; only public API output structures are read.
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {
using Cursor = uint64_t (*)();
using PointerInfo = decltype(&hsa_amd_pointer_info);
using AgentInfo = decltype(&hsa_agent_get_info);
using EventId = decltype(&hsa_amd_signal_get_event_id);
using Create = decltype(&hsa_signal_create);
using Destroy = decltype(&hsa_signal_destroy);
Cursor cursor = nullptr;
PointerInfo pointer_info = nullptr;
AgentInfo agent_info = nullptr;
EventId event_id = nullptr;
Create create_signal = nullptr;
Destroy destroy_signal = nullptr;
void* runtime_handle = nullptr;
constexpr uint64_t uncertain = std::numeric_limits<uint64_t>::max();

uint64_t address(const void* p) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)); }
uint64_t clock_raw() {
    timespec t{};
    return clock_gettime(CLOCK_MONOTONIC_RAW, &t) == 0
        ? uint64_t(t.tv_sec) * UINT64_C(1000000000) + uint64_t(t.tv_nsec) : 0;
}
std::string quote(const char* value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : std::string(value ? value : "")) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) {
            const char* hex = "0123456789abcdef";
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else out << char(c);
    }
    out << '"';
    return out.str();
}
int publish(const std::string& text, char* output, size_t capacity) {
    if (!output || !capacity) return EINVAL;
    output[0] = '\0';
    if (text.size() >= capacity) return ERANGE;
    std::memcpy(output, text.c_str(), text.size() + 1);
    return 0;
}
std::string binding(const void* fn) {
    Dl_info info{};
    const bool ok = fn && dladdr(fn, &info) != 0;
    std::ostringstream out;
    out << "{\"address\":" << address(fn) << ",\"dladdr_success\":" << (ok ? "true" : "false")
        << ",\"module_path\":" << quote(ok ? info.dli_fname : "")
        << ",\"module_base\":" << address(ok ? info.dli_fbase : nullptr)
        << ",\"symbol_name\":" << quote(ok ? info.dli_sname : "") << '}';
    return out.str();
}
template <class F> std::string binding(F fn) { return binding(reinterpret_cast<const void*>(fn)); }
template <class F> F resolve(const char* name) { return reinterpret_cast<F>(dlsym(runtime_handle, name)); }
struct Window {
    unsigned before_attempts = 0, after_attempts = 0;
    uint64_t before, start, end = 0, after = uncertain;
    static uint64_t snapshot(unsigned& attempts) {
        for (attempts = 1; attempts <= 64; ++attempts) {
            const uint64_t value = cursor ? cursor() : uncertain;
            if (value != uncertain || attempts == 64) return value;
            sched_yield();
        }
        return uncertain;
    }
    Window() : before(snapshot(before_attempts)), start(clock_raw()) {}
    void finish() { end = clock_raw(); after = snapshot(after_attempts); }
    std::string fields() const {
        std::ostringstream out;
        out << "\"before_cursor\":" << before << ",\"after_cursor\":" << after
            << ",\"before_cursor_attempts\":" << before_attempts << ",\"after_cursor_attempts\":" << after_attempts
            << ",\"started_monotonic_raw_ns\":" << start << ",\"ended_monotonic_raw_ns\":" << end;
        return out.str();
    }
};
std::string abi() {
    std::ostringstream out;
    out << "{\"amd_signal_size\":" << sizeof(amd_signal_t)
        << ",\"amd_signal_alignment\":" << alignof(amd_signal_t)
        << ",\"pointer_info_size\":" << sizeof(hsa_amd_pointer_info_t)
        << ",\"hsa_agent_size\":" << sizeof(hsa_agent_t)
        << ",\"hsa_signal_size\":" << sizeof(hsa_signal_t)
        << ",\"pointer_size\":" << sizeof(void*) << ",\"size_t_size\":" << sizeof(size_t)
        << ",\"device_attribute\":" << unsigned(HSA_AGENT_INFO_DEVICE)
        << ",\"cpu_device_enum\":" << unsigned(HSA_DEVICE_TYPE_CPU)
        << ",\"pointer_type_unknown\":" << unsigned(HSA_EXT_POINTER_TYPE_UNKNOWN)
        << ",\"pointer_type_hsa\":" << unsigned(HSA_EXT_POINTER_TYPE_HSA)
        << ",\"pointer_type_locked\":" << unsigned(HSA_EXT_POINTER_TYPE_LOCKED)
        << ",\"signal_handle_address_mask\":" << uncertain
        << ",\"address_rule\":\"unhinted_amd_signal_handle\",\"pointer_info_fields\":{";
    bool first = true;
    auto field = [&](const char* name, size_t offset, size_t bytes) {
        if (!first) out << ',';
        first = false;
        out << quote(name) << ":[" << offset << ',' << bytes << ']';
    };
#define FIELD(member, key) field(key, offsetof(hsa_amd_pointer_info_t, member), sizeof(((hsa_amd_pointer_info_t*)nullptr)->member))
    FIELD(size, "returned_size"); FIELD(type, "type");
    FIELD(agentBaseAddress, "agent_base_address"); FIELD(hostBaseAddress, "host_base_address");
    FIELD(sizeInBytes, "size_bytes"); FIELD(agentOwner, "owner_agent");
    FIELD(global_flags, "global_flags"); FIELD(registered, "registered"); FIELD(alloc_flags, "alloc_flags");
#undef FIELD
    out << "}}";
    return out.str();
}
template <class T> bool covers(uint32_t size, size_t offset) {
    return offset <= size && sizeof(T) <= size - offset;
}
std::string sample(uint64_t handle) {
    std::ostringstream out;
    out << "{\"process_id\":" << getpid() << ",\"thread_id\":" << syscall(SYS_gettid)
        << ",\"signal_handle\":" << handle << ",\"abi_signal_address\":" << handle
        << ",\"address_rule\":\"unhinted_amd_signal_handle\",\"api_calls\":[";
    if (!handle || handle % alignof(amd_signal_t)) {
        out << "],\"error\":\"Signal handle is not a nonzero unhinted ABI-aligned address\"}";
        return out.str();
    }
    hsa_amd_pointer_info_t info{};
    info.size = sizeof(info);
    const uint32_t input_size = info.size;
    Window pw;
    const hsa_status_t ps = pointer_info(reinterpret_cast<const void*>(uintptr_t(handle)), &info,
                                        nullptr, nullptr, nullptr);
    pw.finish();
    const bool typed = ps == HSA_STATUS_SUCCESS && covers<hsa_amd_pointer_type_t>(info.size, offsetof(hsa_amd_pointer_info_t, type));
    const bool known = typed && info.type != HSA_EXT_POINTER_TYPE_UNKNOWN;
    const bool owner_known = known && covers<hsa_agent_t>(info.size, offsetof(hsa_amd_pointer_info_t, agentOwner));
    out << "{\"api\":\"hsa_amd_pointer_info\"," << pw.fields()
        << ",\"input\":{\"pointer_address\":" << handle << ",\"info_address\":" << address(&info)
        << ",\"input_size\":" << input_size << ",\"alloc_address\":0,\"num_agents_address\":0,\"accessible_address\":0}"
        << ",\"output\":{\"status_type\":2,\"status\":" << int64_t(ps)
        << ",\"returned_size\":" << (ps == HSA_STATUS_SUCCESS ? std::to_string(info.size) : "null")
        << ",\"type\":" << (typed ? std::to_string(unsigned(info.type)) : "null");
    auto value = [&](const char* name, size_t offset, size_t bytes, const std::string& text) {
        out << ',' << quote(name) << ':' << (known && offset <= info.size && bytes <= info.size - offset ? text : "null");
    };
    // Conditional expressions below are required: unsupported/UNKNOWN outputs
    // are not even evaluated, not merely hidden by the JSON serializer.
#define V(member, key, expression) do { \
    const bool readable = known && covers<decltype(info.member)>(info.size, offsetof(hsa_amd_pointer_info_t, member)); \
    value(key, offsetof(hsa_amd_pointer_info_t, member), sizeof(info.member), readable ? (expression) : "null"); \
} while (false)
    V(agentBaseAddress, "agent_base_address", std::to_string(address(info.agentBaseAddress)));
    V(hostBaseAddress, "host_base_address", std::to_string(address(info.hostBaseAddress)));
    V(sizeInBytes, "size_bytes", std::to_string(info.sizeInBytes));
    V(agentOwner, "owner_agent", std::to_string(info.agentOwner.handle));
    V(global_flags, "global_flags", std::to_string(info.global_flags));
    V(registered, "registered", info.registered ? "true" : "false");
    V(alloc_flags, "alloc_flags", std::to_string(info.alloc_flags));
#undef V
    out << "},\"api_binding\":" << binding(pointer_info) << '}';
    if (owner_known && info.agentOwner.handle) {
        hsa_device_type_t device{};
        Window aw;
        const hsa_status_t as = agent_info(info.agentOwner, HSA_AGENT_INFO_DEVICE, &device);
        aw.finish();
        out << ",{\"api\":\"hsa_agent_get_info\"," << aw.fields()
            << ",\"input\":{\"agent_handle\":" << info.agentOwner.handle
            << ",\"attribute\":" << unsigned(HSA_AGENT_INFO_DEVICE) << ",\"output_address\":" << address(&device) << '}';
        out << ",\"output\":{\"status_type\":2,\"status\":" << int64_t(as)
            << ",\"device_type\":" << (as == HSA_STATUS_SUCCESS ? std::to_string(unsigned(device)) : "null")
            << "},\"api_binding\":" << binding(agent_info) << '}';
    }
    // The caller supplies only an API-created generation still live in the
    // pre-shutdown observer window. No fabricated handle or private read occurs.
    uint32_t id = 0;
    Window ew;
    const hsa_status_t es = event_id(hsa_signal_t{handle}, &id);
    ew.finish();
    out << ",{\"api\":\"hsa_amd_signal_get_event_id\"," << ew.fields()
        << ",\"input\":{\"signal_handle\":" << handle << ",\"output_address\":" << address(&id) << '}'
        << ",\"output\":{\"status_type\":2,\"status\":" << int64_t(es)
        << ",\"event_id\":" << (es == HSA_STATUS_SUCCESS ? std::to_string(id) : "null")
        << "},\"api_binding\":" << binding(event_id) << "}]}";
    return out.str();
}
bool ready() { return runtime_handle && cursor && pointer_info && agent_info && event_id && create_signal && destroy_signal; }
}

extern "C" __attribute__((visibility("default")))
int mlsys_hsa_signal_backing_init(const char* path, uint64_t cursor_address, char* output, size_t capacity) {
    if (!path || !path[0] || !cursor_address || !output || !capacity) return EINVAL;
    // NOLOAD binds the already initialized Runtime; this provider never starts
    // another HSA Runtime or changes its global initialization reference count.
    runtime_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    cursor = reinterpret_cast<Cursor>(uintptr_t(cursor_address));
    pointer_info = nullptr; agent_info = nullptr; event_id = nullptr;
    create_signal = nullptr; destroy_signal = nullptr;
    if (runtime_handle) {
        pointer_info = resolve<PointerInfo>("hsa_amd_pointer_info");
        agent_info = resolve<AgentInfo>("hsa_agent_get_info");
        event_id = resolve<EventId>("hsa_amd_signal_get_event_id");
        create_signal = resolve<Create>("hsa_signal_create");
        destroy_signal = resolve<Destroy>("hsa_signal_destroy");
    }
    std::ostringstream out;
    out << "{\"schema_version\":1,\"kind\":\"public_hsa_signal_backing_provider\",\"ready\":"
        << (ready() ? "true" : "false") << ",\"process_id\":" << getpid()
        << ",\"thread_id\":" << syscall(SYS_gettid)
        << ",\"compiled_source_path\":" << quote(__FILE__)
        << ",\"compiler_version\":" << quote(__VERSION__) << ",\"cplusplus\":" << __cplusplus
        << ",\"abi\":" << abi()
        << ",\"api_bindings\":{\"hsa_amd_pointer_info\":" << binding(pointer_info)
        << ",\"hsa_agent_get_info\":" << binding(agent_info)
        << ",\"hsa_amd_signal_get_event_id\":" << binding(event_id)
        << ",\"hsa_signal_create\":" << binding(create_signal)
        << ",\"hsa_signal_destroy\":" << binding(destroy_signal) << "}}";
    return publish(out.str(), output, capacity);
}

extern "C" __attribute__((visibility("default")))
int mlsys_hsa_signal_backing_sample(uint64_t handle, char* output, size_t capacity) {
    if (!ready()) return ENODEV;
    return publish(sample(handle), output, capacity);
}

extern "C" __attribute__((visibility("default")))
int mlsys_hsa_signal_backing_control(int64_t initial, const uint64_t* consumers, size_t count,
                                    char* output, size_t capacity) {
    if (!ready()) return ENODEV;
    if (count > 4 || (count && !consumers) || !output || !capacity) return EINVAL;
    std::vector<hsa_agent_t> agents;
    for (size_t i = 0; i < count; ++i) {
        if (!consumers[i]) return EINVAL;
        agents.push_back(hsa_agent_t{consumers[i]});
    }
    hsa_signal_t own{};
    Window cw;
    const hsa_status_t cs = create_signal(hsa_signal_value_t(initial), uint32_t(count),
                                         count ? agents.data() : nullptr, &own);
    cw.finish();
    std::ostringstream out;
    out << "{\"kind\":\"owned_ordinary_hsa_signal_control\",\"process_id\":" << getpid()
        << ",\"thread_id\":" << syscall(SYS_gettid) << ",\"create_call\":{\"api\":\"hsa_signal_create\","
        << cw.fields() << ",\"input\":{\"initial_value\":" << initial << ",\"num_consumers\":" << count
        << ",\"output_address\":" << address(&own) << ",\"consumers_address\":"
        << address(count ? agents.data() : nullptr) << ",\"consumer_agents\":[";
    for (size_t i = 0; i < count; ++i) { if (i) out << ','; out << consumers[i]; }
    out << "]},\"output\":{\"status_type\":2,\"status\":" << int64_t(cs)
        << ",\"signal_handle\":" << (cs == HSA_STATUS_SUCCESS ? std::to_string(own.handle) : "null")
        << "},\"api_binding\":" << binding(create_signal) << "},\"samples\":[";
    if (cs == HSA_STATUS_SUCCESS && own.handle) {
        // Only this function's successful create supplies the destroy argument.
        // Runtime-owned signals passed to sample() never reach destroy_signal.
        out << sample(own.handle) << ',' << sample(own.handle) << "],\"destroy_call\":{";
        Window dw;
        const hsa_status_t ds = destroy_signal(own);
        dw.finish();
        out << "\"api\":\"hsa_signal_destroy\"," << dw.fields()
            << ",\"input\":{\"signal_handle\":" << own.handle
            << "},\"output\":{\"status_type\":2,\"status\":" << int64_t(ds)
            << "},\"api_binding\":" << binding(destroy_signal) << "}}";
    } else out << "],\"destroy_call\":null}";
    return publish(out.str(), output, capacity);
}
