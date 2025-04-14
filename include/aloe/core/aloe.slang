#define ALOE_STORAGE_BUFFER_BINDING 0

// This is a C++ File masquerading as a slang file.
const char* SOURCE_STRING = R"(

module aloe;

[[vk::binding(0, 0)]]
public __DynamicResource<__DynamicResourceKind::General> g_buffers[];

namespace aloe {

public struct BufferHandle {
    int64_t id;

    public RWStructuredBuffer<T> get<T>() {
        return (RWStructuredBuffer<T>)g_buffers[int(id)];
    }
};

}

)";